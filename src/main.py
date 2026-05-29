import logging
import random
import os
import json
import numpy as np
import pandas as pd
import torch
import hydra
import mlflow
import mlflow.pytorch
import time

from config import Config
from evaluation.CosineBaselineScorer import CosineBaselineScorer
from evaluation.Evaluator import Evaluator
from evaluation.GNNEvaluator import GNNEvaluator
from evaluation.MLPBaselineScorer import MLPBaselineScorer
from model.RawMLPConcatPredictor import RawMLPConcatPredictor
from preprocess.GNNDataProcessor import GNNDataProcessor
from preprocess.GraphBuilder import GraphBuilder
from train.BaselineTrainer import BaselineTrainer
from train.GNNTraining import GNNTraining
from utils.gpu_manager import acquire_gpu, release_gpu
import warnings

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: Config) -> None:
    # Remove warnings
    import warnings
    warnings.filterwarnings("ignore")
    logging.getLogger("mlflow.pytorch").setLevel(logging.ERROR)
    logging.getLogger("mlflow.utils.requirements_utils").setLevel(logging.ERROR)

    # --- Pre-run: verify extraction data directory and required files ---
    data_path = os.path.abspath(cfg.data.data_dir)
    log.info(f"Checking extraction data directory: {data_path}")

    if not os.path.isdir(data_path):
        raise FileNotFoundError(
            f"\n[FATAL] Extraction directory not found: {data_path}\n"
            f"  run_id  : {cfg.data.run_id}\n"
            f"  base_dir: {cfg.data.base_dir}\n"
            f"Run the GraphAnalysis extraction workflow first, then update "
            f"data.run_id in your config (or pass it as a Hydra override)."
        )

    # Check that key extraction outputs are present
    REQUIRED_FILES = ["user_features", "edges_retweet", "edges_reply", "edges_mention"]
    SUPPORTED_EXTS = [".parquet", ".feather", ".csv"]
    missing = []
    for fname in REQUIRED_FILES:
        found = any(
            os.path.exists(os.path.join(data_path, fname + ext))
            for ext in SUPPORTED_EXTS
        )
        if not found:
            missing.append(fname)

    if missing:
        present = os.listdir(data_path)
        raise FileNotFoundError(
            f"\n[FATAL] Extraction directory exists but is incomplete: {data_path}\n"
            f"  Missing files : {missing}\n"
            f"  Present files : {present}\n"
            f"The extraction may have failed or is still running."
        )

    log.info(f"Extraction directory OK — run_id={cfg.data.run_id}")

    # --- Load Metadata file {metadata.json} ---
    metadata = {}
    metadata_path = os.path.join(data_path, "metadata.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            log.info(f"Loaded metadata from {metadata_path}")
        except Exception as e:
            log.warning(f"Could not load metadata: {e}")

    set_seed(cfg.model.seed)
    log.info(f"Seed set to {cfg.model.seed}")

    # --- GPU Acquisition for Multi-GPU Support ---
    gpu_id, lock_file = acquire_gpu()
    if gpu_id is not None:
        log.info(f"Using GPU: {gpu_id}")
        torch.cuda.set_device(gpu_id)
        # Force MLflow to use the correct device tag if needed
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        # --- MLflow Setup ---
        log.info(f"Setting up MLflow tracking URI: {os.getenv('MLFLOW_TRACKING_URI', cfg.output.mlflow_tracking_uri)}")
        # Read tracking URI from env var first (avoids Hydra URL escaping issues in multirun)
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", cfg.output.mlflow_tracking_uri)
        mlflow.set_tracking_uri(mlflow_uri)
        log.info("Connecting to MLflow and setting experiment...")
        experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", cfg.output.experiment_name)
        try:
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            log.warning(f"set_experiment failed ({e}), attempting to recover...")
            client = mlflow.tracking.MlflowClient()
            exp = client.get_experiment_by_name(experiment_name)
            if exp is not None and exp.lifecycle_stage == 'deleted':
                client.restore_experiment(exp.experiment_id)
                log.info(f"Restored deleted experiment '{experiment_name}'")
            elif exp is None:
                client.create_experiment(experiment_name)
                log.info(f"Created new experiment '{experiment_name}'")
            mlflow.set_experiment(experiment_name)
        log.info("MLflow connection successful!")

        # --- Dynamic check for broken legacy artifact location ---
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is not None:
            art_loc = exp.artifact_location
            if mlflow_uri.startswith("http") and (art_loc.startswith("/") or art_loc.startswith("file:")):
                log.warning(
                    f"\n========================================================================\n"
                    f"[WARNING] MLflow experiment '{experiment_name}' (ID: {exp.experiment_id}) has a local artifact location: '{art_loc}' "
                    f"but tracking URI is remote: '{mlflow_uri}'.\n"
                    f"This experiment was created under a legacy configuration and will cause log_model to fail on the runner.\n"
                    f"RECOMMENDED FIX: Rename your EXPERIMENT_NAME (e.g. to 'gnn-link-prediction-v4') to create a clean experiment.\n"
                    f"========================================================================\n"
                )

        # --- Hydra multirun
        graph_type = cfg.data.graph_type
        encoder_name = cfg.model.encoder
        grad_clip = cfg.model.grad_clip if cfg.model.grad_clip != 0.0 else None

        log.info(f"--- Processing Graph: {graph_type.upper()} | Run: {cfg.data.run_id} ---")
        print(f"    Source: {data_path}")

        collection = metadata.get("collection", "unknown")
        users_processed = metadata.get("users_processed", 0)
        extraction_date = metadata.get("date", "unknown")
        communities_file = metadata.get("communities", None)
        is_community_run = communities_file is not None
        print(f"Communities file: {is_community_run}")
        max_comm = getattr(cfg.data, "max_communities", None)
        
        base_run_name = f"{graph_type}_{encoder_name}"
        if is_community_run:
            run_name = f"{base_run_name}_comm"
            if max_comm is not None:
                run_name += f"_{max_comm}"
        else:
            run_name = base_run_name

        tags = {
            "graph_type": graph_type,
            "encoder": encoder_name,
            "extraction_run_id": str(cfg.data.run_id),
            "mongo_collection": str(collection),
            "users_processed": str(users_processed),
            "is_community_run": str(is_community_run),
            "communities_file": str(communities_file) if is_community_run else "None",
            "allow_self_loops": str(getattr(cfg.data, "allow_self_loops", False)),
        }
        if is_community_run and max_comm is not None:
            tags["max_communities"] = str(max_comm)

        with mlflow.start_run(run_name=run_name, tags=tags):
            run_start_time = time.time()
            note = (
                f"GNN link prediction on {graph_type} graph | collection: {collection} | "
                f"users: {users_processed} | extraction: {extraction_date} | encoder: {encoder_name} | "
                f"mode: {'community (' + str(communities_file) + ')' if is_community_run else 'global'} | "
                f"self_loops: {getattr(cfg.data, 'allow_self_loops', False)}"
            )
            if is_community_run and max_comm is not None:
                note += f" | max_communities: {max_comm}"
                
            mlflow.set_tag("mlflow.note.content", note)

            if metadata:
                if is_community_run and max_comm is not None:
                    metadata["max_communities"] = max_comm

                meta_df = pd.DataFrame([metadata])
                dataset_name = f"{collection}"
                if is_community_run:
                    dataset_name += "_comm"
                    if max_comm is not None:
                        dataset_name += f"_{max_comm}"
                
                dataset = mlflow.data.from_pandas(
                    meta_df,
                    source=metadata_path,
                    name=dataset_name,
                )
                mlflow.log_input(dataset, context="metadata")

            mlflow.log_params({
                "encoder": encoder_name,
                "graph_type": graph_type,
                "decoder": cfg.model.decoder,
                "hidden_dim": cfg.model.hidden_dim,
                "embed_dim": cfg.model.embed_dim,
                "learning_rate": cfg.model.learning_rate,
                "num_epochs": cfg.model.num_epochs,
                "weight_decay": cfg.model.weight_decay,
                "dropout": cfg.model.dropout,
                "neg_ratio": cfg.model.neg_ratio,
                "decoder_hidden": cfg.model.decoder_hidden,
                "decoder_dropout": cfg.model.decoder_dropout,
                "scheduler": cfg.model.scheduler,
                "grad_clip": grad_clip,
                "fusion": cfg.model.fusion,
                "early_stopping_patience": cfg.model.early_stopping_patience,
                "batch_size": cfg.model.batch_size,
                "seed": cfg.model.seed,
                "allow_self_loops": getattr(cfg.data, "allow_self_loops", False),
            })

            with mlflow.start_span(name="pipeline") as pipeline_span:
                pipeline_span.set_attribute("graph_type", graph_type)
                pipeline_span.set_attribute("encoder", encoder_name)
                pipeline_span.set_attribute("collection", collection)

                if is_community_run:
                    from preprocess.CommunityGraphBuilder import CommunityGraphBuilder
                    from preprocess.CommunityGNNDataProcessor import CommunityGNNDataProcessor
                    builder = CommunityGraphBuilder(
                        data_dir=cfg.data.data_dir,
                        graph_dir=cfg.output.graph_dir,
                        graph_type=graph_type,
                        load_graph_if_exists=cfg.data.load_graph_if_exists,
                        min_nodes=cfg.data.get("community_min_nodes", 20),
                        min_edges=cfg.data.get("community_min_edges", 20)
                    )
                else:
                    builder = GraphBuilder(
                        data_dir=cfg.data.data_dir,
                        graph_dir=cfg.output.graph_dir,
                        graph_type=graph_type,
                        load_graph_if_exists=cfg.data.load_graph_if_exists
                    )
                pt_path = builder.build_and_save()

                if is_community_run:
                    processor = CommunityGNNDataProcessor(
                        pt_path,
                        train_ratio=cfg.data.split[0],
                        val_ratio=cfg.data.split[1],
                        test_ratio=cfg.data.split[2],
                        batch_size=cfg.model.batch_size,
                        num_neighbors=list(cfg.model.num_neighbors),
                        graph_split=cfg.data.get("community_graph_split", True),
                        min_edges_for_split=cfg.data.get("community_min_edges_split", 20),
                        max_communities=cfg.data.get("max_communities", None),
                        allow_self_loops=getattr(cfg.data, "allow_self_loops", False)
                    )
                else:
                    processor = GNNDataProcessor(
                        pt_path,
                        train_ratio=cfg.data.split[0],
                        val_ratio=cfg.data.split[1],
                        test_ratio=cfg.data.split[2],
                        batch_size=cfg.model.batch_size,
                        num_neighbors=list(cfg.model.num_neighbors),
                        allow_self_loops=getattr(cfg.data, "allow_self_loops", False)
                    )
                train_data, train_loader, val_data, test_data, full_pos, in_ch, edge_dim, edge_dim_retweet, edge_dim_reply, edge_dim_mention = processor.prepare_data()

                # ---- Models & Baselines execution ----
                if encoder_name == 'cosine':
                    print("\n  >> Running Cosine Baseline...")
                    base_evaluator = Evaluator()
                    cosine_scorer = CosineBaselineScorer()
                    
                    # Evaluate on validation to find the best threshold
                    best_thr = base_evaluator.find_best_threshold(val_data, neg_ratio=1.0, scorer=cosine_scorer, full_pos_edges=full_pos)
                    
                    # Evaluate on test for all neg_ratios
                    for nr in list(cfg.evaluation.neg_ratios):
                        base_evaluator.evaluate_metrics_master(
                            "Cosine", test_data, neg_ratio=nr,
                            scorer=cosine_scorer, threshold=best_thr,
                            full_pos_edges=full_pos
                        )
                    model = cosine_scorer

                elif encoder_name == 'mlp':
                    print("\n  >> Running MLP Baseline...")
                    base_evaluator = Evaluator()
                    mlp_model = RawMLPConcatPredictor(in_dim=in_ch, hidden=cfg.model.hidden_dim)
                    mlp_trainer = BaselineTrainer(mlp_model, lr=cfg.model.learning_rate)
                    trained_mlp = mlp_trainer.train(train_data, val_data, epochs=cfg.model.num_epochs)
                    mlp_scorer = MLPBaselineScorer(trained_mlp)
                    
                    # Evaluate on validation to find the best threshold
                    best_thr = base_evaluator.find_best_threshold(val_data, neg_ratio=1.0, scorer=mlp_scorer, full_pos_edges=full_pos)
                    
                    # Evaluate on test for all neg_ratios
                    for nr in list(cfg.evaluation.neg_ratios):
                        base_evaluator.evaluate_metrics_master(
                            "MLP", test_data, neg_ratio=nr,
                            scorer=mlp_scorer, threshold=best_thr,
                            full_pos_edges=full_pos
                        )
                    model = trained_mlp

                else:
                    # ---- GNN Training + Evaluation ----
                    print(f"\n  >> Training GNN: {encoder_name.upper()}")

                    trainer = GNNTraining(
                        output_dir=cfg.output.output_dir,
                        encoder='late-fuse' if graph_type == 'late_fuse' else encoder_name,
                        decoder=cfg.model.decoder,
                        in_channels=in_ch,
                        hidden_dim=cfg.model.hidden_dim,
                        embed_dim=cfg.model.embed_dim,
                        edge_dim=edge_dim,
                        learning_rate=cfg.model.learning_rate,
                        num_epochs=cfg.model.num_epochs,
                        weight_decay=cfg.model.weight_decay,
                        dropout=cfg.model.dropout,
                        neg_ratio=cfg.model.neg_ratio,
                        decoder_hidden=cfg.model.decoder_hidden,
                        decoder_dropout=cfg.model.decoder_dropout,
                        scheduler=cfg.model.scheduler,
                        grad_clip=grad_clip,
                        fusion=cfg.model.fusion,
                        late_fuse_base_encoder=encoder_name if graph_type == 'late_fuse' else None,
                        edge_dim_retweet=edge_dim_retweet,
                        edge_dim_reply=edge_dim_reply,
                        edge_dim_mention=edge_dim_mention,
                        early_stopping_patience=cfg.model.early_stopping_patience,
                    )

                    model = trainer.train(train_loader, val_data)

                    gnn_eval = GNNEvaluator(model=model)
                    gnn_eval.evaluate(
                        val_data, test_data,
                        neg_ratios=list(cfg.evaluation.neg_ratios),
                        full_pos_edges=full_pos,
                    )

            if model is not None:
                mlflow.pytorch.log_model(
                    model,
                    name="model",
                    registered_model_name=encoder_name,
                )
            
            run_duration = time.time() - run_start_time
            mlflow.log_metric("Duration", round(run_duration, 2))

    finally:
        if gpu_id is not None:
            release_gpu(gpu_id, lock_file)


if __name__ == '__main__':
    main()