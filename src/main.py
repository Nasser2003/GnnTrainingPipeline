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
    # --- Protection : Source folder check ---
    data_path = os.path.abspath(cfg.data.data_dir)
    if not os.path.isdir(data_path):
        log.error(f"Source folder not found : {data_path}")
        print(f"\n[FATAL ERROR] Source folder not found : {data_path}")
        print("Ensure GraphAnalysis extraction has been performed.")
        return

    # --- Load Metadata if exists ---
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
        # Read tracking URI from env var first (avoids Hydra URL escaping issues in multirun)
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", cfg.output.mlflow_tracking_uri)
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("gnn-link-prediction")

        # --- Hydra multirun gère les combinaisons, plus besoin de boucler sur des dicts ---
        graph_type = cfg.data.graph_type
        encoder_name = cfg.model.encoder
        grad_clip = cfg.model.grad_clip if cfg.model.grad_clip != 0.0 else None

        print(f"\n--- Processing Graph: {graph_type.upper()} | Run: {cfg.data.run_id} ---")
        print(f"    Source: {data_path}")

        collection = metadata.get("collection", "unknown")
        users_processed = metadata.get("users_processed", 0)
        extraction_date = metadata.get("date", "unknown")

        with mlflow.start_run(run_name=f"{graph_type}_{encoder_name}", tags={
            "graph_type": graph_type,
            "encoder": encoder_name,
            "extraction_run_id": str(cfg.data.run_id),
            "mongo_collection": str(collection),
            "users_processed": str(users_processed),
        }):
            mlflow.set_tag(
                "mlflow.note.content",
                f"GNN link prediction on {graph_type} graph | collection: {collection} | "
                f"users: {users_processed} | extraction: {extraction_date} | encoder: {encoder_name}"
            )

            if metadata:
                meta_df = pd.DataFrame([metadata])
                dataset = mlflow.data.from_pandas(
                    meta_df,
                    source=metadata_path,
                    name=f"{collection}_{cfg.data.run_id[:8]}",
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
            })

            with mlflow.start_span(name="pipeline") as pipeline_span:
                pipeline_span.set_attribute("graph_type", graph_type)
                pipeline_span.set_attribute("encoder", encoder_name)
                pipeline_span.set_attribute("collection", collection)

                builder = GraphBuilder(
                    data_dir=cfg.data.data_dir,
                    graph_dir=cfg.output.graph_dir,
                    graph_type=graph_type,
                    load_graph_if_exists=cfg.data.load_graph_if_exists
                )
                pt_path = builder.build_and_save()

                processor = GNNDataProcessor(
                    pt_path,
                    train_ratio=cfg.data.split[0],
                    val_ratio=cfg.data.split[1],
                    test_ratio=cfg.data.split[2],
                    batch_size=cfg.model.batch_size,
                    num_neighbors=list(cfg.model.num_neighbors),
                )
                train_data, train_loader, val_data, test_data, full_pos, in_ch, edge_dim = processor.prepare_data()

                # ---- Baselines ----
                if cfg.model.run_baselines:
                    print("\n  >> Evaluating Baselines...")
                    base_evaluator = Evaluator()

                    cosine_scorer = CosineBaselineScorer()
                    base_evaluator.evaluate_metrics_master(
                        "Cosine", test_data, neg_ratio=1.0,
                        scorer=cosine_scorer, threshold=0.5,
                        full_pos_edges=full_pos)

                    mlp_model = RawMLPConcatPredictor(in_dim=in_ch, hidden=cfg.model.hidden_dim)
                    mlp_trainer = BaselineTrainer(mlp_model, lr=cfg.model.learning_rate)
                    trained_mlp = mlp_trainer.train(train_data, val_data, epochs=cfg.model.num_epochs)
                    mlp_scorer = MLPBaselineScorer(trained_mlp)
                    base_evaluator.evaluate_metrics_master(
                        "MLP", test_data, neg_ratio=1.0,
                        scorer=mlp_scorer, threshold=0.5,
                        full_pos_edges=full_pos)
                else:
                    print("\n  >> Skipping Baselines (run_baselines=false)")

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
                    edge_dim_retweet=getattr(train_data, 'edge_dim_retweet', 3),
                    edge_dim_reply=getattr(train_data, 'edge_dim_reply', 3),
                    edge_dim_mention=getattr(train_data, 'edge_dim_mention', 3),
                    early_stopping_patience=cfg.model.early_stopping_patience,
                )

                model = trainer.train(train_loader, val_data)

                gnn_eval = GNNEvaluator(model=model)
                gnn_eval.evaluate(
                    val_data, test_data,
                    neg_ratios=list(cfg.evaluation.neg_ratios),
                    full_pos_edges=full_pos,
                )

            mlflow.pytorch.log_model(
                model,
                name="model",
                registered_model_name=encoder_name,
            )

    finally:
        if gpu_id is not None:
            release_gpu(gpu_id, lock_file)


if __name__ == '__main__':
    main()