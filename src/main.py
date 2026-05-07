import logging
import random

import hydra
import mlflow
import numpy as np
import torch

from config import Config
from gnn.evaluation.CosineBaselineScorer import CosineBaselineScorer
from gnn.evaluation.Evaluator import Evaluator
from gnn.evaluation.GNNEvaluator import GNNEvaluator
from gnn.evaluation.MLPBaselineScorer import MLPBaselineScorer
from gnn.model.RawMLPConcatPredictor import RawMLPConcatPredictor
from gnn.preprocess.GNNDataProcessor import GNNDataProcessor
from gnn.preprocess.GraphBuilder import GraphBuilder
from gnn.train.BaselineTrainer import BaselineTrainer
from gnn.train.GNNTraining import GNNTraining

log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@hydra.main(config_path="../conf", config_name="base_config", version_base=None)
def main(cfg: Config) -> None:
    set_seed(cfg.model.seed)
    log.info(f"Seed set to {cfg.model.seed}")

    mlflow.set_tracking_uri(cfg.output.mlflow_tracking_uri)
    mlflow.set_experiment("gnn-link-prediction")

    enabled_graphs = [k for k, v in cfg.data.graph_type.items() if v]
    enabled_encoders = [k for k, v in cfg.model.encoder.items() if v]

    for graph_type in enabled_graphs:
        log.info(f"\n--- Processing Graph: {graph_type.upper()} ---")

        with mlflow.start_run(run_name=graph_type, tags={"graph_type": graph_type}):
            builder = GraphBuilder(data_dir=cfg.data.data_dir, graph_type=graph_type)
            pt_path = builder.build_and_save()

            processor = GNNDataProcessor(
                pt_path,
                train_ratio=cfg.data.train_ratio,
                val_ratio=cfg.data.val_ratio,
                test_ratio=cfg.data.test_ratio,
                batch_size=cfg.model.batch_size,
                num_neighbors=list(cfg.model.num_neighbors),
            )
            train_data, train_loader, val_data, test_data, full_pos, in_ch, edge_dim = processor.prepare_data()

            log.info(">> Evaluating Baselines")
            base_evaluator = Evaluator()

            with mlflow.start_run(run_name="cosine_baseline", nested=True):
                mlflow.log_param("model", "cosine")
                cosine_scorer = CosineBaselineScorer()
                base_evaluator.evaluate_metrics_master(
                    "Cosine", test_data, neg_ratio=1.0,
                    scorer=cosine_scorer, threshold=0.5,
                    full_pos_edges=full_pos)

            with mlflow.start_run(run_name="mlp_baseline", nested=True):
                mlflow.log_params({
                    "model": "mlp_baseline",
                    "hidden_dim": cfg.model.hidden_dim,
                    "learning_rate": cfg.model.learning_rate,
                    "num_epochs": cfg.model.num_epochs,
                })
                mlp_model = RawMLPConcatPredictor(in_dim=in_ch, hidden=cfg.model.hidden_dim)
                mlp_trainer = BaselineTrainer(mlp_model, lr=cfg.model.learning_rate)
                trained_mlp = mlp_trainer.train(train_data, val_data, epochs=cfg.model.num_epochs)
                mlp_scorer = MLPBaselineScorer(trained_mlp)
                base_evaluator.evaluate_metrics_master(
                    "MLP", test_data, neg_ratio=1.0,
                    scorer=mlp_scorer, threshold=0.5,
                    full_pos_edges=full_pos)

            for encoder_name in enabled_encoders:
                log.info(f">> Training GNN: {encoder_name.upper()}")

                with mlflow.start_run(run_name=encoder_name, nested=True):
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
                        "grad_clip": cfg.model.grad_clip,
                        "fusion": cfg.model.fusion,
                        "early_stopping_patience": cfg.model.early_stopping_patience,
                        "batch_size": cfg.model.batch_size,
                        "seed": cfg.model.seed,
                    })

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
                        grad_clip=cfg.model.grad_clip,
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


if __name__ == '__main__':
    main()
