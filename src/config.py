from __future__ import annotations
from dataclasses import dataclass
from typing import List
from hydra.core.config_store import ConfigStore


@dataclass
class DataConfig:
    base_dir: str
    run_id: str
    data_dir: str
    load_graph_if_exists: bool
    graph_type: str
    split: List[float]

@dataclass
class ModelConfig:
    encoder: str
    decoder: str
    num_epochs: int
    learning_rate: float
    weight_decay: float
    hidden_dim: int
    embed_dim: int
    dropout: float
    neg_ratio: float
    decoder_hidden: int
    decoder_dropout: float
    scheduler: bool
    grad_clip: float
    early_stopping_patience: int
    seed: int
    fusion: str
    batch_size: int
    num_neighbors: List[int]


@dataclass
class EvaluationConfig:
    neg_ratios: List[int]


@dataclass
class OutputConfig:
    output_dir: str
    log_dir: str
    graph_dir: str
    mlflow_tracking_uri: str
    experiment_name: str


@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    output: OutputConfig


cs = ConfigStore.instance()
cs.store(name="config_schema", node=Config)
