from dataclasses import dataclass
from typing import Dict, List

from hydra.core.config_store import ConfigStore


@dataclass
class DataConfig:
    data_dir: str
    graph_type: Dict[str, bool]
    train_ratio: float
    val_ratio: float
    test_ratio: float


@dataclass
class ModelConfig:
    encoder: Dict[str, bool]
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
    mlflow_tracking_uri: str


@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    output: OutputConfig


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
