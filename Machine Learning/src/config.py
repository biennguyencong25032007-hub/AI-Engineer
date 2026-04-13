from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    data_path: str = "data/sample_data.csv"
    target_column: str = "target"
    test_size: float = 0.2
    val_size: float = 0.1
    random_seed: int = 42
    categorical_columns: List[str] = field(default_factory=list)
    numerical_columns: List[str] = field(default_factory=list)
    drop_columns: List[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    input_dim: int = 64
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout_rate: float = 0.3
    batch_norm: bool = True
    residual_blocks: int = 0      # Số residual blocks sau hidden layers
    weight_init: str = "kaiming"  # "kaiming" | "xavier" | "orthogonal"


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10
    lr_scheduler: str = "cosine"       # "cosine" | "onecycle" | "plateau" | "none"
    pos_weight: Optional[float] = None  # class imbalance weight
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    gradient_accumulation_steps: int = 1  # Simulate larger batch sizes


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    device: str = "auto"   # "auto" | "cpu" | "cuda"
    experiment_name: str = "binary_classification"