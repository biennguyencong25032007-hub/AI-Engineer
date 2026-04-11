import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import Tuple
from pathlib import Path

from src.config import Config
from src.preprocessing import Preprocessor
from src.logger import get_logger

logger = get_logger("data_loader")


# ──────────────────────────────────────────────
class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ──────────────────────────────────────────────
def resolve_path(data_path: str) -> Path:
    """
    Resolve data_path thành đường dẫn tuyệt đối.
    Tự động tìm từ thư mục chứa file này (src/) lên project root.
    """
    p = Path(data_path)
    if p.is_absolute() and p.exists():
        return p

    # Thử từ thư mục hiện tại
    if p.exists():
        return p.resolve()

    # Thử từ project root (thư mục cha của src/)
    project_root = Path(__file__).parent.parent
    candidate = project_root / data_path
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"Không tìm thấy file dữ liệu: '{data_path}'\n"
        f"Đã thử:\n"
        f"  - {Path(data_path).resolve()}\n"
        f"  - {candidate}"
    )


# ──────────────────────────────────────────────
def load_data(cfg: Config) -> Tuple[DataLoader, DataLoader, DataLoader, Preprocessor]:
    """
    Reads CSV, preprocesses, splits into train/val/test,
    returns DataLoaders and the fitted Preprocessor.
    """
    data_path = resolve_path(cfg.data.data_path)
    logger.info(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    # Drop unwanted columns
    df.drop(columns=cfg.data.drop_columns, errors="ignore", inplace=True)

    target = cfg.data.target_column
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")

    y = df[target].values.astype(np.float32)
    X_df = df.drop(columns=[target])

    # Train / temp split
    X_train_df, X_temp_df, y_train, y_temp = train_test_split(
        X_df, y,
        test_size=cfg.data.test_size + cfg.data.val_size,
        random_state=cfg.data.random_seed,
        stratify=y,
    )

    # Val / test split
    relative_val = cfg.data.val_size / (cfg.data.test_size + cfg.data.val_size)
    X_val_df, X_test_df, y_val, y_test = train_test_split(
        X_temp_df, y_temp,
        test_size=1 - relative_val,
        random_state=cfg.data.random_seed,
        stratify=y_temp,
    )

    logger.info(f"Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")

    # Fit preprocessor on training data only
    preprocessor = Preprocessor(
        numerical_columns=cfg.data.numerical_columns,
        categorical_columns=cfg.data.categorical_columns,
    )
    X_train = preprocessor.fit_transform(X_train_df, target="")
    X_val   = preprocessor.transform(X_val_df)
    X_test  = preprocessor.transform(X_test_df)

    # Update input_dim automatically
    cfg.model.input_dim = X_train.shape[1]
    logger.info(f"Input dimension: {cfg.model.input_dim}")

    # Check class balance
    pos_ratio = y_train.mean()
    logger.info(f"Positive class ratio (train): {pos_ratio:.3f}")
    if cfg.training.pos_weight is None and pos_ratio < 0.3:
        cfg.training.pos_weight = (1 - pos_ratio) / pos_ratio
        logger.warning(f"Imbalanced dataset detected. Setting pos_weight={cfg.training.pos_weight:.2f}")

    def make_loader(X, y, shuffle):
        ds = TabularDataset(X, y)
        return DataLoader(ds, batch_size=cfg.training.batch_size, shuffle=shuffle, num_workers=0)

    return (
        make_loader(X_train, y_train, shuffle=True),
        make_loader(X_val,   y_val,   shuffle=False),
        make_loader(X_test,  y_test,  shuffle=False),
        preprocessor,
    )