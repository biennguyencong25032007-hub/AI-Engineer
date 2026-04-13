from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from typing import Tuple, List, Optional, Dict
import joblib
from pathlib import Path
from pathlib import Path

from src.logger import get_logger

logger = get_logger("preprocessing")


class Preprocessor:
    """Handles all data preprocessing: imputation, encoding, scaling."""

    def __init__(
        self,
        numerical_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
        drop_columns: Optional[List[str]] = None,
        unknown_category_strategy: str = "encode",  # "encode" | "ignore"
    ):
        self.numerical_columns = numerical_columns or []
        self.categorical_columns = categorical_columns or []
        self.drop_columns = drop_columns or []
        self.unknown_category_strategy = unknown_category_strategy

        self.num_imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_names: List[str] = []
        self._fitted = False

    # ------------------------------------------------------------------
    def _auto_detect_columns(self, df: pd.DataFrame, target: str) -> None:
        """Auto-detect numerical / categorical columns if not provided."""
        cols = [c for c in df.columns if c != target and c not in self.drop_columns]
        if not self.numerical_columns:
            self.numerical_columns = df[cols].select_dtypes(include=np.number).columns.tolist()
        if not self.categorical_columns:
            self.categorical_columns = df[cols].select_dtypes(exclude=np.number).columns.tolist()

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, target: str) -> "Preprocessor":
        self._auto_detect_columns(df, target)
        logger.info(f"Numerical columns  : {self.numerical_columns}")
        logger.info(f"Categorical columns: {self.categorical_columns}")

        if self.numerical_columns:
            self.num_imputer.fit(df[self.numerical_columns])
            self.scaler.fit(df[self.numerical_columns])

        for col in self.categorical_columns:
            le = LabelEncoder()
            # Thêm "__unknown__" để xử lý unseen categories
            categories = df[col].fillna("__missing__").astype(str).unique().tolist()
            if self.unknown_category_strategy == "encode":
                categories = ["__unknown__"] + categories
            le.fit(categories)
            self.label_encoders[col] = le

        # Record final feature order
        self.feature_names = self.numerical_columns + self.categorical_columns
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        assert self._fitted, "Call fit() before transform()."
        parts = []

        if self.numerical_columns:
            num_data = self.num_imputer.transform(df[self.numerical_columns])
            num_data = self.scaler.transform(num_data)
            parts.append(num_data.astype(np.float32))

        for col in self.categorical_columns:
            le = self.label_encoders[col]
            filled = df[col].fillna("__missing__").astype(str)

            if self.unknown_category_strategy == "encode":
                # Map unseen categories to "__unknown__"
                known = set(le.classes_) - {"__unknown__"}
                filled = filled.apply(lambda x: x if x in known else "__unknown__")

            encoded = le.transform(filled).reshape(-1, 1).astype(np.float32)
            parts.append(encoded)

        return np.hstack(parts).astype(np.float32)

    def fit_transform(self, df: pd.DataFrame, target: str) -> np.ndarray:
        return self.fit(df, target).transform(df)

    # ------------------------------------------------------------------
    @property
    def num_features(self) -> int:
        return len(self.numerical_columns)

    @property
    def cat_features(self) -> int:
        return len(self.categorical_columns)

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Preprocessor saved → {path}")

    @staticmethod
    def load(path: str | Path) -> "Preprocessor":
        return joblib.load(path)