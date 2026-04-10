import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from typing import Tuple, List, Optional
import joblib
import os

from src.logger import get_logger

logger = get_logger("preprocessing")


class Preprocessor:
    """Handles all data preprocessing: imputation, encoding, scaling."""

    def __init__(
        self,
        numerical_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
        drop_columns: Optional[List[str]] = None,
    ):
        self.numerical_columns = numerical_columns or []
        self.categorical_columns = categorical_columns or []
        self.drop_columns = drop_columns or []

        self.num_imputer = SimpleImputer(strategy="median")
        self.cat_imputer = SimpleImputer(strategy="most_frequent")
        self.scaler = StandardScaler()
        self.label_encoders: dict = {}
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
            filled = df[col].fillna("__missing__").astype(str)
            le.fit(filled)
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
            parts.append(num_data)

        for col in self.categorical_columns:
            le = self.label_encoders[col]
            filled = df[col].fillna("__missing__").astype(str)
            # Handle unseen categories
            filled = filled.apply(lambda x: x if x in le.classes_ else "__missing__")
            encoded = le.transform(filled).reshape(-1, 1).astype(np.float32)
            parts.append(encoded)

        return np.hstack(parts).astype(np.float32)

    def fit_transform(self, df: pd.DataFrame, target: str) -> np.ndarray:
        return self.fit(df, target).transform(df)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Preprocessor saved → {path}")

    @staticmethod
    def load(path: str) -> "Preprocessor":
        return joblib.load(path)