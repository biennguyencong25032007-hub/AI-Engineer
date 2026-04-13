from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import Config
from src.model import BinaryClassifier
from src.preprocessing import Preprocessor
from src.logger import get_logger

logger = get_logger("predictor")


class Predictor:
    """
    Load trained model + preprocessor và chạy inference trên new data.

    Features:
    - Batch prediction với chunk processing
    - MPS (Apple Silicon) support
    - Confidence scoring
    """

    def __init__(
        self,
        model: BinaryClassifier,
        preprocessor: Preprocessor,
        device: torch.device,
        threshold: float = 0.5,
        batch_size: int = 256,
    ):
        self.model = model.to(device).eval()
        self.preprocessor = preprocessor
        self.device = device
        self.threshold = threshold
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    @classmethod
    def from_checkpoint(
        cls,
        cfg: Config,
        ckpt_path: str | Path,
        preprocessor_path: str | Path,
        threshold: float = 0.5,
        batch_size: int = 256,
    ) -> "Predictor":
        """Load model từ checkpoint và preprocessor."""
        device = _resolve_device(cfg.device)
        preprocessor = Preprocessor.load(preprocessor_path)

        model = BinaryClassifier(cfg.model)
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

        logger.info(f"Model loaded | device={device} | checkpoint={ckpt_path}")
        return cls(model, preprocessor, device, threshold, batch_size)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return probability của positive class cho từng row."""
        X = self.preprocessor.transform(df)
        probs = []

        for i in range(0, len(X), self.batch_size):
            chunk = torch.tensor(X[i:i + self.batch_size], dtype=torch.float32).to(self.device)
            prob_chunk = self.model.predict_proba(chunk).cpu().numpy()
            probs.append(prob_chunk)

        return np.concatenate(probs)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return binary predictions (0 / 1)."""
        return (self.predict_proba(df) >= self.threshold).astype(int)

    # ------------------------------------------------------------------
    def predict_with_confidence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return predictions kèm confidence scores."""
        probs = self.predict_proba(df)

        return pd.DataFrame({
            "probability": probs,
            "prediction": (probs >= self.threshold).astype(int),
            "confidence": np.where(probs >= self.threshold, probs, 1 - probs),
        })

    # ------------------------------------------------------------------
    def predict_single(self, record: dict) -> dict:
        """Predict cho một record (dict of feature → value)."""
        df = pd.DataFrame([record])
        prob = float(self.predict_proba(df)[0])
        label = int(prob >= self.threshold)

        return {
            "probability": prob,
            "prediction": label,
            "confidence": prob if label == 1 else 1 - prob,
        }

    # ------------------------------------------------------------------
    def save_predictions(
        self,
        df: pd.DataFrame,
        output_path: str | Path,
    ) -> None:
        """Lưu predictions ra CSV."""
        result = self.predict_with_confidence(df)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        logger.info(f"Predictions saved → {output_path} ({len(result)} rows)")


# ──────────────────────────────────────────────
def _resolve_device(device_str: str) -> torch.device:
    """Resolve device string to torch.device, auto-detect GPU."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)