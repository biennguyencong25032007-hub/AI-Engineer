import numpy as np
import pandas as pd
import torch

from src.config import Config
from src.model import BinaryClassifier
from src.preprocessing import Preprocessor
from src.logger import get_logger

logger = get_logger("predictor")


class Predictor:
    """Load a trained model + preprocessor and run inference on new data."""

    def __init__(
        self,
        model: BinaryClassifier,
        preprocessor: Preprocessor,
        device: torch.device,
        threshold: float = 0.5,
    ):
        self.model = model.to(device).eval()
        self.preprocessor = preprocessor
        self.device = device
        self.threshold = threshold

    # ------------------------------------------------------------------
    @classmethod
    def from_checkpoint(cls, cfg: Config, ckpt_path: str, preprocessor_path: str) -> "Predictor":
        device = _resolve_device(cfg.device)
        preprocessor = Preprocessor.load(preprocessor_path)

        model = BinaryClassifier(cfg.model)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        logger.info(f"Model loaded from {ckpt_path}")
        return cls(model, preprocessor, device)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return probability of positive class for each row."""
        X = self.preprocessor.transform(df)
        tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        probs = self.model.predict_proba(tensor).cpu().numpy()
        return probs

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return binary predictions (0 / 1)."""
        return (self.predict_proba(df) >= self.threshold).astype(int)

    # ------------------------------------------------------------------
    def predict_single(self, record: dict) -> dict:
        """Predict for a single record (dict of feature → value)."""
        df = pd.DataFrame([record])
        prob = float(self.predict_proba(df)[0])
        label = int(prob >= self.threshold)
        return {"probability": prob, "prediction": label}


# ──────────────────────────────────────────────
def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)