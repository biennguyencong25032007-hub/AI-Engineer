from typing import Dict

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from torch.utils.data import DataLoader

from src.model import BinaryClassifier
from src.logger import get_logger

logger = get_logger("evaluator")


class Evaluator:
    def __init__(self, model: BinaryClassifier, device: torch.device, threshold: float = 0.5):
        self.model = model
        self.device = device
        self.threshold = threshold

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, loader: DataLoader) -> tuple:
        self.model.eval()
        all_probs, all_labels = [], []

        for X, y in loader:
            X = X.to(self.device)
            probs = self.model.predict_proba(X).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y.numpy())

        return np.array(all_probs), np.array(all_labels)

    # ------------------------------------------------------------------
    def evaluate(self, loader: DataLoader, split: str = "test") -> Dict[str, float]:
        probs, labels = self.predict(loader)
        preds = (probs >= self.threshold).astype(int)

        metrics = {
            "accuracy" : accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall"   : recall_score(labels, preds, zero_division=0),
            "f1"       : f1_score(labels, preds, zero_division=0),
            "roc_auc"  : roc_auc_score(labels, probs),
        }

        logger.info(f"── {split.upper()} METRICS ──────────────────────")
        for k, v in metrics.items():
            logger.info(f"  {k:<12}: {v:.4f}")

        cm = confusion_matrix(labels, preds)
        logger.info(f"  Confusion matrix:\n{cm}")
        logger.info(f"\n{classification_report(labels, preds, target_names=['Neg','Pos'])}")

        return metrics

    # ------------------------------------------------------------------
    def find_best_threshold(self, loader: DataLoader) -> float:
        """Sweep thresholds on val set and pick the one with highest F1."""
        probs, labels = self.predict(loader)
        best_thresh, best_f1 = 0.5, 0.0
        for t in np.arange(0.1, 0.91, 0.01):
            preds = (probs >= t).astype(int)
            f1 = f1_score(labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, t
        logger.info(f"Best threshold: {best_thresh:.2f}  (F1={best_f1:.4f})")
        self.threshold = best_thresh
        return best_thresh