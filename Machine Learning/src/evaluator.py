from __future__ import annotations

from typing import Dict
from pathlib import Path

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
    precision_recall_curve,
    average_precision_score,
)
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.model import BinaryClassifier
from src.logger import get_logger

logger = get_logger("evaluator")


class Evaluator:
    """
    Comprehensive evaluation với:
    - Per-class metrics
    - Threshold analysis (Precision-Recall curve)
    - Calibration check
    - Metrics caching
    """

    def __init__(self, model: BinaryClassifier, device: torch.device, threshold: float = 0.5):
        self.model = model
        self.device = device
        self.threshold = threshold
        self._cache: Dict[str, tuple] = {}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, loader: DataLoader) -> tuple:
        """Cache predictions để tránh tính lại."""
        key = id(loader)
        if key not in self._cache:
            self.model.eval()
            all_probs, all_labels = [], []
            for X, y in loader:
                X = X.to(self.device)
                probs = self.model.predict_proba(X).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(y.numpy())
            self._cache[key] = (np.array(all_probs), np.array(all_labels))
        return self._cache[key]

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
            "avg_precision": average_precision_score(labels, probs),
        }

        # Per-class metrics
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        logger.info(f"\n══ {split.upper()} METRICS ══════════════════════════")
        for k, v in metrics.items():
            logger.info(f"  {k:<18}: {v:.4f}")

        cm = confusion_matrix(labels, preds)
        logger.info(f"\n  Confusion Matrix:\n{cm}")
        logger.info(f"\n{classification_report(labels, preds, target_names=['Neg','Pos'], zero_division=0)}")

        return metrics

    # ------------------------------------------------------------------
    def find_best_threshold(self, loader: DataLoader) -> float:
        """Tìm threshold tối ưu dựa trên F1 score."""
        probs, labels = self.predict(loader)
        best_thresh, best_f1 = 0.5, 0.0

        for t in np.arange(0.05, 0.96, 0.01):
            preds = (probs >= t).astype(int)
            f1 = f1_score(labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, t

        logger.info(f"Best threshold: {best_thresh:.2f} (F1={best_f1:.4f})")
        self.threshold = best_thresh
        return best_thresh

    # ------------------------------------------------------------------
    def plot_pr_curve(self, loader: DataLoader, save_path: str | Path = None) -> None:
        """Vẽ Precision-Recall curve."""
        probs, labels = self.predict(loader)
        precision, recall, _ = precision_recall_curve(labels, probs)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(recall, precision, "b-", lw=2, label="PR curve")
        ax.axhline(y=labels.mean(), color="gray", linestyle="--", label="Baseline")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"PR Curve (AP={average_precision_score(labels, probs):.3f})")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"PR curve saved → {save_path}")
        plt.close()

    # ------------------------------------------------------------------
    def clear_cache(self) -> None:
        """Xóa predictions cache."""
        self._cache.clear()