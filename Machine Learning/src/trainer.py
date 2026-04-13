from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from src.config import Config
from src.model import BinaryClassifier
from src.logger import get_logger

logger = get_logger("trainer")


class EarlyStopping:
    """Early stopping với restore best weights."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_loss = float("inf") if mode == "min" else float("-inf")
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if self.mode == "min":
            improved = val_loss < self.best_loss - self.min_delta
        else:
            improved = val_loss > self.best_loss + self.min_delta

        if improved:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ──────────────────────────────────────────────────────────────────────
class Trainer:
    """
    Trainer với:
    - AMP (Automatic Mixed Precision) cho tốc độ nhanh hơn
    - Gradient Accumulation để simulate larger batch sizes
    - OneCycleLR scheduler
    - Comprehensive logging
    """

    def __init__(self, model: BinaryClassifier, cfg: Config, device: torch.device):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device

        # Class weights for imbalanced data
        pos_weight = None
        if cfg.training.pos_weight is not None:
            pos_weight = torch.tensor([cfg.training.pos_weight], device=device)

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.optimizer = AdamW(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )

        self.scaler = GradScaler("cuda" if self.device.type == "cuda" else "cpu")
        self.scheduler = self._build_scheduler()
        self.early_stopping = EarlyStopping(
            patience=cfg.training.early_stopping_patience,
            mode="min",
        )

        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "learning_rate": [],
        }

        self.gradient_accumulation_steps = cfg.training.gradient_accumulation_steps
        Path(cfg.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _build_scheduler(self):
        name = self.cfg.training.lr_scheduler
        if name == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=self.cfg.training.epochs)
        elif name == "onecycle":
            return OneCycleLR(
                self.optimizer,
                max_lr=self.cfg.training.learning_rate,
                total_steps=self.cfg.training.epochs,
                pct_start=0.1,
            )
        elif name == "plateau":
            return ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
            )
        return None

    # ------------------------------------------------------------------
    def _run_epoch(self, loader: DataLoader, train: bool) -> tuple:
        self.model.train(train)
        total_loss, correct, total = 0.0, 0, 0
        accumulation_steps = self.gradient_accumulation_steps

        for batch_idx, (X, y) in enumerate(loader):
            X, y = X.to(self.device), y.to(self.device)

            with autocast("cuda" if self.device.type == "cuda" else "cpu", enabled=self.device.type == "cuda"):
                logits = self.model(X)
                loss = self.criterion(logits, y)
                loss = loss / accumulation_steps

            if train:
                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                    if self.scheduler and isinstance(self.scheduler, OneCycleLR):
                        self.scheduler.step()

            # Metrics
            total_loss += loss.item() * accumulation_steps * len(y)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == y).sum().item()
            total += len(y)

        return total_loss / total, correct / total

    # ------------------------------------------------------------------
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        best_ckpt: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        if best_ckpt is None:
            best_ckpt = os.path.join(self.cfg.training.checkpoint_dir, "best_model.pt")

        best_val_loss = float("inf")
        logger.info(
            f"Starting training | device={self.device} | "
            f"params={self.model.num_parameters:,} | "
            f"grad_accum={self.gradient_accumulation_steps}"
        )

        for epoch in range(1, self.cfg.training.epochs + 1):
            train_loss, train_acc = self._run_epoch(train_loader, train=True)
            val_loss,   val_acc   = self._run_epoch(val_loader,   train=False)

            lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler and not isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step(val_loss)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["learning_rate"].append(lr)

            # Compact logging
            improvement = "↓" if val_loss < best_val_loss else ""
            logger.info(
                f"Epoch {epoch:03d}/{self.cfg.training.epochs} | "
                f"train={train_loss:.4f}/{train_acc:.3f} | "
                f"val={val_loss:.4f}/{val_acc:.3f} | "
                f"lr={lr:.2e} {improvement}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), best_ckpt)
                logger.info(f"  ✓ Saved: {best_ckpt}")

            if self.early_stopping.step(val_loss):
                logger.info(f"Early stopping at epoch {epoch}")
                break

        self.model.load_state_dict(torch.load(best_ckpt, map_location=self.device))
        logger.info(f"Restored best model from epoch {epoch}")
        return self.history