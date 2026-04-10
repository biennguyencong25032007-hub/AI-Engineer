import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader

from src.config import Config
from src.model import BinaryClassifier
from src.logger import get_logger

logger = get_logger("trainer")


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ──────────────────────────────────────────────────────────────────────
class Trainer:
    def __init__(self, model: BinaryClassifier, cfg: Config, device: torch.device):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device

        pos_weight = None
        if cfg.training.pos_weight is not None:
            pos_weight = torch.tensor([cfg.training.pos_weight], device=device)

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.optimizer = Adam(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )
        self.scheduler = self._build_scheduler()
        self.early_stopping = EarlyStopping(patience=cfg.training.early_stopping_patience)
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_acc": []}

        Path(cfg.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _build_scheduler(self):
        name = self.cfg.training.lr_scheduler
        if name == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=self.cfg.training.epochs)
        elif name == "step":
            return StepLR(self.optimizer, step_size=30, gamma=0.1)
        return None

    # ------------------------------------------------------------------
    def _run_epoch(self, loader: DataLoader, train: bool) -> tuple:
        self.model.train(train)
        total_loss, correct, total = 0.0, 0, 0

        with torch.set_grad_enabled(train):
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                logits = self.model(X)
                loss = self.criterion(logits, y)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                total_loss += loss.item() * len(y)
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
        logger.info(f"Starting training | device={self.device} | params={self.model.num_parameters:,}")

        for epoch in range(1, self.cfg.training.epochs + 1):
            train_loss, train_acc = self._run_epoch(train_loader, train=True)
            val_loss,   val_acc   = self._run_epoch(val_loader,   train=False)

            if self.scheduler:
                self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            logger.info(
                f"Epoch {epoch:03d}/{self.cfg.training.epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), best_ckpt)
                logger.info(f"  ✓ Best model saved → {best_ckpt}")

            if self.early_stopping.step(val_loss):
                logger.info(f"Early stopping at epoch {epoch}.")
                break

        # Restore best weights
        self.model.load_state_dict(torch.load(best_ckpt, map_location=self.device))
        return self.history