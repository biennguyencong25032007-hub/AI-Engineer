from typing import List

import torch
import torch.nn as nn

from src.config import ModelConfig


class BinaryClassifier(nn.Module):
    """
    Fully-connected neural network for binary classification.

    Architecture:
        Input → [Linear → BatchNorm → ReLU → Dropout] × N → Linear(1)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = cfg.input_dim

        for out_dim in cfg.hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            if cfg.batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(cfg.dropout_rate))
            in_dim = out_dim

        layers.append(nn.Linear(in_dim, 1))   # raw logit output
        self.network = nn.Sequential(*layers)
        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits with shape (batch,)."""
        return self.network(x).squeeze(1)

    # ------------------------------------------------------------------
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns probabilities in [0, 1]."""
        return torch.sigmoid(self.forward(x))

    # ------------------------------------------------------------------
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)