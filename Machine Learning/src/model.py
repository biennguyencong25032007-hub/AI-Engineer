from typing import List
import math

import torch
import torch.nn as nn

from src.config import ModelConfig


class ResidualBlock(nn.Module):
    """Residual block với skip connection."""

    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class BinaryClassifier(nn.Module):
    """
    Fully-connected neural network for binary classification.

    Architecture:
        Input → [Linear → BatchNorm → GELU → Dropout] × N → [ResidualBlock] × M → Linear(1)

    Features:
        - GELU activation (thay ReLU)
        - Optional residual blocks
        - Configurable initialization
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()

        # Validate config
        assert cfg.input_dim > 0, f"Invalid input_dim: {cfg.input_dim}"
        assert len(cfg.hidden_dims) > 0, "At least one hidden layer required"
        assert all(d > 0 for d in cfg.hidden_dims), "All hidden dims must be positive"

        self.cfg = cfg
        layers: List[nn.Module] = []
        in_dim = cfg.input_dim

        # Input projection
        layers.append(nn.Linear(in_dim, cfg.hidden_dims[0]))
        layers.append(nn.BatchNorm1d(cfg.hidden_dims[0]))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(cfg.dropout_rate))

        # Hidden layers
        for i in range(len(cfg.hidden_dims) - 1):
            out_dim = cfg.hidden_dims[i + 1]
            layers.append(nn.Linear(cfg.hidden_dims[i], out_dim))
            if cfg.batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(cfg.dropout_rate))
            in_dim = out_dim

        # Residual blocks nếu hidden dim đủ lớn
        if cfg.hidden_dims[-1] >= 64 and cfg.residual_blocks > 0:
            for _ in range(cfg.residual_blocks):
                layers.append(ResidualBlock(cfg.hidden_dims[-1], cfg.dropout_rate))

        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                match self.cfg.weight_init:
                    case "kaiming":
                        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    case "xavier":
                        nn.init.xavier_normal_(m.weight)
                    case "orthogonal":
                        nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits with shape (batch,)."""
        return self.network(x).squeeze(-1)

    # ------------------------------------------------------------------
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns probabilities in [0, 1]."""
        return torch.sigmoid(self.forward(x))

    # ------------------------------------------------------------------
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)