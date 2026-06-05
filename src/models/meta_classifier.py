"""
Méta-classifieur (stacking) : MLP sur la concaténation des logits des experts gelés.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class MetaClassifierMLP(nn.Module):
    """
    MLP : ``in_features`` → couches cachées ReLU + dropout → ``num_classes`` logits.

    ``hidden_dims`` vide → une seule couche linéaire ``in_features → num_classes``.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int = 33,
        hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if in_features <= 0:
            raise ValueError("in_features must be positive.")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")

        layers: list[nn.Module] = []
        d_in = in_features
        for h in hidden_dims:
            hi = int(h)
            if hi <= 0:
                continue
            layers.extend(
                [
                    nn.Linear(d_in, hi),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=float(dropout)),
                ]
            )
            d_in = hi
        layers.append(nn.Linear(d_in, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
