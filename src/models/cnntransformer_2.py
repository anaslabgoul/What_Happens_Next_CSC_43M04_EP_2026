"""
TSM (Temporal Shift Module) + 2D CNN backbone for short-clip video classification.

Input: (B, T, C, H, W). Frames are flattened to (B*T, C, H, W) for a standard
ImageNet-style backbone (ResNet-34 by default, or EfficientNet-B0 / ResNet).
Before each residual / MBConv block, channels are shifted along time (zero extra parameters
/ no 3D convs), then 2D convs run unchanged.

See ``CNNTransformer2.from_config`` for Hydra integration (``train.py``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchvision import models
from torchvision.models.efficientnet import MBConv
from torchvision.models.resnet import BasicBlock, Bottleneck


class TemporalShift(nn.Module):
    """Shift 1/fold_div channels forward in time and 1/fold_div backward (TSM)."""

    def __init__(self, n_segment: int = 4, fold_div: int = 8) -> None:
        super().__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*T, C, H, W)
        b_t, c, h, w = x.shape
        t = self.n_segment
        if b_t % t != 0:
            raise ValueError(
                f"TemporalShift: batch {b_t} is not divisible by n_segment={t} "
                "(check num_frames matches the video tensor)."
            )
        b = b_t // t
        fold = c // self.fold_div
        if fold == 0:
            return x

        x_bt = x.view(b, t, c, h, w)
        out = x_bt.clone()
        # Forward shift: past -> present
        out[:, 1:, :fold] = x_bt[:, :-1, :fold]
        # Backward shift: future -> present
        out[:, :-1, fold : 2 * fold] = x_bt[:, 1:, fold : 2 * fold]
        return out.view(b_t, c, h, w)


def _inject_tsm_into_resnet(backbone: nn.Module, n_segment: int, fold_div: int) -> None:
    """Register a TemporalShift pre-hook on every BasicBlock / Bottleneck."""
    shift = TemporalShift(n_segment=n_segment, fold_div=fold_div)

    def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        (inp,) = inputs
        return (shift(inp),)

    for module in backbone.modules():
        if isinstance(module, (BasicBlock, Bottleneck)):
            module.register_forward_pre_hook(hook)


def _inject_tsm_into_efficientnet(backbone: nn.Module, n_segment: int, fold_div: int) -> None:
    """Register a TemporalShift pre-hook on every MBConv (inverted residual) block."""
    shift = TemporalShift(n_segment=n_segment, fold_div=fold_div)

    def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        (inp,) = inputs
        return (shift(inp),)

    for module in backbone.modules():
        if isinstance(module, MBConv):
            module.register_forward_pre_hook(hook)


def _build_backbone(
    backbone: str,
    pretrained: bool,
    freeze_backbone: bool,
) -> tuple[nn.Module, int]:
    """ImageNet trunk with identity head; returns (network, feature_dim)."""
    name = backbone.lower().replace("-", "_")

    if name in ("efficientnet_b0", "efficientnetb0"):
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.efficientnet_b0(weights=weights)
        last_linear = net.classifier[-1]
        assert isinstance(last_linear, nn.Linear)
        dim = last_linear.in_features
        net.classifier = nn.Identity()
    elif name in ("resnet18", "resnet_18"):
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet18(weights=weights)
        dim = net.fc.in_features
        net.fc = nn.Identity()
    elif name in ("resnet34", "resnet_34"):
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet34(weights=weights)
        dim = net.fc.in_features
        net.fc = nn.Identity()
    elif name in ("resnet50", "resnet_50"):
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet50(weights=weights)
        dim = net.fc.in_features
        net.fc = nn.Identity()
    else:
        raise ValueError(
            f"TSM backbone '{backbone}' is not supported. "
            "Use efficientnet_b0, resnet18, resnet34, or resnet50."
        )

    if freeze_backbone:
        for p in net.parameters():
            p.requires_grad = False

    return net, dim


def _inject_tsm_for_backbone_name(
    backbone: nn.Module,
    backbone_name: str,
    n_segment: int,
    fold_div: int,
) -> None:
    key = backbone_name.lower().replace("-", "_")
    if key in ("efficientnet_b0", "efficientnetb0"):
        _inject_tsm_into_efficientnet(backbone, n_segment=n_segment, fold_div=fold_div)
    elif key.startswith("resnet"):
        _inject_tsm_into_resnet(backbone, n_segment=n_segment, fold_div=fold_div)
    else:
        raise ValueError(
            f"Unknown backbone '{backbone_name}' for TSM injection "
            "(expected efficientnet_b0 or a resnet variant)."
        )


class CNNTransformer2(nn.Module):
    """
    TSM + 2D backbone (default ResNet-34): temporal shift before each residual /
    MBConv block, global pooling, linear head.

    Forward:
        ``(B, T, 3, H, W)`` → reshape ``(B*T, 3, H, W)`` → backbone+TSM → ``(B*T, D)``
        → mean over T → dropout → logits ``(B, num_classes)``.
    """

    def __init__(
        self,
        num_classes: int,
        num_frames: int = 4,
        pretrained: bool = True,
        backbone: str = "resnet34",
        fold_div: int = 8,
        freeze_backbone: bool = False,
        head_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.backbone_name = backbone

        self.backbone, feat_dim = _build_backbone(
            backbone=backbone,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
        _inject_tsm_for_backbone_name(
            self.backbone,
            backbone_name=backbone,
            n_segment=num_frames,
            fold_div=fold_div,
        )

        self.head_dropout = nn.Dropout(p=head_dropout)
        self.classifier = nn.Linear(feat_dim, num_classes)
        self._init_classifier()

    def _init_classifier(self) -> None:
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    @classmethod
    def from_config(cls, model_cfg: DictConfig, dataset_num_frames: int | None = None) -> CNNTransformer2:
        """Build from Hydra ``model`` config; ``num_frames`` defaults to ``dataset.num_frames``."""
        mf = model_cfg.get("num_frames")
        num_frames = int(mf) if mf is not None else (dataset_num_frames if dataset_num_frames is not None else 4)
        return cls(
            num_classes=int(model_cfg.num_classes),
            num_frames=num_frames,
            pretrained=bool(model_cfg.get("pretrained", True)),
            backbone=str(model_cfg.get("backbone", "resnet34")),
            fold_div=int(model_cfg.get("fold_div", 8)),
            freeze_backbone=bool(model_cfg.get("freeze_backbone", False)),
            head_dropout=float(model_cfg.get("head_dropout", 0.3)),
        )

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        """
        video_batch: (B, T, C, H, W)
        returns logits: (B, num_classes)
        """
        b, t, c, h, w = video_batch.shape
        if t != self.num_frames:
            raise ValueError(
                f"CNNTransformer2 expects T={self.num_frames} frames, got T={t}. "
                "Align dataset.num_frames with model.num_frames."
            )

        x = video_batch.reshape(b * t, c, h, w)
        x = self.backbone(x)
        x = x.view(b, t, -1).mean(dim=1)
        x = self.head_dropout(x)
        return self.classifier(x)
