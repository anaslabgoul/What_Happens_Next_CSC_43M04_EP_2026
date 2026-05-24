from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoVideoProcessor


# ---------------------------------------------------------------------------
# Simple MLP Probe
# ---------------------------------------------------------------------------

class MLPProbe(nn.Module):
    """
    Probe légère :
        tokens (B, L, D)
        -> LayerNorm
        -> mean pooling sur les tokens
        -> MLP
        -> logits

    Beaucoup plus légère qu'une attentive probe avec blocs transformer,
    donc moins de risque d'overfitting sur petit dataset.
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.token_norm = nn.LayerNorm(feature_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, L, feature_dim)
        Returns:
            logits: (B, num_classes)
        """
        x = self.token_norm(tokens)
        x = x.mean(dim=1)          # global average pooling over tokens -> (B, D)
        logits = self.head(x)      # (B, num_classes)
        return logits


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

class VJEPA2VideoClassifier(nn.Module):
    """
    V-JEPA2 backbone + simple MLP probe.

    Stratégie recommandée sur petit dataset :
        - freeze_backbone=True
        - unfreeze_last_n_layers=0
        - seule la probe est entraînée

    Si besoin :
        - unfreeze_last_n_layers=2..4
        - backbone_lr très faible
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        model_name: str = "facebook/vjepa2-vith-fpc64-256",
        freeze_backbone: bool = True,
        unfreeze_last_n_layers: int = 0,
        # Probe légère
        hidden_dim: int = 512,
        dropout: float = 0.2,
        # Normalisation
        input_norm: str = "imagenet",
        target_num_frames: Optional[int] = None,
        # Compatibilité avec anciennes configs éventuelles
        probe_dim: Optional[int] = None,
        probe_n_heads: int = 8,
        probe_n_layers: int = 4,
        probe_dropout: Optional[float] = None,
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.freeze_backbone = bool(freeze_backbone)
        self.unfreeze_last_n_layers = int(unfreeze_last_n_layers)
        self.input_norm = input_norm
        self.target_num_frames = target_num_frames

        if not pretrained:
            raise ValueError(
                "VJEPA2VideoClassifier est prévu pour utiliser un backbone pretrained. "
                "Mets pretrained: true dans ta config."
            )

        # Backbone
        self.processor = AutoVideoProcessor.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(
            model_name,
            attn_implementation="sdpa",
        )

        self.feature_dim = int(self.backbone.config.hidden_size)

        # Buffers de normalisation
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )

        image_mean = getattr(self.processor, "image_mean", [0.485, 0.456, 0.406])
        image_std = getattr(self.processor, "image_std", [0.229, 0.224, 0.225])
        self.register_buffer(
            "vjepa_mean",
            torch.tensor(image_mean, dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "vjepa_std",
            torch.tensor(image_std, dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )

        # Freeze / unfreeze
        self._apply_freezing()

        # Compatibilité : si une ancienne config passe probe_dropout/probe_dim, on les respecte
        effective_hidden_dim = int(probe_dim) if probe_dim is not None else int(hidden_dim)
        effective_dropout = float(probe_dropout) if probe_dropout is not None else float(dropout)

        # Classifier léger
        self.classifier = MLPProbe(
            feature_dim=self.feature_dim,
            num_classes=num_classes,
            hidden_dim=effective_hidden_dim,
            dropout=effective_dropout,
        )

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(
            f"V-JEPA2: {trainable:,} paramètres entraînables "
            f"/ {total:,} total ({100 * trainable / total:.2f}%)"
        )

    # ------------------------------------------------------------------
    # Freezing
    # ------------------------------------------------------------------

    def _find_transformer_layers(self) -> Optional[nn.ModuleList]:
        expected_layers = int(getattr(self.backbone.config, "num_hidden_layers", 0))
        candidates = []
        for _, module in self.backbone.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) > 0:
                if expected_layers > 0 and len(module) == expected_layers:
                    candidates.append(module)
        if candidates:
            return candidates[-1]

        largest = None
        for _, module in self.backbone.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) > 0:
                if largest is None or len(module) > len(largest):
                    largest = module
        return largest

    def _apply_freezing(self) -> None:
        if not self.freeze_backbone:
            return

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.unfreeze_last_n_layers <= 0:
            return

        layers = self._find_transformer_layers()
        if layers is None:
            print("Warning: impossible de trouver les layers Transformer. Backbone entièrement gelé.")
            return

        n = min(self.unfreeze_last_n_layers, len(layers))
        for layer in layers[-n:]:
            for p in layer.parameters():
                p.requires_grad = True

        print(f"V-JEPA2: unfroze last {n} transformer layers.")

    # ------------------------------------------------------------------
    # Video preparation
    # ------------------------------------------------------------------

    def _prepare_video(self, video_batch: torch.Tensor) -> torch.Tensor:
        """
        Convertit (B, T, C, H, W) normalisé ImageNet -> normalisé V-JEPA2.
        """
        x = video_batch

        if self.input_norm == "imagenet":
            x = x * self.imagenet_std + self.imagenet_mean
            x = x.clamp(0.0, 1.0)
        elif self.input_norm == "none":
            x = x.clamp(0.0, 1.0)
        else:
            raise ValueError(
                f"input_norm={self.input_norm!r} non supporté. Utilise 'imagenet' ou 'none'."
            )

        if self.target_num_frames is not None:
            b, t, c, h, w = x.shape
            if t != self.target_num_frames:
                x_perm = x.permute(0, 2, 3, 4, 1)  # (B, C, H, W, T)
                x_perm = F.interpolate(
                    x_perm,
                    size=(h, w, self.target_num_frames),
                    mode="trilinear",
                    align_corners=False,
                )
                x = x_perm.permute(0, 4, 1, 2, 3)

        x = (x - self.vjepa_mean) / self.vjepa_std
        return x

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _encode(self, video_batch: torch.Tensor) -> torch.Tensor:
        """
        Extrait les tokens du backbone.
        Returns:
            tokens: (B, L, feature_dim)
        """
        pixel_values = self._prepare_video(video_batch)
        backbone_trainable = any(p.requires_grad for p in self.backbone.parameters())

        if backbone_trainable:
            outputs = self.backbone(
                pixel_values_videos=pixel_values,
                skip_predictor=True,
            )
        else:
            with torch.no_grad():
                outputs = self.backbone(
                    pixel_values_videos=pixel_values,
                    skip_predictor=True,
                )

        return outputs.last_hidden_state

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video_batch: (B, T, C, H, W)
        Returns:
            logits: (B, num_classes)
        """
        tokens = self._encode(video_batch)
        return self.classifier(tokens)

    def forward_multiclip(self, clips: torch.Tensor, n_clips: int) -> torch.Tensor:
        """
        Inférence multi-clip avec moyenne des logits.

        Args:
            clips:   (B * n_clips, T, C, H, W)
            n_clips: nombre de clips par vidéo

        Returns:
            logits: (B, num_classes)
        """
        bn = clips.shape[0]
        if bn % n_clips != 0:
            raise ValueError(
                f"clips.shape[0]={bn} doit être divisible par n_clips={n_clips}"
            )

        b = bn // n_clips
        logits_all = self.forward(clips)              # (B*N, num_classes)
        logits_all = logits_all.view(b, n_clips, -1) # (B, N, num_classes)
        return logits_all.mean(dim=1)

    # ------------------------------------------------------------------
    # Optimizer param groups
    # ------------------------------------------------------------------

    def get_param_groups(
        self,
        probe_lr: float = 1e-4,
        backbone_lr: float = 1e-5,
        weight_decay: float = 0.05,
    ) -> list[dict]:
        """
        Retourne des groupes de paramètres séparés pour backbone et probe.
        """
        probe_params = list(self.classifier.parameters())
        backbone_trainable = [p for p in self.backbone.parameters() if p.requires_grad]

        groups = [
            {
                "params": probe_params,
                "lr": probe_lr,
                "weight_decay": weight_decay,
            }
        ]

        if backbone_trainable:
            groups.append(
                {
                    "params": backbone_trainable,
                    "lr": backbone_lr,
                    "weight_decay": weight_decay,
                }
            )

        return groups