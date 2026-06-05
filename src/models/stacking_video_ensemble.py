"""
Modèle vidéo unique pour le stacking : ``(B,T,C,H,W)`` en [0,1] → logits (33 classes).

Utilisé pour sauvegarder un checkpoint compatible avec ``create_submission.py``
(``model_state_dict`` + entrées ``[0,1]`` comme à l’entraînement stacking).
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from models.meta_classifier import MetaClassifierMLP
from train_crops import normalize_video_batch


@runtime_checkable
class _ExpertLike(Protocol):
    model: nn.Module
    use_imagenet_norm: bool


class StackingVideoEnsemble(nn.Module):
    """
    Enchaîne les experts (chacun avec sa normalisation) puis le méta-MLP.

    L’entrée ``video`` doit être dans **[0, 1]** (Resize + ToTensor sans normalisation globale).
    """

    def __init__(
        self,
        experts: nn.ModuleList,
        use_imagenet_norm_per_expert: torch.Tensor,
        meta: MetaClassifierMLP,
    ) -> None:
        super().__init__()
        self.experts = experts
        self.register_buffer("use_imagenet_norm_per_expert", use_imagenet_norm_per_expert.clone())
        self.meta = meta

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        parts: List[torch.Tensor] = []
        for i, module in enumerate(self.experts):
            use_in = bool(self.use_imagenet_norm_per_expert[i].item())
            x = normalize_video_batch(video, use_in)
            parts.append(module(x))
        return self.meta(torch.cat(parts, dim=1))


def stacking_video_ensemble_from_trained_parts(
    experts: List[_ExpertLike],
    meta: MetaClassifierMLP,
) -> StackingVideoEnsemble:
    """Construit l’ensemble à partir des experts gelés (``FrozenExpert``) et du MLP entraîné."""
    if not experts:
        raise ValueError("experts vide")
    mods = nn.ModuleList([e.model for e in experts])
    flags = torch.tensor([e.use_imagenet_norm for e in experts], dtype=torch.bool)
    return StackingVideoEnsemble(mods, flags, meta)


def load_stacking_video_ensemble_from_checkpoint(ckpt: Dict[str, Any]) -> StackingVideoEnsemble:
    """Reconstruit l’ensemble depuis un checkpoint sauvegardé par ``train_stacking.py``."""
    from stacking_common import build_one_stacking_base_model

    if "expert_init_configs" not in ckpt or "model_state_dict" not in ckpt:
        raise KeyError("Checkpoint stacking vidéo invalide (manque expert_init_configs ou model_state_dict).")

    expert_cfgs = ckpt["expert_init_configs"]
    mods: list[nn.Module] = []
    uses: list[bool] = []
    for d in expert_cfgs:
        cfg = OmegaConf.create(d)
        m = build_one_stacking_base_model(cfg)
        mods.append(m)
        uses.append(bool(cfg.model.get("pretrained", True)))

    em = nn.ModuleList(mods)
    flags = torch.tensor(uses, dtype=torch.bool)
    hidden_raw = ckpt.get("meta_hidden_dims", [256, 128])
    hidden_dims = [int(x) for x in hidden_raw]
    meta = MetaClassifierMLP(
        in_features=int(ckpt["in_features"]),
        num_classes=int(ckpt["num_classes"]),
        hidden_dims=hidden_dims,
        dropout=float(ckpt.get("meta_dropout", 0.3)),
    )
    ens = StackingVideoEnsemble(em, flags, meta)
    ens.load_state_dict(ckpt["model_state_dict"], strict=True)
    return ens
