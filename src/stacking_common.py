"""
Utilitaires partagés pour le stacking : chargement des experts gelés depuis des checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from models.cnn_specialized_agent import CNNTransformerSpecializedAgent
from train_crops import build_model as build_model_train_crops
from train import build_model as build_model_train


@dataclass
class FrozenExpert:
    """Un modèle gelé + le booléen de normalisation utilisé à l’entraînement."""

    name: str
    path: Path
    model: nn.Module
    use_imagenet_norm: bool
    num_logits: int


def cfg_from_checkpoint(ckpt: Dict[str, Any], path: Path) -> DictConfig:
    if ckpt.get("config") is not None:
        return OmegaConf.create(ckpt["config"])
    if ckpt.get("model_config") is not None:
        mc = OmegaConf.create(ckpt["model_config"])
        nf = int(ckpt.get("num_frames", 4))
        return OmegaConf.create({"model": mc, "dataset": {"num_frames": nf}})
    raise ValueError(
        f"Checkpoint {path} : ni 'config' ni 'model_config' — impossible de reconstruire le modèle."
    )


def build_one_stacking_base_model(cfg: DictConfig) -> nn.Module:
    """Reconstruction alignée sur ``train_crops`` + agents spécialisés + repli ``train.py``."""
    name = str(cfg.model.name)

    if name in ("cnntransformer_specialized", "cnn_transformer_specialized"):
        nf = int(cfg.dataset.num_frames)
        return CNNTransformerSpecializedAgent.from_config(cfg.model, dataset_num_frames=nf)

    try:
        return build_model_train_crops(cfg)
    except ValueError:
        return build_model_train(cfg)


def load_frozen_experts(
    paths: List[str | Path],
    device: torch.device,
) -> Tuple[List[FrozenExpert], int, int, List[Dict[str, Any]]]:
    """
    Charge les checkpoints, gèle les poids.

    Returns:
        experts, total_logits_dim, common_num_frames, expert_init_configs
        (``expert_init_configs`` = configs Hydra sérialisées pour reconstruire l’architecture).
    """
    experts: List[FrozenExpert] = []
    expert_init_configs: List[Dict[str, Any]] = []
    total_logits = 0
    frames_set: set[int] = set()

    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint introuvable : {path}")

        ckpt: Dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
        sub_cfg = cfg_from_checkpoint(ckpt, path)
        expert_init_configs.append(OmegaConf.to_container(sub_cfg, resolve=True))
        frames_set.add(int(sub_cfg.dataset.num_frames))
        model = build_one_stacking_base_model(sub_cfg)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        use_imagenet = bool(sub_cfg.model.get("pretrained", True))
        n_logits = int(ckpt.get("num_classes", sub_cfg.model.get("num_classes", 0)))
        if n_logits <= 0:
            if isinstance(model, CNNTransformerSpecializedAgent):
                n_logits = len(model.specialized_classes) + 1
            else:
                fc = getattr(model, "classifier", None)
                if isinstance(fc, nn.Linear):
                    n_logits = int(fc.out_features)
                else:
                    raise ValueError(f"Impossible d’inférer num_logits pour {path}")

        mname = str(ckpt.get("model_name", sub_cfg.model.name))
        experts.append(
            FrozenExpert(
                name=mname,
                path=path,
                model=model,
                use_imagenet_norm=use_imagenet,
                num_logits=n_logits,
            )
        )
        total_logits += n_logits

    if not experts:
        raise ValueError("stacking.model_paths est vide : fournissez au moins un checkpoint.")

    if len(frames_set) > 1:
        raise ValueError(
            f"Les experts n’ont pas le même dataset.num_frames dans leurs checkpoints : {sorted(frames_set)}. "
            "Ré-entraînez ou choisissez des checkpoints compatibles."
        )
    common_nf = int(next(iter(frames_set)))

    return experts, total_logits, common_nf, expert_init_configs
