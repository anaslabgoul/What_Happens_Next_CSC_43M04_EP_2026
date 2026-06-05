"""
Entraînement d’un **méta-classifieur (stacking)** sur la concaténation des logits
de plusieurs modèles vidéo **gelés** (même pipeline données + augmentations que ``train_crops.py``).

Le **meilleur** checkpoint sauvegardé (``training.checkpoint_path``) est un **modèle vidéo complet**
``StackingVideoEnsemble`` : ``(B,T,C,H,W)`` en **[0,1]** → 33 logits, utilisable avec
``create_submission.py`` (même format que ``train_crops`` : ``model_state_dict`` + ``config``).

Chaque batch (train) :
  1. forte augmentation (une technique aléatoire par batch, comme ``train_crops``) ;
  2. passage dans chaque expert gelé avec la normalisation lue depuis son checkpoint ;
  3. concaténation des logits → ``MetaClassifierMLP`` ;
  4. rétropropagation **uniquement** sur le MLP.

Lancer depuis ``src/``::

    python train_stacking.py \\
      stacking.model_paths='["/path/general.pt","/path/a1.pt",...]'
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from models.meta_classifier import MetaClassifierMLP
from models.stacking_video_ensemble import stacking_video_ensemble_from_trained_parts
from stacking_common import FrozenExpert, load_frozen_experts
from train_crops import apply_augment_technique, normalize_video_batch
from utils import set_seed


@torch.no_grad()
def concat_frozen_logits(
    experts: List[FrozenExpert],
    video_batch: torch.Tensor,
) -> torch.Tensor:
    """``video_batch`` : (B,T,C,H,W) dans [0, 1] ; retour (B, sum_logits)."""
    parts: List[torch.Tensor] = []
    for ex in experts:
        x = normalize_video_batch(video_batch, ex.use_imagenet_norm)
        logits = ex.model(x)
        parts.append(logits)
    return torch.cat(parts, dim=1)


def train_one_epoch_stacking(
    experts: List[FrozenExpert],
    meta: MetaClassifierMLP,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: DictConfig,
    image_size: int,
    epoch_idx: int,
    total_epochs: int,
) -> Tuple[float, float]:
    """Même logique de progression (~10 lignes / epoch) que ``train_crops.train_one_epoch``."""
    meta.train()
    running_loss = 0.0
    correct = 0
    total = 0

    total_batches = len(data_loader)
    progress_points = {
        max(1, min(total_batches, int(round(k * total_batches / 10))))
        for k in range(1, 11)
    }

    tr = cfg.training
    crop_scale = (float(tr.get("crop_scale_min", 0.45)), float(tr.get("crop_scale_max", 1.0)))
    crop_ratio = tuple(tr.get("crop_ratio", (0.82, 1.18)))
    if len(crop_ratio) != 2:
        crop_ratio = (0.82, 1.18)
    crop_ratio_t = (float(crop_ratio[0]), float(crop_ratio[1]))

    aug_techniques: Tuple[str, ...] = (
        "resized_crop",
        "affine_translate",
        "photometric",
        "gaussian_blur",
        "random_erasing",
    )

    for batch_idx, (video_batch, labels) in enumerate(data_loader, start=1):
        video_batch = video_batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        tech = random.choice(aug_techniques)
        video_batch = apply_augment_technique(
            video_batch,
            tech,
            image_size=image_size,
            crop_scale=crop_scale,
            crop_ratio=crop_ratio_t,
            affine_translate_frac=float(tr.get("affine_translate_frac", 0.06)),
            grayscale_prob=float(tr.get("grayscale_prob", 0.18)),
            sharpness_delta=float(tr.get("sharpness_delta", 0.55)),
            color_brightness=float(tr.get("color_jitter_brightness", 0.35)),
            color_contrast=float(tr.get("color_jitter_contrast", 0.35)),
            color_saturation=float(tr.get("color_jitter_saturation", 0.38)),
            color_hue=float(tr.get("color_jitter_hue", 0.06)),
            blur_sigma_lo=float(tr.get("blur_sigma_min", 0.15)),
            blur_sigma_hi=float(tr.get("blur_sigma_max", 1.1)),
            erase_area_lo=float(tr.get("erase_area_min", 0.02)),
            erase_area_hi=float(tr.get("erase_area_max", 0.12)),
        )
        video_batch = video_batch.clamp(0.0, 1.0)

        with torch.no_grad():
            meta_in = concat_frozen_logits(experts, video_batch)

        optimizer.zero_grad()
        logits = meta(meta_in)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += labels.size(0)

        if batch_idx in progress_points:
            running_average_loss = running_loss / max(total, 1)
            running_accuracy = correct / max(total, 1)
            print(
                f"Epoch {epoch_idx + 1}/{total_epochs} | "
                f"step {batch_idx}/{total_batches} | "
                f"train loss {running_average_loss:.4f} acc {running_accuracy:.4f}",
                flush=True,
            )

    average_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return average_loss, accuracy


@torch.no_grad()
def evaluate_stacking(
    experts: List[FrozenExpert],
    meta: MetaClassifierMLP,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Validation : pas d’augmentation ; tenseurs [0,1] puis normalisation par expert."""
    meta.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for video_batch, labels in data_loader:
        video_batch = video_batch.to(device)
        labels = labels.to(device)

        meta_in = concat_frozen_logits(experts, video_batch)
        logits = meta(meta_in)
        loss = loss_fn(logits, labels)

        running_loss += float(loss.item()) * labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += labels.size(0)

    average_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return average_loss, accuracy


def _stacking_video_checkpoint_payload(
    cfg: DictConfig,
    ensemble: nn.Module,
    experts: List[FrozenExpert],
    expert_init_configs: List[Dict[str, Any]],
    in_features: int,
    hidden_dims: List[int],
    dropout: float,
    val_acc: float,
    image_size: int,
) -> Dict[str, Any]:
    """Checkpoint style ``train_crops`` + métadonnées pour ``create_submission``."""
    cfg_cont = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_cont, dict):
        cfg_cont = {}
    cfg_cont = dict(cfg_cont)
    cfg_cont["model"] = dict(cfg_cont.get("model", {}))
    cfg_cont["model"]["name"] = "stacking_video_ensemble"
    cfg_cont["model"]["pretrained"] = False
    cfg_cont["model"]["num_classes"] = int(cfg.num_classes)

    return {
        "model_state_dict": ensemble.state_dict(),
        "model_name": "stacking_video_ensemble",
        "num_classes": int(cfg.num_classes),
        "num_frames": int(cfg.dataset.num_frames),
        "pretrained": False,
        "video_input_raw_01": True,
        "in_features": int(in_features),
        "meta_hidden_dims": list(hidden_dims),
        "meta_dropout": float(dropout),
        "expert_init_configs": expert_init_configs,
        "base_checkpoint_paths": [str(e.path) for e in experts],
        "base_model_names": [e.name for e in experts],
        "base_num_logits": [e.num_logits for e in experts],
        "image_size": int(image_size),
        "val_accuracy": float(val_acc),
        "config": cfg_cont,
    }


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg), flush=True)

    set_seed(int(cfg.dataset.seed))

    device_str = str(cfg.training.device)
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.", flush=True)
        device_str = "cpu"
    device = torch.device(device_str)

    raw_paths = OmegaConf.to_container(cfg.stacking.get("model_paths"), resolve=True)
    if not isinstance(raw_paths, (list, tuple)) or len(raw_paths) == 0:
        raise ValueError(
            "Définissez stacking.model_paths avec la liste des checkpoints, "
            "ex: stacking.model_paths='[\"/a.pt\",\"/b.pt\"]'"
        )
    paths = [str(p) for p in raw_paths]

    experts, in_features, expert_num_frames, expert_init_configs = load_frozen_experts(paths, device)
    if int(cfg.dataset.num_frames) != expert_num_frames:
        print(
            f"[train_stacking] Alignement dataset.num_frames {cfg.dataset.num_frames} → {expert_num_frames} "
            f"(imposé par les checkpoints experts).",
            flush=True,
        )
        try:
            OmegaConf.set_struct(cfg, False)
            cfg.dataset.num_frames = expert_num_frames
            OmegaConf.set_struct(cfg, True)
        except Exception:
            cfg.dataset.num_frames = expert_num_frames

    print(
        f"Experts gelés ({len(experts)}): "
        + ", ".join(f"{e.name}({e.num_logits})" for e in experts)
        + f" → concat = {in_features} logits.",
        flush=True,
    )

    train_dir = Path(cfg.dataset.train_dir).resolve()
    val_dir = Path(cfg.dataset.val_dir).resolve()

    train_samples = collect_video_samples(train_dir)
    val_samples = collect_video_samples(val_dir)

    max_samples = cfg.dataset.get("max_samples")
    if max_samples is not None:
        ms = int(max_samples)
        train_samples = train_samples[:ms]
        val_samples = val_samples[:ms]

    image_size = int(cfg.training.get("image_size", 224))
    resize_totensor = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    train_dataset = VideoFrameDataset(
        root_dir=train_dir,
        num_frames=int(cfg.dataset.num_frames),
        transform=resize_totensor,
        sample_list=train_samples,
    )
    val_dataset = VideoFrameDataset(
        root_dir=val_dir,
        num_frames=int(cfg.dataset.num_frames),
        transform=resize_totensor,
        sample_list=val_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    hidden_raw = cfg.stacking.get("meta_hidden_dims", [256, 128])
    hidden_dims = [int(x) for x in OmegaConf.to_container(hidden_raw, resolve=True)]
    dropout = float(cfg.stacking.get("meta_dropout", 0.3))

    meta = MetaClassifierMLP(
        in_features=in_features,
        num_classes=int(cfg.num_classes),
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(meta.parameters(), lr=float(cfg.training.lr))

    best_val_accuracy = 0.0
    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()

    n_train = len(train_dataset)
    n_batches = len(train_loader)
    print(
        f"train_stacking: {n_train} train clips, {n_batches} batches/epoch, device={device} "
        f"— jusqu’à 10 lignes train / epoch (comme train_crops).",
        flush=True,
    )

    for epoch in range(int(cfg.training.epochs)):
        train_loss, train_acc = train_one_epoch_stacking(
            experts,
            meta,
            train_loader,
            loss_fn,
            optimizer,
            device,
            cfg,
            image_size=image_size,
            epoch_idx=epoch,
            total_epochs=int(cfg.training.epochs),
        )
        val_loss, val_acc = evaluate_stacking(experts, meta, val_loader, loss_fn, device)

        print(
            f"Epoch {epoch + 1}/{cfg.training.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}",
            flush=True,
        )

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            ensemble = stacking_video_ensemble_from_trained_parts(experts, meta)
            torch.save(
                _stacking_video_checkpoint_payload(
                    cfg,
                    ensemble,
                    experts,
                    expert_init_configs,
                    in_features,
                    hidden_dims,
                    dropout,
                    val_acc,
                    image_size,
                ),
                checkpoint_path,
            )
            print(
                f"  Saved stacking video model (experts+meta) to {checkpoint_path} "
                f"(val acc={val_acc:.4f}) — compatible create_submission.py",
                flush=True,
            )

    print(f"Done. Best validation accuracy: {best_val_accuracy:.4f}", flush=True)


if __name__ == "__main__":
    main()
