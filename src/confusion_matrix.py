#!/usr/bin/env python3
"""
Compute la matrice de confusion (split validation) pour un checkpoint vidéo.

Écrit une **heatmap PNG** (couleur = confusion ; par défaut chaque ligne est
normalisée sur la vraie classe pour mieux voir les confusions) et un **CSV**
numérique (même dossier par défaut).

Réutilise le même jeu de validation que ``evaluate.py`` (``dataset.val_dir``).

Exemple (depuis ``src/`` ; si les données ne sont pas sous ``hydra:runtime.cwd``,
passe un ``val_dir`` absolu)::

    python confusion_matrix.py \\
        training.checkpoint_path=/Data/anas/ResNet34-TSM-38.pt \\
        dataset.val_dir=/Data/anas/processed_data/val2/val

Sorties par défaut (dans le parent de ``val_dir``) : ``confusion_matrix.png``,
``confusion_matrix.csv``. Overrides::

    python confusion_matrix.py \\
        training.checkpoint_path=best_model.pt \\
        training.confusion_heatmap_output=/chemin/heatmap.png \\
        training.confusion_matrix_output=/chemin/matrice.csv \\
        training.confusion_heatmap_row_normalize=false \\
        training.confusion_matrix_write_csv=false
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from dataset.video_dataset import VideoFrameDataset, _parse_class_index, collect_video_samples
from evaluate import load_model_from_checkpoint
from utils import build_transforms, set_seed


def _class_index_to_folder_name(val_dir: Path) -> Dict[int, str]:
    """Mappe index de classe -> nom du dossier (ex. ``017_Basketball``)."""
    val_dir = val_dir.resolve()
    class_dirs = [p for p in sorted(val_dir.iterdir()) if p.is_dir()]
    fallback_index = {p.name: i for i, p in enumerate(class_dirs)}
    out: Dict[int, str] = {}
    for class_dir in class_dirs:
        parsed = _parse_class_index(class_dir.name)
        idx = parsed if parsed is not None else fallback_index[class_dir.name]
        out[idx] = class_dir.name
    return out


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[List[int], List[int]]:
    y_true: List[int] = []
    y_pred: List[int] = []
    for video_batch, labels in loader:
        video_batch = video_batch.to(device)
        labels = labels.to(device)
        logits = model(video_batch)
        pred = logits.argmax(dim=1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
    return y_true, y_pred


def confusion_counts(
    y_true: List[int], y_pred: List[int], num_classes: int
) -> List[List[int]]:
    """Matrice C[true][pred] = nombre d'exemples."""
    cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t][p] += 1
    return cm


def _short_class_label(i: int, class_names: Dict[int, str], max_len: int = 28) -> str:
    name = class_names.get(i, str(i))
    if len(name) <= max_len:
        return name
    return name[: max_len - 1] + "…"


def write_confusion_csv(
    path: Path,
    cm: List[List[int]],
    class_names: Dict[int, str],
    num_classes: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["true_class_index", "true_class_folder"] + [str(j) for j in range(num_classes)]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(num_classes):
            name = class_names.get(i, str(i))
            w.writerow([i, name] + [cm[i][j] for j in range(num_classes)])
    print(f"Matrice de confusion (CSV) : {path}", flush=True)


def write_confusion_heatmap(
    path: Path,
    cm: List[List[int]],
    class_names: Dict[int, str],
    num_classes: int,
    *,
    row_normalize: bool,
    title: str,
) -> None:
    """Écrit une heatmap PNG ; ``row_normalize`` = couleurs par ligne (rappel par vraie classe)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(cm, dtype=np.float64)
    if row_normalize:
        sums = arr.sum(axis=1, keepdims=True)
        sums = np.maximum(sums, 1e-12)
        display = arr / sums
        cbar_label = "Fraction (ligne = vraie classe)"
        vmax = 1.0
    else:
        display = arr
        cbar_label = "Nombre d'échantillons"
        vmax = None

    tick_labels = [_short_class_label(i, class_names) for i in range(num_classes)]
    side = max(10.0, num_classes * 0.42)
    fig, ax = plt.subplots(figsize=(side, side))
    im = ax.imshow(
        display,
        interpolation="nearest",
        cmap="Blues",
        aspect="equal",
        vmin=0.0,
        vmax=vmax,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(cbar_label, rotation=270, labelpad=18)

    ax.set_title(title)
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Vraie classe")

    ticks = np.arange(num_classes)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(tick_labels, rotation=75, ha="right", fontsize=max(5, 9 - num_classes // 8))
    ax.set_yticklabels(tick_labels, fontsize=max(5, 9 - num_classes // 8))

    # Grille légère entre cellules
    ax.set_xticks(np.arange(num_classes + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(num_classes + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.35)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap : {path}", flush=True)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    set_seed(int(cfg.dataset.seed))

    device_str = cfg.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint introuvable : {checkpoint_path}")

    raw: Dict[str, Any] = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in raw:
        if "meta_state_dict" in raw:
            raise SystemExit(
                "Ancien checkpoint « méta seul » (meta_state_dict). "
                "Relance train_stacking.py pour obtenir un stacking_video_ensemble "
                "(model_state_dict + video_input_raw_01)."
            )
        raise SystemExit(f"Checkpoint sans 'model_state_dict'. Clés : {sorted(raw.keys())}")

    model = load_model_from_checkpoint(raw, device)

    num_classes = int(cfg.model.num_classes)
    if "config" in raw and raw["config"] is not None:
        mcfg = OmegaConf.create(raw["config"])
        num_classes = int(mcfg.model.num_classes)
    elif raw.get("num_classes") is not None:
        num_classes = int(raw["num_classes"])

    if bool(raw.get("video_input_raw_01")) or str(raw.get("model_name", "")) == "stacking_video_ensemble":
        image_sz = int(raw.get("image_size", cfg.training.get("image_size", 224)))
        eval_transform = transforms.Compose(
            [
                transforms.Resize((image_sz, image_sz)),
                transforms.ToTensor(),
            ]
        )
    else:
        pretrained_used = bool(raw.get("pretrained", cfg.model.pretrained))
        eval_transform = build_transforms(is_training=False, use_imagenet_norm=pretrained_used)

    val_dir = Path(cfg.dataset.val_dir).resolve()
    val_samples = collect_video_samples(val_dir)
    max_samples = cfg.dataset.get("max_samples")
    if max_samples is not None:
        val_samples = val_samples[: int(max_samples)]

    num_frames = int(raw.get("num_frames", cfg.dataset.num_frames))
    val_dataset = VideoFrameDataset(
        root_dir=val_dir,
        num_frames=num_frames,
        transform=eval_transform,
        sample_list=val_samples,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    y_true, y_pred = collect_predictions(model, val_loader, device)
    cm = confusion_counts(y_true, y_pred, num_classes)

    correct = sum(cm[i][i] for i in range(num_classes))
    total = sum(sum(row) for row in cm)
    print(f"Échantillons validation : {total}")
    print(f"Exactitude (diagonale) : {correct / max(total, 1):.4f}")

    default_csv = val_dir.parent / "confusion_matrix.csv"
    csv_str = cfg.training.get("confusion_matrix_output")
    csv_path = Path(str(csv_str)).resolve() if csv_str else default_csv.resolve()

    default_png = val_dir.parent / "confusion_matrix.png"
    png_str = cfg.training.get("confusion_heatmap_output")
    png_path = Path(str(png_str)).resolve() if png_str else default_png.resolve()

    row_norm = bool(cfg.training.get("confusion_heatmap_row_normalize", True))
    title_raw = cfg.training.get("confusion_heatmap_title")
    title = (
        str(title_raw)
        if title_raw not in (None, "", "null", "None")
        else f"Matrice de confusion — {checkpoint_path.name}"
    )

    class_names = _class_index_to_folder_name(val_dir)
    if bool(cfg.training.get("confusion_matrix_write_csv", True)):
        write_confusion_csv(csv_path, cm, class_names, num_classes)
    write_confusion_heatmap(
        png_path,
        cm,
        class_names,
        num_classes,
        row_normalize=row_norm,
        title=title,
    )


if __name__ == "__main__":
    main()
