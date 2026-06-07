"""
Entraînement avec loss pondérée par classe, sans oversampling complet.

Points clés :
- train/val sur dossiers séparés
- class-weighted CrossEntropyLoss
- augmentations synchronisées au niveau du clip
- support du mixed precision bf16
- support du resume depuis checkpoint complet
- support de max_samples avec sous-échantillonnage aléatoire
- support de train_augment_repeats depuis la config
- support de dataset.focus_classes pour entraîner un spécialiste de cluster sans copier les fichiers
- param groups séparés pour V-JEPA / ConvNeXt / autres modèles
- support scheduler cosine avec LR final séparé backbone/head
- support wandb
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F_t
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader

try:
    import wandb
except Exception:
    wandb = None

from dataset.video_dataset import (
    VideoFrameDataset,
    _list_frame_paths,
    _parse_class_index,
    _pick_frame_indices,
    collect_video_samples,
)
from models.cnn_baseline import CNNBaseline
from models.cnn_lstm import CNNLSTM
from models.vjepa import VJEPA2VideoClassifier
from utils import build_transforms, set_seed

try:
    from models.convnext import ConvNeXtTemporalAttention
except Exception:
    ConvNeXtTemporalAttention = None


# ---------------------------------------------------------------------
# Utils naming / W&B
# ---------------------------------------------------------------------


def _fmt_float_for_name(x: float) -> str:
    s = f"{x:.0e}" if (x != 0 and (abs(x) < 1e-2 or abs(x) >= 1e2)) else f"{x}"
    return s.replace(".", "p").replace("-", "m")


def build_wandb_run_name(cfg: DictConfig) -> str:
    forced = cfg.training.get("wandb_run_name", None)
    if forced:
        return str(forced)

    model_name = str(cfg.model.get("name", "model"))
    num_frames = int(cfg.dataset.get("num_frames", 0))
    image_size = int(cfg.training.get("image_size", 0))
    batch_size = int(cfg.training.get("batch_size", 0))

    parts = [
        model_name,
        f"{num_frames}f",
        f"{image_size}",
        f"bs{batch_size}",
    ]

    if model_name == "vjepa":
        freeze_backbone = bool(cfg.model.get("freeze_backbone", False))
        unfreeze_last_n_layers = int(cfg.model.get("unfreeze_last_n_layers", 0))
        hidden_dim = int(cfg.model.get("hidden_dim", 512))
        dropout = float(cfg.model.get("dropout", 0.0))
        head_lr = float(cfg.training.get("head_lr", cfg.training.get("lr", 0.0)))
        backbone_lr = float(cfg.training.get("backbone_lr", 0.0))
        target_num_frames = cfg.model.get("target_num_frames", None)
        frame_resampling = str(cfg.model.get("frame_resampling", "none"))

        if freeze_backbone and unfreeze_last_n_layers == 0:
            parts.append("probeonly")
        else:
            parts.append(f"u{unfreeze_last_n_layers}")

        parts.extend(
            [
                f"tgt{target_num_frames}",
                frame_resampling,
                f"h{hidden_dim}",
                f"do{str(dropout).replace('.', 'p')}",
                f"hlr{_fmt_float_for_name(head_lr)}",
                f"blr{_fmt_float_for_name(backbone_lr)}",
            ]
        )
    else:
        lr = float(cfg.training.get("lr", 0.0))
        wd = float(cfg.training.get("weight_decay", 0.0))
        parts.extend(
            [
                f"lr{_fmt_float_for_name(lr)}",
                f"wd{_fmt_float_for_name(wd)}",
            ]
        )

    return "_".join(parts)


# ---------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------


def build_model(cfg: DictConfig) -> nn.Module:
    """Create the model described by cfg.model.name."""
    name = str(cfg.model.name)
    num_classes = int(cfg.model.num_classes)
    pretrained = bool(cfg.model.pretrained)

    if name == "cnn_baseline":
        return CNNBaseline(num_classes=num_classes, pretrained=pretrained)

    if name == "cnn_lstm":
        hidden = int(cfg.model.get("lstm_hidden_size", 512))
        return CNNLSTM(
            num_classes=num_classes,
            pretrained=pretrained,
            lstm_hidden_size=hidden,
        )

    if name == "convnext":
        if ConvNeXtTemporalAttention is None:
            raise ImportError(
                "models.convnext.ConvNeXtTemporalAttention is not available."
            )

        return ConvNeXtTemporalAttention(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=float(cfg.model.get("dropout", 0.35)),
            attention_hidden_dim=int(cfg.model.get("attention_hidden_dim", 256)),
            freeze_backbone=bool(cfg.model.get("freeze_backbone", False)),
            unfreeze_last_block=bool(cfg.model.get("unfreeze_last_block", False)),
            unfreeze_last_stage=bool(cfg.model.get("unfreeze_last_stage", False)),
            unfreeze_last_n_blocks=int(cfg.model.get("unfreeze_last_n_blocks", 0)),
        )

    if name == "vjepa":
        return VJEPA2VideoClassifier(
            num_classes=num_classes,
            pretrained=pretrained,
            model_name=str(
                cfg.model.get(
                    "model_name",
                    "facebook/vjepa2-vitl-fpc16-256-ssv2",
                )
            ),
            freeze_backbone=bool(cfg.model.get("freeze_backbone", True)),
            unfreeze_last_n_layers=int(cfg.model.get("unfreeze_last_n_layers", 0)),
            hidden_dim=int(cfg.model.get("hidden_dim", 512)),
            dropout=float(cfg.model.get("dropout", 0.0)),
            input_norm=str(cfg.model.get("input_norm", "imagenet")),
            target_num_frames=cfg.model.get("target_num_frames", 16),
            frame_resampling=str(cfg.model.get("frame_resampling", "repeat")),
        )

    raise ValueError(f"Unknown model.name: {name}")


# ---------------------------------------------------------------------
# Dataset / augmentations
# ---------------------------------------------------------------------


def subsample_samples(
    samples: List[Tuple[Path, int]],
    max_samples: Optional[int],
    seed: int,
) -> List[Tuple[Path, int]]:
    """Random subsample instead of taking the first max_samples entries."""
    if max_samples is None:
        return samples

    max_samples = int(max_samples)
    if max_samples <= 0 or len(samples) <= max_samples:
        return samples

    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    keep = indices[:max_samples]
    return [samples[i] for i in keep]


def filter_samples_by_classes(
    samples: List[Tuple[Path, int]],
    focus_classes: Optional[List[int]],
) -> List[Tuple[Path, int]]:
    """
    Keep only samples whose label belongs to focus_classes.

    Used to train a specialist on a confusion cluster without copying files.
    Example:
        dataset.focus_classes: [8, 9, 11, 14, 30]
    """
    if not focus_classes:
        return samples

    focus_set = {int(c) for c in focus_classes}
    return [(path, label) for path, label in samples if int(label) in focus_set]


def get_focus_classes(cfg: DictConfig) -> Optional[List[int]]:
    """
    Read optional dataset.focus_classes from Hydra config.

    Returns None if absent/empty.
    """
    raw_focus = cfg.dataset.get("focus_classes", None)
    if raw_focus is None:
        return None

    focus_classes = [int(c) for c in raw_focus]
    if len(focus_classes) == 0:
        return None

    return focus_classes


def restrict_predictions_to_classes(
    logits: torch.Tensor,
    focus_classes: Optional[List[int]],
) -> torch.Tensor:
    """
    Return predictions in original class-id space.

    If focus_classes is None:
        standard argmax over all logits.
    Else:
        argmax only among logits[:, focus_classes], then map back to original IDs.

    This is useful for specialist training/eval:
        - loss is still full 33-way CE with original labels
        - accuracy/checkpoint selection is computed only inside the cluster
    """
    if not focus_classes:
        return logits.argmax(dim=1)

    focus_tensor = torch.tensor(
        [int(c) for c in focus_classes],
        dtype=torch.long,
        device=logits.device,
    )
    restricted_logits = logits.index_select(dim=1, index=focus_tensor)
    restricted_pred_local = restricted_logits.argmax(dim=1)
    return focus_tensor.index_select(dim=0, index=restricted_pred_local)


def _affine_translate_pil_sync(
    pil_frames: List[Image.Image],
    max_frac: float,
) -> List[Image.Image]:
    """Small zero-angle translation; same (tx, ty) in pixels for every frame."""
    if max_frac <= 0 or not pil_frames:
        return pil_frames

    w, h = pil_frames[0].size
    max_dx = int(max_frac * w)
    max_dy = int(max_frac * h)
    if max_dx <= 0 and max_dy <= 0:
        return pil_frames

    tx = random.randint(-max_dx, max_dx) if max_dx > 0 else 0
    ty = random.randint(-max_dy, max_dy) if max_dy > 0 else 0
    if tx == 0 and ty == 0:
        return pil_frames

    out: List[Image.Image] = []
    for pil in pil_frames:
        out.append(
            F_t.affine(
                pil,
                angle=0.0,
                translate=(tx, ty),
                scale=1.0,
                shear=(0.0, 0.0),
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=0,
            )
        )
    return out


def _horizontal_flip_pil_sync(
    pil_frames: List[Image.Image],
) -> List[Image.Image]:
    """Horizontal flip synchronisé sur toutes les frames du clip."""
    return [F_t.hflip(pil) for pil in pil_frames]


def _swap_horizontal_direction_label(label: int) -> int:
    """
    Swap des classes directionnelles gauche/droite après horizontal flip.

    18: Pulling something from left to right
    19: Pulling something from right to left
    """
    if label == 18:
        return 19
    if label == 19:
        return 18
    return label


def _color_jitter_pil_one(
    pil: Image.Image,
    b: float,
    c: float,
    s: float,
    hue: float,
) -> Image.Image:
    """Apply brightness / contrast / saturation / hue multipliers."""
    x = F_t.adjust_brightness(pil, b)
    x = F_t.adjust_contrast(x, c)
    x = F_t.adjust_saturation(x, s)
    x = F_t.adjust_hue(x, hue)
    return x


def _sample_color_jitter_factors(
    brightness: float,
    contrast: float,
    saturation: float,
    hue: float,
) -> Tuple[float, float, float, float]:
    """Sample multiplicative brightness/contrast/saturation and additive hue."""
    b = 1.0 + (random.random() * 2.0 - 1.0) * brightness
    c = 1.0 + (random.random() * 2.0 - 1.0) * contrast
    s = 1.0 + (random.random() * 2.0 - 1.0) * saturation
    h = (random.random() * 2.0 - 1.0) * hue
    h = max(-0.5, min(0.5, h))
    return b, c, s, h


def _photometric_pil_clip(
    pil_frames: List[Image.Image],
    grayscale_prob: float,
    sharpness_delta: float,
    brightness: float,
    contrast: float,
    saturation: float,
    hue: float,
) -> List[Image.Image]:
    """Same photometric rollouts on every frame. No flip."""
    b, c, s, h = _sample_color_jitter_factors(
        brightness,
        contrast,
        saturation,
        hue,
    )
    sharp = 1.0 + (random.random() * 2.0 - 1.0) * sharpness_delta
    sharp = max(0.05, sharp)
    do_gray = random.random() < grayscale_prob

    out: List[Image.Image] = []
    for pil in pil_frames:
        x = _color_jitter_pil_one(pil, b, c, s, h)
        x = F_t.adjust_sharpness(x, sharp)
        if do_gray:
            x = F_t.rgb_to_grayscale(x, num_output_channels=3)
        out.append(x)
    return out


def _maybe_gaussian_blur_tensor_sync(
    tensors_01: List[torch.Tensor],
    prob: float,
    sigma_lo: float,
    sigma_hi: float,
) -> None:
    """In-place gaussian blur; same sigma for each frame."""
    if prob <= 0 or not tensors_01 or random.random() >= prob:
        return

    sigma = random.uniform(sigma_lo, sigma_hi)
    k = int(random.choice([3, 5]))
    for i, t in enumerate(tensors_01):
        tensors_01[i] = F_t.gaussian_blur(
            t,
            kernel_size=[k, k],
            sigma=[sigma, sigma],
        )


def _random_erasing_sync(
    tensors_01: List[torch.Tensor],
    prob: float,
    area_lo: float,
    area_hi: float,
) -> None:
    """Same random rectangle on every frame; values in [0, 1] before normalize."""
    if prob <= 0 or not tensors_01 or random.random() >= prob:
        return

    c, h, w = tensors_01[0].shape
    if h < 4 or w < 4:
        return

    img_area = float(h * w)
    for _ in range(12):
        er_area = random.uniform(area_lo, area_hi) * img_area
        aspect = random.uniform(0.35, 1.0 / 0.35)
        eh = max(2, int(round((er_area * aspect) ** 0.5)))
        ew = max(2, int(round(er_area / float(eh))))
        eh = min(eh, h - 1)
        ew = min(ew, w - 1)

        if eh < 2 or ew < 2 or eh >= h or ew >= w:
            continue

        top = random.randint(0, h - eh)
        left = random.randint(0, w - ew)
        noise = torch.rand(c, eh, ew)

        for t in tensors_01:
            t[:, top : top + eh, left : left + ew] = noise
        return


class VideoFrameDatasetZoomAugment(VideoFrameDataset):
    """
    Dataset train avec augmentation synchronisée au niveau du clip :
    zoom/crop partagé, couleurs, netteté, flou léger, erasing synchronisé,
    translation — sans flip.
    """

    def __init__(
        self,
        root_dir: str | Path,
        num_frames: int,
        transform_eval_style: transforms.Compose,
        sample_list: List[Tuple[Path, int]] | None = None,
        image_size: int = 224,
        zoom_prob: float = 0.65,
        crop_scale: Tuple[float, float] = (0.45, 1.0),
        crop_ratio: Tuple[float, float] = (0.82, 1.18),
        use_imagenet_norm: bool = True,
        augment_repeats: int = 1,
        affine_translate_frac: float = 0.06,
        grayscale_prob: float = 0.18,
        sharpness_delta: float = 0.55,
        color_brightness: float = 0.35,
        color_contrast: float = 0.35,
        color_saturation: float = 0.38,
        color_hue: float = 0.06,
        blur_prob: float = 0.22,
        blur_sigma_lo: float = 0.15,
        blur_sigma_hi: float = 1.1,
        erase_prob: float = 0.28,
        erase_area_lo: float = 0.02,
        erase_area_hi: float = 0.12,
        horizontal_flip_prob: float = 0.5,
    ) -> None:
        super().__init__(
            root_dir=root_dir,
            num_frames=num_frames,
            transform=transform_eval_style,
            sample_list=sample_list,
        )
        self._n_base = len(self.samples)
        self.augment_repeats = max(1, int(augment_repeats))
        self.image_size = int(image_size)
        self.zoom_prob = float(zoom_prob)
        self.crop_scale = crop_scale
        self.crop_ratio = crop_ratio
        self.affine_translate_frac = float(affine_translate_frac)
        self.grayscale_prob = float(grayscale_prob)
        self.sharpness_delta = float(sharpness_delta)
        self.color_brightness = float(color_brightness)
        self.color_contrast = float(color_contrast)
        self.color_saturation = float(color_saturation)
        self.color_hue = float(color_hue)
        self.blur_prob = float(blur_prob)
        self.blur_sigma_lo = float(blur_sigma_lo)
        self.blur_sigma_hi = float(blur_sigma_hi)
        self.erase_prob = float(erase_prob)
        self.erase_area_lo = float(erase_area_lo)
        self.erase_area_hi = float(erase_area_hi)
        self.horizontal_flip_prob = float(horizontal_flip_prob)

        if use_imagenet_norm:
            self.normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        else:
            self.normalize = transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            )

    def __len__(self) -> int:
        return self._n_base * self.augment_repeats

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        real_index = index % self._n_base
        video_dir, label = self.samples[real_index]
        frame_paths = _list_frame_paths(video_dir)
        frame_indices = _pick_frame_indices(len(frame_paths), self.num_frames)

        pil_frames: List[Image.Image] = []
        for frame_index in frame_indices:
            path = frame_paths[frame_index]
            with Image.open(path) as image:
                pil_frames.append(image.convert("RGB"))

        ref = pil_frames[0]
        do_zoom = random.random() < self.zoom_prob
        resized_list: List[Image.Image] = []

        if do_zoom:
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                ref,
                scale=self.crop_scale,
                ratio=self.crop_ratio,
            )
            for pil in pil_frames:
                cropped = F_t.crop(pil, i, j, h, w)
                resized_list.append(
                    F_t.resize(
                        cropped,
                        [self.image_size, self.image_size],
                        interpolation=transforms.InterpolationMode.BILINEAR,
                    )
                )
        else:
            for pil in pil_frames:
                resized_list.append(
                    F_t.resize(
                        pil,
                        [self.image_size, self.image_size],
                        interpolation=transforms.InterpolationMode.BILINEAR,
                    )
                )

        do_hflip = random.random() < self.horizontal_flip_prob
        if do_hflip:
            resized_list = _horizontal_flip_pil_sync(resized_list)
            label = _swap_horizontal_direction_label(int(label))

        resized_list = _affine_translate_pil_sync(
            resized_list,
            self.affine_translate_frac,
        )
        resized_list = _photometric_pil_clip(
            resized_list,
            grayscale_prob=self.grayscale_prob,
            sharpness_delta=self.sharpness_delta,
            brightness=self.color_brightness,
            contrast=self.color_contrast,
            saturation=self.color_saturation,
            hue=self.color_hue,
        )

        tensors_01: List[torch.Tensor] = [F_t.to_tensor(pil) for pil in resized_list]

        _maybe_gaussian_blur_tensor_sync(
            tensors_01,
            prob=self.blur_prob,
            sigma_lo=self.blur_sigma_lo,
            sigma_hi=self.blur_sigma_hi,
        )
        _random_erasing_sync(
            tensors_01,
            prob=self.erase_prob,
            area_lo=self.erase_area_lo,
            area_hi=self.erase_area_hi,
        )

        frames_chw = [self.normalize(t) for t in tensors_01]
        video_tensor = torch.stack(frames_chw, dim=0)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return video_tensor, label_tensor


def _eval_style_resize_only_transform(
    image_size: int,
    use_imagenet_norm: bool,
) -> transforms.Compose:
    """Resize + ToTensor + Normalize."""
    if use_imagenet_norm:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    else:
        normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )


# ---------------------------------------------------------------------
# Class weights / reporting
# ---------------------------------------------------------------------


def count_videos_per_class(samples: List[Tuple[Path, int]]) -> Dict[int, int]:
    """Return {class_index: number of video entries in samples}."""
    counts: Dict[int, int] = defaultdict(int)
    for _, label in samples:
        counts[int(label)] += 1
    return dict(counts)


def build_class_weights(
    samples: List[Tuple[Path, int]],
    num_classes: int,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Build class weights for CrossEntropyLoss.

    alpha=1.0 -> full inverse frequency.
    alpha=0.5 -> inverse sqrt frequency.
    alpha=0.25 -> softer balancing.
    """
    counts = count_videos_per_class(samples)
    weights = torch.ones(num_classes, dtype=torch.float32)

    present_classes = sorted(counts.keys())
    if not present_classes:
        return weights

    total = sum(counts.values())
    n_present = len(present_classes)

    for class_index in present_classes:
        if 0 <= class_index < num_classes:
            inverse_frequency = total / (n_present * counts[class_index])
            weights[class_index] = float(inverse_frequency ** alpha)

    present_weights = torch.tensor(
        [
            weights[class_index].item()
            for class_index in present_classes
            if 0 <= class_index < num_classes
        ],
        dtype=torch.float32,
    )

    if present_weights.numel() > 0:
        weights = weights / present_weights.mean()

    return weights


def build_class_index_to_name(train_dir: Path) -> Dict[int, str]:
    """Map class index to folder name."""
    class_dirs = [p for p in sorted(train_dir.iterdir()) if p.is_dir()]
    fallback_index = {p.name: i for i, p in enumerate(class_dirs)}

    mapping: Dict[int, str] = {}
    for class_dir in class_dirs:
        parsed = _parse_class_index(class_dir.name)
        class_index = parsed if parsed is not None else fallback_index[class_dir.name]
        mapping[class_index] = class_dir.name

    return mapping


def print_class_video_counts(
    counts: Dict[int, int],
    class_names: Dict[int, str],
    title: str,
) -> None:
    """Print per-class video counts and summary statistics."""
    print(title)
    if not counts:
        print("  (no samples)")
        return

    values = list(counts.values())
    for class_index in sorted(counts.keys()):
        name = class_names.get(class_index, f"class_{class_index}")
        print(f"  [{class_index:03d}] {name}: {counts[class_index]} videos")

    print(
        f"  → total: {sum(values)} videos | "
        f"min={min(values)} max={max(values)} | {len(counts)} classes"
    )


# ---------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------


def _checkpoint_payload(
    cfg: DictConfig,
    model: nn.Module,
    val_acc: float,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "model_name": str(cfg.model.name),
        "num_classes": int(cfg.model.num_classes),
        "pretrained": bool(cfg.model.pretrained),
        "num_frames": int(cfg.dataset.num_frames),
        "val_accuracy": float(val_acc),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }

    if str(cfg.model.name) == "cnn_lstm":
        payload["lstm_hidden_size"] = int(cfg.model.get("lstm_hidden_size", 512))

    return payload


def save_checkpoint(
    cfg: DictConfig,
    model: nn.Module,
    val_acc: float,
    checkpoint_path: Path,
    reason: str,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_checkpoint_payload(cfg, model, val_acc), checkpoint_path)
    print(
        f"  Saved new best model to {checkpoint_path} "
        f"({reason}, val acc={val_acc:.4f})"
    )


# ---------------------------------------------------------------------
# Optimizer / scheduler
# ---------------------------------------------------------------------


def build_optimizer(
    model: nn.Module,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """
    Build AdamW with two parameter groups:
    - backbone group: pretrained visual/video backbone
    - head group: classifier/probe/custom heads
    """
    backbone_params: List[nn.Parameter] = []
    head_params: List[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # ConvNeXt backbone
        if name.startswith("features."):
            backbone_params.append(param)

        # Generic backbone naming
        elif name.startswith("backbone."):
            backbone_params.append(param)

        # V-JEPA video classification model:
        # pretrained classifier/head should use head_lr
        elif name.startswith("vjepa_model.classifier"):
            head_params.append(param)

        # Any other V-JEPA internals are backbone
        elif name.startswith("vjepa_model."):
            backbone_params.append(param)

        # Custom classifier heads / MLP / temporal heads
        else:
            head_params.append(param)

    param_groups = []

    if backbone_params:
        param_groups.append(
            {
                "params": backbone_params,
                "lr": float(backbone_lr),
                "weight_decay": float(weight_decay),
                "name": "backbone",
            }
        )

    if head_params:
        param_groups.append(
            {
                "params": head_params,
                "lr": float(head_lr),
                "weight_decay": float(weight_decay),
                "name": "head",
            }
        )

    if not param_groups:
        raise ValueError("No trainable parameters found.")

    optimizer = torch.optim.AdamW(param_groups)

    print(
        f"Optimizer: AdamW | "
        f"backbone params={sum(p.numel() for p in backbone_params)} "
        f"lr={backbone_lr} | "
        f"head params={sum(p.numel() for p in head_params)} "
        f"lr={head_lr} | "
        f"weight_decay={weight_decay}"
    )

    return optimizer


def build_cosine_lambda_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    final_backbone_lr: Optional[float] = None,
    final_head_lr: Optional[float] = None,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Cosine decay multiplicatif compatible avec plusieurs param groups.

    Pour chaque groupe :
      lr(epoch) va de initial_lr vers final_lr.

    On identifie les groupes via group["name"] si disponible :
      - "backbone" -> final_backbone_lr
      - "head" -> final_head_lr
    """
    total_epochs = max(1, int(total_epochs))

    def make_lambda(initial_lr: float, final_lr: float):
        initial_lr = float(initial_lr)
        final_lr = float(final_lr)

        if initial_lr <= 0:
            return lambda epoch: 1.0

        final_factor = final_lr / initial_lr

        def lr_lambda(epoch: int) -> float:
            progress = min(max(epoch, 0), total_epochs) / total_epochs
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return final_factor + (1.0 - final_factor) * cosine

        return lr_lambda

    lambdas = []
    for group in optimizer.param_groups:
        group_name = str(group.get("name", "head"))
        initial_lr = float(group["lr"])

        if group_name == "backbone":
            final_lr = (
                float(final_backbone_lr)
                if final_backbone_lr is not None
                else initial_lr * 0.1
            )
        else:
            final_lr = (
                float(final_head_lr)
                if final_head_lr is not None
                else initial_lr * 0.1
            )

        lambdas.append(make_lambda(initial_lr, final_lr))

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambdas,
    )


def build_scheduler(
    cfg: DictConfig,
    optimizer: torch.optim.Optimizer,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    scheduler_name = str(cfg.training.get("scheduler", "none")).lower()

    if scheduler_name in {"none", "null", ""}:
        return None

    if scheduler_name == "cosine":
        cosine_t_max = int(cfg.training.get("cosine_t_max", cfg.training.epochs))

        final_backbone_lr = cfg.training.get("final_backbone_lr", None)
        final_head_lr = cfg.training.get("final_head_lr", None)

        scheduler = build_cosine_lambda_scheduler(
            optimizer=optimizer,
            total_epochs=cosine_t_max,
            final_backbone_lr=(
                float(final_backbone_lr) if final_backbone_lr is not None else None
            ),
            final_head_lr=(float(final_head_lr) if final_head_lr is not None else None),
        )

        print(
            f"Scheduler: cosine | "
            f"T_max={cosine_t_max} | "
            f"final_backbone_lr={final_backbone_lr} | "
            f"final_head_lr={final_head_lr}"
        )
        return scheduler

    if scheduler_name == "plateau":
        factor = float(cfg.training.get("lr_decay_factor", 0.5))
        patience = int(cfg.training.get("lr_patience", 1))

        print(
            f"Scheduler: ReduceLROnPlateau | "
            f"factor={factor} | patience={patience}"
        )

        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor,
            patience=patience,
        )

    raise ValueError(f"Unknown scheduler: {scheduler_name}")


# ---------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch_idx: int,
    total_epochs: int,
    cfg: DictConfig,
    use_bf16: bool = False,
    grad_clip_norm: Optional[float] = None,
    val_loader: Optional[DataLoader] = None,
    checkpoint_path: Optional[Path] = None,
    best_val_accuracy: float = 0.0,
    mid_epoch_validation: bool = True,
    run: Any = None,
) -> Tuple[float, float, float]:
    """
    Train for one epoch.

    If mid_epoch_validation=True:
      validate at half epoch and save checkpoint if best.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    total_batches = len(data_loader)
    half_point = max(1, total_batches // 2)
    progress_points = {
        max(1, min(total_batches, int(round(k * total_batches / 10))))
        for k in range(1, 11)
    }

    autocast_enabled = bool(use_bf16 and device.type == "cuda")

    for batch_idx, (video_batch, labels) in enumerate(data_loader, start=1):
        video_batch = video_batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if autocast_enabled
            else nullcontext()
        )

        with amp_ctx:
            logits = model(video_batch)
            loss = loss_fn(logits, labels)

        loss.backward()

        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()

        running_loss += float(loss.item()) * labels.size(0)
        focus_classes = get_focus_classes(cfg)
        predictions = restrict_predictions_to_classes(logits, focus_classes)
        correct += int((predictions == labels).sum().item())
        total += labels.size(0)

        if (
            mid_epoch_validation
            and val_loader is not None
            and checkpoint_path is not None
            and batch_idx == half_point
        ):
            val_loss_mid, val_acc_mid = evaluate_epoch(
                model=model,
                data_loader=val_loader,
                loss_fn=loss_fn,
                device=device,
                use_bf16=use_bf16,
                focus_classes=get_focus_classes(cfg),
            )

            print(
                f"Epoch {epoch_idx + 1}/{total_epochs} | mid-epoch "
                f"(after batch {batch_idx}/{total_batches}) | "
                f"val loss {val_loss_mid:.4f} acc {val_acc_mid:.4f}"
            )

            if run is not None:
                run.log(
                    {
                        "epoch": epoch_idx + 1,
                        "mid_val/loss": val_loss_mid,
                        "mid_val/acc": val_acc_mid,
                    }
                )

            if val_acc_mid > best_val_accuracy:
                best_val_accuracy = val_acc_mid
                save_checkpoint(
                    cfg=cfg,
                    model=model,
                    val_acc=val_acc_mid,
                    checkpoint_path=checkpoint_path,
                    reason="mid-epoch",
                )

                if run is not None:
                    run.summary["best_val_acc"] = val_acc_mid
                    run.summary["best_checkpoint_path"] = str(checkpoint_path)

            model.train()

        if batch_idx in progress_points:
            running_average_loss = running_loss / max(total, 1)
            running_accuracy = correct / max(total, 1)

            print(
                f"Epoch {epoch_idx + 1}/{total_epochs} | "
                f"step {batch_idx}/{total_batches} | "
                f"train loss {running_average_loss:.4f} acc {running_accuracy:.4f}"
            )

    average_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return average_loss, accuracy, best_val_accuracy


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    use_bf16: bool = False,
    focus_classes: Optional[List[int]] = None,
) -> Tuple[float, float]:
    """Returns (average loss, top-1 accuracy) on validation loader."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    autocast_enabled = bool(use_bf16 and device.type == "cuda")

    for video_batch, labels in data_loader:
        video_batch = video_batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if autocast_enabled
            else nullcontext()
        )

        with amp_ctx:
            logits = model(video_batch)
            loss = loss_fn(logits, labels)

        running_loss += float(loss.item()) * labels.size(0)
        focus_classes = get_focus_classes(cfg)
        predictions = restrict_predictions_to_classes(logits, focus_classes)
        correct += int((predictions == labels).sum().item())
        total += labels.size(0)

    average_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return average_loss, accuracy


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    use_wandb = bool(cfg.training.get("use_wandb", True))
    run = None

    if use_wandb and wandb is not None:
        run_name = build_wandb_run_name(cfg)
        run = wandb.init(
            project=str(cfg.training.get("wandb_project", "what-happens-next")),
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        print(f"W&B run name: {run_name}")
    elif use_wandb and wandb is None:
        print("W&B requested but wandb is not installed. Continuing without W&B.")

    try:
        seed = int(cfg.dataset.seed)
        set_seed(seed)

        device_str = str(cfg.training.device)
        if device_str == "cuda" and not torch.cuda.is_available():
            print("CUDA not available; using CPU.")
            device_str = "cpu"
        device = torch.device(device_str)

        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        use_bf16 = bool(cfg.training.get("use_bf16", False))
        if use_bf16 and device.type == "cuda":
            if not torch.cuda.is_bf16_supported():
                print(
                    "BF16 requested but not supported on this GPU. "
                    "Falling back to full precision."
                )
                use_bf16 = False

        train_dir = Path(cfg.dataset.train_dir).resolve()
        val_dir = Path(cfg.dataset.val_dir).resolve()

        train_samples = collect_video_samples(train_dir)
        val_samples = collect_video_samples(val_dir)

        focus_classes = get_focus_classes(cfg)
        if focus_classes:
            print(f"\n=== Specialist mode: focus_classes={focus_classes} ===")
            before_train = len(train_samples)
            before_val = len(val_samples)

            train_samples = filter_samples_by_classes(train_samples, focus_classes)
            val_samples = filter_samples_by_classes(val_samples, focus_classes)

            print(f"Train samples: {before_train} -> {len(train_samples)}")
            print(f"Val samples:   {before_val} -> {len(val_samples)}")

            if len(train_samples) == 0:
                raise ValueError(
                    f"No train samples left after filtering focus_classes={focus_classes}."
                )
            if len(val_samples) == 0:
                raise ValueError(
                    f"No val samples left after filtering focus_classes={focus_classes}."
                )

        max_samples = cfg.dataset.get("max_samples")
        train_samples = subsample_samples(train_samples, max_samples, seed=seed)
        val_samples = subsample_samples(val_samples, max_samples, seed=seed + 1)

        class_names = build_class_index_to_name(train_dir)
        original_counts = count_videos_per_class(train_samples)
        print_class_video_counts(
            original_counts,
            class_names,
            "\n=== Train set: original videos per class ===",
        )

        if focus_classes:
            print(
                "\n=== Train set: specialist subset, no oversampling ===\n"
                "Using only focus_classes videos + class-weighted CrossEntropyLoss.\n"
                "Train/val accuracy is computed with argmax restricted to focus_classes."
            )
        else:
            print(
                "\n=== Train set: no oversampling ===\n"
                "Using original train videos + class-weighted CrossEntropyLoss."
            )

        use_imagenet_norm = bool(cfg.model.pretrained)
        image_size = int(cfg.training.get("image_size", 224))
        zoom_prob = float(cfg.training.get("crop_zoom_prob", 0.65))
        crop_scale_lo = float(cfg.training.get("crop_scale_min", 0.45))
        crop_scale_hi = float(cfg.training.get("crop_scale_max", 1.0))
        train_augment_repeats = int(cfg.training.get("train_augment_repeats", 1))

        eval_transform = build_transforms(
            is_training=False,
            use_imagenet_norm=use_imagenet_norm,
            image_size=image_size,
        )

        placeholder_eval = _eval_style_resize_only_transform(
            image_size=image_size,
            use_imagenet_norm=use_imagenet_norm,
        )

        train_dataset = VideoFrameDatasetZoomAugment(
            root_dir=train_dir,
            num_frames=int(cfg.dataset.num_frames),
            transform_eval_style=placeholder_eval,
            sample_list=train_samples,
            image_size=image_size,
            zoom_prob=zoom_prob,
            crop_scale=(crop_scale_lo, crop_scale_hi),
            use_imagenet_norm=use_imagenet_norm,
            augment_repeats=train_augment_repeats,
            affine_translate_frac=float(cfg.training.get("affine_translate_frac", 0.06)),
            grayscale_prob=float(cfg.training.get("grayscale_prob", 0.18)),
            sharpness_delta=float(cfg.training.get("sharpness_delta", 0.55)),
            color_brightness=float(cfg.training.get("color_jitter_brightness", 0.35)),
            color_contrast=float(cfg.training.get("color_jitter_contrast", 0.35)),
            color_saturation=float(cfg.training.get("color_jitter_saturation", 0.38)),
            color_hue=float(cfg.training.get("color_jitter_hue", 0.06)),
            blur_prob=float(cfg.training.get("blur_prob", 0.22)),
            blur_sigma_lo=float(cfg.training.get("blur_sigma_min", 0.15)),
            blur_sigma_hi=float(cfg.training.get("blur_sigma_max", 1.1)),
            erase_prob=float(cfg.training.get("erase_prob", 0.28)),
            erase_area_lo=float(cfg.training.get("erase_area_min", 0.02)),
            erase_area_hi=float(cfg.training.get("erase_area_max", 0.12)),
            horizontal_flip_prob=float(cfg.training.get("horizontal_flip_prob", 0.5)),
        )

        print(
            f"\nTrain (original, weighted loss): {len(train_samples)} clips "
            f"→ {len(train_dataset)} samples/epoch | "
            f"augment_repeats={train_augment_repeats} | "
            f"aug sync (color, zoom, blur, erase, translate, hflip), "
            f"hflip_prob={float(cfg.training.get('horizontal_flip_prob', 0.5))}."
        )

        val_dataset = VideoFrameDataset(
            root_dir=val_dir,
            num_frames=int(cfg.dataset.num_frames),
            transform=eval_transform,
            sample_list=val_samples,
        )

        num_workers = int(cfg.training.num_workers)

        train_loader_kwargs = {
            "dataset": train_dataset,
            "batch_size": int(cfg.training.batch_size),
            "shuffle": True,
            "num_workers": num_workers,
            "pin_memory": (device.type == "cuda"),
        }
        val_loader_kwargs = {
            "dataset": val_dataset,
            "batch_size": int(cfg.training.batch_size),
            "shuffle": False,
            "num_workers": num_workers,
            "pin_memory": (device.type == "cuda"),
        }

        if num_workers > 0:
            train_loader_kwargs["persistent_workers"] = True
            val_loader_kwargs["persistent_workers"] = True

        train_loader = DataLoader(**train_loader_kwargs)
        val_loader = DataLoader(**val_loader_kwargs)

        model = build_model(cfg).to(device)

        best_val_accuracy = 0.0
        resume_path = cfg.training.get("resume_from_checkpoint", None)

        if resume_path:
            resume_path = Path(str(resume_path)).resolve()
            checkpoint = torch.load(resume_path, map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            best_val_accuracy = float(checkpoint.get("val_accuracy", 0.0))
            print(f"Loaded checkpoint from {resume_path}")
            print(f"Resumed best_val_accuracy = {best_val_accuracy:.4f}")

        class_weight_alpha = float(cfg.training.get("class_weight_alpha", 0.5))
        label_smoothing = float(cfg.training.get("label_smoothing", 0.0))
        class_weights = build_class_weights(
            train_samples,
            num_classes=int(cfg.model.num_classes),
            alpha=class_weight_alpha,
        ).to(device)

        print(
            f"Loss: weighted CrossEntropyLoss | "
            f"class_weight_alpha={class_weight_alpha} | "
            f"label_smoothing={label_smoothing}"
        )
        print("Class weights:", class_weights.detach().cpu().tolist())

        loss_fn = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )

        backbone_lr = float(cfg.training.get("backbone_lr", 1e-5))
        head_lr = float(cfg.training.get("head_lr", cfg.training.get("lr", 3e-4)))
        weight_decay = float(cfg.training.get("weight_decay", 1e-4))

        optimizer = build_optimizer(
            model=model,
            backbone_lr=backbone_lr,
            head_lr=head_lr,
            weight_decay=weight_decay,
        )

        scheduler = build_scheduler(cfg, optimizer)

        grad_clip_norm = cfg.training.get("grad_clip_norm", None)
        if grad_clip_norm is not None:
            grad_clip_norm = float(grad_clip_norm)

        checkpoint_path = Path(cfg.training.checkpoint_path).resolve()
        do_mid_val = bool(cfg.training.get("mid_epoch_validation", True))

        for epoch in range(int(cfg.training.epochs)):
            train_loss, train_acc, best_val_accuracy = train_one_epoch(
                model=model,
                data_loader=train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device,
                epoch_idx=epoch,
                total_epochs=int(cfg.training.epochs),
                cfg=cfg,
                use_bf16=use_bf16,
                grad_clip_norm=grad_clip_norm,
                val_loader=val_loader,
                checkpoint_path=checkpoint_path,
                best_val_accuracy=best_val_accuracy,
                mid_epoch_validation=do_mid_val,
                run=run,
            )

            val_loss, val_acc = evaluate_epoch(
                model=model,
                data_loader=val_loader,
                loss_fn=loss_fn,
                device=device,
                use_bf16=use_bf16,
                focus_classes=focus_classes,
            )

            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                save_checkpoint(
                    cfg=cfg,
                    model=model,
                    val_acc=val_acc,
                    checkpoint_path=checkpoint_path,
                    reason="end-epoch",
                )

                if run is not None:
                    run.summary["best_val_acc"] = val_acc
                    run.summary["best_checkpoint_path"] = str(checkpoint_path)

            print(
                f"Epoch {epoch + 1}/{cfg.training.epochs} | "
                f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.4f}"
            )

            if scheduler is not None:
                scheduler_name = str(cfg.training.get("scheduler", "none")).lower()
                if scheduler_name == "plateau":
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            log_dict = {
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "best_val_acc": best_val_accuracy,
            }
            if focus_classes:
                log_dict["specialist/focus_classes"] = ",".join(str(c) for c in focus_classes)

            for i, group in enumerate(optimizer.param_groups):
                group_name = str(group.get("name", f"group_{i}"))
                log_dict[f"lr/{group_name}"] = group["lr"]

            if run is not None:
                run.log(log_dict)

        print(f"Done. Best validation accuracy: {best_val_accuracy:.4f}")

    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()