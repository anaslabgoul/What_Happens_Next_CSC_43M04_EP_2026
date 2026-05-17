"""
Entraînement identique à ``train_equilibre.py``, avec **équilibrage des classes** sur le train,
mais **sans** multiplier les clips par 5 après équilibrage (``augment_repeats=1``).

Avant l'entraînement, les classes minoritaires sont sur-échantillonnées (répétition de
vidéos avec remplacement) jusqu'à atteindre le nombre de vidéos de la classe majoritaire.
Les comptes par classe **originaux** et **après équilibrage** sont affichés.

Les augmentations fortes (zoom, couleur, flou, erasing, translation — sans flip) restent
inchangées ; chaque clip équilibré est vu une fois par epoch.

Lancer depuis ``src/``::

    python train_equilibre2.py
    python train_equilibre2.py experiment=cnn_lstm
"""

from __future__ import annotations

import random
from collections import defaultdict
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

from dataset.video_dataset import (
    VideoFrameDataset,
    _list_frame_paths,
    _parse_class_index,
    _pick_frame_indices,
    collect_video_samples,
)
from models.cnn_baseline import CNNBaseline
from models.cnn_lstm import CNNLSTM
from models.cnntransformer import CNNTransformer
from models.cnntransformer_2 import CNNTransformer_2
from models.ResNet3DBaseline import ResNet3DBaseline
from utils import build_transforms, set_seed


def build_model(cfg: DictConfig) -> nn.Module:
    """Create the model described by cfg.model.name."""
    name = cfg.model.name
    num_classes = cfg.model.num_classes
    pretrained = cfg.model.pretrained

    if name == "cnn_baseline":
        return CNNBaseline(num_classes=num_classes, pretrained=pretrained)
    if name == "cnn_lstm":
        hidden = cfg.model.get("lstm_hidden_size", 512)
        return CNNLSTM(
            num_classes=num_classes,
            pretrained=pretrained,
            lstm_hidden_size=int(hidden),
        )
    if name == "ResNet3DBaseline":
        return ResNet3DBaseline(
            num_classes=num_classes,
            pretrained=pretrained,
        )
    if name in ("cnntransformer", "cnn_transformer"):
        return build_cnntransformer_model(cfg.model)

    if name == "cnntransformer_2":
        return CNNTransformer_2(
            num_classes=num_classes,
            pretrained=pretrained
        )


    raise ValueError(f"Unknown model.name: {name}")


def _affine_translate_pil_sync(
    pil_frames: List[Image.Image],
    max_frac: float,
) -> List[Image.Image]:
    """Small zero-angle translation; same (tx, ty) in pixels for every frame (no flips)."""
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


def _color_jitter_pil_one(pil: Image.Image, b: float, c: float, s: float, hue: float) -> Image.Image:
    """Apply brightness / contrast / saturation / hue multipliers (same recipe on each frame)."""
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
    """Sample multiplicative brightness/contrast/saturation and additive hue (PIL range)."""
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
    """Same photometric rollouts on every frame (temporal consistency). No horizontal flip."""
    b, c, s, h = _sample_color_jitter_factors(brightness, contrast, saturation, hue)
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
    """In-place: same sigma, odd kernel, applied to each C×H×W tensor in [0, 1]."""
    if prob <= 0 or not tensors_01 or random.random() >= prob:
        return
    sigma = random.uniform(sigma_lo, sigma_hi)
    k = int(random.choice([3, 5]))
    for i, t in enumerate(tensors_01):
        tensors_01[i] = F_t.gaussian_blur(t, kernel_size=[k, k], sigma=[sigma, sigma])


def _random_erasing_sync(
    tensors_01: List[torch.Tensor],
    prob: float,
    area_lo: float,
    area_hi: float,
) -> None:
    """Same random rectangle on every frame; values in [0,1] before normalize."""
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
    Dataset train avec **forte augmentation** : zoom/crop partagé, couleurs, netteté,
    flou léger, erasing synchronisé, translation — **sans aucun flip** (sens spatial préservé).

    ``augment_repeats`` multiplie ``__len__`` pour revoir chaque vidéo plusieurs fois par
    epoch avec de nouveaux tirages aléatoires (défaut 5).
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
        augment_repeats: int = 5,
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
        if use_imagenet_norm:
            self.normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        else:
            self.normalize = transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
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

        resized_list = _affine_translate_pil_sync(
            resized_list, self.affine_translate_frac
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
    image_size: int, use_imagenet_norm: bool
) -> transforms.Compose:
    """Resize + ToTensor + Normalize (placeholder parent API; train aug is in the dataset)."""
    if use_imagenet_norm:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    else:
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )


def count_videos_per_class(samples: List[Tuple[Path, int]]) -> Dict[int, int]:
    """Return {class_index: number of unique video entries in samples}."""
    counts: Dict[int, int] = defaultdict(int)
    for _, label in samples:
        counts[int(label)] += 1
    return dict(counts)


def build_class_index_to_name(train_dir: Path) -> Dict[int, str]:
    """Map class index to folder name (e.g. 017_Opening something)."""
    class_dirs = [p for p in sorted(train_dir.iterdir()) if p.is_dir()]
    fallback_index = {p.name: i for i, p in enumerate(class_dirs)}
    mapping: Dict[int, str] = {}
    for class_dir in class_dirs:
        parsed = _parse_class_index(class_dir.name)
        class_index = parsed if parsed is not None else fallback_index[class_dir.name]
        mapping[class_index] = class_dir.name
    return mapping


def balance_video_samples(
    samples: List[Tuple[Path, int]],
    *,
    seed: int,
) -> List[Tuple[Path, int]]:
    """
    Oversample minority classes with replacement until every class has the same
    number of videos as the majority class.
    """
    by_class: Dict[int, List[Tuple[Path, int]]] = defaultdict(list)
    for item in samples:
        by_class[int(item[1])].append(item)

    if not by_class:
        return list(samples)

    target = max(len(pool) for pool in by_class.values())
    rng = random.Random(seed)
    balanced: List[Tuple[Path, int]] = []

    for class_index in sorted(by_class.keys()):
        pool = list(by_class[class_index])
        n = len(pool)
        balanced.extend(pool)
        if n < target:
            for _ in range(target - n):
                balanced.append(rng.choice(pool))

    rng.shuffle(balanced)
    return balanced


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


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch_idx: int,
    total_epochs: int,
    val_loader: Optional[DataLoader] = None,
    cfg: Optional[DictConfig] = None,
    checkpoint_path: Optional[Path] = None,
    best_val_accuracy: float = 0.0,
    mid_epoch_validation: bool = True,
) -> Tuple[float, float, float]:
    """
    Entraîne sur toute l'epoch. Si ``mid_epoch_validation`` et ``val_loader`` sont fournis,
    après ``len(data_loader) // 2`` batches : validation, affichage, sauvegarde si meilleur.

    Retourne (train_loss, train_acc, best_val_accuracy_mis_à_jour).
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

    for batch_idx, (video_batch, labels) in enumerate(data_loader, start=1):
        video_batch = video_batch.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(video_batch)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += labels.size(0)

        if (
            mid_epoch_validation
            and val_loader is not None
            and cfg is not None
            and checkpoint_path is not None
            and batch_idx == half_point
        ):
            val_loss_mid, val_acc_mid = evaluate_epoch(
                model, val_loader, loss_fn, device
            )
            print(
                f"Epoch {epoch_idx + 1}/{total_epochs} | mid-epoch "
                f"(after batch {batch_idx}/{total_batches}) | "
                f"val loss {val_loss_mid:.4f} acc {val_acc_mid:.4f}"
            )
            if val_acc_mid > best_val_accuracy:
                best_val_accuracy = val_acc_mid
                torch.save(
                    _checkpoint_payload(cfg, model, val_acc_mid),
                    checkpoint_path,
                )
                print(
                    f"  Saved new best model to {checkpoint_path} "
                    f"(mid-epoch val acc={val_acc_mid:.4f})"
                )
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
) -> Tuple[float, float]:
    """Returns (average loss, top-1 accuracy) on the validation loader."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for video_batch, labels in data_loader:
        video_batch = video_batch.to(device)
        labels = labels.to(device)

        logits = model(video_batch)
        loss = loss_fn(logits, labels)

        running_loss += float(loss.item()) * labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += labels.size(0)

    average_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return average_loss, accuracy


def _checkpoint_payload(cfg: DictConfig, model: nn.Module, val_acc: float) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "model_name": cfg.model.name,
        "num_classes": int(cfg.model.num_classes),
        "pretrained": bool(cfg.model.pretrained),
        "num_frames": int(cfg.dataset.num_frames),
        "val_accuracy": val_acc,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    if cfg.model.name == "cnn_lstm":
        payload["lstm_hidden_size"] = int(cfg.model.get("lstm_hidden_size", 512))
    return payload


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    seed = int(cfg.dataset.seed)
    set_seed(seed)

    device_str = cfg.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    train_dir = Path(cfg.dataset.train_dir).resolve()
    val_dir = Path(cfg.dataset.val_dir).resolve()

    train_samples = collect_video_samples(train_dir)
    val_samples = collect_video_samples(val_dir)

    max_samples = cfg.dataset.get("max_samples")
    if max_samples is not None:
        max_samples = int(max_samples)
        train_samples = train_samples[:max_samples]
        val_samples = val_samples[:max_samples]

    class_names = build_class_index_to_name(train_dir)
    original_counts = count_videos_per_class(train_samples)
    print_class_video_counts(
        original_counts,
        class_names,
        "\n=== Train set: original videos per class ===",
    )

    train_samples_balanced = balance_video_samples(train_samples, seed=seed)
    balanced_counts = count_videos_per_class(train_samples_balanced)
    print_class_video_counts(
        balanced_counts,
        class_names,
        "\n=== Train set: videos per class after balancing (oversampling) ===",
    )

    use_imagenet_norm = bool(cfg.model.pretrained)
    image_size = int(cfg.training.get("image_size", 224))
    zoom_prob = float(cfg.training.get("crop_zoom_prob", 0.65))
    crop_scale_lo = float(cfg.training.get("crop_scale_min", 0.45))
    crop_scale_hi = float(cfg.training.get("crop_scale_max", 1.0))
    train_augment_repeats = 1

    eval_transform = build_transforms(
        is_training=False,
        use_imagenet_norm=use_imagenet_norm,
        image_size=image_size,
    )
    placeholder_eval = _eval_style_resize_only_transform(
        image_size=image_size, use_imagenet_norm=use_imagenet_norm
    )

    train_dataset = VideoFrameDatasetZoomAugment(
        root_dir=train_dir,
        num_frames=int(cfg.dataset.num_frames),
        transform_eval_style=placeholder_eval,
        sample_list=train_samples_balanced,
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
    )
    print(
        f"\nTrain (balanced): {len(train_samples_balanced)} clips "
        f"→ {len(train_dataset)} samples/epoch | strong aug (color, zoom, blur, erase, "
        f"translate), no flips."
    )
    val_dataset = VideoFrameDataset(
        root_dir=val_dir,
        num_frames=int(cfg.dataset.num_frames),
        transform=eval_transform,
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

    model = build_model(cfg).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.training.lr))

    best_val_accuracy = 0.0
    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()
    do_mid_val = bool(cfg.training.get("mid_epoch_validation", True))

    for epoch in range(int(cfg.training.epochs)):
        train_loss, train_acc, best_val_accuracy = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            epoch,
            int(cfg.training.epochs),
            val_loader=val_loader,
            cfg=cfg,
            checkpoint_path=checkpoint_path,
            best_val_accuracy=best_val_accuracy,
            mid_epoch_validation=do_mid_val,
        )
        val_loss, val_acc = evaluate_epoch(model, val_loader, loss_fn, device)

        print(
            f"Epoch {epoch + 1}/{cfg.training.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(
                _checkpoint_payload(cfg, model, val_acc),
                checkpoint_path,
            )
            print(
                f"  Saved new best model to {checkpoint_path} (val acc={val_acc:.4f})"
            )

    print(f"Done. Best validation accuracy: {best_val_accuracy:.4f}")


if __name__ == "__main__":
    main()
