"""
Entraînement comme ``train.py`` avec **forte augmentation** vidéo sur le train (sans miroir).

Les données proviennent de ``dataset.train_dir`` (p.ex. ``processed_data/val2/train``).

**Techniques d’augmentation** (mêmes effets que la version précédente, mais **une seule
technique tirée au hasard par batch** parmi) :
  1. ``RandomResizedCrop`` cohérent dans le temps (même recadrage sur toutes les frames) ;
  2. **translation** affine (angle=0), identique sur toute la séquence ;
  3. **photométrie** : jitter luminosité / contraste / saturation / teinte, netteté,
     niveau de gris (mêmes paramètres sur chaque frame du clip) ;
  4. **flou gaussien** léger sur tenseur (même sigma / noyau sur toutes les frames) ;
  5. **Random Erasing** synchronisé (même rectangle sur toutes les frames).

Chaque batch est vu **une fois** par epoch (plus de multiplicateur ``train_augment_repeats``).

Pendant l’epoch : comme ``train.py``, jusqu’à **10** lignes de progression (loss / acc
cumulés sur les batches déjà vus, aux pas ~10 %, 20 %, …, 100 % du loader). Au lancement :
affichage de la config puis une ligne récapitulative (taille du train, device).
À la fin de chaque epoch : même ligne récapitulative que ``train.py`` (train loss / acc et
val loss / acc), puis enregistrement du meilleur checkpoint sur ``dataset.val_dir`` si la
précision validation augmente. Les logs Hydra sont aussi écrits dans
``outputs/<date>_<heure>/train_crops.log``.

Lancer depuis ``src/``::

    python train_crops.py
    python train_crops.py experiment=cnn_lstm
    python train_crops.py experiment=maxvit_lstm
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F_t
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchvision.transforms import RandomResizedCrop

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from models.cnn_baseline import CNNBaseline
from models.cnn_frame_temporal import CNNFrameTemporal
from models.cnn_lstm import CNNLSTM
from models.cnntransformer import CNNTransformer
from models.cnntransformer_2 import CNNTransformer2
from models.maxvit_lstm import MaxVitLSTM
from utils import build_transforms, set_seed


# Une technique par batch (même famille que l’ancien pipeline, sans flip).
AUG_TECHNIQUES: Tuple[str, ...] = (
    "resized_crop",
    "affine_translate",
    "photometric",
    "gaussian_blur",
    "random_erasing",
)


def build_model(cfg: DictConfig) -> nn.Module:
    """Create the model described by cfg.model.name."""
    name = cfg.model.name
    num_classes = cfg.model.num_classes
    pretrained = cfg.model.pretrained

    if name == "cnn_baseline":
        return CNNBaseline(num_classes=num_classes, pretrained=pretrained)
    if name == "cnn_lstm":
        return CNNLSTM.from_config(cfg.model)
    if name == "cnn_frame_temporal":
        return CNNFrameTemporal.from_config(cfg.model)

    if name in ("cnntransformer", "cnn_transformer"):
        return CNNTransformer.from_config(
            cfg.model,
            dataset_num_frames=int(cfg.dataset.num_frames),
        )

    if name == "cnntransformer_2":
        return CNNTransformer2.from_config(
            cfg.model,
            dataset_num_frames=int(cfg.dataset.num_frames),
        )

    if name == "maxvit_lstm":
        return MaxVitLSTM.from_config(cfg.model)

    raise ValueError(f"Unknown model.name: {name}")


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


def _color_jitter_tensor_one(
    x: torch.Tensor, b: float, c: float, s: float, hue: float
) -> torch.Tensor:
    """Apply brightness / contrast / saturation / hue (same recipe as PIL path)."""
    y = F_t.adjust_brightness(x, b)
    y = F_t.adjust_contrast(y, c)
    y = F_t.adjust_saturation(y, s)
    y = F_t.adjust_hue(y, hue)
    return y


def _photometric_tensor_clip(
    clip_tchw: torch.Tensor,
    grayscale_prob: float,
    sharpness_delta: float,
    brightness: float,
    contrast: float,
    saturation: float,
    hue: float,
) -> torch.Tensor:
    """T, C, H, W in [0,1]. Same rollouts on every frame (temporal consistency)."""
    b, c, s, h = _sample_color_jitter_factors(brightness, contrast, saturation, hue)
    sharp = 1.0 + (random.random() * 2.0 - 1.0) * sharpness_delta
    sharp = max(0.05, sharp)
    do_gray = random.random() < grayscale_prob

    t_dim = clip_tchw.shape[0]
    out_list: List[torch.Tensor] = []
    for ti in range(t_dim):
        y = _color_jitter_tensor_one(clip_tchw[ti], b, c, s, h)
        y = F_t.adjust_sharpness(y, sharp)
        if do_gray:
            y = F_t.rgb_to_grayscale(y, num_output_channels=3)
        out_list.append(y)
    return torch.stack(out_list, dim=0)


def _random_erasing_sync_tensors(
    tensors_01: List[torch.Tensor],
    area_lo: float,
    area_hi: float,
) -> None:
    """Same random rectangle on every frame; values in [0,1]. In-place."""
    if not tensors_01:
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
        noise = torch.rand(c, eh, ew, device=tensors_01[0].device, dtype=tensors_01[0].dtype)
        for t in tensors_01:
            t[:, top : top + eh, left : left + ew] = noise
        return


def normalize_video_batch(x_btchw: torch.Tensor, use_imagenet_norm: bool) -> torch.Tensor:
    """x: B,T,C,H,W in [0,1] → normalized like ImageNet or 0.5 stats."""
    if use_imagenet_norm:
        mean = torch.tensor(
            [0.485, 0.456, 0.406], device=x_btchw.device, dtype=x_btchw.dtype
        ).view(1, 1, 3, 1, 1)
        std = torch.tensor(
            [0.229, 0.224, 0.225], device=x_btchw.device, dtype=x_btchw.dtype
        ).view(1, 1, 3, 1, 1)
    else:
        mean = torch.full(
            (1, 1, 3, 1, 1), 0.5, device=x_btchw.device, dtype=x_btchw.dtype
        )
        std = torch.full(
            (1, 1, 3, 1, 1), 0.5, device=x_btchw.device, dtype=x_btchw.dtype
        )
    return (x_btchw - mean) / std


def apply_augment_technique(
    x: torch.Tensor,
    technique: str,
    *,
    image_size: int,
    crop_scale: Tuple[float, float],
    crop_ratio: Tuple[float, float],
    affine_translate_frac: float,
    grayscale_prob: float,
    sharpness_delta: float,
    color_brightness: float,
    color_contrast: float,
    color_saturation: float,
    color_hue: float,
    blur_sigma_lo: float,
    blur_sigma_hi: float,
    erase_area_lo: float,
    erase_area_hi: float,
) -> torch.Tensor:
    """
    Apply exactly one augmentation family to x (B, T, C, H, W), values in [0, 1].
    Temporal consistency is preserved within each clip (same b).
    """
    bsz, t_dim, _, h, w = x.shape
    out = x.clone()

    if technique == "resized_crop":
        for bi in range(bsz):
            ref = out[bi, 0]
            i, j, hh, ww = RandomResizedCrop.get_params(ref, scale=list(crop_scale), ratio=list(crop_ratio))
            for ti in range(t_dim):
                out[bi, ti] = F_t.resized_crop(
                    out[bi, ti],
                    i,
                    j,
                    hh,
                    ww,
                    size=[image_size, image_size],
                    interpolation=transforms.InterpolationMode.BILINEAR,
                )
        return out

    if technique == "affine_translate":
        if affine_translate_frac <= 0:
            return out
        for bi in range(bsz):
            max_dx = int(affine_translate_frac * w)
            max_dy = int(affine_translate_frac * h)
            if max_dx <= 0 and max_dy <= 0:
                continue
            tx = random.randint(-max_dx, max_dx) if max_dx > 0 else 0
            ty = random.randint(-max_dy, max_dy) if max_dy > 0 else 0
            if tx == 0 and ty == 0:
                continue
            for ti in range(t_dim):
                out[bi, ti] = F_t.affine(
                    out[bi, ti],
                    angle=0.0,
                    translate=[float(tx), float(ty)],
                    scale=1.0,
                    shear=[0.0, 0.0],
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    fill=[0.0, 0.0, 0.0],
                )
        return out

    if technique == "photometric":
        for bi in range(bsz):
            out[bi] = _photometric_tensor_clip(
                out[bi],
                grayscale_prob=grayscale_prob,
                sharpness_delta=sharpness_delta,
                brightness=color_brightness,
                contrast=color_contrast,
                saturation=color_saturation,
                hue=color_hue,
            )
        return out

    if technique == "gaussian_blur":
        sigma = random.uniform(blur_sigma_lo, blur_sigma_hi)
        k = int(random.choice([3, 5]))
        for bi in range(bsz):
            for ti in range(t_dim):
                out[bi, ti] = F_t.gaussian_blur(
                    out[bi, ti], kernel_size=[k, k], sigma=[sigma, sigma]
                )
        return out

    if technique == "random_erasing":
        for bi in range(bsz):
            clip_tensors = [out[bi, ti].clone() for ti in range(t_dim)]
            _random_erasing_sync_tensors(
                clip_tensors,
                area_lo=erase_area_lo,
                area_hi=erase_area_hi,
            )
            for ti in range(t_dim):
                out[bi, ti] = clip_tensors[ti]
        return out

    raise ValueError(f"Unknown augmentation technique: {technique}")


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: DictConfig,
    image_size: int,
    use_imagenet_norm: bool,
    epoch_idx: int,
    total_epochs: int,
) -> Tuple[float, float]:
    """Entraîne une epoch ; une technique d’aug. aléatoire par batch ; logs ~10 % du loader."""
    model.train()
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

    for batch_idx, (video_batch, labels) in enumerate(data_loader, start=1):
        video_batch = video_batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        tech = random.choice(AUG_TECHNIQUES)
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
        video_batch = normalize_video_batch(video_batch, use_imagenet_norm)

        optimizer.zero_grad()
        logits = model(video_batch)
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
    if cfg.model.name in ("cnn_lstm", "cnn_frame_temporal", "cnntransformer", "cnntransformer_2"):
        payload["model_config"] = OmegaConf.to_container(cfg.model, resolve=True)
    return payload


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg), flush=True)

    set_seed(int(cfg.dataset.seed))

    device_str = cfg.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.", flush=True)
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

    use_imagenet_norm = bool(cfg.model.pretrained)
    image_size = int(cfg.training.get("image_size", 224))

    eval_transform = build_transforms(
        is_training=False,
        use_imagenet_norm=use_imagenet_norm,
        image_size=image_size,
    )
    train_resize_totensor = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    train_dataset = VideoFrameDataset(
        root_dir=train_dir,
        num_frames=int(cfg.dataset.num_frames),
        transform=train_resize_totensor,
        sample_list=train_samples,
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

    n_train = len(train_dataset)
    n_batches = len(train_loader)
    print(
        f"train_crops: {n_train} train clips, {n_batches} batches/epoch, "
        f"device={device} — up to 10 train loss/acc lines per epoch (like train.py).",
        flush=True,
    )

    for epoch in range(int(cfg.training.epochs)):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            cfg,
            image_size=image_size,
            use_imagenet_norm=use_imagenet_norm,
            epoch_idx=epoch,
            total_epochs=int(cfg.training.epochs),
        )
        val_loss, val_acc = evaluate_epoch(model, val_loader, loss_fn, device)

        print(
            f"Epoch {epoch + 1}/{cfg.training.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}",
            flush=True,
        )

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(
                _checkpoint_payload(cfg, model, val_acc),
                checkpoint_path,
            )
            print(
                f"  Saved new best model to {checkpoint_path} (val acc={val_acc:.4f})",
                flush=True,
            )

    print(f"Done. Best validation accuracy: {best_val_accuracy:.4f}", flush=True)


if __name__ == "__main__":
    main()
