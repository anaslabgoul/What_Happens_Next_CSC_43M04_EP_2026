"""
Evaluate a saved checkpoint on the full validation split.

This version avoids Hydra/checkpoint config mismatches:
- the model is rebuilt from the config saved inside the checkpoint;
- num_frames, image_size, normalization, target_num_frames and frame_resampling
  therefore match the checkpoint training setup;
- Hydra is still used for runtime options such as checkpoint_path, batch_size,
  num_workers and device.

Example from project root:

    python src/evaluate.py \
      training.checkpoint_path=/path/to/best_model.pt \
      training.batch_size=1 \
      training.num_workers=4
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from train_weighted import build_model
from utils import build_transforms, set_seed


def _get_nested(cfg: DictConfig, dotted_key: str, default: Any = None) -> Any:
    """Safe dotted-key getter for OmegaConf DictConfig."""
    cur: Any = cfg
    for part in dotted_key.split("."):
        if cur is None or not hasattr(cur, "get"):
            return default
        cur = cur.get(part, default)
    return cur


def _prepare_cuda(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print(f"Loading checkpoint: {checkpoint_path}", flush=True)
    checkpoint: Dict[str, Any] = torch.load(checkpoint_path, map_location="cpu")
    if "config" not in checkpoint or checkpoint["config"] is None:
        raise ValueError(
            "Checkpoint has no 'config' entry. Train with the current training "
            "script so the full Hydra config is saved with the weights."
        )
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint has no 'model_state_dict' entry.")
    return checkpoint


def build_model_from_checkpoint(
    checkpoint: Dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, DictConfig]:
    """Rebuild model from checkpoint config and load checkpoint weights."""
    ckpt_cfg = OmegaConf.create(checkpoint["config"])

    print("=== CHECKPOINT MODEL CFG ===")
    print(OmegaConf.to_yaml(ckpt_cfg.model), flush=True)

    model = build_model(ckpt_cfg)
    missing, unexpected = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    if missing or unexpected:
        print(f"Missing keys: {missing}", flush=True)
        print(f"Unexpected keys: {unexpected}", flush=True)

    model.to(device)
    model.eval()
    print(f"Model loaded on device: {device}", flush=True)
    return model, ckpt_cfg


def build_validation_loader(
    hydra_cfg: DictConfig,
    ckpt_cfg: DictConfig,
    checkpoint: Dict[str, Any],
    device: torch.device,
) -> DataLoader:
    """Build validation loader with checkpoint preprocessing settings."""
    pretrained_used = bool(checkpoint.get("pretrained", _get_nested(ckpt_cfg, "model.pretrained", True)))
    num_frames = int(checkpoint.get("num_frames", _get_nested(ckpt_cfg, "dataset.num_frames", hydra_cfg.dataset.num_frames)))
    image_size = int(_get_nested(ckpt_cfg, "training.image_size", _get_nested(hydra_cfg, "training.image_size", 224)))

    # Use the validation directory saved in the checkpoint. This is the safest
    # way to reproduce the checkpoint's stored validation accuracy.
    val_dir = Path(str(_get_nested(ckpt_cfg, "dataset.val_dir", hydra_cfg.dataset.val_dir))).resolve()

    print("=== EVAL SETTINGS ===", flush=True)
    print(f"val_dir: {val_dir}", flush=True)
    print(f"num_frames: {num_frames}", flush=True)
    print(f"image_size: {image_size}", flush=True)
    print(f"pretrained normalization: {pretrained_used}", flush=True)
    print(
        "model target_num_frames: "
        f"{_get_nested(ckpt_cfg, 'model.target_num_frames', None)}",
        flush=True,
    )
    print(
        "model frame_resampling: "
        f"{_get_nested(ckpt_cfg, 'model.frame_resampling', None)}",
        flush=True,
    )

    eval_transform = build_transforms(
        is_training=False,
        use_imagenet_norm=pretrained_used,
        image_size=image_size,
    )

    val_samples = collect_video_samples(val_dir)
    max_samples = hydra_cfg.dataset.get("max_samples", None)
    if max_samples is not None:
        val_samples = val_samples[: int(max_samples)]
        print(f"Using max_samples={max_samples}: {len(val_samples)} samples", flush=True)

    val_dataset = VideoFrameDataset(
        root_dir=val_dir,
        num_frames=num_frames,
        transform=eval_transform,
        sample_list=val_samples,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(hydra_cfg.training.batch_size),
        shuffle=False,
        num_workers=int(hydra_cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(int(hydra_cfg.training.num_workers) > 0),
    )

    print(
        f"Starting evaluation: {len(val_dataset)} samples, "
        f"batch_size={int(hydra_cfg.training.batch_size)}, "
        f"num_workers={int(hydra_cfg.training.num_workers)}, "
        f"{len(val_loader)} batches",
        flush=True,
    )
    return val_loader


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    n_batches = len(val_loader)
    log_interval = max(1, n_batches // 10)

    model.eval()
    for batch_idx, (video_batch, labels) in enumerate(val_loader, start=1):
        video_batch = video_batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(video_batch)

        predictions_top1 = logits.argmax(dim=1)
        correct_top1 += int((predictions_top1 == labels).sum().item())

        k = min(5, logits.shape[1])
        _, predictions_topk = logits.topk(k, dim=1, largest=True, sorted=True)
        matches_top5 = predictions_topk.eq(labels.view(-1, 1)).any(dim=1)
        correct_top5 += int(matches_top5.sum().item())

        total += labels.size(0)

        if batch_idx == 1 or batch_idx % log_interval == 0 or batch_idx == n_batches:
            running_top1 = correct_top1 / max(total, 1)
            running_top5 = correct_top5 / max(total, 1)
            print(
                f"[EVAL] batch {batch_idx}/{n_batches} | "
                f"seen={total} | top1={running_top1:.4f} | top5={running_top5:.4f}",
                flush=True,
            )

    top1_accuracy = correct_top1 / max(total, 1)
    top5_accuracy = correct_top5 / max(total, 1)
    return top1_accuracy, top5_accuracy


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print("=== HYDRA RUNTIME CFG ===")
    print(OmegaConf.to_yaml(cfg), flush=True)

    set_seed(int(cfg.dataset.seed))

    device_str = str(cfg.training.device)
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.", flush=True)
        device_str = "cpu"
    device = torch.device(device_str)
    _prepare_cuda(device)

    checkpoint_path = Path(str(cfg.training.checkpoint_path)).resolve()
    checkpoint = load_checkpoint(checkpoint_path)

    saved_val = checkpoint.get("val_accuracy", None)
    if saved_val is not None:
        print(f"Checkpoint saved val_accuracy: {float(saved_val):.4f}", flush=True)

    model, ckpt_cfg = build_model_from_checkpoint(checkpoint, device)
    val_loader = build_validation_loader(cfg, ckpt_cfg, checkpoint, device)

    top1_accuracy, top5_accuracy = evaluate(model, val_loader, device)

    print("=== FINAL RESULTS ===", flush=True)
    print(f"Validation samples: {len(val_loader.dataset)}", flush=True)
    print(f"Top-1 accuracy: {top1_accuracy:.4f}", flush=True)
    print(f"Top-5 accuracy: {top5_accuracy:.4f}", flush=True)


if __name__ == "__main__":
    main()
