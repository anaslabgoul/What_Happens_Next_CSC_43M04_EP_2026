#!/usr/bin/env python3
"""
Evaluate a saved checkpoint on the full validation split.

Outputs:
- top-1 accuracy
- top-5 accuracy
- predictions CSV
- raw confusion matrix CSV
- row-normalized confusion matrix CSV
- per-class metrics CSV
- top confusion pairs CSV

Key point:
- The MODEL and PREPROCESSING are rebuilt from the config saved inside the checkpoint.
- The current Hydra config is used mainly for:
  - training.checkpoint_path
  - training.batch_size
  - training.num_workers
  - training.device
  - optional evaluation.output_dir
  - optional evaluation.use_checkpoint_val_dir

This avoids config mismatches such as:
    Hydra current cfg = cnn_baseline / num_frames=8 / image_size default
    checkpoint cfg     = vjepa        / num_frames=4 / image_size=256
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from dataset.video_dataset import (
    VideoFrameDataset,
    collect_video_samples,
    _parse_class_index,
)
from utils import build_transforms, set_seed


def import_build_model():
    """
    Robust import for build_model.

    Prefer train_weighted because your current V-JEPA training code lives there.
    Fallbacks are kept only to avoid breaking older scripts/checkpoints.
    """
    errors = []

    for module_name in ("train_weighted", "train2w", "train"):
        try:
            module = __import__(module_name, fromlist=["build_model"])
            return module.build_model
        except Exception as exc:
            errors.append(f"{module_name}: {type(exc).__name__}: {exc}")

    joined = "\n".join(errors)
    raise ImportError(
        f"Could not import build_model from train_weighted/train2w/train:\n{joined}"
    )


build_model = import_build_model()


def cfg_get(cfg: DictConfig, dotted_key: str, default: Any) -> Any:
    cur: Any = cfg
    for part in dotted_key.split("."):
        if cur is None:
            return default
        try:
            if part not in cur:
                return default
            cur = cur[part]
        except Exception:
            return default
    return cur


def get_checkpoint_cfg(checkpoint: Dict[str, Any]) -> DictConfig:
    if "config" not in checkpoint or checkpoint["config"] is None:
        raise ValueError(
            "Checkpoint has no 'config' entry. You need a checkpoint saved with "
            "the full Hydra config."
        )
    return OmegaConf.create(checkpoint["config"])


def build_model_from_checkpoint(
    checkpoint: Dict[str, Any],
    device: torch.device,
) -> Tuple[torch.nn.Module, DictConfig]:
    """
    Rebuild the model from the Hydra config stored in the checkpoint,
    load weights, move to device, set eval mode.
    """
    ckpt_cfg = get_checkpoint_cfg(checkpoint)

    model = build_model(ckpt_cfg)
    missing, unexpected = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=False,
    )

    if missing:
        print(f"[WARN] Missing keys when loading checkpoint: {len(missing)}")
        print("       First missing keys:", missing[:10])
    if unexpected:
        print(f"[WARN] Unexpected keys when loading checkpoint: {len(unexpected)}")
        print("       First unexpected keys:", unexpected[:10])

    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint did not load cleanly. Do not evaluate: architecture/config mismatch."
        )

    model.to(device)
    model.eval()
    return model, ckpt_cfg


def build_class_index_to_name(val_dir: Path, num_classes: int) -> Dict[int, str]:
    """
    Build {class_index: folder_name} from validation directory.

    Falls back to class_XXX if a class folder is missing.
    """
    mapping = {i: f"class_{i:03d}" for i in range(num_classes)}

    if not val_dir.exists():
        return mapping

    class_dirs = [p for p in sorted(val_dir.iterdir()) if p.is_dir()]
    fallback_index = {p.name: i for i, p in enumerate(class_dirs)}

    for class_dir in class_dirs:
        parsed = _parse_class_index(class_dir.name)
        class_index = parsed if parsed is not None else fallback_index[class_dir.name]
        if 0 <= class_index < num_classes:
            mapping[class_index] = class_dir.name

    return mapping


def write_predictions_csv(
    output_path: Path,
    rows: List[Dict[str, Any]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "video_dir",
        "true_label",
        "true_name",
        "pred_label",
        "pred_name",
        "correct",
        "top1_confidence",
        "top5_labels",
        "top5_names",
        "top5_scores",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_matrix_csv(
    output_path: Path,
    matrix: List[List[float]],
    class_names: Dict[int, str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_classes = len(matrix)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        header = ["true\\pred"]
        for j in range(num_classes):
            header.append(f"{j:03d}_{class_names.get(j, f'class_{j:03d}')}")
        writer.writerow(header)

        for i in range(num_classes):
            row_name = f"{i:03d}_{class_names.get(i, f'class_{i:03d}')}"
            writer.writerow([row_name] + matrix[i])


def compute_per_class_metrics(
    confusion: torch.Tensor,
    class_names: Dict[int, str],
) -> List[Dict[str, Any]]:
    num_classes = confusion.shape[0]
    rows: List[Dict[str, Any]] = []

    for c in range(num_classes):
        tp = int(confusion[c, c].item())
        support = int(confusion[c, :].sum().item())
        predicted_as_c = int(confusion[:, c].sum().item())

        recall = tp / support if support > 0 else 0.0
        precision = tp / predicted_as_c if predicted_as_c > 0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        # Main wrong prediction for this true class
        row = confusion[c, :].clone()
        row[c] = 0
        main_conf_count = int(row.max().item())
        main_conf_pred = int(row.argmax().item()) if main_conf_count > 0 else -1

        rows.append(
            {
                "class_id": c,
                "class_name": class_names.get(c, f"class_{c:03d}"),
                "support": support,
                "true_positive": tp,
                "predicted_as_class": predicted_as_c,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "main_confused_with_id": main_conf_pred,
                "main_confused_with_name": (
                    class_names.get(main_conf_pred, f"class_{main_conf_pred:03d}")
                    if main_conf_pred >= 0
                    else ""
                ),
                "main_confusion_count": main_conf_count,
                "main_confusion_rate_within_true_class": (
                    main_conf_count / support if support > 0 else 0.0
                ),
            }
        )

    return rows


def write_per_class_metrics_csv(
    output_path: Path,
    rows: List[Dict[str, Any]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "class_id",
        "class_name",
        "support",
        "true_positive",
        "predicted_as_class",
        "precision",
        "recall",
        "f1",
        "main_confused_with_id",
        "main_confused_with_name",
        "main_confusion_count",
        "main_confusion_rate_within_true_class",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_top_confusions(
    confusion: torch.Tensor,
    class_names: Dict[int, str],
    top_n: int = 50,
) -> List[Dict[str, Any]]:
    num_classes = confusion.shape[0]
    rows: List[Dict[str, Any]] = []

    supports = confusion.sum(dim=1)

    for true_id in range(num_classes):
        support = int(supports[true_id].item())

        for pred_id in range(num_classes):
            if true_id == pred_id:
                continue

            count = int(confusion[true_id, pred_id].item())
            if count <= 0:
                continue

            rows.append(
                {
                    "true_id": true_id,
                    "true_name": class_names.get(true_id, f"class_{true_id:03d}"),
                    "pred_id": pred_id,
                    "pred_name": class_names.get(pred_id, f"class_{pred_id:03d}"),
                    "count": count,
                    "true_support": support,
                    "rate_within_true_class": count / support if support > 0 else 0.0,
                }
            )

    rows.sort(
        key=lambda r: (
            r["count"],
            r["rate_within_true_class"],
        ),
        reverse=True,
    )

    return rows[:top_n]


def write_top_confusions_csv(
    output_path: Path,
    rows: List[Dict[str, Any]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "true_id",
        "true_name",
        "pred_id",
        "pred_name",
        "count",
        "true_support",
        "rate_within_true_class",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_write_confusion_png(
    output_path: Path,
    confusion_norm: torch.Tensor,
    class_names: Dict[int, str],
) -> None:
    """
    Optional PNG plot. If matplotlib is unavailable, silently skip.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable, skipping PNG confusion matrix: {exc}")
        return

    num_classes = confusion_norm.shape[0]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(confusion_norm.numpy(), interpolation="nearest")
    fig.colorbar(im, ax=ax)

    tick_labels = [
        f"{i:02d}" for i in range(num_classes)
    ]

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
    ax.set_yticklabels(tick_labels, fontsize=6)

    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Row-normalized confusion matrix")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print("=== HYDRA RUNTIME CFG ===")
    print(OmegaConf.to_yaml(cfg))

    set_seed(int(cfg.dataset.seed))

    device_str = str(cfg.training.device)
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    checkpoint_path = Path(str(cfg.training.checkpoint_path)).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}", flush=True)
    raw: Dict[str, Any] = torch.load(checkpoint_path, map_location="cpu")

    model, ckpt_cfg = build_model_from_checkpoint(raw, device)

    num_classes = int(ckpt_cfg.model.get("num_classes", raw.get("num_classes", cfg.num_classes)))

    print("=== CHECKPOINT MODEL CFG ===")
    print(OmegaConf.to_yaml(ckpt_cfg.model))
    print("=== CHECKPOINT DATA/TRAIN SUMMARY ===")
    print(f"ckpt val_accuracy: {raw.get('val_accuracy', '<not stored>')}")
    print(f"ckpt dataset.num_frames: {ckpt_cfg.dataset.get('num_frames')}")
    print(f"ckpt training.image_size: {ckpt_cfg.training.get('image_size', '<missing>')}")
    print(f"ckpt model.target_num_frames: {ckpt_cfg.model.get('target_num_frames', '<missing>')}")
    print(f"ckpt model.frame_resampling: {ckpt_cfg.model.get('frame_resampling', '<missing>')}")
    print(f"num_classes: {num_classes}")

    # Use checkpoint preprocessing, not the current Hydra default.
    pretrained_used = bool(raw.get("pretrained", ckpt_cfg.model.get("pretrained", True)))
    image_size = int(ckpt_cfg.training.get("image_size", cfg_get(cfg, "training.image_size", 224)))
    num_frames = int(raw.get("num_frames", ckpt_cfg.dataset.get("num_frames", cfg.dataset.num_frames)))

    eval_transform = build_transforms(
        is_training=False,
        use_imagenet_norm=pretrained_used,
        image_size=image_size,
    )

    # By default, evaluate on the validation folder saved in checkpoint.
    # To force current Hydra val_dir:
    #   +evaluation.use_checkpoint_val_dir=false
    use_checkpoint_val_dir = bool(cfg_get(cfg, "evaluation.use_checkpoint_val_dir", True))
    if use_checkpoint_val_dir:
        val_dir = Path(str(ckpt_cfg.dataset.val_dir)).resolve()
    else:
        val_dir = Path(str(cfg.dataset.val_dir)).resolve()

    print(f"Using val_dir: {val_dir}")
    print(
        f"Using image_size={image_size}, num_frames={num_frames}, "
        f"pretrained_norm={pretrained_used}"
    )

    val_samples = collect_video_samples(val_dir)
    class_names = build_class_index_to_name(val_dir, num_classes)

    max_samples = cfg.dataset.get("max_samples")
    if max_samples is not None:
        val_samples = val_samples[: int(max_samples)]

    val_dataset = VideoFrameDataset(
        root_dir=val_dir,
        num_frames=num_frames,
        transform=eval_transform,
        sample_list=val_samples,
    )

    batch_size = int(cfg.training.batch_size)
    num_workers = int(cfg.training.num_workers)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    output_dir_cfg = cfg_get(cfg, "evaluation.output_dir", None)
    if output_dir_cfg is None:
        output_dir = checkpoint_path.parent / f"eval_{checkpoint_path.stem}"
    else:
        output_dir = Path(str(output_dir_cfg)).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing evaluation artifacts to: {output_dir}")

    print(
        f"Starting evaluation: {len(val_dataset)} samples, "
        f"batch_size={batch_size}, {len(val_loader)} batches",
        flush=True,
    )

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    prediction_rows: List[Dict[str, Any]] = []

    log_interval = max(1, len(val_loader) // 10)

    with torch.no_grad():
        sample_offset = 0

        for batch_idx, (video_batch, labels) in enumerate(val_loader, start=1):
            video_batch = video_batch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(video_batch)

            probs = torch.softmax(logits, dim=1)
            predictions_top1 = logits.argmax(dim=1)

            correct_top1 += int((predictions_top1 == labels).sum().item())

            k = min(5, logits.shape[1])
            top5_scores, predictions_top5 = probs.topk(k, dim=1, largest=True, sorted=True)
            matches_top5 = predictions_top5.eq(labels.view(-1, 1)).any(dim=1)
            correct_top5 += int(matches_top5.sum().item())

            # Update confusion matrix and prediction rows.
            batch_size_actual = labels.size(0)
            labels_cpu = labels.detach().cpu()
            pred_cpu = predictions_top1.detach().cpu()
            top5_cpu = predictions_top5.detach().cpu()
            top5_scores_cpu = top5_scores.detach().cpu()

            for local_i in range(batch_size_actual):
                true_id = int(labels_cpu[local_i].item())
                pred_id = int(pred_cpu[local_i].item())

                if 0 <= true_id < num_classes and 0 <= pred_id < num_classes:
                    confusion[true_id, pred_id] += 1

                sample_index = sample_offset + local_i
                video_dir = (
                    str(val_samples[sample_index][0])
                    if sample_index < len(val_samples)
                    else ""
                )

                top5_ids = [int(x) for x in top5_cpu[local_i].tolist()]
                top5_names = [class_names.get(x, f"class_{x:03d}") for x in top5_ids]
                top5_vals = [float(x) for x in top5_scores_cpu[local_i].tolist()]

                prediction_rows.append(
                    {
                        "video_dir": video_dir,
                        "true_label": true_id,
                        "true_name": class_names.get(true_id, f"class_{true_id:03d}"),
                        "pred_label": pred_id,
                        "pred_name": class_names.get(pred_id, f"class_{pred_id:03d}"),
                        "correct": int(true_id == pred_id),
                        "top1_confidence": float(probs[local_i, pred_id].detach().cpu().item()),
                        "top5_labels": " ".join(str(x) for x in top5_ids),
                        "top5_names": " | ".join(top5_names),
                        "top5_scores": " ".join(f"{x:.6f}" for x in top5_vals),
                    }
                )

            sample_offset += batch_size_actual
            total += batch_size_actual

            if batch_idx % log_interval == 0 or batch_idx == len(val_loader):
                top1_so_far = correct_top1 / max(total, 1)
                print(
                    f"[EVAL] batch {batch_idx}/{len(val_loader)} | "
                    f"{total}/{len(val_dataset)} samples | "
                    f"top1_so_far={top1_so_far:.4f}",
                    flush=True,
                )

    top1_accuracy = correct_top1 / max(total, 1)
    top5_accuracy = correct_top5 / max(total, 1)

    print(f"Validation samples: {len(val_dataset)}")
    print(f"Top-1 accuracy: {top1_accuracy:.4f}")
    print(f"Top-5 accuracy: {top5_accuracy:.4f}")

    # Raw confusion matrix.
    confusion_raw_list = confusion.tolist()

    # Row-normalized confusion matrix: each row sums to 1 when support > 0.
    support = confusion.sum(dim=1, keepdim=True).clamp_min(1)
    confusion_norm = confusion.float() / support.float()
    confusion_norm_list = confusion_norm.tolist()

    per_class_rows = compute_per_class_metrics(confusion, class_names)
    top_confusion_rows = compute_top_confusions(confusion, class_names, top_n=100)

    write_predictions_csv(output_dir / "predictions.csv", prediction_rows)
    write_matrix_csv(output_dir / "confusion_raw.csv", confusion_raw_list, class_names)
    write_matrix_csv(output_dir / "confusion_normalized_by_true.csv", confusion_norm_list, class_names)
    write_per_class_metrics_csv(output_dir / "per_class_metrics.csv", per_class_rows)
    write_top_confusions_csv(output_dir / "top_confusions.csv", top_confusion_rows)
    maybe_write_confusion_png(output_dir / "confusion_normalized_by_true.png", confusion_norm, class_names)

    # Small terminal summary.
    print("\n=== WORST RECALL CLASSES ===")
    by_recall = sorted(
        [r for r in per_class_rows if int(r["support"]) > 0],
        key=lambda r: (float(r["recall"]), -int(r["support"])),
    )
    for r in by_recall[:10]:
        print(
            f"class {int(r['class_id']):03d} | "
            f"recall={float(r['recall']):.3f} | "
            f"support={int(r['support'])} | "
            f"name={r['class_name']} | "
            f"main confusion -> {r['main_confused_with_id']} "
            f"({r['main_confused_with_name']}), "
            f"count={r['main_confusion_count']}"
        )

    print("\n=== TOP CONFUSIONS ===")
    for r in top_confusion_rows[:20]:
        print(
            f"{int(r['true_id']):03d} ({r['true_name']}) -> "
            f"{int(r['pred_id']):03d} ({r['pred_name']}) | "
            f"count={int(r['count'])} | "
            f"rate={float(r['rate_within_true_class']):.3f}"
        )

    print(f"\nSaved artifacts in: {output_dir}")


if __name__ == "__main__":
    main()