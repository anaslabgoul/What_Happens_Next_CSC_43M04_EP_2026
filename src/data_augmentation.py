from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np


def _video_seed(video_dir: Path) -> int:
    """Create a stable seed from the video folder path."""
    digest = hashlib.sha256(str(video_dir).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def _color_cast_params(video_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate deterministic color-cast parameters for one video.
    Same video folder always gets the same cast.
    """
    rng = np.random.default_rng(_video_seed(video_dir))
    # Channel multipliers (B, G, R): mild cast to preserve semantics
    scale = rng.uniform(0.8, 1.2, size=3).astype(np.float32)
    # Channel additive offsets (B, G, R)
    shift = rng.uniform(-20.0, 20.0, size=3).astype(np.float32)
    return scale, shift


def _apply_color_cast(image: np.ndarray, scale: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """Apply channel-wise cast and clamp to valid uint8 range."""
    out = image.astype(np.float32)
    out = out * scale.reshape(1, 1, 3) + shift.reshape(1, 1, 3)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def _augment_video_folder(video_dir: Path, overwrite: bool = False) -> bool:
    """
    Create <video_dir>_new and write color-cast augmented jpg/jpeg frames.
    Returns True if created, False if skipped.
    """
    new_video_dir = video_dir.with_name(f"{video_dir.name}_new")

    if new_video_dir.exists():
        if not overwrite:
            return False
    else:
        new_video_dir.mkdir(parents=True, exist_ok=True)

    scale, shift = _color_cast_params(video_dir)
    frame_paths = sorted(
        p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}
    )

    for frame_path in frame_paths:
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        aug_image = _apply_color_cast(image, scale, shift)
        output_path = new_video_dir / frame_path.name
        cv2.imwrite(str(output_path), aug_image)

    return True


def augment_split(split_dir: Path, overwrite: bool = False) -> tuple[int, int]:
    """
    Traverse split structure:
      split_dir / <class_name> / <video_folder> / *.jpg
    For each original video folder, create <video_folder>_new with color-cast frames.
    """
    created = 0
    skipped = 0

    if not split_dir.exists():
        print(f"[WARN] Split folder not found: {split_dir}")
        return created, skipped

    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        for video_dir in sorted(class_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            if video_dir.name.endswith("_new"):
                continue

            did_create = _augment_video_folder(video_dir, overwrite=overwrite)
            if did_create:
                created += 1
            else:
                skipped += 1

    return created, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Duplicate video folders with color-cast augmentation for train/val splits."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "processed_data" / "val2",
        help="Path containing train/ and val/ folders.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing *_new folders if they already exist.",
    )
    args = parser.parse_args()

    train_dir = args.data_root / "train"
    val_dir = args.data_root / "val"

    train_created, train_skipped = augment_split(train_dir, overwrite=args.overwrite)
    val_created, val_skipped = augment_split(val_dir, overwrite=args.overwrite)

    print("\nAugmentation finished.")
    print(f"Train: created {train_created}, skipped {train_skipped}")
    print(f"Val:   created {val_created}, skipped {val_skipped}")


if __name__ == "__main__":
    main()
