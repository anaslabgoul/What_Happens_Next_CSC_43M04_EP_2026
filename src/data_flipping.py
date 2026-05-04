from __future__ import annotations

import argparse
from pathlib import Path

import cv2


# Classes that must swap labels after horizontal flip.
CLASS_SWAP = {
    "018_Pulling_something_from_left_to_right": "019_Pulling_something_from_right_to_left",
    "019_Pulling_something_from_right_to_left": "018_Pulling_something_from_left_to_right",
}


def _target_class_name(source_class_name: str) -> str:
    """Return destination class for flipped video."""
    return CLASS_SWAP.get(source_class_name, source_class_name)


def _flip_video_folder(
    video_dir: Path, source_class_dir: Path, split_dir: Path, overwrite: bool = False
) -> bool:
    """
    Create a horizontally flipped copy of one video folder.
    New folder name: <video_name>_flipped
    Destination class is remapped if class is direction-sensitive.
    """
    dest_class_name = _target_class_name(source_class_dir.name)
    dest_class_dir = split_dir / dest_class_name
    dest_class_dir.mkdir(parents=True, exist_ok=True)

    new_video_dir = dest_class_dir / f"{video_dir.name}_flipped"

    if new_video_dir.exists():
        if not overwrite:
            return False
    else:
        new_video_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(
        p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}
    )

    for frame_path in frame_paths:
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        flipped = cv2.flip(image, 1)  # Horizontal flip
        output_path = new_video_dir / frame_path.name
        cv2.imwrite(str(output_path), flipped)

    return True


def flip_split(split_dir: Path, overwrite: bool = False) -> tuple[int, int]:
    """
    Traverse split structure:
      split_dir / <class_name> / <video_folder> / *.jpg
    Create horizontally flipped videos as <video_folder>_flipped.
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
            if video_dir.name.endswith("_flipped"):
                continue

            did_create = _flip_video_folder(
                video_dir=video_dir,
                source_class_dir=class_dir,
                split_dir=split_dir,
                overwrite=overwrite,
            )
            if did_create:
                created += 1
            else:
                skipped += 1

    return created, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create horizontal-flip video copies with class remapping for left/right actions."
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
        help="Overwrite existing *_flipped folders if they already exist.",
    )
    args = parser.parse_args()

    train_dir = args.data_root / "train"
    val_dir = args.data_root / "val"

    train_created, train_skipped = flip_split(train_dir, overwrite=args.overwrite)
    val_created, val_skipped = flip_split(val_dir, overwrite=args.overwrite)

    print("\nFlipping finished.")
    print(f"Train: created {train_created}, skipped {train_skipped}")
    print(f"Val:   created {val_created}, skipped {val_skipped}")
    print("\nClass remapping used for horizontal flip:")
    for src, dst in CLASS_SWAP.items():
        print(f"  {src} -> {dst}")


if __name__ == "__main__":
    main()
