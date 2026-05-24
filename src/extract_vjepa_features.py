from __future__ import annotations

import random
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional, Tuple

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from models.vjepa import VJEPA2VideoClassifier
from utils import build_transforms, set_seed


def subsample_samples(
    samples: List[Tuple[Path, int]],
    max_samples: Optional[int],
    seed: int,
) -> List[Tuple[Path, int]]:
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


def build_vjepa_model(cfg: DictConfig) -> VJEPA2VideoClassifier:
    return VJEPA2VideoClassifier(
        num_classes=int(cfg.model.num_classes),
        pretrained=bool(cfg.model.pretrained),
        model_name=str(cfg.model.get("model_name", "facebook/vjepa2-vith-fpc64-256")),
        freeze_backbone=True,
        unfreeze_last_n_layers=0,
        hidden_dim=int(cfg.model.get("hidden_dim", 256)),
        dropout=float(cfg.model.get("dropout", 0.2)),
        input_norm=str(cfg.model.get("input_norm", "imagenet")),
        target_num_frames=cfg.model.get("target_num_frames", None),
    )


@torch.no_grad()
def pooled_features_from_model(
    model: VJEPA2VideoClassifier,
    video_batch: torch.Tensor,
) -> torch.Tensor:
    """
    Retourne des features (B, D) prêtes pour une probe MLP.
    Essaie d'utiliser model.extract_features si présent.
    Sinon, retombe sur _encode + token_norm + mean pooling.
    """
    if hasattr(model, "extract_features"):
        feats = model.extract_features(video_batch)
        return feats

    if not hasattr(model, "_encode"):
        raise AttributeError(
            "Le modèle V-JEPA n'a ni extract_features() ni _encode(). "
            "Ajoute extract_features() dans vjepa.py."
        )

    tokens = model._encode(video_batch)  # type: ignore[attr-defined]

    if hasattr(model, "classifier") and hasattr(model.classifier, "token_norm"):
        tokens = model.classifier.token_norm(tokens)

    feats = tokens.mean(dim=1)
    return feats


@torch.no_grad()
def extract_split(
    model: VJEPA2VideoClassifier,
    data_loader: DataLoader,
    device: torch.device,
    use_bf16: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()

    features_all = []
    labels_all = []

    autocast_enabled = bool(use_bf16 and device.type == "cuda")

    for video_batch, labels in data_loader:
        video_batch = video_batch.to(device, non_blocking=True)

        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if autocast_enabled
            else nullcontext()
        )

        with amp_ctx:
            feats = pooled_features_from_model(model, video_batch)

        features_all.append(feats.float().cpu())
        labels_all.append(labels.cpu())

    features = torch.cat(features_all, dim=0)
    labels = torch.cat(labels_all, dim=0)
    return features, labels


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

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

    use_bf16 = bool(cfg.features.get("use_bf16", cfg.training.get("use_bf16", False)))
    if use_bf16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        print("BF16 requested but not supported on this GPU. Falling back to fp32.")
        use_bf16 = False

    output_dir = Path(str(cfg.features.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_size = int(cfg.training.get("image_size", 224))
    use_imagenet_norm = bool(str(cfg.model.get("input_norm", "imagenet")) == "imagenet")

    # Extraction déterministe : pas d'augmentation random ici.
    transform = build_transforms(
        is_training=False,
        use_imagenet_norm=use_imagenet_norm,
        image_size=image_size,
    )

    batch_size = int(cfg.features.get("batch_size", 4))
    num_workers = int(cfg.features.get("num_workers", cfg.training.get("num_workers", 4)))
    max_samples = cfg.dataset.get("max_samples", None)

    model = build_vjepa_model(cfg).to(device)
    model.eval()

    for split_name, split_dir, split_seed in [
        ("train", Path(cfg.dataset.train_dir).resolve(), seed),
        ("val", Path(cfg.dataset.val_dir).resolve(), seed + 1),
    ]:
        samples = collect_video_samples(split_dir)
        samples = subsample_samples(samples, max_samples=max_samples, seed=split_seed)

        dataset = VideoFrameDataset(
            root_dir=split_dir,
            num_frames=int(cfg.dataset.num_frames),
            transform=transform,
            sample_list=samples,
        )

        loader_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": num_workers,
            "pin_memory": (device.type == "cuda"),
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True

        data_loader = DataLoader(**loader_kwargs)

        print(
            f"\nExtracting {split_name}: {len(dataset)} clips | "
            f"batch_size={batch_size} | num_workers={num_workers} | bf16={use_bf16}"
        )

        features, labels = extract_split(
            model=model,
            data_loader=data_loader,
            device=device,
            use_bf16=use_bf16,
        )

        payload = {
            "features": features,               # (N, D)
            "labels": labels.long(),           # (N,)
            "paths": [str(path) for path, _ in samples],
            "feature_dim": int(features.shape[1]),
            "num_samples": int(features.shape[0]),
            "config": OmegaConf.to_container(cfg, resolve=True),
            "split": split_name,
        }

        out_path = output_dir / f"{split_name}_features.pt"
        torch.save(payload, out_path)
        print(f"Saved {split_name} features to {out_path}")
        print(f"  features shape = {tuple(features.shape)}")
        print(f"  labels shape   = {tuple(labels.shape)}")

    print(f"\nDone. Feature files saved under: {output_dir}")


if __name__ == "__main__":
    main()