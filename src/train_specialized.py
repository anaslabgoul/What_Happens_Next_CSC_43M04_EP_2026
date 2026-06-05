"""
Entraînement d’un **agent TSM spécialisé** (``CNNTransformerSpecializedAgent``) :
``K`` classes d’intérêt + une classe « Autre » (indice ``K``).

- Même famille d’augmentations que ``train_crops.py`` (une technique aléatoire par batch).
- **Équilibrage strict** : pour les classes choisies, on prend **exactement**
  ``N = min(effectif de chaque classe spécialisée)`` échantillons par classe spécialisée
  (sous-échantillonnage aléatoire au-delà). La classe « Autre » reçoit aussi **N** exemples
  (uniformément répartis sur les classes globales non spécialisées présentes ; si le pool
  « Autre » est plus petit, ``N`` est réduit pour toutes les classes afin de rester équilibré).

Lancer depuis ``src/``::

    python train_specialized.py experiment=specialized_agent model.specialized_classes=[3,5,12]
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import hydra
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from models.cnn_specialized_agent import CNNTransformerSpecializedAgent, local_to_global_mapping
from train_crops import apply_augment_technique, evaluate_epoch, normalize_video_batch, train_one_epoch
from utils import build_transforms, set_seed


def _group_by_class(samples: List[Tuple[Path, int]]) -> Dict[int, List[Path]]:
    buckets: Dict[int, List[Path]] = defaultdict(list)
    for video_dir, y in samples:
        buckets[int(y)].append(video_dir)
    return dict(buckets)


def _labels_present(samples: List[Tuple[Path, int]]) -> set[int]:
    return {int(y) for _, y in samples}


def _n_min_specialized(counts: Mapping[int, int], specialized: Sequence[int]) -> int:
    """Nombre cible d’échantillons par classe (spécialisées + Autre) = minimum des effectifs des classes choisies."""
    vals = [int(counts[c]) for c in specialized]
    if not vals:
        return 0
    return min(vals)
def _sample_other_uniform(
    by_class: Mapping[int, List[Path]],
    non_specialized_globals: Sequence[int],
    N: int,
    rng: random.Random,
) -> List[Tuple[Path, int]]:
    """
    Choisit jusqu’à ``N`` paires ``(video_dir, global_class_id)``,
    en visant une répartition uniforme sur les classes non spécialisées **avec données**,
    puis en réallouant les quotas impossibles vers les pools encore disponibles.
    """
    if N <= 0:
        return []

    pools: Dict[int, List[Path]] = {}
    for c in non_specialized_globals:
        lst = list(by_class.get(int(c), []))
        if lst:
            pools[int(c)] = list(lst)

    if not pools:
        return []

    class_ids = list(pools.keys())
    rng.shuffle(class_ids)
    M = len(class_ids)
    base = N // M
    rem = N % M
    quotas: Dict[int, int] = {c: base + (1 if i < rem else 0) for i, c in enumerate(class_ids)}

    out: List[Tuple[Path, int]] = []

    for c in class_ids:
        need = quotas[c]
        pool = pools[c]
        rng.shuffle(pool)
        take = min(need, len(pool))
        for j in range(take):
            out.append((pool[j], c))
        pools[c] = pool[take:]

    deficit = N - len(out)
    guard = 0
    max_guard = max(N * M, 200)
    while deficit > 0 and any(pools.get(c) for c in class_ids):
        guard += 1
        if guard > max_guard:
            break
        active = [c for c in class_ids if pools.get(c)]
        rng.shuffle(active)
        for c in active:
            if deficit <= 0:
                break
            pl = pools[c]
            if not pl:
                continue
            path = pl.pop(rng.randrange(len(pl)))
            out.append((path, c))
            deficit -= 1

    return out


def build_balanced_specialized_samples(
    samples: List[Tuple[Path, int]],
    specialized_order: List[int],
    rng: random.Random,
) -> Tuple[List[Tuple[Path, int]], int, Dict[str, Any]]:
    """
    Construit une liste ``(video_dir, label_local)`` équilibrée.

    Returns:
        balanced_samples, N, diagnostics dict.
    """
    K = len(specialized_order)
    other_local = K
    spec_set = set(specialized_order)
    by_class = _group_by_class(samples)

    counts = {c: len(by_class.get(c, [])) for c in specialized_order}
    if any(counts[c] == 0 for c in specialized_order):
        missing = [c for c in specialized_order if counts[c] == 0]
        raise RuntimeError(
            f"Classes spécialisées sans aucun échantillon dans ce split : {missing}. "
            "Ajoutez des données ou retirez ces classes de model.specialized_classes."
        )

    n_specialized_min = _n_min_specialized(counts, specialized_order)
    if n_specialized_min <= 0:
        raise RuntimeError("Cible N <= 0 : impossible d’équilibrer (effectifs spécialisés vides ou invalides).")

    present_globals = _labels_present(samples)
    non_spec_sorted = sorted(g for g in present_globals if g not in spec_set)

    other_pairs = _sample_other_uniform(by_class, non_spec_sorted, n_specialized_min, rng)
    n_other = len(other_pairs)
    n_final = min(n_specialized_min, n_other)
    if n_final <= 0:
        raise RuntimeError(
            "Impossible d’équilibrer : aucun échantillon « Autre » disponible "
            "(toutes les classes du split sont spécialisées ou vides)."
        )

    balanced: List[Tuple[Path, int]] = []

    for local_idx, g in enumerate(specialized_order):
        paths = list(by_class[g])
        rng.shuffle(paths)
        chosen = paths[:n_final]
        if len(chosen) < n_final:
            raise RuntimeError(f"Internal error: class {g} a moins de n_final={n_final} échantillons.")
        for p in chosen:
            balanced.append((p, local_idx))

    for path, _global_y in other_pairs[:n_final]:
        balanced.append((path, other_local))

    diag: Dict[str, Any] = {
        "N_specialized_min": n_specialized_min,
        "N_final": n_final,
        "specialized_counts_raw": dict(counts),
        "other_requested": n_specialized_min,
        "other_pool_hits": n_other,
        "non_specialized_globals": non_spec_sorted,
    }
    return balanced, n_final, diag


def build_model(cfg: DictConfig) -> CNNTransformerSpecializedAgent:
    name = str(cfg.model.name)
    if name not in ("cnntransformer_specialized", "cnn_transformer_specialized"):
        raise ValueError(
            f"train_specialized.py attend model.name=cnntransformer_specialized, got {name!r}."
        )
    return CNNTransformerSpecializedAgent.from_config(
        cfg.model,
        dataset_num_frames=int(cfg.dataset.num_frames),
    )


def _checkpoint_payload(
    cfg: DictConfig,
    model: CNNTransformerSpecializedAgent,
    val_acc: float,
) -> Dict[str, Any]:
    spec = model.specialized_classes
    K = len(spec)
    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "model_name": str(cfg.model.name),
        "num_classes": K + 1,
        "specialized_classes": list(spec),
        "label_mapping_local_to_global": local_to_global_mapping(spec),
        "pretrained": bool(cfg.model.pretrained),
        "num_frames": int(cfg.dataset.num_frames),
        "val_accuracy": val_acc,
        "global_num_classes": int(cfg.get("num_classes", 33)),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "model_config": OmegaConf.to_container(cfg.model, resolve=True),
    }
    return payload


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg), flush=True)

    set_seed(int(cfg.dataset.seed))
    rng = random.Random(int(cfg.dataset.seed) + 7919)

    device_str = str(cfg.training.device)
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
        ms = int(max_samples)
        train_samples = train_samples[:ms]
        val_samples = val_samples[:ms]

    raw_spec = OmegaConf.to_container(cfg.model.get("specialized_classes"), resolve=True)
    if not isinstance(raw_spec, (list, tuple)) or len(raw_spec) == 0:
        raise ValueError(
            "Définissez model.specialized_classes en CLI, "
            "ex: model.specialized_classes=[3,5,12]"
        )
    specialized_order = []
    seen: set[int] = set()
    for x in raw_spec:
        xi = int(x)
        if xi not in seen:
            seen.add(xi)
            specialized_order.append(xi)

    train_balanced, _, diag_tr = build_balanced_specialized_samples(
        train_samples, specialized_order, rng
    )
    val_rng = random.Random(int(cfg.dataset.seed) + 9203)
    val_balanced, _, diag_va = build_balanced_specialized_samples(
        val_samples, specialized_order, val_rng
    )

    if diag_tr["N_final"] < diag_tr["N_specialized_min"]:
        print(
            f"[train_specialized] Avertissement train : N réduit {diag_tr['N_specialized_min']} → {diag_tr['N_final']} "
            f"(« Autre » : {diag_tr['other_pool_hits']} échantillons disponibles sur le minimum des classes spécialisées).",
            flush=True,
        )
    if diag_va["N_final"] < diag_va["N_specialized_min"]:
        print(
            f"[train_specialized] Avertissement val : N réduit {diag_va['N_specialized_min']} → {diag_va['N_final']} "
            f"(« Autre » : {diag_va['other_pool_hits']} échantillons).",
            flush=True,
        )

    K = len(specialized_order)
    print(
        f"Agent spécialisé K={K} classes + Autre ; ordre global {specialized_order} ; "
        f"N = min(effectifs classes choisies) ; "
        f"N_train={diag_tr['N_final']} (cible min spécialisées {diag_tr['N_specialized_min']}) "
        f"N_val={diag_va['N_final']} (cible min spécialisées {diag_va['N_specialized_min']}) ; "
        f"taille train={len(train_balanced)} val={len(val_balanced)}.",
        flush=True,
    )
    print(f"Diag train: {diag_tr}", flush=True)
    print(f"Diag val: {diag_va}", flush=True)

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
        sample_list=train_balanced,
    )
    val_dataset = VideoFrameDataset(
        root_dir=val_dir,
        num_frames=int(cfg.dataset.num_frames),
        transform=eval_transform,
        sample_list=val_balanced,
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
        f"train_specialized: {n_train} train clips, {n_batches} batches/epoch, device={device}.",
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
