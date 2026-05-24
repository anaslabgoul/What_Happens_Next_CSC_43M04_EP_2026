from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict

import hydra
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset


class FeatureDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self.features = features.float()
        self.labels = labels.long()

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]


class ProbeMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_class_weights_from_labels(
    labels: torch.Tensor,
    num_classes: int,
    alpha: float = 0.5,
) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_classes).float()
    weights = torch.ones(num_classes, dtype=torch.float32)

    present = counts > 0
    if present.any():
        total = counts[present].sum()
        n_present = present.sum()
        inv_freq = total / (n_present * counts[present])
        weights[present] = inv_freq.pow(alpha)

        weights = weights / weights[present].mean()

    return weights


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_bf16: bool = False,
    grad_clip_norm: float | None = None,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    autocast_enabled = bool(use_bf16 and device.type == "cuda")

    for features, labels in data_loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if autocast_enabled
            else nullcontext()
        )

        with amp_ctx:
            logits = model(features)
            loss = loss_fn(logits, labels)

        loss.backward()

        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()

        running_loss += float(loss.item()) * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total += labels.size(0)

    return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    use_bf16: bool = False,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    autocast_enabled = bool(use_bf16 and device.type == "cuda")

    for features, labels in data_loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if autocast_enabled
            else nullcontext()
        )

        with amp_ctx:
            logits = model(features)
            loss = loss_fn(logits, labels)

        running_loss += float(loss.item()) * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total += labels.size(0)

    return running_loss / max(total, 1), correct / max(total, 1)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    run = wandb.init(
        project=str(cfg.training.get("wandb_project", "what-happens-next")),
        name=str(cfg.training.get("wandb_run_name", "vjepa_probe_from_features")),
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    try:
        device_str = str(cfg.training.device)
        if device_str == "cuda" and not torch.cuda.is_available():
            print("CUDA not available; using CPU.")
            device_str = "cpu"
        device = torch.device(device_str)

        use_bf16 = bool(cfg.training.get("use_bf16", False))
        if use_bf16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
            print("BF16 requested but not supported on this GPU. Falling back to fp32.")
            use_bf16 = False

        train_file = Path(str(cfg.features.train_file)).resolve()
        val_file = Path(str(cfg.features.val_file)).resolve()

        train_payload: Dict[str, Any] = torch.load(train_file, map_location="cpu")
        val_payload: Dict[str, Any] = torch.load(val_file, map_location="cpu")

        train_features = train_payload["features"].float()
        train_labels = train_payload["labels"].long()
        val_features = val_payload["features"].float()
        val_labels = val_payload["labels"].long()

        input_dim = int(train_features.shape[1])
        num_classes = max(
            int(cfg.get("num_classes", 0)),
            int(train_labels.max().item()) + 1,
            int(val_labels.max().item()) + 1,
        )

        print(f"Train features: {tuple(train_features.shape)}")
        print(f"Val features:   {tuple(val_features.shape)}")
        print(f"Input dim:      {input_dim}")
        print(f"Num classes:    {num_classes}")

        train_dataset = FeatureDataset(train_features, train_labels)
        val_dataset = FeatureDataset(val_features, val_labels)

        num_workers = int(cfg.training.get("num_workers", 0))

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

        model = ProbeMLP(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=int(cfg.probe.get("hidden_dim", 256)),
            dropout=float(cfg.probe.get("dropout", 0.2)),
        ).to(device)

        class_weight_alpha = float(cfg.training.get("class_weight_alpha", 0.25))
        label_smoothing = float(cfg.training.get("label_smoothing", 0.0))
        class_weights = build_class_weights_from_labels(
            train_labels,
            num_classes=num_classes,
            alpha=class_weight_alpha,
        ).to(device)

        print("Class weights:", class_weights.detach().cpu().tolist())

        loss_fn = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg.training.lr),
            weight_decay=float(cfg.training.get("weight_decay", 1e-4)),
        )

        scheduler = None
        lr_decay_factor = cfg.training.get("lr_decay_factor", None)
        lr_patience = cfg.training.get("lr_patience", None)
        if lr_decay_factor is not None and lr_patience is not None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=float(lr_decay_factor),
                patience=int(lr_patience),
            )

        grad_clip_norm = cfg.training.get("grad_clip_norm", None)
        if grad_clip_norm is not None:
            grad_clip_norm = float(grad_clip_norm)

        checkpoint_path = Path(str(cfg.training.checkpoint_path)).resolve()
        best_val_acc = 0.0

        for epoch in range(int(cfg.training.epochs)):
            train_loss, train_acc = train_one_epoch(
                model=model,
                data_loader=train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device,
                use_bf16=use_bf16,
                grad_clip_norm=grad_clip_norm,
            )

            val_loss, val_acc = evaluate_epoch(
                model=model,
                data_loader=val_loader,
                loss_fn=loss_fn,
                device=device,
                use_bf16=use_bf16,
            )

            if scheduler is not None:
                scheduler.step(val_loss)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "input_dim": input_dim,
                        "num_classes": num_classes,
                        "val_accuracy": val_acc,
                        "config": OmegaConf.to_container(cfg, resolve=True),
                    },
                    checkpoint_path,
                )
                print(f"  Saved new best probe to {checkpoint_path} (val acc={val_acc:.4f})")
                run.summary["best_val_acc"] = val_acc
                run.summary["best_checkpoint_path"] = str(checkpoint_path)

            print(
                f"Epoch {epoch + 1}/{cfg.training.epochs} | "
                f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.4f}"
            )

            log_dict = {
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "best_val_acc": best_val_acc,
                "lr": optimizer.param_groups[0]["lr"],
            }
            run.log(log_dict)

        print(f"Done. Best validation accuracy: {best_val_acc:.4f}")

    finally:
        run.finish()


if __name__ == "__main__":
    main()