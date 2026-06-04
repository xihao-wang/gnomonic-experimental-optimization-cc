from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tracker_pipeline.strongsort.association_veto_model import (
    DEFAULT_FEATURE_NAMES,
    AssociationVetoMLP,
    save_association_veto_checkpoint,
)


def load_npz_files(paths: List[Path]):
    features = []
    labels = []
    for path in paths:
        data = np.load(path, allow_pickle=True)
        features.append(data["features"].astype(np.float32))
        labels.append(data["label"].astype(np.float32))
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def load_datasets(args) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    if args.train_input:
        if not args.val_input:
            raise ValueError("--val-input is required when --train-input is used.")
        x_train, y_train = load_npz_files(args.train_input)
        x_val, y_val = load_npz_files(args.val_input)
        return x_train, y_train, x_val, y_val, "scene_split"

    if not args.input:
        raise ValueError("Use either --train-input/--val-input or legacy --input.")
    x, y = load_npz_files(args.input)
    if x.size == 0:
        raise RuntimeError("No training samples found.")
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(y))
    val_count = max(1, int(len(indices) * args.val_fraction))
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx], "random_split"


def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            probs = torch.sigmoid(logits)
            pred = (probs >= 0.5).float()
            correct += int((pred == y).sum().item())
            total += int(y.numel())
            loss_sum += float(loss.item()) * int(y.numel())
    return loss_sum / max(total, 1), correct / max(total, 1)


def train(args):
    x_train, y_train, x_val, y_val, split_mode = load_datasets(args)
    if x_train.size == 0 or x_val.size == 0:
        raise RuntimeError("Training and validation sets must both contain samples.")

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-6] = 1.0
    x_train_norm = (x_train - mean) / std
    x_val_norm = (x_val - mean) / std

    train_ds = TensorDataset(
        torch.from_numpy(x_train_norm.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    val_ds = TensorDataset(
        torch.from_numpy(x_val_norm.astype(np.float32)),
        torch.from_numpy(y_val.astype(np.float32)),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device)
    model = AssociationVetoMLP(input_dim=x_train.shape[1], hidden_dim=args.hidden_dim).to(device)
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(yb.numel())
            total += int(yb.numel())

        val_loss, val_acc = evaluate(model, val_loader, device)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % args.log_every == 0 or epoch == 1 or epoch == args.epochs:
            print(
                f"epoch={epoch:03d} train_loss={total_loss / max(total, 1):.5f} "
                f"val_loss={val_loss:.5f} val_acc={val_acc:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_association_veto_checkpoint(
        args.output,
        model.cpu(),
        feature_mean=mean,
        feature_std=std,
        hidden_dim=args.hidden_dim,
        feature_names=DEFAULT_FEATURE_NAMES,
        metadata={
            "split_mode": split_mode,
            "train_files": [str(path) for path in (args.train_input or args.input or [])],
            "val_files": [str(path) for path in (args.val_input or [])],
            "train_samples": int(len(y_train)),
            "val_samples": int(len(y_val)),
            "train_positive_samples": int(y_train.sum()),
            "train_negative_samples": int(len(y_train) - y_train.sum()),
            "val_positive_samples": int(y_val.sum()),
            "val_negative_samples": int(len(y_val) - y_val.sum()),
            "best_val_loss": float(best_val),
        },
    )
    print(f"saved: {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small association-veto MLP.")
    parser.add_argument("--train-input", type=Path, nargs="+", help="Scene-level training NPZ files.")
    parser.add_argument("--val-input", type=Path, nargs="+", help="Scene-level validation NPZ files.")
    parser.add_argument("--input", type=Path, nargs="+", help="Legacy mode: one or more NPZ files, split by --val-fraction.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
