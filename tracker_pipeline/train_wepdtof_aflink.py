from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _rotated_corners(cx: float, cy: float, w: float, h: float, angle_deg: float) -> np.ndarray:
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    corners = np.array(
        [
            [-w / 2.0, -h / 2.0],
            [w / 2.0, -h / 2.0],
            [w / 2.0, h / 2.0],
            [-w / 2.0, h / 2.0],
        ],
        dtype=float,
    )
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=float)
    return corners @ rot.T + np.array([cx, cy], dtype=float)


def _axis_tlwh_from_rotated(cx: float, cy: float, w: float, h: float, angle_deg: float) -> tuple[float, float, float, float]:
    corners = _rotated_corners(cx, cy, w, h, angle_deg)
    x1, y1 = corners.min(axis=0)
    x2, y2 = corners.max(axis=0)
    return float(x1), float(y1), float(x2 - x1), float(y2 - y1)


def _frame_number(image_id: str) -> int:
    digits = "".join(ch for ch in image_id.split("_")[-1] if ch.isdigit())
    if not digits:
        raise ValueError(f"Cannot infer frame number from image_id={image_id!r}")
    return int(digits)


def _write_scene_gt(annotation_json: Path, out_file: Path) -> None:
    data = json.loads(annotation_json.read_text())
    rows = []
    for ann in data.get("annotations", []):
        if int(ann.get("category_id", 1)) != 1:
            continue
        bbox = ann.get("bbox", [])
        if len(bbox) < 5:
            continue
        cx, cy, w, h, angle = map(float, bbox[:5])
        person_id = int(ann["person_id"])
        if person_id <= 0 or w <= 0 or h <= 0:
            continue
        frame = _frame_number(str(ann["image_id"]))
        x, y, wa, ha = _axis_tlwh_from_rotated(cx, cy, w, h, angle)
        rows.append((frame, person_id, x, y, wa, ha))

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w") as f:
        for frame, person_id, x, y, w, h in sorted(rows):
            f.write(f"{frame},{person_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,1,1\n")


def prepare_aflink_root(annotations_dir: Path, output_root: Path, scenes: Iterable[str]) -> None:
    for scene in scenes:
        ann = annotations_dir / f"{scene}.json"
        if not ann.exists():
            raise FileNotFoundError(ann)
        gt_dir = output_root / scene / "gt"
        _write_scene_gt(ann, gt_dir / "gt_train_half.txt")
        _write_scene_gt(ann, gt_dir / "gt_val_half.txt")


def _load_aflink_modules(aflink_code: Path):
    sys.path.insert(0, str(aflink_code))
    import AFLink.dataset as dataset_mod
    from AFLink.dataset import LinkData
    from AFLink.model import PostLinker

    return dataset_mod, LinkData, PostLinker


def _make_loader(dataset_mod, LinkData, root: Path, scenes: List[str], mode: str, batch_size: int, num_workers: int):
    dataset_mod.SEQ["train"] = scenes
    dataset = LinkData(str(root), mode)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(mode == "train"), num_workers=num_workers, drop_last=(mode == "train"))


def _flatten_batch(pair1, pair2, pair3, pair4, label, device: torch.device):
    pairs_1 = torch.cat((pair1[0], pair2[0], pair3[0], pair4[0]), dim=0).to(device)
    pairs_2 = torch.cat((pair1[1], pair2[1], pair3[1], pair4[1]), dim=0).to(device)
    labels = torch.cat(label, dim=0).to(device)
    return pairs_1, pairs_2, labels


@torch.no_grad()
def validate(model, loader, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    correct = 0
    for pair1, pair2, pair3, pair4, label in loader:
        x1, x2, y = _flatten_batch(pair1, pair2, pair3, pair4, label, device)
        logits = model(x1, x2)
        loss = loss_fn(logits, y)
        total_loss += float(loss.item())
        pred = logits.argmax(dim=1)
        total += int(y.numel())
        correct += int((pred == y).sum().item())
    model.train()
    return total_loss / max(len(loader), 1), correct / max(total, 1)


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    annotations_dir = args.wepdtof_root / "annotations"
    all_scenes = sorted(set(args.train_scenes + args.val_scenes))
    prepare_aflink_root(annotations_dir, args.aflink_data_root, all_scenes)

    dataset_mod, LinkData, PostLinker = _load_aflink_modules(args.aflink_code)
    train_loader = _make_loader(dataset_mod, LinkData, args.aflink_data_root, args.train_scenes, "train", args.batch_size, args.num_workers)
    val_loader = _make_loader(dataset_mod, LinkData, args.aflink_data_root, args.val_scenes, "val", args.val_batch_size, args.num_workers)

    model = PostLinker().to(device)
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_path = args.output

    for epoch in range(1, args.epochs + 1):
        loss_sum = 0.0
        for pair1, pair2, pair3, pair4, label in train_loader:
            optimizer.zero_grad()
            x1, x2, y = _flatten_batch(pair1, pair2, pair3, pair4, label, device)
            loss = loss_fn(model(x1, x2), y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())
        scheduler.step()

        val_loss, val_acc = validate(model, val_loader, device)
        train_loss = loss_sum / max(len(train_loader), 1)
        print(f"epoch={epoch:03d} train_loss={train_loss:.5f} val_loss={val_loss:.5f} val_acc={val_acc:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)

    last_path = args.output.with_name(args.output.stem + "_last.pth")
    torch.save(model.state_dict(), last_path)
    print(f"saved_best: {best_path}")
    print(f"saved_last : {last_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AFLink PostLinker on WEPDTOF annotations.")
    parser.add_argument("--wepdtof-root", type=Path, default=Path("/home/wang/下载/WEPDTOF/WEPDTOF"))
    parser.add_argument("--aflink-code", type=Path, default=Path("/home/wang/桌面/suivi-changement-apparence"))
    parser.add_argument("--aflink-data-root", type=Path, default=_project_root() / "tracker_pipeline/data/aflink_wepdtof_mot")
    parser.add_argument("--output", type=Path, default=_project_root() / "tracker_pipeline/data/aflink/AFLink_wepdtof_train4_valkindergarten.pth")
    parser.add_argument("--train-scenes", nargs="+", default=["jewelry_store", "jewelry_store_2", "tech_store", "printing_store"])
    parser.add_argument("--val-scenes", nargs="+", default=["kindergarten"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
