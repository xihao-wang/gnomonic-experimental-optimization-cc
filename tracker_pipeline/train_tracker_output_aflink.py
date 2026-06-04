from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset


@dataclass
class Tracklet:
    scene: str
    track_id: int
    rows: np.ndarray  # frame, x, y, w, h
    gt_id: int
    purity: float

    @property
    def start(self) -> int:
        return int(self.rows[0, 0])

    @property
    def end(self) -> int:
        return int(self.rows[-1, 0])

    @property
    def first_xy(self) -> np.ndarray:
        return self.rows[0, 1:3]

    @property
    def last_xy(self) -> np.ndarray:
        return self.rows[-1, 1:3]


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _rotated_corners(cx: float, cy: float, w: float, h: float, angle_deg: float) -> np.ndarray:
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    pts = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]], dtype=float)
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=float)
    return pts @ rot.T + np.array([cx, cy], dtype=float)


def _axis_tlwh(cx: float, cy: float, w: float, h: float, angle_deg: float) -> Tuple[float, float, float, float]:
    corners = _rotated_corners(cx, cy, w, h, angle_deg)
    x1, y1 = corners.min(axis=0)
    x2, y2 = corners.max(axis=0)
    return float(x1), float(y1), float(x2 - x1), float(y2 - y1)


def _frame_number(image_id: str) -> int:
    digits = "".join(ch for ch in image_id.split("_")[-1] if ch.isdigit())
    if not digits:
        raise ValueError(f"Cannot infer frame number from {image_id!r}")
    return int(digits)


def _load_gt(annotation_json: Path) -> Dict[int, List[Tuple[int, np.ndarray]]]:
    data = json.loads(annotation_json.read_text())
    by_frame: Dict[int, List[Tuple[int, np.ndarray]]] = {}
    for ann in data.get("annotations", []):
        if int(ann.get("category_id", 1)) != 1:
            continue
        bbox = ann.get("bbox", [])
        if len(bbox) < 5:
            continue
        cx, cy, w, h, angle = map(float, bbox[:5])
        if w <= 0 or h <= 0:
            continue
        frame = _frame_number(str(ann["image_id"]))
        gt_id = int(ann["person_id"])
        by_frame.setdefault(frame, []).append((gt_id, np.array(_axis_tlwh(cx, cy, w, h, angle), dtype=float)))
    return by_frame


def _load_mot(path: Path) -> Dict[int, List[np.ndarray]]:
    by_id: Dict[int, List[np.ndarray]] = {}
    if not path.exists() or path.stat().st_size == 0:
        return by_id
    arr = np.loadtxt(path, delimiter=",")
    if arr.ndim == 1:
        arr = arr[None, :]
    for row in arr:
        frame, tid = int(row[0]), int(row[1])
        x, y, w, h = map(float, row[2:6])
        if w <= 0 or h <= 0:
            continue
        by_id.setdefault(tid, []).append(np.array([frame, x, y, w, h], dtype=float))
    return {tid: sorted(rows, key=lambda r: r[0]) for tid, rows in by_id.items()}


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return 0.0 if union <= 0 else inter / union


def _split_by_time_gap(rows: Sequence[np.ndarray], max_gap: int) -> List[np.ndarray]:
    chunks: List[List[np.ndarray]] = []
    cur: List[np.ndarray] = []
    prev = None
    for row in rows:
        frame = int(row[0])
        if prev is not None and frame - prev > max_gap and cur:
            chunks.append(cur)
            cur = []
        cur.append(row)
        prev = frame
    if cur:
        chunks.append(cur)
    return [np.stack(chunk, axis=0) for chunk in chunks]


def _assign_tracklet_gt(rows: np.ndarray, gt_by_frame: Dict[int, List[Tuple[int, np.ndarray]]], iou_thr: float) -> Tuple[int, float]:
    counts: Dict[int, int] = {}
    matched = 0
    for row in rows:
        frame = int(row[0])
        pred_box = row[1:5]
        best_id, best_iou = 0, 0.0
        for gt_id, gt_box in gt_by_frame.get(frame, []):
            score = _iou(pred_box, gt_box)
            if score > best_iou:
                best_id, best_iou = gt_id, score
        if best_iou >= iou_thr and best_id > 0:
            counts[best_id] = counts.get(best_id, 0) + 1
            matched += 1
    if not counts:
        return 0, 0.0
    gt_id, count = max(counts.items(), key=lambda kv: kv[1])
    return int(gt_id), float(count) / max(len(rows), 1)


def _collect_tracklets(
    scene: str,
    result_txt: Path,
    annotation_json: Path,
    iou_thr: float,
    min_len: int,
    purity_thr: float,
    max_internal_gap: int,
) -> List[Tracklet]:
    gt_by_frame = _load_gt(annotation_json)
    by_id = _load_mot(result_txt)
    out: List[Tracklet] = []
    for tid, rows in by_id.items():
        for chunk in _split_by_time_gap(rows, max_internal_gap):
            if len(chunk) < min_len:
                continue
            gt_id, purity = _assign_tracklet_gt(chunk, gt_by_frame, iou_thr)
            if gt_id > 0 and purity >= purity_thr:
                out.append(Tracklet(scene=scene, track_id=int(tid), rows=chunk, gt_id=gt_id, purity=purity))
    return out


def _fill_or_cut(x: np.ndarray, input_len: int, former: bool) -> np.ndarray:
    if len(x) >= input_len:
        return x[-input_len:] if former else x[:input_len]
    pad = np.zeros((input_len - len(x), x.shape[1]), dtype=float)
    return np.concatenate([pad, x], axis=0) if former else np.concatenate([x, pad], axis=0)


def _transform_pair(a: np.ndarray, b: np.ndarray, input_len: int) -> Tuple[np.ndarray, np.ndarray]:
    a = _fill_or_cut(a, input_len, True)
    b = _fill_or_cut(b, input_len, False)
    both = np.concatenate([a, b], axis=0)
    mid = (both.max(axis=0) + both.min(axis=0)) / 2.0
    div = (both.max(axis=0) - both.min(axis=0)) / 2.0 + 1e-5
    return ((a - mid) / div)[None, :, :].astype(np.float32), ((b - mid) / div)[None, :, :].astype(np.float32)


def _temporal_gap(a: Tracklet, b: Tracklet) -> int:
    return b.start - a.end


def _spatial_gap(a: Tracklet, b: Tracklet) -> float:
    return float(np.linalg.norm(a.last_xy - b.first_xy))


def _candidate_pairs(tracklets: List[Tracklet], min_gap: int, max_gap: int, max_dist: float):
    by_scene: Dict[str, List[Tracklet]] = {}
    for t in tracklets:
        by_scene.setdefault(t.scene, []).append(t)
    for scene_tracks in by_scene.values():
        for a in scene_tracks:
            for b in scene_tracks:
                if a is b:
                    continue
                gap = _temporal_gap(a, b)
                if min_gap <= gap <= max_gap and _spatial_gap(a, b) <= max_dist:
                    yield a, b


def _artificial_positive(tracklet: Tracklet, input_len: int, rng: random.Random, max_gap: int) -> Tuple[np.ndarray, np.ndarray] | None:
    rows = tracklet.rows
    if len(rows) < max(input_len * 2, 60):
        return None
    cut = rng.randint(input_len, len(rows) - input_len)
    gap = rng.randint(3, min(max_gap, max(3, len(rows) // 5)))
    left = rows[:cut]
    right = rows[min(len(rows), cut + gap) :]
    if len(left) < input_len // 2 or len(right) < input_len // 2:
        return None
    return left, right


def _build_samples(tracklets: List[Tracklet], args: argparse.Namespace, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    positives: List[Tuple[np.ndarray, np.ndarray]] = []
    negatives: List[Tuple[np.ndarray, np.ndarray]] = []

    for a, b in _candidate_pairs(tracklets, args.min_temporal_gap, args.max_temporal_gap, args.max_spatial_gap):
        if a.gt_id == b.gt_id:
            positives.append((a.rows, b.rows))
        else:
            negatives.append((a.rows, b.rows))

    for t in tracklets:
        for _ in range(args.artificial_splits_per_tracklet):
            pair = _artificial_positive(t, args.input_len, rng, args.max_temporal_gap)
            if pair is not None:
                positives.append(pair)

    rng.shuffle(positives)
    rng.shuffle(negatives)
    if args.max_positive_pairs > 0:
        positives = positives[: args.max_positive_pairs]
    target_neg = min(len(negatives), max(args.min_negative_pairs, int(len(positives) * args.neg_pos_ratio)))
    negatives = negatives[:target_neg]

    x1, x2, y = [], [], []
    for label, pairs in [(1, positives), (0, negatives)]:
        for a, b in pairs:
            ta, tb = _transform_pair(a, b, args.input_len)
            x1.append(ta)
            x2.append(tb)
            y.append(label)
    if not y:
        raise RuntimeError("No samples built. Relax thresholds or check result/GT paths.")
    idx = list(range(len(y)))
    rng.shuffle(idx)
    return np.stack([x1[i] for i in idx]), np.stack([x2[i] for i in idx]), np.array([y[i] for i in idx], dtype=np.int64)


class PairDataset(Dataset):
    def __init__(self, x1: np.ndarray, x2: np.ndarray, y: np.ndarray):
        self.x1 = torch.from_numpy(x1)
        self.x2 = torch.from_numpy(x2)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x1[idx], self.x2[idx], self.y[idx]


def _load_postlinker(aflink_code: Path):
    sys.path.insert(0, str(aflink_code))
    from AFLink.model import PostLinker

    return PostLinker


def _result_path(root: Path, scene: str) -> Path:
    direct = root / scene / "result.txt"
    if direct.exists():
        return direct
    matches = sorted((root / scene).glob("*/result.txt"))
    if matches:
        return matches[-1]
    raise FileNotFoundError(f"No result.txt found for {scene} under {root}")


def _collect_for_scenes(scenes: Sequence[str], result_root: Path, wepdtof_root: Path, args: argparse.Namespace) -> List[Tracklet]:
    out: List[Tracklet] = []
    for scene in scenes:
        result = _result_path(result_root, scene)
        ann = wepdtof_root / "annotations" / f"{scene}.json"
        scene_tracklets = _collect_tracklets(
            scene,
            result,
            ann,
            args.iou_threshold,
            args.min_tracklet_len,
            args.min_purity,
            args.max_internal_gap,
        )
        print(f"{scene}: tracklets={len(scene_tracklets)} result={result}")
        out.extend(scene_tracklets)
    return out


@torch.no_grad()
def _validate(model, loader, device) -> Tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    loss_sum, total, correct = 0.0, 0, 0
    for x1, x2, y in loader:
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        logits = model(x1, x2)
        loss_sum += float(loss_fn(logits, y).item())
        pred = logits.argmax(dim=1)
        total += int(y.numel())
        correct += int((pred == y).sum().item())
    model.train()
    return loss_sum / max(len(loader), 1), correct / max(total, 1)


def train(args: argparse.Namespace) -> None:
    rng_seed = int(args.seed)
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)

    train_tracklets = _collect_for_scenes(args.train_scenes, args.train_result_root, args.wepdtof_root, args)
    val_tracklets = _collect_for_scenes(args.val_scenes, args.val_result_root, args.wepdtof_root, args)

    x1_train, x2_train, y_train = _build_samples(train_tracklets, args, rng_seed)
    x1_val, x2_val, y_val = _build_samples(val_tracklets, args, rng_seed + 1)
    print(f"train samples={len(y_train)} pos={int(y_train.sum())} neg={int((y_train == 0).sum())}")
    print(f"val   samples={len(y_val)} pos={int(y_val.sum())} neg={int((y_val == 0).sum())}")

    train_loader = DataLoader(PairDataset(x1_train, x2_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(PairDataset(x1_val, x2_val, y_val), batch_size=args.val_batch_size, shuffle=False)

    device = torch.device(args.device)
    PostLinker = _load_postlinker(args.aflink_code)
    model = PostLinker().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    best_loss = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for x1, x2, y in train_loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x1, x2), y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())
        scheduler.step()
        val_loss, val_acc = _validate(model, val_loader, device)
        print(f"epoch={epoch:03d} train_loss={loss_sum / max(len(train_loader), 1):.5f} val_loss={val_loss:.5f} val_acc={val_acc:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), args.output)
    print(f"saved_best: {args.output}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train AFLink on tracker-output tracklet pairs.")
    root = _project_root()
    p.add_argument("--wepdtof-root", type=Path, default=Path("/home/wang/下载/WEPDTOF/WEPDTOF"))
    p.add_argument("--aflink-code", type=Path, default=Path("/home/wang/桌面/suivi-changement-apparence"))
    p.add_argument("--train-result-root", type=Path, default=root / "tracker_pipeline/results/dedup/trainval")
    p.add_argument("--val-result-root", type=Path, default=root / "tracker_pipeline/results/dedup/old_model")
    p.add_argument("--train-scenes", nargs="+", default=["jewelry_store", "jewelry_store_2", "tech_store", "printing_store"])
    p.add_argument("--val-scenes", nargs="+", default=["kindergarten"])
    p.add_argument("--output", type=Path, default=root / "tracker_pipeline/data/aflink/AFLink_tracker_output_train4_valkindergarten.pth")
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--val-batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--min-tracklet-len", type=int, default=10)
    p.add_argument("--min-purity", type=float, default=0.6)
    p.add_argument("--max-internal-gap", type=int, default=3)
    p.add_argument("--min-temporal-gap", type=int, default=1)
    p.add_argument("--max-temporal-gap", type=int, default=120)
    p.add_argument("--max-spatial-gap", type=float, default=350.0)
    p.add_argument("--input-len", type=int, default=30)
    p.add_argument("--artificial-splits-per-tracklet", type=int, default=3)
    p.add_argument("--neg-pos-ratio", type=float, default=2.0)
    p.add_argument("--min-negative-pairs", type=int, default=100)
    p.add_argument("--max-positive-pairs", type=int, default=20000)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
