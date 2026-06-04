from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tracker_pipeline.strongsort.association_veto_model import DEFAULT_FEATURE_NAMES


Box = Tuple[float, float, float, float]


def tlwh_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return 0.0 if union <= 0 else float(inter / union)


def load_gt(gt_path: Path) -> Dict[int, List[Tuple[int, Box]]]:
    by_frame: Dict[int, List[Tuple[int, Box]]] = {}
    with gt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            frame = int(float(parts[0]))
            gt_id = int(float(parts[1]))
            box = tuple(float(v) for v in parts[2:6])
            by_frame.setdefault(frame, []).append((gt_id, box))
    return by_frame


def match_gt_id(box: Sequence[float], gt_items: List[Tuple[int, Box]], threshold: float) -> Optional[int]:
    best_id = None
    best_iou = 0.0
    for gt_id, gt_box in gt_items:
        score = tlwh_iou(box, gt_box)
        if score > best_iou:
            best_iou = score
            best_id = gt_id
    if best_iou >= threshold:
        return best_id
    return None


def matrix_value(matrix, row: int, col: int, default: float = np.nan) -> float:
    if matrix is None:
        return default
    try:
        value = matrix[row][col]
    except (IndexError, TypeError):
        return default
    if value is None:
        return default
    return float(value)


def build_pairs(debug_jsonl: Path, gt_path: Path, output_npz: Path, iou_threshold: float, stage_prefix: str) -> None:
    gt_by_frame = load_gt(gt_path)
    features: List[List[float]] = []
    labels: List[float] = []
    frames: List[int] = []
    track_ids: List[int] = []
    detection_indices: List[int] = []

    with debug_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if not str(record.get("stage", "")).startswith(stage_prefix):
                continue
            frame = record.get("frame")
            if frame is None:
                continue
            base_cost = record.get("base_cost")
            temporal_cost = record.get("temporal_cost")
            gating_distance = record.get("gating_distance")
            iou_matrix = record.get("iou")
            track_tlwhs = record.get("track_tlwhs") or []
            detection_tlwhs = record.get("detection_tlwhs") or []
            ages = record.get("track_ages") or []
            tids = record.get("track_ids") or []
            det_indices = record.get("detection_indices") or []
            gt_items = gt_by_frame.get(int(frame), [])

            for row, track_box in enumerate(track_tlwhs):
                if row >= len(ages) or row >= len(tids):
                    continue
                track_gt = match_gt_id(track_box, gt_items, iou_threshold)
                for col, det_box in enumerate(detection_tlwhs):
                    if col >= len(det_indices):
                        continue
                    reid_distance = matrix_value(base_cost, row, col)
                    kalman_distance = matrix_value(gating_distance, row, col)
                    pair_iou = matrix_value(iou_matrix, row, col)
                    temporal = matrix_value(temporal_cost, row, col, default=1.0)
                    if not np.isfinite(reid_distance) or not np.isfinite(kalman_distance):
                        continue
                    temporal_score = 0.0 if temporal >= 0.999999 else 1.0 - float(temporal)
                    det_gt = match_gt_id(det_box, gt_items, iou_threshold)
                    label = 1.0 if track_gt is not None and track_gt == det_gt else 0.0
                    features.append([
                        temporal_score,
                        reid_distance,
                        kalman_distance,
                        pair_iou,
                        float(ages[row]),
                    ])
                    labels.append(label)
                    frames.append(int(frame))
                    track_ids.append(int(tids[row]))
                    detection_indices.append(int(det_indices[col]))

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        features=np.asarray(features, dtype=np.float32),
        label=np.asarray(labels, dtype=np.float32),
        frame=np.asarray(frames, dtype=np.int32),
        track_id=np.asarray(track_ids, dtype=np.int32),
        detection_index=np.asarray(detection_indices, dtype=np.int32),
        feature_names=np.asarray(DEFAULT_FEATURE_NAMES),
    )
    num_pos = int(np.sum(labels))
    print(f"saved: {output_npz}")
    print(f"samples={len(labels)} positives={num_pos} negatives={len(labels) - num_pos}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build association-veto MLP pairs from StrongSORT matching debug.")
    parser.add_argument("--matching-debug-jsonl", required=True, type=Path)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--stage-prefix", default="appearance_metric")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_pairs(
        debug_jsonl=args.matching_debug_jsonl,
        gt_path=args.gt,
        output_npz=args.output,
        iou_threshold=args.iou_threshold,
        stage_prefix=args.stage_prefix,
    )
