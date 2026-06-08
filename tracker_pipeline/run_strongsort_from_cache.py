#!/usr/bin/env python3
"""Run StrongSORT association from cached detections and ReID features."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tracker_pipeline.process_video_strongsort import _load_temporal_model, _resolve
from tracker_pipeline.strongsort import nn_matching
from tracker_pipeline.strongsort.association_veto_model import load_association_veto_scorer
from tracker_pipeline.strongsort.detection import Detection as StrongSortDetection
from tracker_pipeline.strongsort.opts import opt as strongsort_opt
from tracker_pipeline.strongsort.tracker import Tracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run StrongSORT from cached detection/ReID features.")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-name", default=None)

    parser.add_argument("--matching-threshold", type=float, default=0.2)
    parser.add_argument("--max-iou-distance", type=float, default=0.7)
    parser.add_argument("--max-age", type=int, default=100)
    parser.add_argument("--n-init", type=int, default=3)
    parser.add_argument("--nn-budget", type=int, default=100)
    parser.add_argument("--ltm-stm", action="store_true")
    parser.add_argument("--memory-aware", action="store_true")
    parser.add_argument("--memory-init", action="store_true")
    parser.add_argument("--topk", action="store_true")

    parser.add_argument("--learned-temporal", action="store_true")
    parser.add_argument("--fuse-learned-temporal", action="store_true")
    parser.add_argument(
        "--tracker-model",
        "--temporal-model-ckpt",
        dest="tracker_model",
        default="tracker_pipeline/learnable_model/gt_l15_infonce_track.pt",
    )
    parser.add_argument("--temporal-device", default="cpu")
    parser.add_argument("--learned-temporal-alpha", type=float, default=1.0)
    parser.add_argument("--learned-temporal-min-scale", type=float, default=0.02)
    parser.add_argument("--learned-temporal-max-correction", type=float, default=0.02)
    parser.add_argument("--learned-temporal-veto-cost", type=float, default=0.8)
    parser.add_argument("--matching-debug-jsonl", default=None)
    parser.add_argument("--association-veto-model", default=None)
    parser.add_argument("--association-veto-device", default="cpu")
    parser.add_argument("--association-veto-threshold", type=float, default=0.7)
    return parser.parse_args()


def _load_cache(path: Path):
    data = np.load(path, allow_pickle=True)
    metadata = {}
    if "metadata" in data:
        raw = data["metadata"].item()
        metadata = json.loads(str(raw))
    return {
        "frames": data["frames"].astype(np.int32),
        "tlwhs": data["tlwhs"].astype(np.float32),
        "confidences": data["confidences"].astype(np.float32),
        "features": data["features"].astype(np.float32),
        "metadata": metadata,
    }


def _make_output_dir(base_dir: Path, sequence_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base_dir / f"{sequence_name}_strongsort_cache_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def _write_mot_result_txt(path: Path, frame_idx: int, tracker: Tracker) -> None:
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            x, y, w, h = [float(v) for v in track.to_tlwh()]
            conf = 1.0 if track.match_confidence is None else float(track.match_confidence)
            writer.writerow([
                frame_idx,
                int(track.track_id),
                f"{x:.3f}",
                f"{y:.3f}",
                f"{w:.3f}",
                f"{h:.3f}",
                f"{conf:.6f}",
                -1,
                -1,
                -1,
            ])


def _write_tracks_csv(path: Path, frame_idx: int, tracker: Tracker) -> None:
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["frame", "track_id", "x", "y", "w", "h", "match_confidence", "time_since_update"])
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            x, y, w, h = [float(v) for v in track.to_tlwh()]
            writer.writerow([
                frame_idx,
                int(track.track_id),
                f"{x:.3f}",
                f"{y:.3f}",
                f"{w:.3f}",
                f"{h:.3f}",
                "" if track.match_confidence is None else f"{float(track.match_confidence):.6f}",
                int(track.time_since_update),
            ])


def _write_detections_mot_txt(path: Path, frame_idx: int, detections: List[StrongSortDetection]) -> None:
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        for det in detections:
            x, y, w, h = [float(v) for v in det.tlwh]
            if w <= 0.0 or h <= 0.0:
                continue
            writer.writerow([
                frame_idx,
                -1,
                f"{x:.3f}",
                f"{y:.3f}",
                f"{w:.3f}",
                f"{h:.3f}",
                f"{float(det.confidence):.6f}",
                -1,
                -1,
                -1,
            ])


def _build_tracker(args: argparse.Namespace) -> Tracker:
    strongsort_opt.enable_stm_ltm = bool(args.ltm_stm)
    strongsort_opt.enable_memory_matching = bool(args.memory_aware)
    strongsort_opt.enable_topk_matching = bool(args.topk)
    strongsort_opt.enable_memory_init_control = bool(args.memory_init)
    strongsort_opt.MC = False

    temporal_model = None
    if args.learned_temporal:
        temporal_model = _load_temporal_model(_resolve(args.tracker_model), args.temporal_device)

    association_veto_model = None
    if args.association_veto_model:
        association_veto_model = load_association_veto_scorer(
            _resolve(args.association_veto_model),
            device=args.association_veto_device,
        )

    matching_debug_jsonl = None
    if args.matching_debug_jsonl:
        matching_debug_jsonl = _resolve(args.matching_debug_jsonl)
        matching_debug_jsonl.parent.mkdir(parents=True, exist_ok=True)
        if matching_debug_jsonl.exists():
            matching_debug_jsonl.unlink()

    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine",
        matching_threshold=args.matching_threshold,
        budget=args.nn_budget,
    )
    return Tracker(
        metric,
        max_iou_distance=args.max_iou_distance,
        max_age=args.max_age,
        n_init=args.n_init,
        temporal_model=temporal_model,
        temporal_alpha=args.learned_temporal_alpha,
        fuse_temporal_model=args.fuse_learned_temporal,
        temporal_max_correction=args.learned_temporal_max_correction,
        temporal_min_scale=args.learned_temporal_min_scale,
        temporal_veto_cost=args.learned_temporal_veto_cost,
        association_veto_model=association_veto_model,
        association_veto_threshold=args.association_veto_threshold,
        matching_debug_jsonl=str(matching_debug_jsonl) if matching_debug_jsonl else None,
    )


def run(args: argparse.Namespace) -> Path:
    cache_path = _resolve(str(args.cache))
    cache = _load_cache(cache_path)
    metadata = cache["metadata"]
    sequence_name = args.sequence_name or metadata.get("sequence") or cache_path.stem
    output_dir = _make_output_dir(_resolve(str(args.output_dir)), sequence_name)
    result_path = output_dir / "result.txt"
    tracks_csv = output_dir / "tracks.csv"
    detections_mot = output_dir / "detections_mot.txt"

    by_frame: Dict[int, List[int]] = defaultdict(list)
    for idx, frame_idx in enumerate(cache["frames"]):
        by_frame[int(frame_idx)].append(idx)

    tracker = _build_tracker(args)
    max_frame = int(max(metadata.get("num_frames", 0), int(cache["frames"].max()) if len(cache["frames"]) else 0))

    for frame_idx in range(1, max_frame + 1):
        indices = by_frame.get(frame_idx, [])
        detections = [
            StrongSortDetection(cache["tlwhs"][idx], float(cache["confidences"][idx]), cache["features"][idx])
            for idx in indices
        ]
        tracker.predict()
        tracker.matching_debug_frame = frame_idx
        tracker.update(detections)
        _write_detections_mot_txt(detections_mot, frame_idx, detections)
        _write_tracks_csv(tracks_csv, frame_idx, tracker)
        _write_mot_result_txt(result_path, frame_idx, tracker)

    with (output_dir / "metadata.txt").open("w") as f:
        f.write(f"input_cache={cache_path}\n")
        f.write(f"sequence={sequence_name}\n")
        f.write(f"mot_result_txt={result_path}\n")
        f.write(f"tracks_csv={tracks_csv}\n")
        f.write(f"detections_mot_txt={detections_mot}\n")
        f.write(f"learned_temporal={args.learned_temporal}\n")
        f.write(f"fuse_learned_temporal={args.fuse_learned_temporal}\n")
        f.write(f"tracker_model={_resolve(args.tracker_model) if args.learned_temporal else ''}\n")
        f.write(f"learned_temporal_alpha={args.learned_temporal_alpha}\n")
        f.write(f"learned_temporal_min_scale={args.learned_temporal_min_scale}\n")
        f.write(f"learned_temporal_max_correction={args.learned_temporal_max_correction}\n")
        f.write(f"learned_temporal_veto_cost={args.learned_temporal_veto_cost}\n")
        f.write(f"matching_threshold={args.matching_threshold}\n")
        f.write(f"max_iou_distance={args.max_iou_distance}\n")
        f.write(f"max_age={args.max_age}\n")
        f.write(f"n_init={args.n_init}\n")

    print(f"done: {output_dir}")
    return output_dir


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
