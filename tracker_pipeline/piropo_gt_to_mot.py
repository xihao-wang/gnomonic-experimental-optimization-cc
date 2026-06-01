#!/usr/bin/env python3
"""Convert PIROPO point annotations to MOT-style bbox GT.

PIROPO ground-truth CSV files store one frame id followed by one or more
person points:

    frame,x1,y1,x2,y2,...

This project needs MOT-style bounding boxes:

    frame,id,x,y,w,h,conf,class,visibility

Since the official annotation is point-based, this script uses the current
tracker/detector output boxes as bbox proposals and assigns each GT point to
one proposal on the same frame. The resulting file is therefore a bbox GT
derived from official person points plus the current box geometry.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class GTPoint:
    frame: int
    gt_id: int
    x: float
    y: float


@dataclass(frozen=True)
class TrackBox:
    frame: int
    track_id: int
    x: float
    y: float
    w: float
    h: float
    confidence: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    def contains(self, px: float, py: float, padding: float = 0.0) -> bool:
        return (
            self.x - padding <= px <= self.x + self.w + padding
            and self.y - padding <= py <= self.y + self.h + padding
        )

    def center_distance(self, px: float, py: float) -> float:
        return math.hypot(self.cx - px, self.cy - py)


def load_piropo_points(path: Path) -> List[GTPoint]:
    points: List[GTPoint] = []
    with path.open(newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            values = [v.strip() for v in row if v.strip() != ""]
            if len(values) < 3:
                continue
            frame = int(float(values[0]))
            coords = values[1:]
            for idx in range(0, len(coords) - 1, 2):
                x = float(coords[idx])
                y = float(coords[idx + 1])
                if x == 0.0 and y == 0.0:
                    continue
                points.append(GTPoint(frame=frame, gt_id=idx // 2 + 1, x=x, y=y))
    return points


def load_track_boxes(path: Path) -> List[TrackBox]:
    boxes: List[TrackBox] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"frame", "track_id", "x", "y", "w", "h"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"tracks.csv missing columns: {sorted(missing)}")
        for row in reader:
            boxes.append(
                TrackBox(
                    frame=int(float(row["frame"])),
                    track_id=int(float(row["track_id"])),
                    x=float(row["x"]),
                    y=float(row["y"]),
                    w=float(row["w"]),
                    h=float(row["h"]),
                    confidence=float(row.get("match_confidence") or row.get("confidence") or 1.0),
                )
            )
    return boxes


def group_boxes_by_frame(boxes: Iterable[TrackBox]) -> Dict[int, List[TrackBox]]:
    grouped: Dict[int, List[TrackBox]] = {}
    for box in boxes:
        grouped.setdefault(box.frame, []).append(box)
    return grouped


def match_point_to_box(
    point: GTPoint,
    candidates: Sequence[TrackBox],
    used_indices: set[int],
    point_padding: float,
    max_center_distance: float,
) -> Tuple[Optional[int], Optional[TrackBox], str, float]:
    available = [(idx, box) for idx, box in enumerate(candidates) if idx not in used_indices]
    if not available:
        return None, None, "no_candidate", float("inf")

    inside = [
        (idx, box, box.center_distance(point.x, point.y))
        for idx, box in available
        if box.contains(point.x, point.y, padding=point_padding)
    ]
    if inside:
        idx, box, dist = min(inside, key=lambda item: item[2])
        return idx, box, "inside", dist

    idx, box, dist = min(
        ((idx, box, box.center_distance(point.x, point.y)) for idx, box in available),
        key=lambda item: item[2],
    )
    if dist <= max_center_distance:
        return idx, box, "nearest", dist
    return None, None, "too_far", dist


def score_frame_offset(
    points: Sequence[GTPoint],
    boxes_by_frame: Dict[int, List[TrackBox]],
    offset: int,
    point_padding: float,
    max_center_distance: float,
    sample_limit: int,
) -> Tuple[int, float]:
    matched = 0
    total_dist = 0.0
    sampled = 0
    for point in points:
        candidates = boxes_by_frame.get(point.frame + offset, [])
        if not candidates:
            continue
        _, box, reason, dist = match_point_to_box(
            point, candidates, set(), point_padding, max_center_distance
        )
        sampled += 1
        if box is not None and reason in {"inside", "nearest"}:
            matched += 1
            total_dist += dist
        if sampled >= sample_limit:
            break
    mean_dist = total_dist / matched if matched else float("inf")
    return matched, mean_dist


def infer_frame_offset(
    points: Sequence[GTPoint],
    boxes: Sequence[TrackBox],
    point_padding: float,
    max_center_distance: float,
    search_radius: int,
    sample_limit: int,
) -> int:
    if not points or not boxes:
        return 0
    boxes_by_frame = group_boxes_by_frame(boxes)
    gt_min = min(p.frame for p in points)
    track_min = min(b.frame for b in boxes)
    base = track_min - gt_min
    best_offset = base
    best_score = (-1, float("inf"))
    for offset in range(base - search_radius, base + search_radius + 1):
        score = score_frame_offset(
            points, boxes_by_frame, offset, point_padding, max_center_distance, sample_limit
        )
        if score[0] > best_score[0] or (score[0] == best_score[0] and score[1] < best_score[1]):
            best_score = score
            best_offset = offset
    return best_offset


def convert(
    points: Sequence[GTPoint],
    boxes: Sequence[TrackBox],
    frame_offset: int,
    point_padding: float,
    max_center_distance: float,
) -> Tuple[List[Tuple[int, int, TrackBox]], List[dict]]:
    boxes_by_frame = group_boxes_by_frame(boxes)
    matched_rows: List[Tuple[int, int, TrackBox]] = []
    diagnostics: List[dict] = []
    points_by_frame: Dict[int, List[GTPoint]] = {}
    for point in points:
        points_by_frame.setdefault(point.frame, []).append(point)

    for gt_frame in sorted(points_by_frame):
        track_frame = gt_frame + frame_offset
        candidates = boxes_by_frame.get(track_frame, [])
        used_indices: set[int] = set()
        for point in points_by_frame[gt_frame]:
            idx, box, reason, dist = match_point_to_box(
                point, candidates, used_indices, point_padding, max_center_distance
            )
            if idx is not None:
                used_indices.add(idx)
            if box is not None:
                matched_rows.append((track_frame, point.gt_id, box))
            diagnostics.append(
                {
                    "gt_frame": gt_frame,
                    "track_frame": track_frame,
                    "gt_id": point.gt_id,
                    "point_x": point.x,
                    "point_y": point.y,
                    "matched": int(box is not None),
                    "reason": reason,
                    "distance": "" if math.isinf(dist) else f"{dist:.3f}",
                    "track_id": "" if box is None else box.track_id,
                    "bbox_x": "" if box is None else f"{box.x:.3f}",
                    "bbox_y": "" if box is None else f"{box.y:.3f}",
                    "bbox_w": "" if box is None else f"{box.w:.3f}",
                    "bbox_h": "" if box is None else f"{box.h:.3f}",
                }
            )
    return matched_rows, diagnostics


def write_mot_gt(path: Path, rows: Sequence[Tuple[int, int, TrackBox]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for frame, gt_id, box in sorted(rows, key=lambda item: (item[0], item[1])):
            f.write(
                f"{frame},{gt_id},{box.x:.2f},{box.y:.2f},{box.w:.2f},{box.h:.2f},1,1,1\n"
            )


def write_diagnostics(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "gt_frame",
        "track_frame",
        "gt_id",
        "point_x",
        "point_y",
        "matched",
        "reason",
        "distance",
        "track_id",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PIROPO point GT CSV to MOT gt.txt.")
    parser.add_argument("--piropo-gt-csv", required=True, type=Path)
    parser.add_argument("--tracks-csv", required=True, type=Path)
    parser.add_argument("--output-gt-txt", required=True, type=Path)
    parser.add_argument("--diagnostics-csv", default=None, type=Path)
    parser.add_argument("--frame-offset", default=None, type=int)
    parser.add_argument("--auto-frame-offset", action="store_true")
    parser.add_argument("--offset-search-radius", default=300, type=int)
    parser.add_argument("--offset-sample-limit", default=500, type=int)
    parser.add_argument("--point-padding", default=8.0, type=float)
    parser.add_argument("--max-center-distance", default=500.0, type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points = load_piropo_points(args.piropo_gt_csv)
    boxes = load_track_boxes(args.tracks_csv)
    if args.auto_frame_offset:
        frame_offset = infer_frame_offset(
            points,
            boxes,
            args.point_padding,
            args.max_center_distance,
            args.offset_search_radius,
            args.offset_sample_limit,
        )
    else:
        frame_offset = int(args.frame_offset or 0)

    matched_rows, diagnostics = convert(
        points,
        boxes,
        frame_offset,
        args.point_padding,
        args.max_center_distance,
    )
    write_mot_gt(args.output_gt_txt, matched_rows)
    diagnostics_csv = args.diagnostics_csv or args.output_gt_txt.with_name("gt_point_match_diagnostics.csv")
    write_diagnostics(diagnostics_csv, diagnostics)

    total_points = len(points)
    matched = sum(int(row["matched"]) for row in diagnostics)
    print(f"PIROPO points: {total_points}")
    print(f"Track boxes: {len(boxes)}")
    print(f"Frame offset: {frame_offset}")
    print(f"Matched GT points: {matched}/{total_points} ({matched / total_points * 100:.2f}%)")
    print(f"Wrote MOT GT: {args.output_gt_txt}")
    print(f"Wrote diagnostics: {diagnostics_csv}")


if __name__ == "__main__":
    main()
