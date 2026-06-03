#!/usr/bin/env python3
"""Filter MOT GT boxes by overlap with detector/tracker output boxes.

This creates a detection-conditioned GT file: GT boxes are kept only if at
least one box from a reference MOT result overlaps them in the same frame.
Use this for tracker-association diagnostics, not for standard MOT scores.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MotRow:
    frame: int
    obj_id: int
    x: float
    y: float
    w: float
    h: float
    parts: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep GT boxes that overlap same-frame detection/tracker boxes."
    )
    parser.add_argument("--gt", required=True, type=Path, help="Original MOT gt.txt.")
    parser.add_argument(
        "--reference",
        required=True,
        type=Path,
        help="MOT-format detector/tracker result used to decide visible GT boxes.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Filtered GT output txt.")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--ignore-gt-mark-zero",
        action="store_true",
        help="If set, GT rows with mark/conf column <= 0 are ignored before filtering.",
    )
    parser.add_argument(
        "--reference-min-score",
        type=float,
        default=None,
        help="Optional minimum reference score/confidence from column 7.",
    )
    return parser.parse_args()


def parse_mot_rows(path: Path, *, min_score: float | None = None) -> list[MotRow]:
    rows: list[MotRow] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            raise ValueError(f"{path}:{line_no}: expected at least 6 MOT columns")
        try:
            frame = int(float(parts[0]))
            obj_id = int(float(parts[1]))
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: invalid MOT numeric fields") from exc
        if min_score is not None and len(parts) >= 7:
            try:
                if float(parts[6]) < min_score:
                    continue
            except ValueError:
                pass
        rows.append(MotRow(frame, obj_id, x, y, w, h, parts))
    return rows


def iou(a: MotRow, b: MotRow) -> float:
    ax1, ay1 = a.x, a.y
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx1, by1 = b.x, b.y
    bx2, by2 = b.x + b.w, b.y + b.h

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, a.w) * max(0.0, a.h)
    area_b = max(0.0, b.w) * max(0.0, b.h)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def gt_is_marked(row: MotRow) -> bool:
    if len(row.parts) < 7:
        return True
    try:
        return float(row.parts[6]) > 0.0
    except ValueError:
        return True


def main() -> None:
    args = parse_args()
    if not args.gt.exists():
        raise FileNotFoundError(args.gt)
    if not args.reference.exists():
        raise FileNotFoundError(args.reference)

    gt_rows = parse_mot_rows(args.gt)
    ref_rows = parse_mot_rows(args.reference, min_score=args.reference_min_score)

    ref_by_frame: dict[int, list[MotRow]] = defaultdict(list)
    for row in ref_rows:
        ref_by_frame[row.frame].append(row)

    kept: list[MotRow] = []
    skipped_unmarked = 0
    for gt_row in gt_rows:
        if args.ignore_gt_mark_zero and not gt_is_marked(gt_row):
            skipped_unmarked += 1
            continue
        refs = ref_by_frame.get(gt_row.frame, [])
        if any(iou(gt_row, ref) >= args.iou_threshold for ref in refs):
            kept.append(gt_row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(",".join(row.parts) for row in kept) + ("\n" if kept else ""),
        encoding="utf-8",
    )

    print(f"GT input rows       : {len(gt_rows)}")
    print(f"Reference rows      : {len(ref_rows)}")
    print(f"Skipped unmarked GT : {skipped_unmarked}")
    print(f"Kept GT rows        : {len(kept)}")
    print(f"Removed GT rows     : {len(gt_rows) - skipped_unmarked - len(kept)}")
    print(f"IoU threshold       : {args.iou_threshold}")
    print(f"Output              : {args.output}")


if __name__ == "__main__":
    main()
