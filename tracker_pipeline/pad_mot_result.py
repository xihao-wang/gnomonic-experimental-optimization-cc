#!/usr/bin/env python3
"""Apply simple bbox padding to a MOT-format result file.

This is intentionally post-processing only: it does not change detection or
tracking code. It is useful for testing whether tighter boxes are hurting
TrackEval IoU.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Pad MOT result bounding boxes.")
    parser.add_argument("--input", required=True, type=Path, help="Input MOT result txt.")
    parser.add_argument("--output", required=True, type=Path, help="Output padded MOT result txt.")
    parser.add_argument("--pad-x", type=float, default=0.15, help="Padding per side as a fraction of width.")
    parser.add_argument("--pad-y", type=float, default=0.05, help="Padding per side as a fraction of height.")
    parser.add_argument("--image-width", type=float, default=None, help="Optional image width for clipping.")
    parser.add_argument("--image-height", type=float, default=None, help="Optional image height for clipping.")
    return parser.parse_args()


def fmt(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def pad_box(x: float, y: float, w: float, h: float, pad_x: float, pad_y: float):
    new_x = x - pad_x * w
    new_y = y - pad_y * h
    new_w = w * (1.0 + 2.0 * pad_x)
    new_h = h * (1.0 + 2.0 * pad_y)
    return new_x, new_y, new_w, new_h


def clip_box(x: float, y: float, w: float, h: float, image_width, image_height):
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    if image_width is not None:
        x1 = max(0.0, min(float(image_width), x1))
        x2 = max(0.0, min(float(image_width), x2))
    if image_height is not None:
        y1 = max(0.0, min(float(image_height), y1))
        y2 = max(0.0, min(float(image_height), y2))
    return x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)


def main():
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_lines = []
    changed = 0
    for line in args.input.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.strip().split(",")
        if len(parts) < 6:
            out_lines.append(line)
            continue
        try:
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
        except ValueError:
            out_lines.append(line)
            continue
        x, y, w, h = pad_box(x, y, w, h, args.pad_x, args.pad_y)
        x, y, w, h = clip_box(x, y, w, h, args.image_width, args.image_height)
        parts[2] = fmt(x)
        parts[3] = fmt(y)
        parts[4] = fmt(w)
        parts[5] = fmt(h)
        out_lines.append(",".join(parts))
        changed += 1

    args.output.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))
    print(f"Wrote: {args.output}")
    print(f"Rows padded: {changed}")


if __name__ == "__main__":
    main()
