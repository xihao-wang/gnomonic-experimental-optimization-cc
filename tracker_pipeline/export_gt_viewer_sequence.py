#!/usr/bin/env python3
"""Export gnomonic tracking outputs to the old GT-edit viewer format.

The legacy viewer in ``suivi-changement-apparence`` expects a MOT-like sequence:

    sequence_dir/
      img1/000001.jpg
      gt/
      detections.npy
      result.txt

This tool builds that structure from a gnomonic image sequence and the
``tracks.csv`` produced by ``tracker_pipeline/process_video_strongsort.py``.

Notes:
    * MOT ``result.txt`` and ``gt.txt`` store axis-aligned boxes only.
    * ``rotated_boxes.jsonl`` stores a sidecar radial rotated box reconstructed
      from ``x,y,w,h`` and the fisheye image center. This matches the current
      video visualization logic.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from detection_pipeline.backprojection import build_radial_bbox


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def _collect_images(image_dir: Path) -> List[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    images = sorted([p for p in image_dir.iterdir() if p.is_file() and _is_image(p)], key=lambda p: p.name)
    if not images:
        raise FileNotFoundError(f"No image files found in: {image_dir}")
    return images


def _read_tracks_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"tracks.csv not found: {path}")
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"frame", "track_id", "x", "y", "w", "h"}
    missing = required.difference(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return rows


def _safe_float(row: Dict[str, str], key: str, default=None):
    value = row.get(key)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _safe_int(row: Dict[str, str], key: str, default=0) -> int:
    value = row.get(key)
    if value in (None, ""):
        return default
    return int(float(value))


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(src.resolve(), dst)
    else:
        raise ValueError(f"Unsupported image mode: {mode}")


def _write_img1(images: List[Path], output_dir: Path, mode: str) -> None:
    img1 = output_dir / "img1"
    img1.mkdir(parents=True, exist_ok=True)
    for idx, image_path in enumerate(images, start=1):
        dst = img1 / f"{idx:06d}{image_path.suffix.lower()}"
        _link_or_copy(image_path, dst, mode)


def _write_seqinfo(output_dir: Path, name: str, images: List[Path], fps: float) -> None:
    first = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Could not read first image: {images[0]}")
    height, width = first.shape[:2]
    with (output_dir / "seqinfo.ini").open("w") as f:
        f.write("[Sequence]\n")
        f.write(f"name={name}\n")
        f.write("imDir=img1\n")
        f.write(f"frameRate={int(round(fps))}\n")
        f.write(f"seqLength={len(images)}\n")
        f.write(f"imWidth={width}\n")
        f.write(f"imHeight={height}\n")
        f.write(f"imExt={images[0].suffix.lower()}\n")


def _write_result_txt(rows: Iterable[Dict[str, str]], output_path: Path) -> None:
    with output_path.open("w") as f:
        for row in rows:
            frame = _safe_int(row, "frame")
            track_id = _safe_int(row, "track_id")
            x = _safe_float(row, "x", 0.0)
            y = _safe_float(row, "y", 0.0)
            w = _safe_float(row, "w", 0.0)
            h = _safe_float(row, "h", 0.0)
            conf = _safe_float(row, "match_confidence", 1.0)
            f.write(f"{frame},{track_id},{x:.3f},{y:.3f},{w:.3f},{h:.3f},{conf:.6f},-1,-1,-1\n")


def _write_detection_npy(rows: Iterable[Dict[str, str]], output_path: Path, feature_dim: int) -> None:
    det_rows = []
    for idx, row in enumerate(rows):
        frame = _safe_int(row, "frame")
        x = _safe_float(row, "x", 0.0)
        y = _safe_float(row, "y", 0.0)
        w = _safe_float(row, "w", 0.0)
        h = _safe_float(row, "h", 0.0)
        conf = _safe_float(row, "match_confidence", 1.0)
        base = [frame, idx, x, y, w, h, conf, -1.0, -1.0, -1.0]
        det_rows.append(base + [0.0] * feature_dim)
    arr = np.asarray(det_rows, dtype=np.float32)
    np.save(output_path, arr)


def _write_empty_scores(frames: Iterable[int], output_path: Path) -> None:
    with output_path.open("w") as f:
        for frame in sorted(set(frames)):
            f.write(json.dumps({"frame": int(frame)}) + "\n")


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


def _reconstruct_radial_rotated_box(row: Dict[str, str], image_shape) -> Dict:
    x = _safe_float(row, "x", 0.0)
    y = _safe_float(row, "y", 0.0)
    w = _safe_float(row, "w", 0.0)
    h = _safe_float(row, "h", 0.0)
    center = (x + w / 2.0, y + h / 2.0)
    fisheye_center = (image_shape[1] / 2.0, image_shape[0] / 2.0)
    radial = build_radial_bbox(center, w, h, fisheye_center)
    return {
        "center_x": float(radial["center"][0]),
        "center_y": float(radial["center"][1]),
        "width": float(radial["size"][0]),
        "height": float(radial["size"][1]),
        "angle": float(radial["angle"]),
        "corners": _jsonable(radial.get("corners")),
        "source": "reconstructed_from_axis_tlwh_and_fisheye_center",
    }


def _write_rotated_sidecar(rows: Iterable[Dict[str, str]], output_path: Path, image_shape) -> None:
    with output_path.open("w") as f:
        for row in rows:
            obj = {
                "frame": _safe_int(row, "frame"),
                "track_id": _safe_int(row, "track_id"),
                "axis_aligned_tlwh": [
                    _safe_float(row, "x", 0.0),
                    _safe_float(row, "y", 0.0),
                    _safe_float(row, "w", 0.0),
                    _safe_float(row, "h", 0.0),
                ],
                "rotated_box": _reconstruct_radial_rotated_box(row, image_shape),
            }
            f.write(json.dumps(obj) + "\n")


def _write_readme(output_dir: Path, old_repo: str) -> None:
    with (output_dir / "README_GT_VIEWER.txt").open("w") as f:
        f.write("MOT-like export for the old debug_match_viewer GT edit mode.\n\n")
        f.write("Run from the old suivi-changement-apparence repository:\n\n")
        f.write(f"cd {old_repo}\n")
        f.write("python3 tools/debug_match_viewer.py \\\n")
        f.write(f"  --sequence_dir {output_dir.resolve()} \\\n")
        f.write(f"  --detection_file {str((output_dir / 'detections.npy').resolve())} \\\n")
        f.write(f"  --result_file {str((output_dir / 'result.txt').resolve())} \\\n")
        f.write(f"  --temporal_scores_file {str((output_dir / 'empty_scores.jsonl').resolve())} \\\n")
        f.write("  --gt_edit \\\n")
        f.write("  --gt_seed tracks\n\n")
        f.write("Important: MOT result/gt files are axis-aligned. A radial rotated-box sidecar is\n")
        f.write("stored in rotated_boxes.jsonl, reconstructed from x,y,w,h and the fisheye center.\n")
        f.write("It matches the current visualization logic but is not editable by the old viewer.\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export gnomonic tracks to legacy GT viewer format.")
    parser.add_argument("--image-dir", required=True, help="Original fisheye image sequence directory.")
    parser.add_argument("--tracks-csv", required=True, help="tracks.csv from process_video_strongsort.py.")
    parser.add_argument(
        "--output-sequence",
        default=None,
        help="Output MOT-like sequence directory. Defaults to <tracks_dir>/viewer_sequence.",
    )
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--feature-dim", type=int, default=8, help="Dummy feature dimension for viewer detections.npy.")
    parser.add_argument(
        "--image-mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="Use symlinks by default to avoid duplicating image data.",
    )
    parser.add_argument(
        "--old-repo",
        default="/home/wang/桌面/suivi-changement-apparence",
        help="Path printed in the generated viewer command.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    tracks_csv = Path(args.tracks_csv)
    output_dir = Path(args.output_sequence) if args.output_sequence else tracks_csv.parent / "viewer_sequence"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "gt").mkdir(exist_ok=True)

    images = _collect_images(image_dir)
    first_image = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
    if first_image is None:
        raise RuntimeError(f"Could not read first image: {images[0]}")
    rows = _read_tracks_csv(tracks_csv)
    frames = [_safe_int(row, "frame") for row in rows]

    _write_img1(images, output_dir, args.image_mode)
    _write_seqinfo(output_dir, output_dir.name, images, args.fps)
    _write_result_txt(rows, output_dir / "result.txt")
    _write_detection_npy(rows, output_dir / "detections.npy", args.feature_dim)
    _write_empty_scores(frames, output_dir / "empty_scores.jsonl")
    _write_rotated_sidecar(rows, output_dir / "rotated_boxes.jsonl", first_image.shape)
    _write_readme(output_dir, args.old_repo)

    print(f"images       : {len(images)}")
    print(f"track rows   : {len(rows)}")
    print(f"output       : {output_dir}")
    print(f"viewer readme: {output_dir / 'README_GT_VIEWER.txt'}")


if __name__ == "__main__":
    main()
