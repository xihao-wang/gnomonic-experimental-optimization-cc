#!/usr/bin/env python3
"""Convert WEPDTOF JSON annotations to the rotated GT viewer format.

The WEPDTOF annotations already contain person IDs and rotated boxes, so this
converter does not match GT points against tracker output. It directly creates:

    viewer_sequence/
      img1/000001.jpg
      gt/gt.txt
      gt/gt_work_rotated.json
      gt/gt_rotated.jsonl
      rotated_boxes.jsonl

``gt.txt`` uses axis-aligned enclosing boxes for MOT-style evaluation/training.
The JSON sidecars keep the original rotated boxes for visual inspection/editing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _collect_images(image_dir: Path) -> List[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS], key=lambda p: p.name)
    if not images:
        raise FileNotFoundError(f"No images found in: {image_dir}")
    return images


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(src.resolve(), dst)
    else:
        raise ValueError(f"Unsupported image mode: {mode}")


def _rotated_corners(cx: float, cy: float, w: float, h: float, angle_deg: float) -> np.ndarray:
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    local = np.array(
        [
            [-w / 2.0, -h / 2.0],
            [w / 2.0, -h / 2.0],
            [w / 2.0, h / 2.0],
            [-w / 2.0, h / 2.0],
        ],
        dtype=np.float32,
    )
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    return local @ rot.T + np.array([cx, cy], dtype=np.float32)


def _axis_tlwh_from_rotated(cx: float, cy: float, w: float, h: float, angle_deg: float) -> Tuple[float, float, float, float]:
    corners = _rotated_corners(cx, cy, w, h, angle_deg)
    x1, y1 = corners.min(axis=0)
    x2, y2 = corners.max(axis=0)
    return float(x1), float(y1), float(x2 - x1), float(y2 - y1)


def _frame_number_from_image_id(image_id: str, image_id_to_frame: Dict[str, int]) -> int:
    if image_id in image_id_to_frame:
        return image_id_to_frame[image_id]
    digits = "".join(ch for ch in image_id.split("_")[-1] if ch.isdigit())
    if not digits:
        raise ValueError(f"Could not infer frame number from image_id={image_id!r}")
    return int(digits)


def _load_wepdtof_annotations(
    annotation_json: Path,
    image_id_to_frame: Dict[str, int],
    id_offset: int,
) -> Dict[int, List[dict]]:
    data = json.loads(annotation_json.read_text())
    boxes_by_frame: Dict[int, List[dict]] = {}
    for ann in data.get("annotations", []):
        if int(ann.get("category_id", 1)) != 1:
            continue
        bbox = ann.get("bbox", [])
        if len(bbox) < 5:
            continue
        cx, cy, w, h, angle = map(float, bbox[:5])
        person_id = int(ann["person_id"]) + int(id_offset)
        frame = _frame_number_from_image_id(str(ann["image_id"]), image_id_to_frame)
        if w <= 0 or h <= 0 or person_id <= 0:
            continue
        boxes_by_frame.setdefault(frame, []).append(
            {
                "gt_id": person_id,
                "cx": cx,
                "cy": cy,
                "w": w,
                "h": h,
                "angle": angle,
            }
        )
    return boxes_by_frame


def _write_img1(images: List[Path], output_dir: Path, mode: str) -> Dict[str, int]:
    img1 = output_dir / "img1"
    img1.mkdir(parents=True, exist_ok=True)
    image_id_to_frame: Dict[str, int] = {}
    for frame, src in enumerate(images, start=1):
        dst = img1 / f"{frame:06d}{src.suffix.lower()}"
        _link_or_copy(src, dst, mode)
        image_id_to_frame[src.stem] = frame
    return image_id_to_frame


def _write_seqinfo(output_dir: Path, scene: str, images: List[Path], fps: float) -> None:
    first = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Could not read image: {images[0]}")
    h, w = first.shape[:2]
    with (output_dir / "seqinfo.ini").open("w") as f:
        f.write("[Sequence]\n")
        f.write(f"name={scene}\n")
        f.write("imDir=img1\n")
        f.write(f"frameRate={int(round(fps))}\n")
        f.write(f"seqLength={len(images)}\n")
        f.write(f"imWidth={w}\n")
        f.write(f"imHeight={h}\n")
        f.write(f"imExt={images[0].suffix.lower()}\n")


def _write_gt_files(boxes_by_frame: Dict[int, List[dict]], output_dir: Path) -> None:
    gt_dir = output_dir / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)

    work = {"format": "rotated_gt_work_v1", "frames": {}}
    with (gt_dir / "gt.txt").open("w") as gt_txt, (gt_dir / "gt_rotated.jsonl").open("w") as rot_gt, (
        output_dir / "rotated_boxes.jsonl"
    ).open("w") as seed_rot:
        for frame, boxes in sorted(boxes_by_frame.items()):
            work["frames"][str(frame)] = []
            for box in sorted(boxes, key=lambda b: b["gt_id"]):
                x, y, w_axis, h_axis = _axis_tlwh_from_rotated(
                    box["cx"], box["cy"], box["w"], box["h"], box["angle"]
                )
                corners = _rotated_corners(box["cx"], box["cy"], box["w"], box["h"], box["angle"]).tolist()

                gt_txt.write(
                    f"{frame},{box['gt_id']},{x:.2f},{y:.2f},{w_axis:.2f},{h_axis:.2f},1,1,1\n"
                )
                work["frames"][str(frame)].append(box)

                rot_row = {
                    "frame": frame,
                    "gt_id": box["gt_id"],
                    "center_x": box["cx"],
                    "center_y": box["cy"],
                    "width": box["w"],
                    "height": box["h"],
                    "angle": box["angle"],
                    "corners": corners,
                    "axis_tlwh": [x, y, w_axis, h_axis],
                }
                rot_gt.write(json.dumps(rot_row) + "\n")

                seed_row = {
                    "frame": frame,
                    "track_id": box["gt_id"],
                    "axis_aligned_tlwh": [x, y, w_axis, h_axis],
                    "rotated_box": {
                        "center_x": box["cx"],
                        "center_y": box["cy"],
                        "width": box["w"],
                        "height": box["h"],
                        "angle": box["angle"],
                    },
                }
                seed_rot.write(json.dumps(seed_row) + "\n")

    (gt_dir / "gt_work_rotated.json").write_text(json.dumps(work, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert WEPDTOF JSON annotations to rotated viewer GT.")
    parser.add_argument("--frames-dir", required=True, type=Path)
    parser.add_argument("--annotation-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--image-mode", choices=["copy", "symlink"], default="symlink")
    parser.add_argument(
        "--id-offset",
        type=int,
        default=0,
        help="Offset added to each WEPDTOF person_id, useful when merging scenes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = _collect_images(args.frames_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_id_to_frame = _write_img1(images, args.output_dir, args.image_mode)
    _write_seqinfo(args.output_dir, args.frames_dir.name, images, args.fps)
    boxes_by_frame = _load_wepdtof_annotations(args.annotation_json, image_id_to_frame, args.id_offset)
    _write_gt_files(boxes_by_frame, args.output_dir)

    ann_count = sum(len(v) for v in boxes_by_frame.values())
    id_count = len({box["gt_id"] for boxes in boxes_by_frame.values() for box in boxes})
    print(f"Wrote viewer sequence: {args.output_dir}")
    print(f"Images: {len(images)}")
    print(f"GT frames: {len(boxes_by_frame)}")
    print(f"GT boxes: {ann_count}")
    print(f"GT IDs: {id_count}")


if __name__ == "__main__":
    main()
