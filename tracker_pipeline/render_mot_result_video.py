#!/usr/bin/env python3
"""Render fisheye tracking videos from MOT result.txt files."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detection_pipeline.backprojection import build_radial_bbox, draw_rotated_bbox

TRACK_COLORS = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (0, 255, 255),
    (255, 0, 255),
    (255, 128, 0),
    (128, 0, 255),
    (0, 128, 255),
    (0, 200, 100),
    (200, 200, 0),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render videos from WEPDTOF frames and MOT result files.")
    parser.add_argument("--result-root", type=Path, required=True, help="Root containing <seq>/*/result.txt.")
    parser.add_argument("--wepdtof-root", type=Path, default=Path("/home/wang/下载/WEPDTOF/WEPDTOF"))
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=[
            "convenience_store",
            "empty_store",
            "exhibition",
            "it_office",
            "large_office",
            "large_office_2",
            "warehouse",
        ],
    )
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--trail-length", type=int, default=30)
    parser.add_argument("--no-h264-copy", action="store_true")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _collect_frames(frames_dir: Path) -> List[Path]:
    frames = sorted([p for p in frames_dir.iterdir() if p.is_file() and _is_image(p)], key=lambda p: p.name)
    if not frames:
        raise FileNotFoundError(f"No frames found in {frames_dir}")
    return frames


def _load_mot(path: Path) -> Dict[int, List[Tuple[int, float, float, float, float, float]]]:
    rows: Dict[int, List[Tuple[int, float, float, float, float, float]]] = defaultdict(list)
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            raise ValueError(f"{path}:{line_no}: expected at least 6 MOT columns")
        frame = int(float(parts[0]))
        track_id = int(float(parts[1]))
        x, y, w, h = [float(v) for v in parts[2:6]]
        conf = 1.0
        if len(parts) >= 7:
            try:
                conf = float(parts[6])
            except ValueError:
                pass
        if w > 0 and h > 0:
            rows[frame].append((track_id, x, y, w, h, conf))
    return rows


def _color(track_id: int) -> Tuple[int, int, int]:
    return TRACK_COLORS[int(track_id) % len(TRACK_COLORS)]


def _axis_tlwh_to_radial_bbox(tlwh: Sequence[float], frame_shape: Sequence[int], confidence: float = 1.0) -> dict:
    x, y, w, h = [float(v) for v in tlwh]
    center = (x + w / 2.0, y + h / 2.0)
    fisheye_center = (frame_shape[1] / 2.0, frame_shape[0] / 2.0)
    bbox = build_radial_bbox(center, w, h, fisheye_center)
    bbox["confidence"] = float(confidence)
    bbox["class_name"] = "track"
    return bbox


def _draw_tracks(frame, tracks, trail_history: Dict[int, List[Tuple[int, int]]], trail_len: int) -> None:
    for track_id, x, y, w, h, conf in tracks:
        color = _color(track_id)
        bbox = _axis_tlwh_to_radial_bbox((x, y, w, h), frame.shape, confidence=conf)
        draw_rotated_bbox(frame, bbox, color=color, thickness=2, draw_label=False)

        cx, cy = int(round(x + w / 2.0)), int(round(y + h / 2.0))
        cv2.circle(frame, (cx, cy), 4, color, -1)
        cv2.putText(frame, f"ID {track_id}", (cx + 6, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        if trail_len > 0:
            trail = trail_history[track_id]
            trail.append((cx, cy))
            if len(trail) > trail_len:
                del trail[0]
            for idx in range(1, len(trail)):
                cv2.line(frame, trail[idx - 1], trail[idx], color, 1)


def _transcode_h264(input_video: Path, output_video: Path) -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_video),
        ],
        check=True,
    )
    return True


def _find_result(result_root: Path, seq: str) -> Path:
    direct = result_root / seq / "result.txt"
    if direct.exists():
        return direct
    matches = sorted((result_root / seq).glob("*/result.txt"))
    if not matches:
        raise FileNotFoundError(f"No result.txt found for {seq} under {result_root}")
    return matches[-1]


def render_sequence(result_txt: Path, frames_dir: Path, fps: float, trail_len: int, h264: bool) -> None:
    out_dir = result_txt.parent
    video_path = out_dir / "tracked_fisheye.mp4"
    h264_path = out_dir / "tracked_fisheye_h264.mp4"
    frames = _collect_frames(frames_dir)
    tracks_by_frame = _load_mot(result_txt)

    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"Could not read first frame: {frames[0]}")
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {video_path}")

    trail_history: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for frame_idx, frame_path in enumerate(frames, start=1):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"WARNING: skipping unreadable frame: {frame_path}")
            continue
        vis = frame.copy()
        _draw_tracks(vis, tracks_by_frame.get(frame_idx, []), trail_history, trail_len)
        writer.write(vis)
    writer.release()

    if h264:
        try:
            _transcode_h264(video_path, h264_path)
        except subprocess.CalledProcessError as exc:
            print(f"WARNING: H.264 transcode failed for {video_path}: {exc}")

    print(video_path)


def main() -> None:
    args = parse_args()
    result_root = _resolve(args.result_root)
    h264 = not args.no_h264_copy
    for seq in args.sequences:
        result_txt = _find_result(result_root, seq)
        frames_dir = args.wepdtof_root / "frames" / seq
        render_sequence(result_txt, frames_dir, args.fps, args.trail_length, h264)


if __name__ == "__main__":
    main()
