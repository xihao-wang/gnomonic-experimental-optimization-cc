"""End-to-end fisheye tracking with source-aware detections + FastReID + StrongSORT.

This is a parallel runner for validating the new gnomonic-to-tracker bridge. It
keeps the existing ByteTrack entry point untouched.

Run from project root, for example:
    python3 tracker_pipeline/process_video_strongsort.py \
        --input detection_pipeline/videos/Meeting1.mp4 \
        --yolo-model detection_pipeline/models/yolov8n.pt \
        --preset yolo_grid \
        --device cpu \
        --fastreid-device cpu \
        --max-frames 100
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from detection_pipeline.backprojection import build_radial_bbox, draw_rotated_bbox
from detection_pipeline.config import get_cfg as get_detection_cfg
from detection_pipeline.pipeline import DetectionPipeline
from image_composer.presets import get_preset
from tracker_pipeline.gnomonic_adapter import build_tracker_detections, _tracking_tlwh_from_bbox
from tracker_pipeline.reid import FastReIDFeatureExtractor
from tracker_pipeline.strongsort import nn_matching
from tracker_pipeline.strongsort.detection import Detection as StrongSortDetection
from tracker_pipeline.strongsort.temporal_model import TemporalAttentionScorer
from tracker_pipeline.strongsort.tracker import Tracker
from tracker_pipeline.strongsort.opts import opt as strongsort_opt


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


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _PROJECT_ROOT / p


def _build_detection_cfg(args: argparse.Namespace):
    cfg = get_detection_cfg()
    cfg.PROJECTION.PRESET = args.preset
    if args.preset is None:
        cfg.PROJECTION.PROJ_NBR = args.proj_nbr
        cfg.PROJECTION.FOV_H = args.fov_h
        cfg.PROJECTION.FOV_V = args.fov_v
        cfg.PROJECTION.LATITUDE = args.latitude
        cfg.PROJECTION.LON_0 = args.lon_0
        cfg.PROJECTION.LON_STEP = args.lon_step
        cfg.PROJECTION.GRID = [args.grid_rows, args.grid_cols]
        cfg.PROJECTION.COMP_SIZE = [args.comp_width, args.comp_height]
        cfg.PROJECTION.TARGET_MP = args.target_mp

    cfg.YOLO.MODEL = str(_resolve(args.yolo_model))
    cfg.YOLO.DEVICE = args.device
    cfg.YOLO.CONFIDENCE_THRESHOLD = args.confidence

    cfg.NMS.STAGE1.ENABLED = not args.disable_stage1_nms
    cfg.NMS.STAGE1.IOU_THRESHOLD = args.stage1_iou
    cfg.NMS.STAGE2.ENABLED = not args.disable_stage2_nms
    cfg.NMS.STAGE2.SIGMA = args.stage2_sigma
    cfg.NMS.STAGE2.SCORE_THRESHOLD = args.stage2_score_threshold

    cfg.BACKPROJECTION.ENABLED = True
    cfg.OUTPUT.SAVE_COMPOSITE = False
    cfg.OUTPUT.SAVE_COMPOSITE_VIZ = False
    cfg.OUTPUT.SAVE_FISHEYE_VIZ = False
    cfg.OUTPUT.SAVE_LATTICE_VIZ = False
    cfg.VERBOSE = args.verbose
    return cfg


def _build_projection_config(det_cfg, input_path: Path) -> Dict:
    if det_cfg.PROJECTION.PRESET:
        proj_cfg = get_preset(det_cfg.PROJECTION.PRESET)
        if proj_cfg is None:
            raise ValueError(f"Unknown projection preset: {det_cfg.PROJECTION.PRESET}")
    else:
        proj_cfg = {
            "proj_nbr": det_cfg.PROJECTION.PROJ_NBR,
            "fov_h": det_cfg.PROJECTION.FOV_H,
            "fov_v": det_cfg.PROJECTION.FOV_V,
            "latitude": det_cfg.PROJECTION.LATITUDE,
            "lon_0": det_cfg.PROJECTION.LON_0,
            "lon_step": det_cfg.PROJECTION.LON_STEP,
            "grid": det_cfg.PROJECTION.GRID,
            "comp_sz": det_cfg.PROJECTION.COMP_SIZE,
            "target_mp": det_cfg.PROJECTION.TARGET_MP,
        }
    proj_cfg["img_path"] = str(input_path)
    return proj_cfg


def _make_output_dir(base_dir: Path, input_path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base_dir / f"{input_path.stem}_strongsort_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _collect_image_sequence(input_path: Path) -> List[Path]:
    if input_path.is_file() and _is_image_file(input_path):
        return [input_path]
    if not input_path.is_dir():
        return []
    frames = [p for p in input_path.iterdir() if p.is_file() and _is_image_file(p)]
    return sorted(frames, key=lambda p: p.name)


def _color(track_id: int) -> Tuple[int, int, int]:
    return TRACK_COLORS[int(track_id) % len(TRACK_COLORS)]


def _axis_tlwh_to_radial_bbox(tlwh: Sequence[float], frame_shape: Sequence[int], confidence: float = 1.0) -> Dict:
    x, y, w, h = [float(v) for v in tlwh]
    center = (x + w / 2.0, y + h / 2.0)
    fisheye_center = (frame_shape[1] / 2.0, frame_shape[0] / 2.0)
    bbox = build_radial_bbox(center, w, h, fisheye_center)
    bbox["confidence"] = float(confidence)
    bbox["class_name"] = "track"
    return bbox


def _draw_tracks(frame: np.ndarray, tracker: Tracker, trail_history: Dict[int, List[Tuple[int, int]]], trail_len: int) -> None:
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        tid = int(track.track_id)
        tlwh = track.to_tlwh()
        color = _color(tid)
        bbox = _axis_tlwh_to_radial_bbox(tlwh, frame.shape, confidence=track.match_confidence or 1.0)
        draw_rotated_bbox(frame, bbox, color=color, thickness=2, draw_label=False)

        x, y, w, h = [float(v) for v in tlwh]
        cx, cy = int(round(x + w / 2.0)), int(round(y + h / 2.0))
        cv2.circle(frame, (cx, cy), 4, color, -1)
        label = f"ID {tid}"
        if track.match_confidence is not None:
            label += f" {track.match_confidence:.2f}"
        cv2.putText(frame, label, (cx + 6, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        if trail_len > 0:
            trail = trail_history[tid]
            trail.append((cx, cy))
            if len(trail) > trail_len:
                del trail[0]
            for idx in range(1, len(trail)):
                cv2.line(frame, trail[idx - 1], trail[idx], color, 1)


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


def _write_mot_result_txt(path: Path, frame_idx: int, tracker: Tracker) -> None:
    """Append confirmed tracks in MOTChallenge result format.

    Format:
        frame, id, x, y, w, h, conf, -1, -1, -1
    """
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


def _write_detections_mot_txt(path: Path, frame_idx: int, fisheye_bboxes: Sequence[Dict]) -> None:
    """Append tracker-input detections in MOTChallenge-like format.

    Format:
        frame, -1, x, y, w, h, conf, -1, -1, -1
    """
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        for bbox in fisheye_bboxes or []:
            try:
                x, y, w, h = [float(v) for v in _tracking_tlwh_from_bbox(bbox)]
            except (KeyError, TypeError, ValueError):
                continue
            if w <= 0.0 or h <= 0.0:
                continue
            conf = float(bbox.get("confidence", bbox.get("source_confidence", 1.0)))
            writer.writerow([
                frame_idx,
                -1,
                f"{x:.3f}",
                f"{y:.3f}",
                f"{w:.3f}",
                f"{h:.3f}",
                f"{conf:.6f}",
                -1,
                -1,
                -1,
            ])


def _transcode_h264(input_video: Path, output_video: Path) -> bool:
    """Create a broadly compatible H.264 MP4 copy with ffmpeg if available."""
    if shutil.which("ffmpeg") is None:
        return False
    cmd = [
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
    ]
    subprocess.run(cmd, check=True)
    return True


def _load_temporal_model(checkpoint_path: Path, device: str) -> TemporalAttentionScorer:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Temporal model checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(str(checkpoint_path), map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    model = TemporalAttentionScorer(
        feature_dim=int(ckpt.get("feature_dim", 2048)),
        hidden_dim=int(ckpt.get("hidden_dim", 256)),
        num_heads=int(ckpt.get("num_heads", 4)),
        history_len=int(ckpt.get("history_len", 5)),
        long_history_len=int(ckpt.get("long_history_len", 15)),
        use_long_memory=bool(ckpt.get("use_long_memory", True)),
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _make_strongsort_detections(adapter_detections) -> List[StrongSortDetection]:
    return [
        StrongSortDetection(det.tlwh, det.confidence, det.feature)
        for det in adapter_detections
        if det.feature is not None and np.asarray(det.feature).size > 0 and det.tlwh[2] > 0 and det.tlwh[3] > 0
    ]


def run(args: argparse.Namespace) -> Path:
    input_path = _resolve(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    image_sequence = _collect_image_sequence(input_path)
    input_is_sequence = bool(image_sequence)
    if input_path.is_dir() and not image_sequence:
        raise FileNotFoundError(f"No image frames found in directory: {input_path}")

    output_dir = _make_output_dir(_resolve(args.output_dir), input_path)
    tracks_csv = output_dir / "tracks.csv"
    mot_result_txt = output_dir / "result.txt"
    detections_mot_txt = output_dir / "detections_mot.txt"
    video_out_path = output_dir / "tracked_fisheye.mp4"
    matching_debug_jsonl = None
    if args.matching_debug_jsonl:
        matching_debug_jsonl = _resolve(args.matching_debug_jsonl)
        matching_debug_jsonl.parent.mkdir(parents=True, exist_ok=True)
        if matching_debug_jsonl.exists():
            matching_debug_jsonl.unlink()

    det_cfg = _build_detection_cfg(args)
    projection_input_path = image_sequence[0] if input_is_sequence else input_path
    proj_cfg = _build_projection_config(det_cfg, projection_input_path)
    pipeline = DetectionPipeline(det_cfg)

    extractor = FastReIDFeatureExtractor(
        fastreid_root=str(_resolve(args.fastreid_root)),
        config_file=str(_resolve(args.fastreid_config)),
        weights=str(_resolve(args.fastreid_weights)),
        device=args.fastreid_device,
        batch_size=args.fastreid_batch_size,
    )

    strongsort_opt.enable_stm_ltm = bool(args.ltm_stm)
    strongsort_opt.enable_memory_matching = bool(args.memory_aware)
    strongsort_opt.enable_topk_matching = bool(args.topk)
    strongsort_opt.enable_memory_init_control = bool(args.memory_init)
    strongsort_opt.MC = False

    temporal_model = None
    if args.learned_temporal:
        temporal_model = _load_temporal_model(_resolve(args.tracker_model), args.temporal_device)

    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine",
        matching_threshold=args.matching_threshold,
        budget=args.nn_budget,
    )
    tracker = Tracker(
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
        matching_debug_jsonl=str(matching_debug_jsonl) if matching_debug_jsonl else None,
    )

    cap = None
    if input_is_sequence:
        first_frame = cv2.imread(str(image_sequence[0]))
        if first_frame is None:
            raise RuntimeError(f"Could not read first image frame: {image_sequence[0]}")
        fps = float(args.sequence_fps)
        height, width = first_frame.shape[:2]
        total_frames = len(image_sequence)
    else:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open input video: {input_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(
        str(video_out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {video_out_path}")

    trail_history: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    frame_idx = 0
    processed = 0

    print("=" * 80)
    print("GNOMONIC + FASTREID + STRONGSORT")
    print("=" * 80)
    print(f"input : {input_path}")
    print(f"output: {output_dir}")
    input_kind = "image sequence" if input_is_sequence else "video"
    print(f"input kind: {input_kind}")
    print(f"video     : {width}x{height} @ {fps:.2f} fps, frames={total_frames}")

    while True:
        if input_is_sequence:
            if frame_idx >= len(image_sequence):
                break
            frame_path = image_sequence[frame_idx]
            frame = cv2.imread(str(frame_path))
            if frame is None:
                print(f"WARNING: skipping unreadable frame: {frame_path}")
                frame_idx += 1
                continue
            ok = True
        else:
            ok, frame = cap.read()
            if not ok:
                break
        frame_idx += 1
        if args.max_frames is not None and processed >= args.max_frames:
            break
        if frame_idx % args.process_every != 0:
            writer.write(frame)
            continue

        fisheye_bboxes, composite_image, raw_detections = pipeline.run(
            fisheye_image=frame,
            projection_config=proj_cfg,
            return_visuals=True,
        )
        adapter_dets = build_tracker_detections(
            fisheye_bboxes,
            frame if args.reid_crop_source == "fisheye" else composite_image,
            feature_extractor=extractor,
            require_features=True,
            crop_pad=args.reid_crop_pad,
            reid_source=args.reid_crop_source,
        )
        detections = _make_strongsort_detections(adapter_dets)
        _write_detections_mot_txt(detections_mot_txt, frame_idx, fisheye_bboxes)

        tracker.predict()
        tracker.matching_debug_frame = frame_idx
        tracker.update(detections)

        vis = frame.copy()
        _draw_tracks(vis, tracker, trail_history, args.trail_length)
        writer.write(vis)
        _write_tracks_csv(tracks_csv, frame_idx, tracker)
        _write_mot_result_txt(mot_result_txt, frame_idx, tracker)

        processed += 1
        if args.log_every and processed % args.log_every == 0:
            active_ids = [int(t.track_id) for t in tracker.tracks if t.is_confirmed() and t.time_since_update <= 1]
            print(
                f"frame={frame_idx} processed={processed} raw={len(raw_detections)} "
                f"final={len(fisheye_bboxes)} tracks={active_ids}"
            )

    if cap is not None:
        cap.release()
    writer.release()

    h264_video_path = output_dir / "tracked_fisheye_h264.mp4"
    h264_created = False
    if not args.no_h264_copy:
        try:
            h264_created = _transcode_h264(video_out_path, h264_video_path)
        except subprocess.CalledProcessError as exc:
            print(f"WARNING: H.264 transcode failed: {exc}")

    with (output_dir / "metadata.txt").open("w") as f:
        f.write("Gnomonic + FastReID + StrongSORT run\n")
        f.write(f"input={input_path}\n")
        f.write(f"input_kind={input_kind}\n")
        if input_is_sequence:
            f.write(f"image_sequence_frames={len(image_sequence)}\n")
        f.write(f"output_video={video_out_path}\n")
        f.write(f"output_video_h264={h264_video_path if h264_created else ''}\n")
        f.write(f"tracks_csv={tracks_csv}\n")
        f.write(f"mot_result_txt={mot_result_txt}\n")
        f.write(f"detections_mot_txt={detections_mot_txt}\n")
        f.write(f"processed_frames={processed}\n")
        f.write(f"projection_preset={args.preset}\n")
        f.write(f"yolo_model={_resolve(args.yolo_model)}\n")
        f.write(f"fastreid_root={_resolve(args.fastreid_root)}\n")
        f.write(f"fastreid_config={_resolve(args.fastreid_config)}\n")
        f.write(f"fastreid_weights={_resolve(args.fastreid_weights)}\n")
        f.write(f"reid_crop_source={args.reid_crop_source}\n")
        f.write(f"ltm_stm={args.ltm_stm}\n")
        f.write(f"memory_aware={args.memory_aware}\n")
        f.write(f"topk={args.topk}\n")
        f.write(f"learned_temporal={args.learned_temporal}\n")
        f.write(f"fuse_learned_temporal={args.fuse_learned_temporal}\n")
        f.write(f"tracker_model={_resolve(args.tracker_model) if args.learned_temporal else ''}\n")
        f.write(f"learned_temporal_alpha={args.learned_temporal_alpha}\n")
        f.write(f"learned_temporal_min_scale={args.learned_temporal_min_scale}\n")
        f.write(f"learned_temporal_max_correction={args.learned_temporal_max_correction}\n")
        f.write(f"learned_temporal_veto_cost={args.learned_temporal_veto_cost}\n")
        f.write(f"matching_debug_jsonl={matching_debug_jsonl or ''}\n")

    if h264_created:
        print(f"h264 : {h264_video_path}")
    print(f"done: {output_dir}")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run source-aware gnomonic detections through FastReID + StrongSORT.")
    parser.add_argument("--input", default="detection_pipeline/videos/Meeting1.mp4")
    parser.add_argument("--output-dir", default="tracker_pipeline/results")

    parser.add_argument("--preset", default="yolo_grid")
    parser.add_argument("--proj-nbr", type=int, default=6)
    parser.add_argument("--fov-h", type=float, default=60.0)
    parser.add_argument("--fov-v", type=float, default=85.0)
    parser.add_argument("--latitude", type=float, default=45.0)
    parser.add_argument("--lon-0", type=float, default=0.0)
    parser.add_argument("--lon-step", type=float, default=60.0)
    parser.add_argument("--grid-rows", type=int, default=2)
    parser.add_argument("--grid-cols", type=int, default=3)
    parser.add_argument("--comp-width", type=int, default=640)
    parser.add_argument("--comp-height", type=int, default=640)
    parser.add_argument("--target-mp", default="auto")

    parser.add_argument("--yolo-model", default="detection_pipeline/models/yolov9e.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--disable-stage1-nms", action="store_true")
    parser.add_argument("--stage1-iou", type=float, default=0.8)
    parser.add_argument("--disable-stage2-nms", action="store_true")
    parser.add_argument("--stage2-sigma", type=float, default=0.2)
    parser.add_argument("--stage2-score-threshold", type=float, default=0.3)

    parser.add_argument("--fastreid-root", default="tracker_pipeline/third_party/fast-reid")
    parser.add_argument("--fastreid-config", default="tracker_pipeline/reid/configs/bagtricks_S50.yml")
    parser.add_argument("--fastreid-weights", default="tracker_pipeline/reid/weights/duke_bot_S50.pth")
    parser.add_argument("--fastreid-device", default="cpu")
    parser.add_argument("--fastreid-batch-size", type=int, default=32)
    parser.add_argument("--reid-crop-pad", type=int, default=0)
    parser.add_argument(
        "--reid-crop-source",
        choices=("fisheye", "composite"),
        default="fisheye",
        help="Image source for ReID crops. 'fisheye' tracks directly after fisheye NMS; 'composite' uses the selected perspective source crop.",
    )

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
        help="Path to the learned temporal tracker checkpoint. --temporal-model-ckpt is kept as a legacy alias.",
    )
    parser.add_argument("--temporal-device", default="cpu")
    parser.add_argument("--learned-temporal-alpha", type=float, default=1.0)
    parser.add_argument("--learned-temporal-min-scale", type=float, default=0.02)
    parser.add_argument("--learned-temporal-max-correction", type=float, default=0.02)
    parser.add_argument(
        "--learned-temporal-veto-cost",
        type=float,
        default=0.8,
        help="Veto learned temporal pairs above this cost. Use a negative value to disable veto.",
    )
    parser.add_argument(
        "--matching-debug-jsonl",
        default=None,
        help="Optional JSONL path for per-frame matching costs and Kalman gating diagnostics.",
    )

    parser.add_argument("--process-every", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--sequence-fps", type=float, default=10.0)
    parser.add_argument("--trail-length", type=int, default=30)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--no-h264-copy", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
