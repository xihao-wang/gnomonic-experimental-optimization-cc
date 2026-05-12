"""
Tracker pipeline entry point.

Processes a fisheye video frame-by-frame:
  1. Run the detection pipeline (gnomonic projection → YOLO → backproject)
  2. Feed fisheye bounding boxes into ByteTrack
  3. Annotate the fisheye frame with tracked IDs and motion trails
  4. Write annotated video to the output session directory

Run from project root:
    python tracker_pipeline/process_video.py

Author: Yassir Zardoua — y.zardoua@caplogy.com | yassirzardoua@gmail.com
"""

import sys
from pathlib import Path
import cv2
import numpy as np
from collections import defaultdict
from datetime import datetime

# ---- project root on sys.path (mandatory for all runnable scripts) ----------
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# ---- internal imports -------------------------------------------------------
import tracker_pipeline.config as cfg
from tracker_pipeline.lib.byte_tracker import ByteTracker, STrack
from detection_pipeline.config import get_cfg as get_detection_cfg
from detection_pipeline.pipeline import DetectionPipeline
from detection_pipeline.backprojection import draw_rotated_bbox, build_radial_bbox
from image_composer.presets import get_preset


# ============================================================================
# Helpers
# ============================================================================

def _track_color(track_id: int) -> tuple:
    """Return a BGR color from the palette for a given track ID."""
    return cfg.TRACK_COLORS[track_id % len(cfg.TRACK_COLORS)]


def _find_next_session(base_dir: Path, stem: str) -> int:
    n = 1
    while (base_dir / f"{stem}_session_{n}").exists():
        n += 1
    return n


def _build_detection_cfg(model_path: str):
    """Build a YACS config for DetectionPipeline from tracker_pipeline/config.py values."""
    det_cfg = get_detection_cfg()

    # Projection
    if cfg.PROJECTION_PRESET:
        det_cfg.PROJECTION.PRESET = cfg.PROJECTION_PRESET
    else:
        det_cfg.PROJECTION.PRESET      = None
        det_cfg.PROJECTION.PROJ_NBR    = cfg.PROJ_NBR
        det_cfg.PROJECTION.FOV_H       = cfg.FOV_H
        det_cfg.PROJECTION.FOV_V       = cfg.FOV_V
        det_cfg.PROJECTION.LATITUDE    = cfg.LATITUDE
        det_cfg.PROJECTION.LON_0       = cfg.LON_0
        det_cfg.PROJECTION.LON_STEP    = cfg.LON_STEP
        det_cfg.PROJECTION.GRID        = cfg.GRID
        det_cfg.PROJECTION.COMP_SIZE   = tuple(cfg.COMP_SIZE)
        det_cfg.PROJECTION.TARGET_MP   = cfg.TARGET_MP

    # YOLO
    det_cfg.YOLO.MODEL                = model_path
    det_cfg.YOLO.DEVICE               = cfg.YOLO_DEVICE
    det_cfg.YOLO.CONFIDENCE_THRESHOLD = cfg.YOLO_CONFIDENCE

    # NMS
    det_cfg.NMS.STAGE1.ENABLED        = cfg.NMS_STAGE1_ENABLED
    det_cfg.NMS.STAGE1.IOU_THRESHOLD  = cfg.NMS_STAGE1_IOU_THRESHOLD
    det_cfg.NMS.STAGE2.ENABLED        = cfg.NMS_STAGE2_ENABLED
    det_cfg.NMS.STAGE2.SIGMA          = cfg.NMS_STAGE2_SIGMA
    det_cfg.NMS.STAGE2.SCORE_THRESHOLD = cfg.NMS_STAGE2_SCORE_THRESHOLD

    # Backprojection always on
    det_cfg.BACKPROJECTION.ENABLED    = True

    # Disable file I/O (tracker pipeline handles output)
    det_cfg.OUTPUT.SAVE_COMPOSITE     = False
    det_cfg.OUTPUT.SAVE_COMPOSITE_VIZ = False
    det_cfg.OUTPUT.SAVE_FISHEYE_VIZ   = False
    det_cfg.OUTPUT.SAVE_LATTICE_VIZ   = False

    det_cfg.VERBOSE = False

    return det_cfg


def _build_projection_config(det_cfg, project_root: Path) -> dict:
    """Build projection config dict once (reused every frame)."""
    if det_cfg.PROJECTION.PRESET:
        proj_cfg = get_preset(det_cfg.PROJECTION.PRESET)
        if proj_cfg is None:
            raise ValueError(f"Unknown projection preset: {det_cfg.PROJECTION.PRESET}")
    else:
        proj_cfg = {
            'proj_nbr':  det_cfg.PROJECTION.PROJ_NBR,
            'fov_h':     det_cfg.PROJECTION.FOV_H,
            'fov_v':     det_cfg.PROJECTION.FOV_V,
            'latitude':  det_cfg.PROJECTION.LATITUDE,
            'lon_0':     det_cfg.PROJECTION.LON_0,
            'lon_step':  det_cfg.PROJECTION.LON_STEP,
            'grid':      det_cfg.PROJECTION.GRID,
            'comp_sz':   det_cfg.PROJECTION.COMP_SIZE,
            'target_mp': det_cfg.PROJECTION.TARGET_MP,
        }
    # img_path is only required when no image is passed in memory
    proj_cfg['img_path'] = str(project_root / cfg.INPUT_VIDEO)
    return proj_cfg


def _draw_track(frame: np.ndarray, track: STrack,
                trail_history: dict) -> None:
    """Annotate one confirmed track onto the fisheye frame."""
    tid = track.track_id
    color = _track_color(tid)
    bbox = track.to_fisheye_bbox()

    # Rotated bounding box
    draw_rotated_bbox(frame, bbox, color=color,
                      thickness=cfg.BBOX_THICKNESS, draw_label=False)

    # Centroid
    cx, cy = bbox['center']
    if cfg.SHOW_CENTROID:
        cv2.circle(frame, (int(cx), int(cy)), 4, color, -1)

    # Motion trail
    if cfg.TRAIL_LENGTH > 0:
        history = trail_history[tid]
        history.append((int(cx), int(cy)))
        if len(history) > cfg.TRAIL_LENGTH:
            history.pop(0)
        for i in range(1, len(history)):
            alpha = i / len(history)
            c = tuple(int(v * alpha) for v in color)
            cv2.line(frame, history[i - 1], history[i], c, 1)

    # Label: ID [conf]
    if cfg.SHOW_IDS:
        label = f"ID {tid}"
        if cfg.SHOW_CONFIDENCE:
            label += f" {bbox['confidence']:.2f}"
        label_x = int(cx) + 6
        label_y = int(cy) - 6
        cv2.putText(frame, label, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def _write_metadata(path: Path, input_path: Path, session_dir: Path,
                    fps: int, width: int, height: int, total_frames: int,
                    det_cfg, tracker: ByteTracker):
    """Write session metadata to a text file."""
    with open(path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TRACKER PIPELINE — SESSION METADATA\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp   : {datetime.now().isoformat()}\n")
        f.write(f"Session dir : {session_dir}\n\n")

        f.write("INPUT VIDEO\n")
        f.write(f"  Path       : {input_path}\n")
        f.write(f"  Resolution : {width}x{height}\n")
        f.write(f"  FPS        : {fps}\n")
        f.write(f"  Frames     : {total_frames}\n\n")

        f.write("PROJECTION\n")
        if det_cfg.PROJECTION.PRESET:
            f.write(f"  Preset     : {det_cfg.PROJECTION.PRESET}\n")
        else:
            f.write(f"  Projections: {det_cfg.PROJECTION.PROJ_NBR}\n")
            f.write(f"  FOV H/V    : {det_cfg.PROJECTION.FOV_H} / {det_cfg.PROJECTION.FOV_V} deg\n")
            f.write(f"  Latitude   : {det_cfg.PROJECTION.LATITUDE} deg\n")
            f.write(f"  Grid       : {det_cfg.PROJECTION.GRID}\n")
        f.write(f"  Composite  : {det_cfg.PROJECTION.COMP_SIZE}\n\n")

        f.write("DETECTION (YOLO)\n")
        f.write(f"  Model      : {cfg.YOLO_MODEL}\n")
        f.write(f"  Confidence : {cfg.YOLO_CONFIDENCE}\n")
        f.write(f"  NMS s1 IoU : {cfg.NMS_STAGE1_IOU_THRESHOLD} (enabled={cfg.NMS_STAGE1_ENABLED})\n")
        f.write(f"  NMS s2 sig : {cfg.NMS_STAGE2_SIGMA}  thr={cfg.NMS_STAGE2_SCORE_THRESHOLD} "
                f"(enabled={cfg.NMS_STAGE2_ENABLED})\n\n")

        f.write("TRACKER (ByteTrack)\n")
        f.write(f"  High conf  : {tracker.track_high_conf}\n")
        f.write(f"  Low conf   : {tracker.track_low_conf}\n")
        f.write(f"  Max age    : {tracker.max_age}\n")
        f.write(f"  Min hits   : {tracker.min_hits}\n")
        f.write(f"  IoU thr    : {tracker.iou_threshold}\n")
        f.write(f"  Min area   : {tracker.min_box_area} px2\n\n")

        f.write("PROCESSING\n")
        f.write(f"  Every N frames : {cfg.PROCESS_EVERY_N_FRAMES}\n")
        f.write(f"  Max frames     : {cfg.MAX_FRAMES if cfg.MAX_FRAMES else 'all'}\n")
        f.write(f"  Trail length   : {cfg.TRAIL_LENGTH}\n")
        f.write("=" * 80 + "\n")


# ============================================================================
# Main
# ============================================================================

def process_video():
    project_root = Path(__file__).parent.parent
    input_path   = project_root / cfg.INPUT_VIDEO
    base_out_dir = project_root / cfg.OUTPUT_DIR

    if not input_path.exists():
        print(f"ERROR: Input video not found: {input_path}")
        return

    base_out_dir.mkdir(parents=True, exist_ok=True)
    stem        = input_path.stem
    session_n   = _find_next_session(base_out_dir, stem)
    session_dir = base_out_dir / f"{stem}_session_{session_n}"
    session_dir.mkdir(parents=True, exist_ok=True)

    tracked_video_path   = session_dir / "tracked_fisheye.mp4"
    composite_video_path = session_dir / "composite_detections.mp4"
    metadata_path        = session_dir / "metadata.txt"

    print("=" * 80)
    print("TRACKER PIPELINE")
    print("=" * 80)
    print(f"Input  : {input_path}")
    print(f"Session: {session_dir}")
    print()

    # Open video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {input_path}")
        return

    fps          = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video  : {width}x{height} @ {fps} fps, {total_frames} frames")
    print()

    # Build detection pipeline
    model_path = str(project_root / cfg.YOLO_MODEL)
    det_cfg    = _build_detection_cfg(model_path)
    proj_cfg   = _build_projection_config(det_cfg, project_root)

    print("Initializing detection pipeline...")
    pipeline = DetectionPipeline(det_cfg)
    print(f"  Model : {cfg.YOLO_MODEL}")
    print(f"  Device: {pipeline.detector.device}")
    print()

    # Build tracker
    STrack.reset_id_counter()
    tracker = ByteTracker(
        track_high_conf = cfg.TRACK_HIGH_CONF,
        track_low_conf  = cfg.TRACK_LOW_CONF,
        max_age         = cfg.MAX_AGE,
        min_hits        = cfg.MIN_HITS,
        iou_threshold   = cfg.IOU_THRESHOLD,
        min_box_area    = cfg.MIN_BOX_AREA,
    )

    # Write metadata
    _write_metadata(metadata_path, input_path, session_dir,
                    fps, width, height, total_frames, det_cfg, tracker)

    # Video writers
    fourcc          = cv2.VideoWriter_fourcc(*'mp4v')
    tracked_writer  = cv2.VideoWriter(str(tracked_video_path), fourcc, fps, (width, height))

    composite_writer = None
    if cfg.SAVE_COMPOSITE_VIDEO:
        comp_w, comp_h = det_cfg.PROJECTION.COMP_SIZE
        composite_writer = cv2.VideoWriter(
            str(composite_video_path), fourcc, fps, (comp_w, comp_h))

    # Per-track centroid trail history
    trail_history = defaultdict(list)

    frame_idx       = 0
    processed_count = 0

    print("Processing frames...")
    print("-" * 80)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if cfg.MAX_FRAMES is not None and processed_count >= cfg.MAX_FRAMES:
                break

            if frame_idx % cfg.PROCESS_EVERY_N_FRAMES != 0:
                tracked_writer.write(frame)
                frame_idx += 1
                continue

            # ---- Run detection pipeline --------------------------------
            # metrics mode (fisheye_image + projection_config both provided)
            # returns (converted_bboxes, composite, raw_dets) with return_visuals=True
            # Evaluator format: {center_x, center_y, width, height, angle, confidence, class_name}
            eval_bboxes, composite, _ = pipeline.run(
                fisheye_image=frame,
                projection_config=proj_cfg,
                return_visuals=True,
            )

            # Convert evaluator format → backprojection format:
            # {center (tuple), size (tuple), angle, corners, confidence, class_name}
            # Corners are needed for AABB IoU matching in the tracker.
            fh, fw = frame.shape[:2]
            fisheye_center = (fw / 2.0, fh / 2.0)
            fisheye_bboxes = []
            for b in eval_bboxes:
                rbbox = build_radial_bbox(
                    (b['center_x'], b['center_y']),
                    b['width'], b['height'],
                    fisheye_center,
                )
                rbbox['confidence'] = b['confidence']
                rbbox['class_name'] = b['class_name']
                fisheye_bboxes.append(rbbox)

            # ---- Run tracker -------------------------------------------
            confirmed_tracks = tracker.update(fisheye_bboxes)

            # ---- Annotate frame ----------------------------------------
            annotated = frame.copy()
            for track in confirmed_tracks:
                _draw_track(annotated, track, trail_history)

            tracked_writer.write(annotated)

            # ---- Optional: composite video (raw YOLO detections) -------
            if cfg.SAVE_COMPOSITE_VIDEO and composite_writer is not None and composite is not None:
                composite_writer.write(composite)

            # ---- Real-time preview -------------------------------------
            if cfg.DISPLAY_FRAMES:
                cv2.imshow("Tracked Fisheye (Q to quit)", annotated)
                if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                    print("\nDisplay closed by user.")
                    break

            processed_count += 1
            if (cfg.LOG_EVERY_N_FRAMES is not None
                    and processed_count % cfg.LOG_EVERY_N_FRAMES == 0):
                print(f"  Frame {frame_idx:5d} | processed {processed_count} | "
                      f"detections {len(fisheye_bboxes)} | "
                      f"tracks {len(confirmed_tracks)}")

            frame_idx += 1

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        cap.release()
        tracked_writer.release()
        if composite_writer:
            composite_writer.release()
        if cfg.DISPLAY_FRAMES:
            cv2.destroyAllWindows()

    print("-" * 80)
    print(f"Done. Processed {processed_count} frames.")
    print(f"  Tracked video : {tracked_video_path}")
    print(f"  Metadata      : {metadata_path}")
    print("=" * 80)


if __name__ == "__main__":
    process_video()
