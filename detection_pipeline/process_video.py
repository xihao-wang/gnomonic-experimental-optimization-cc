"""
Video processing script for detection pipeline.

Processes a video file frame-by-frame and saves annotated output video
with detections overlaid on the fisheye image.

Configuration is controlled entirely via detection_pipeline/config.py
(VIDEO section for video-specific params, all other sections for pipeline params).

Run from project root:
    python detection_pipeline/process_video.py
"""

import sys
from pathlib import Path
import cv2
import numpy as np

_parent_dir = Path(__file__).parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from detection_pipeline.config import get_cfg
from detection_pipeline.pipeline import DetectionPipeline
from detection_pipeline.backprojection import draw_rotated_bbox
from image_composer.multi_persp import draw_fov_on_fisheye, generate_rainbow_colors


def find_next_session_number(base_dir: Path, video_stem: str) -> int:
    session_num = 1
    while True:
        if not (base_dir / f"{video_stem}_session_{session_num}").exists():
            return session_num
        session_num += 1


def write_metadata_file(metadata_path: Path, cfg, pipeline, input_path: Path,
                        fps: int, width: int, height: int, total_frames: int):
    from datetime import datetime

    with open(metadata_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("VIDEO PROCESSING METADATA\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")

        f.write("INPUT VIDEO:\n")
        f.write(f"  Path: {input_path}\n")
        f.write(f"  Resolution: {width}x{height}\n")
        f.write(f"  FPS: {fps}\n")
        f.write(f"  Total frames: {total_frames}\n\n")

        f.write("PROJECTION CONFIGURATION:\n")
        if cfg.PROJECTION.PRESET:
            f.write(f"  Preset: {cfg.PROJECTION.PRESET}\n")
        else:
            f.write(f"  Preset: None (manual)\n")
            f.write(f"  Projections: {cfg.PROJECTION.PROJ_NBR}\n")
            f.write(f"  FOV: {cfg.PROJECTION.FOV_H}° x {cfg.PROJECTION.FOV_V}°\n")
            f.write(f"  Latitude: {cfg.PROJECTION.LATITUDE}°\n")
            f.write(f"  Longitude start/step: {cfg.PROJECTION.LON_0}° / {cfg.PROJECTION.LON_STEP}°\n")
            f.write(f"  Grid: {cfg.PROJECTION.GRID}\n")
        f.write(f"  Composite size: {cfg.PROJECTION.COMP_SIZE}\n")
        f.write(f"  Target MP: {cfg.PROJECTION.TARGET_MP}\n\n")

        f.write("YOLO DETECTION:\n")
        f.write(f"  Model: {cfg.YOLO.MODEL}\n")
        f.write(f"  Device: {pipeline.detector.device}\n")
        f.write(f"  Confidence threshold: {cfg.YOLO.CONFIDENCE_THRESHOLD}\n")
        f.write(f"  Max detections: {cfg.YOLO.MAX_DETECTIONS}\n\n")

        f.write("NMS:\n")
        f.write(f"  Stage 1 enabled: {cfg.NMS.STAGE1.ENABLED}")
        if cfg.NMS.STAGE1.ENABLED:
            f.write(f"  (IoU={cfg.NMS.STAGE1.IOU_THRESHOLD})")
        f.write(f"\n  Stage 2 enabled: {cfg.NMS.STAGE2.ENABLED}")
        if cfg.NMS.STAGE2.ENABLED:
            f.write(f"  (sigma={cfg.NMS.STAGE2.SIGMA}, score_thresh={cfg.NMS.STAGE2.SCORE_THRESHOLD})")
        f.write("\n\n")

        f.write("BACKPROJECTION:\n")
        f.write(f"  Enabled: {cfg.BACKPROJECTION.ENABLED}\n")
        f.write(f"  Lattice height samples: {cfg.BACKPROJECTION.LATTICE_HEIGHT_SAMPLES}\n\n")

        f.write("PROCESSING OPTIONS:\n")
        f.write(f"  Process every N frames: {cfg.VIDEO.PROCESS_EVERY_N_FRAMES}\n")
        f.write(f"  Max frames: {cfg.VIDEO.MAX_FRAMES if cfg.VIDEO.MAX_FRAMES > 0 else 'All'}\n")
        f.write(f"  Save composite video: {cfg.VIDEO.SAVE_COMPOSITE}\n\n")

        f.write("VISUALIZATION:\n")
        f.write(f"  BBox color (BGR): {cfg.VIDEO.BBOX_COLOR}\n")
        f.write(f"  BBox thickness: {cfg.VIDEO.BBOX_THICKNESS}\n")
        f.write(f"  Show labels: {cfg.VIDEO.SHOW_LABELS}\n")
        f.write(f"  Show confidence: {cfg.VIDEO.SHOW_CONFIDENCE}\n\n")

        f.write("=" * 80 + "\n")


def process_video():
    cfg = get_cfg()

    project_root = Path(__file__).parent.parent
    input_path = project_root / cfg.VIDEO.INPUT_PATH
    base_output_dir = project_root / cfg.VIDEO.OUTPUT_DIR
    base_output_dir.mkdir(parents=True, exist_ok=True)

    input_stem = Path(cfg.VIDEO.INPUT_PATH).stem
    session_num = find_next_session_number(base_output_dir, input_stem)
    session_dir = base_output_dir / f"{input_stem}_session_{session_num}"
    session_dir.mkdir(parents=True, exist_ok=True)

    fisheye_video_path = session_dir / "fisheye_detections.mp4"
    composite_video_path = session_dir / "composite_detections.mp4"
    metadata_path = session_dir / "metadata.txt"

    if not input_path.exists():
        print(f"ERROR: Input video not found: {input_path}")
        return

    print("=" * 80)
    print("VIDEO PROCESSING")
    print("=" * 80)
    print(f"Input:   {input_path}")
    print(f"Session: {session_dir}")
    print()

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {input_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print()

    # Disable per-frame file saving (video writer handles output)
    cfg.OUTPUT.SAVE_COMPOSITE = False
    cfg.OUTPUT.SAVE_COMPOSITE_VIZ = False
    cfg.OUTPUT.SAVE_FISHEYE_VIZ = False
    cfg.OUTPUT.SAVE_LATTICE_VIZ = False
    cfg.VERBOSE = False

    print("Initializing detection pipeline...")
    pipeline = DetectionPipeline(cfg)
    print(f"  Model: {cfg.YOLO.MODEL}")
    print(f"  Device: {pipeline.detector.device}")
    print(f"  NMS Stage 1: {'Enabled' if cfg.NMS.STAGE1.ENABLED else 'Disabled'}")
    print(f"  NMS Stage 2: {'Enabled' if cfg.NMS.STAGE2.ENABLED else 'Disabled'}")
    print()

    write_metadata_file(metadata_path, cfg, pipeline, input_path, fps, width, height, total_frames)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fisheye_writer = cv2.VideoWriter(str(fisheye_video_path), fourcc, fps, (width, height))

    composite_writer = None
    if cfg.VIDEO.SAVE_COMPOSITE:
        comp_width, comp_height = cfg.PROJECTION.COMP_SIZE
        composite_writer = cv2.VideoWriter(str(composite_video_path), fourcc, fps, (comp_width, comp_height))

    frame_idx = 0
    processed_frames = 0

    print("Processing frames...")
    print("-" * 80)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if cfg.VIDEO.MAX_FRAMES > 0 and processed_frames >= cfg.VIDEO.MAX_FRAMES:
                break

            if frame_idx % cfg.VIDEO.PROCESS_EVERY_N_FRAMES != 0:
                fisheye_writer.write(frame)
                frame_idx += 1
                continue

            temp_frame_path = session_dir / "temp_frame.jpg"
            cv2.imwrite(str(temp_frame_path), frame)
            pipeline.cfg.INPUT.IMAGE_PATH = str(temp_frame_path)

            detections, composite, metadata, _, fisheye_bboxes = pipeline.run()

            annotated_fisheye = frame.copy()
            if fisheye_bboxes:
                for bbox in fisheye_bboxes:
                    draw_rotated_bbox(
                        annotated_fisheye,
                        bbox,
                        color=cfg.VIDEO.BBOX_COLOR,
                        thickness=cfg.VIDEO.BBOX_THICKNESS,
                        draw_label=cfg.VIDEO.SHOW_LABELS
                    )

            # Optional overlay: per-projection FOV outline in unique colours.
            if cfg.OUTPUT.SAVE_FISHEYE_PROJ_BORDERS and metadata and metadata.get("proj_list"):
                fh, fw = annotated_fisheye.shape[:2]
                fcx, fcy = fw // 2, fh // 2
                fr = min(fcx, fcy)
                proj_list = metadata["proj_list"]
                proj_colors = generate_rainbow_colors(len(proj_list))
                for pparams, pcolor in zip(proj_list, proj_colors):
                    annotated_fisheye = draw_fov_on_fisheye(
                        annotated_fisheye, fcx, fcy, fr,
                        pparams["longitude"], pparams["latitude"],
                        pparams["fov_h"], pparams["fov_v"],
                        color=pcolor,
                        thickness=cfg.OUTPUT.PROJ_BORDERS_THICKNESS,
                    )

            fisheye_writer.write(annotated_fisheye)

            if cfg.VIDEO.DISPLAY_FRAMES:
                cv2.imshow('Fisheye Detections (Press Q to quit)', annotated_fisheye)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("\n\nDisplay window closed by user")
                    break

            if cfg.VIDEO.SAVE_COMPOSITE and composite is not None:
                annotated_composite = composite.copy()
                comp_h, comp_w = composite.shape[:2]

                for det in detections:
                    cx = int(det['x'] * comp_w)
                    cy = int(det['y'] * comp_h)
                    bw = int(det['w'] * comp_w)
                    bh = int(det['h'] * comp_h)
                    x1, y1 = max(0, cx - bw // 2), max(0, cy - bh // 2)
                    x2, y2 = min(comp_w, cx + bw // 2), min(comp_h, cy + bh // 2)

                    cv2.rectangle(annotated_composite, (x1, y1), (x2, y2),
                                  cfg.VIDEO.BBOX_COLOR, cfg.VIDEO.BBOX_THICKNESS)
                    if cfg.VIDEO.SHOW_LABELS:
                        label = f"{det['class_name']} {det['confidence']:.2f}"
                        cv2.putText(annotated_composite, label, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, cfg.VIDEO.BBOX_COLOR, 1)

                composite_writer.write(annotated_composite)

            processed_frames += 1
            if cfg.VIDEO.VERBOSE and processed_frames % 10 == 0:
                n_det = len(fisheye_bboxes) if fisheye_bboxes else 0
                print(f"  Processed {processed_frames}/{total_frames // cfg.VIDEO.PROCESS_EVERY_N_FRAMES} "
                      f"frames ({n_det} detections)")

            frame_idx += 1

            if temp_frame_path.exists():
                temp_frame_path.unlink()

    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user")

    finally:
        cap.release()
        fisheye_writer.release()
        if composite_writer:
            composite_writer.release()
        if cfg.VIDEO.DISPLAY_FRAMES:
            cv2.destroyAllWindows()
        temp_frame_path = session_dir / "temp_frame.jpg"
        if temp_frame_path.exists():
            temp_frame_path.unlink()

    print("-" * 80)
    print(f"\nProcessing complete!")
    print(f"  Processed: {processed_frames} frames")
    print(f"  Session directory: {session_dir}")
    print(f"  Fisheye video: {fisheye_video_path.name}")
    if cfg.VIDEO.SAVE_COMPOSITE:
        print(f"  Composite video: {composite_video_path.name}")
    print(f"  Metadata: {metadata_path.name}")
    print("=" * 80)


if __name__ == "__main__":
    process_video()
