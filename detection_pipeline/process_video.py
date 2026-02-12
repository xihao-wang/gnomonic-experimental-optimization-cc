"""
Video processing script for detection pipeline.

Processes a video file frame-by-frame and saves annotated output video
with detections overlaid on the fisheye image.

Run from project root:
    python detection_pipeline/process_video.py
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Add parent directory to path
_parent_dir = Path(__file__).parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from detection_pipeline.config import get_cfg
from detection_pipeline.pipeline import DetectionPipeline
from detection_pipeline.backprojection import draw_rotated_bbox


# ============================================================================
# CONFIGURATION - Edit these parameters
# ============================================================================

# Input video path (relative to project root)
INPUT_VIDEO = "detection_pipeline/videos/sandra_caplogy_gs.mp4"

# Output directory (relative to project root)
OUTPUT_DIR = "detection_pipeline/results/video_demos"

# Session-based output structure:
# detection_pipeline/results/video_demos/{video_name}_session_N/
#   - fisheye_detections.mp4 (main output with detections on fisheye)
#   - composite_detections.mp4 (optional: detections on composite after stage 1 NMS)
#   - metadata.txt (projection config, NMS params, etc.)

# Save composite video (with stage 1 NMS only, before backprojection)
SAVE_COMPOSITE_VIDEO = True

# Projection configuration
# Option 1: Use preset (set to preset name, manual params below will be ignored)
# Option 2: Use manual params (set to None, then configure params below)
PROJECTION_PRESET = "wide_angle"  # Set to "yolo_grid", "default", "high_coverage", etc. or None

# Manual projection parameters (only used if PROJECTION_PRESET = None)
PROJ_NBR = 4
FOV_H = 70.0  # degrees
FOV_V = 70.0  # degrees
LATITUDE = 45.0
LON_0 = 0.0
LON_STEP = 40.0
GRID = [2, 2]  # [rows, cols]
COMP_SIZE = [640, 640]  # [width, height]
TARGET_MP = "auto"

# YOLO configuration
YOLO_MODEL = "models/yolov8m.pt"
YOLO_CONFIDENCE = 0.25  # Initial filter - removes very weak detections
YOLO_DEVICE = "cuda"  # None for auto-detect, "cuda" or "cpu"

# NMS configuration
NMS_STAGE1_ENABLED = True
NMS_STAGE1_IOU_THRESHOLD = 0.8  # High threshold - keeps overlapping boxes for Stage 2

NMS_STAGE2_ENABLED = True
NMS_STAGE2_SIGMA = 0.2  # Gaussian decay parameter (0.1=aggressive, 0.4=gentle)
NMS_STAGE2_SCORE_THRESHOLD = 0.3  # Filter decayed scores - removes heavily penalized duplicates
# Note: Confidence shown on fisheye is AFTER Soft-NMS decay
# A bbox with original conf=0.9 that overlaps heavily might decay to 0.2-0.4

# Visualization
BBOX_COLOR = (0, 255, 0)  # Green in BGR
BBOX_THICKNESS = 2
SHOW_LABELS = True
SHOW_CONFIDENCE = True

# Processing options
PROCESS_EVERY_N_FRAMES = 1  # Process every Nth frame (1 = all frames)
MAX_FRAMES = None  # Maximum frames to process (None = all frames)
VERBOSE = True
DISPLAY_FRAMES = True  # Show real-time preview window (press 'q' to quit)

# ============================================================================
# END CONFIGURATION
# ============================================================================


def find_next_session_number(base_dir: Path, video_stem: str) -> int:
    """Find the next available session number."""
    session_num = 1
    while True:
        session_dir = base_dir / f"{video_stem}_session_{session_num}"
        if not session_dir.exists():
            return session_num
        session_num += 1


def write_metadata_file(metadata_path: Path, cfg, pipeline, input_path: Path,
                        fps: int, width: int, height: int, total_frames: int):
    """Write metadata file with all configuration parameters."""
    from datetime import datetime

    with open(metadata_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("VIDEO PROCESSING METADATA\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")

        # Input video info
        f.write("INPUT VIDEO:\n")
        f.write(f"  Path: {input_path}\n")
        f.write(f"  Resolution: {width}x{height}\n")
        f.write(f"  FPS: {fps}\n")
        f.write(f"  Total frames: {total_frames}\n\n")

        # Projection configuration
        f.write("PROJECTION CONFIGURATION:\n")
        if cfg.PROJECTION.PRESET:
            f.write(f"  Preset: {cfg.PROJECTION.PRESET}\n")
        else:
            f.write(f"  Preset: None (manual configuration)\n")
            f.write(f"  Number of projections: {cfg.PROJECTION.PROJ_NBR}\n")
            f.write(f"  FOV horizontal: {cfg.PROJECTION.FOV_H} degrees\n")
            f.write(f"  FOV vertical: {cfg.PROJECTION.FOV_V} degrees\n")
            f.write(f"  Latitude: {cfg.PROJECTION.LATITUDE} degrees\n")
            f.write(f"  Longitude start: {cfg.PROJECTION.LON_0} degrees\n")
            f.write(f"  Longitude step: {cfg.PROJECTION.LON_STEP} degrees\n")
            f.write(f"  Grid layout: {cfg.PROJECTION.GRID}\n")
        f.write(f"  Composite size: {cfg.PROJECTION.COMP_SIZE}\n")
        f.write(f"  Target megapixels: {cfg.PROJECTION.TARGET_MP}\n\n")

        # YOLO configuration
        f.write("YOLO DETECTION:\n")
        f.write(f"  Model: {cfg.YOLO.MODEL}\n")
        f.write(f"  Device: {pipeline.detector.device}\n")
        f.write(f"  Confidence threshold: {cfg.YOLO.CONFIDENCE_THRESHOLD}\n")
        f.write(f"  Max detections: {cfg.YOLO.MAX_DETECTIONS}\n\n")

        # NMS configuration
        f.write("NON-MAXIMUM SUPPRESSION (NMS):\n")
        f.write(f"  Stage 1 (Composite Image - Standard NMS):\n")
        f.write(f"    Enabled: {cfg.NMS.STAGE1.ENABLED}\n")
        if cfg.NMS.STAGE1.ENABLED:
            f.write(f"    IoU threshold: {cfg.NMS.STAGE1.IOU_THRESHOLD}\n")
        f.write(f"  Stage 2 (Fisheye Image - Soft-NMS):\n")
        f.write(f"    Enabled: {cfg.NMS.STAGE2.ENABLED}\n")
        if cfg.NMS.STAGE2.ENABLED:
            f.write(f"    Sigma: {cfg.NMS.STAGE2.SIGMA}\n")
            f.write(f"    Score threshold: {cfg.NMS.STAGE2.SCORE_THRESHOLD}\n")
        f.write("\n")

        # Backprojection configuration
        f.write("BACKPROJECTION:\n")
        f.write(f"  Enabled: {cfg.BACKPROJECTION.ENABLED}\n")
        f.write(f"  Lattice height samples: {cfg.BACKPROJECTION.LATTICE_HEIGHT_SAMPLES}\n\n")

        # Processing options
        f.write("PROCESSING OPTIONS:\n")
        f.write(f"  Process every N frames: {PROCESS_EVERY_N_FRAMES}\n")
        f.write(f"  Max frames: {MAX_FRAMES if MAX_FRAMES else 'All'}\n")
        f.write(f"  Save composite video: {SAVE_COMPOSITE_VIDEO}\n\n")

        # Visualization
        f.write("VISUALIZATION:\n")
        f.write(f"  BBox color (BGR): {BBOX_COLOR}\n")
        f.write(f"  BBox thickness: {BBOX_THICKNESS}\n")
        f.write(f"  Show labels: {SHOW_LABELS}\n")
        f.write(f"  Show confidence: {SHOW_CONFIDENCE}\n\n")

        f.write("=" * 80 + "\n")


def process_video():
    """Process video file and save annotated output."""

    # Build paths
    project_root = Path(__file__).parent.parent
    input_path = project_root / INPUT_VIDEO
    base_output_dir = project_root / OUTPUT_DIR

    # Create base output directory
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Find next available session number
    input_stem = Path(INPUT_VIDEO).stem
    session_num = find_next_session_number(base_output_dir, input_stem)
    session_dir = base_output_dir / f"{input_stem}_session_{session_num}"
    session_dir.mkdir(parents=True, exist_ok=True)

    # Output paths
    fisheye_video_path = session_dir / "fisheye_detections.mp4"
    composite_video_path = session_dir / "composite_detections.mp4"
    metadata_path = session_dir / "metadata.txt"

    # Check input video exists
    if not input_path.exists():
        print(f"ERROR: Input video not found: {input_path}")
        return

    print("=" * 80)
    print("VIDEO PROCESSING")
    print("=" * 80)
    print(f"Input:   {input_path}")
    print(f"Session: {session_dir}")
    print()

    # Open input video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {input_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print()

    # Configure detection pipeline
    cfg = get_cfg()

    # Projection settings
    if PROJECTION_PRESET:
        cfg.PROJECTION.PRESET = PROJECTION_PRESET
    else:
        cfg.PROJECTION.PRESET = None
        cfg.PROJECTION.PROJ_NBR = PROJ_NBR
        cfg.PROJECTION.FOV_H = FOV_H
        cfg.PROJECTION.FOV_V = FOV_V
        cfg.PROJECTION.LATITUDE = LATITUDE
        cfg.PROJECTION.LON_0 = LON_0
        cfg.PROJECTION.LON_STEP = LON_STEP
        cfg.PROJECTION.GRID = GRID
        cfg.PROJECTION.COMP_SIZE = tuple(COMP_SIZE)
        cfg.PROJECTION.TARGET_MP = TARGET_MP

    # YOLO settings (resolve model path relative to project root)
    model_path = project_root / YOLO_MODEL
    cfg.YOLO.MODEL = str(model_path)
    cfg.YOLO.DEVICE = YOLO_DEVICE
    cfg.YOLO.CONFIDENCE_THRESHOLD = YOLO_CONFIDENCE

    # NMS settings
    cfg.NMS.STAGE1.ENABLED = NMS_STAGE1_ENABLED
    cfg.NMS.STAGE1.IOU_THRESHOLD = NMS_STAGE1_IOU_THRESHOLD
    cfg.NMS.STAGE2.ENABLED = NMS_STAGE2_ENABLED
    cfg.NMS.STAGE2.SIGMA = NMS_STAGE2_SIGMA
    cfg.NMS.STAGE2.SCORE_THRESHOLD = NMS_STAGE2_SCORE_THRESHOLD

    # Backprojection settings
    cfg.BACKPROJECTION.ENABLED = True

    # Output settings (disable file saving for video processing)
    cfg.OUTPUT.SAVE_COMPOSITE = False
    cfg.OUTPUT.SAVE_COMPOSITE_VIZ = False
    cfg.OUTPUT.SAVE_FISHEYE_VIZ = False
    cfg.OUTPUT.SAVE_LATTICE_VIZ = False

    cfg.VERBOSE = False  # Disable verbose for per-frame processing

    # Initialize pipeline once
    print("Initializing detection pipeline...")
    pipeline = DetectionPipeline(cfg)
    print(f"  Model: {YOLO_MODEL}")
    print(f"  Device: {pipeline.detector.device}")
    print(f"  NMS Stage 1: {'Enabled' if NMS_STAGE1_ENABLED else 'Disabled'}")
    print(f"  NMS Stage 2: {'Enabled' if NMS_STAGE2_ENABLED else 'Disabled'}")
    print()

    # Save metadata file
    write_metadata_file(metadata_path, cfg, pipeline, input_path, fps, width, height, total_frames)

    # Create video writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fisheye_writer = cv2.VideoWriter(str(fisheye_video_path), fourcc, fps, (width, height))

    composite_writer = None
    if SAVE_COMPOSITE_VIDEO:
        # Composite size from config
        comp_width, comp_height = cfg.PROJECTION.COMP_SIZE
        composite_writer = cv2.VideoWriter(str(composite_video_path), fourcc, fps, (comp_width, comp_height))

    # Process frames
    frame_idx = 0
    processed_frames = 0

    print("Processing frames...")
    print("-" * 80)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Check max frames limit - stop completely if reached
            if MAX_FRAMES is not None and processed_frames >= MAX_FRAMES:
                break

            # Check if we should process this frame
            if frame_idx % PROCESS_EVERY_N_FRAMES != 0:
                fisheye_writer.write(frame)  # Write original frame without processing
                frame_idx += 1
                continue

            # Save frame temporarily
            temp_frame_path = session_dir / "temp_frame.jpg"
            cv2.imwrite(str(temp_frame_path), frame)

            # Update pipeline config with current frame
            pipeline.cfg.INPUT.IMAGE_PATH = str(temp_frame_path)

            # Run detection
            detections, composite, metadata, _, fisheye_bboxes = pipeline.run()

            # Draw detections on fisheye frame
            annotated_fisheye = frame.copy()
            if fisheye_bboxes:
                for bbox in fisheye_bboxes:
                    draw_rotated_bbox(
                        annotated_fisheye,
                        bbox,
                        color=BBOX_COLOR,
                        thickness=BBOX_THICKNESS,
                        draw_label=SHOW_LABELS
                    )

            # Write fisheye video
            fisheye_writer.write(annotated_fisheye)

            # Display frame in real-time
            if DISPLAY_FRAMES:
                cv2.imshow('Fisheye Detections (Press Q to quit)', annotated_fisheye)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("\n\nDisplay window closed by user")
                    break

            # Draw detections on composite (Stage 1 NMS only, before backprojection)
            if SAVE_COMPOSITE_VIDEO and composite is not None:
                annotated_composite = composite.copy()
                comp_h, comp_w = composite.shape[:2]

                for det in detections:
                    # Convert normalized coords to pixels
                    cx = int(det['x'] * comp_w)
                    cy = int(det['y'] * comp_h)
                    bw = int(det['w'] * comp_w)
                    bh = int(det['h'] * comp_h)

                    x1 = max(0, cx - bw // 2)
                    y1 = max(0, cy - bh // 2)
                    x2 = min(comp_w, cx + bw // 2)
                    y2 = min(comp_h, cy + bh // 2)

                    cv2.rectangle(annotated_composite, (x1, y1), (x2, y2), BBOX_COLOR, BBOX_THICKNESS)

                    if SHOW_LABELS:
                        label = f"{det['class_name']} {det['confidence']:.2f}"
                        cv2.putText(annotated_composite, label, (x1, y1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, BBOX_COLOR, 1)

                composite_writer.write(annotated_composite)

            processed_frames += 1
            if VERBOSE and processed_frames % 10 == 0:
                print(f"  Processed {processed_frames}/{total_frames // PROCESS_EVERY_N_FRAMES} frames "
                      f"({len(fisheye_bboxes) if fisheye_bboxes else 0} detections)")

            frame_idx += 1

            # Cleanup temp frame
            if temp_frame_path.exists():
                temp_frame_path.unlink()

    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user")

    finally:
        # Cleanup
        cap.release()
        fisheye_writer.release()
        if composite_writer:
            composite_writer.release()

        # Close display window
        if DISPLAY_FRAMES:
            cv2.destroyAllWindows()

        # Remove temp frame if exists
        temp_frame_path = session_dir / "temp_frame.jpg"
        if temp_frame_path.exists():
            temp_frame_path.unlink()

    print("-" * 80)
    print(f"\nProcessing complete!")
    print(f"  Processed: {processed_frames} frames")
    print(f"  Session directory: {session_dir}")
    print(f"  Fisheye video: {fisheye_video_path.name}")
    if SAVE_COMPOSITE_VIDEO:
        print(f"  Composite video: {composite_video_path.name}")
    print(f"  Metadata: {metadata_path.name}")
    print("=" * 80)


if __name__ == "__main__":
    process_video()
