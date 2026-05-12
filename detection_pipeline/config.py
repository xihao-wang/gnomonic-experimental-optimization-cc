"""
YACS configuration for detection pipeline.

This module defines all configurable parameters for the detection pipeline using YACS.
Configuration hierarchy:
  - INPUT: Fisheye image and preprocessing
  - PROJECTION: Composite image generation settings
  - YOLO: Detection model and inference parameters
  - NMS: Two-stage non-maximum suppression
  - BACKPROJECTION: Fisheye coordinate transformation
  - OUTPUT: Visualization and result storage
  - VIDEO: Video processing parameters

Usage:
    from detection_pipeline.config import get_cfg
    cfg = get_cfg()  # Get default config
    cfg.freeze()     # Prevent accidental modifications
"""

from yacs.config import CfgNode as CN

# ============================================================================
# Create config object
# ============================================================================

_C = CN()

# ============================================================================
# INPUT: Fisheye Image Configuration
# ============================================================================

_C.INPUT = CN()

# Path to fisheye image to process
_C.INPUT.IMAGE_PATH = "fisheye-sample.png"

# ============================================================================
# PROJECTION: Composite Image Configuration
# ============================================================================

_C.PROJECTION = CN()

# Preset to use: "yolo_grid", "default", "high_coverage", "horizon", "wide_angle", "high_res"
# Set to None to use manual parameters below
_C.PROJECTION.PRESET = None

# Override preset's image path (None = use preset's default)
_C.PROJECTION.CUSTOM_IMAGE = None

# ---- Manual configuration (used when PRESET is None) ----

# Number of projections to generate
_C.PROJECTION.PROJ_NBR = 6

# Field of view in degrees
_C.PROJECTION.FOV_H = 60.0  # Horizontal
_C.PROJECTION.FOV_V = 85.0  # Vertical

# Camera positioning
_C.PROJECTION.LATITUDE = 45.0  # 0=nadir (straight down), 90=horizon
_C.PROJECTION.LON_0 = 0.0
_C.PROJECTION.LON_STEP = 60.0  # 360 / PROJ_NBR for uniform spacing

# Grid layout: (rows, cols) or None for automatic
_C.PROJECTION.GRID = (2, 3)

# Composite image size (width, height) - should match YOLO input size
_C.PROJECTION.COMP_SIZE = (640, 640)

# Target megapixels per projection: 'auto' (recommended) or float value
# 'auto' means: comp_size / grid_dims (efficient, no resizing needed)
_C.PROJECTION.TARGET_MP = 'auto'

# ============================================================================
# YOLO: Detection Model Configuration
# ============================================================================

_C.YOLO = CN()

# YOLO model path (relative to project root)
# Available models in models/: yolov8{n,s,m,l,x}.pt, yolov9{c,e}.pt,
#                               yolo11{n,s}.pt, yolo12{n,s,x}.pt, etc.
_C.YOLO.MODEL = "models/yolov9e.pt"

# Device: "cuda", "cpu", or None for auto-detection
_C.YOLO.DEVICE = None

# Confidence threshold (pre-NMS filter — removes very weak detections)
_C.YOLO.CONFIDENCE_THRESHOLD = 0.25

# Maximum number of detections to keep per image
_C.YOLO.MAX_DETECTIONS = 300

# ============================================================================
# NMS: Non-Maximum Suppression Configuration
# ============================================================================

_C.NMS = CN()

# ---- Stage 1: Standard NMS on Composite Image ----
_C.NMS.STAGE1 = CN()

# Applied on composite detections before backprojection
_C.NMS.STAGE1.ENABLED = True

# High threshold keeps more boxes since Stage 2 NMS follows
_C.NMS.STAGE1.IOU_THRESHOLD = 0.8

# ---- Stage 2: Soft-NMS on Fisheye Image ----
_C.NMS.STAGE2 = CN()

# Applied on fisheye detections after backprojection
_C.NMS.STAGE2.ENABLED = True

# Gaussian decay: score ← score * exp((-IoU²) / sigma)
# 0.1=aggressive suppression, 0.2=moderate, 0.4=gentle
_C.NMS.STAGE2.SIGMA = 0.2

# Discard detections below this score after Gaussian decay
_C.NMS.STAGE2.SCORE_THRESHOLD = 0.3

# ============================================================================
# BACKPROJECTION: Fisheye Coordinate Transformation
# ============================================================================

_C.BACKPROJECTION = CN()

# Map detections from composite to fisheye coordinates
_C.BACKPROJECTION.ENABLED = True

# Lattice grid points along bbox height for distortion visualization
_C.BACKPROJECTION.LATTICE_HEIGHT_SAMPLES = 10

# ============================================================================
# OUTPUT: Visualization and Results (single-image pipeline)
# ============================================================================

_C.OUTPUT = CN()

_C.OUTPUT.SAVE_COMPOSITE = True
_C.OUTPUT.SAVE_COMPOSITE_VIZ = True
_C.OUTPUT.SAVE_FISHEYE_VIZ = True
_C.OUTPUT.SAVE_LATTICE_VIZ = True
_C.OUTPUT.SAVE_DIR = "results"

# ============================================================================
# VIDEO: Video Processing Parameters
# ============================================================================

_C.VIDEO = CN()

# Input video path (relative to project root)
_C.VIDEO.INPUT_PATH = "detection_pipeline/videos/sandra_caplogy_gs.mp4"

# Output directory (relative to project root)
_C.VIDEO.OUTPUT_DIR = "detection_pipeline/results/video_demos"

# Save composite video alongside the fisheye output
_C.VIDEO.SAVE_COMPOSITE = True

# Process every Nth frame (1 = all frames, 2 = every other frame, etc.)
_C.VIDEO.PROCESS_EVERY_N_FRAMES = 1

# Maximum frames to process (0 = all frames)
_C.VIDEO.MAX_FRAMES = 0

# Show real-time preview window (press Q to quit)
_C.VIDEO.DISPLAY_FRAMES = True

# Bounding box rendering
_C.VIDEO.BBOX_COLOR = (0, 255, 0)  # BGR
_C.VIDEO.BBOX_THICKNESS = 2
_C.VIDEO.SHOW_LABELS = True
_C.VIDEO.SHOW_CONFIDENCE = True

# Print progress every 10 processed frames
_C.VIDEO.VERBOSE = True

# ============================================================================
# DEBUG / VERBOSE (pipeline-level)
# ============================================================================

_C.VERBOSE = False


# ============================================================================
# Public API
# ============================================================================

def get_cfg():
    return _C.clone()


def get_cfg_as_dict(cfg):
    return CN.to_py(cfg)


if __name__ == "__main__":
    cfg = get_cfg()
    print(cfg)
