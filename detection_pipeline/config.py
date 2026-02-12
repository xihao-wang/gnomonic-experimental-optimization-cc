"""
YACS configuration for detection pipeline.

This module defines all configurable parameters for the detection pipeline using YACS.
Configuration hierarchy:
  - INPUT: Fisheye image and preprocessing
  - PROJECTION: Composite image generation settings
  - YOLO: Detection model and inference parameters
  - OUTPUT: Visualization and result storage

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

# Preset to use for projection: "yolo_grid", "default", "high_coverage", "horizon", "wide_angle", "high_res"
_C.PROJECTION.PRESET = "yolo_grid"

# Override preset's image path (None = use preset's default)
_C.PROJECTION.CUSTOM_IMAGE = None

# ---- Manual configuration (used when PRESET is None) ----

# Number of projections to generate
_C.PROJECTION.PROJ_NBR = 9

# Field of view in degrees
_C.PROJECTION.FOV_H = 60.0  # Horizontal
_C.PROJECTION.FOV_V = 60.0  # Vertical

# Camera positioning
_C.PROJECTION.LATITUDE = 45.0  # 0=nadir (straight down), 90=horizon
_C.PROJECTION.LON_0 = 0.0     # Starting longitude
_C.PROJECTION.LON_STEP = 40.0 # Longitude step between projections

# Grid layout: (rows, cols) or None for automatic
_C.PROJECTION.GRID = None

# Composite image size (width, height) - should match YOLO input size
_C.PROJECTION.COMP_SIZE = (640, 640)

# Target megapixels per projection: 'auto' (recommended) or float value
# 'auto' means: comp_size / grid_dims (efficient, no resizing needed)
_C.PROJECTION.TARGET_MP = 'auto'

# ============================================================================
# YOLO: Detection Model Configuration
# ============================================================================

_C.YOLO = CN()

# YOLO model to use: path to .pt file in models/ folder at project root
# Available: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
#            yolo11n.pt, yolo11s.pt, yolo12n.pt, yolo12s.pt, yolo12x.pt, etc.
# Use relative path from project root
_C.YOLO.MODEL = "models/yolo12x.pt"

# Device to run YOLO on: "cuda" or "cpu"
# Set to None for auto-detection (GPU if available, else CPU)
_C.YOLO.DEVICE = None

# ---- Detection Parameters ----

# Confidence threshold for person class detections (0-1)
# Initial filter to remove very weak detections before NMS stages
_C.YOLO.CONFIDENCE_THRESHOLD = 0.25

# Maximum number of detections to keep per image
_C.YOLO.MAX_DETECTIONS = 300

# ============================================================================
# NMS: Non-Maximum Suppression Configuration
# ============================================================================

_C.NMS = CN()

# ---- Stage 1: Standard NMS on Composite Image ----
_C.NMS.STAGE1 = CN()

# Enable/disable Stage 1 NMS (applied on composite detections before backprojection)
# Note: YOLO always applies internal NMS, this controls the threshold
_C.NMS.STAGE1.ENABLED = True

# IoU threshold for standard NMS on composite image
# High threshold (0.8) keeps more bboxes since Stage 2 NMS follows
_C.NMS.STAGE1.IOU_THRESHOLD = 0.8

# ---- Stage 2: Soft-NMS on Fisheye Image ----
_C.NMS.STAGE2 = CN()

# Enable/disable Stage 2 Soft-NMS (applied on fisheye detections after backprojection)
_C.NMS.STAGE2.ENABLED = True

# Sigma parameter for Gaussian Soft-NMS: score ← score * exp((-IoU²)/sigma)
# Literature suggests: 0.1 (aggressive), 0.2 (moderate), 0.4 (gentle)
# Lower values suppress overlapping boxes more aggressively
_C.NMS.STAGE2.SIGMA = 0.2

# Score threshold for Soft-NMS (applied to decayed scores)
# Detections with score below this threshold after Gaussian decay are discarded
# Note: Scores are DECAYED by Soft-NMS, so threshold should be lower than initial confidence
# Reasonable values: 0.2-0.4 (removes heavily penalized duplicates)
# Suggested alternatives: 0.2, 0.3, 0.4, 0.5
_C.NMS.STAGE2.SCORE_THRESHOLD = 0.3

# ============================================================================
# BACKPROJECTION: Fisheye Coordinate Transformation
# ============================================================================

_C.BACKPROJECTION = CN()

# Enable backprojection of detections to fisheye coordinates
_C.BACKPROJECTION.ENABLED = True

# Number of lattice points to sample along bbox height
# Width samples are calculated automatically based on bbox aspect ratio
# Higher values = more detailed distortion visualization (e.g., 10, 15, 20)
_C.BACKPROJECTION.LATTICE_HEIGHT_SAMPLES = 10

# ============================================================================
# OUTPUT: Visualization and Results
# ============================================================================

_C.OUTPUT = CN()

# Save intermediate composite image before YOLO
_C.OUTPUT.SAVE_COMPOSITE = True

# Save visualization of detections on composite image
_C.OUTPUT.SAVE_COMPOSITE_VIZ = True

# Save visualization of detections on original fisheye image (after backprojection)
_C.OUTPUT.SAVE_FISHEYE_VIZ = True

# Save lattice visualization for each backprojected bbox (shows distortion)
_C.OUTPUT.SAVE_LATTICE_VIZ = True

# Output directory for visualizations and results
_C.OUTPUT.SAVE_DIR = "results"

# ============================================================================
# DEBUG / VERBOSE
# ============================================================================

_C.VERBOSE = False  # Print detailed pipeline information


# ============================================================================
# Public API Functions
# ============================================================================

def get_cfg():
    """
    Get a copy of the default configuration.

    Returns:
        yacs.config.CfgNode: Configuration object
    """
    return _C.clone()


def get_cfg_as_dict(cfg):
    """
    Convert YACS config object to dictionary.

    Args:
        cfg: YACS config object

    Returns:
        dict: Configuration as dictionary
    """
    return CN.to_py(cfg)


if __name__ == "__main__":
    """Print default configuration when run as script."""
    cfg = get_cfg()
    print(cfg)
