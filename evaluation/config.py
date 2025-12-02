"""
YACS configuration for evaluation metrics and dataset handling.

This module defines all configurable parameters for dataset loading, visualization,
and evaluation using YACS.

Configuration hierarchy:
  - DATASETS: Dataset-specific configurations (BOMNI, PIROPO, etc.)
  - METRICS: Evaluation metrics to compute
  - OUTPUT: Result storage and visualization settings
  - VISUALIZATION: Annotation visualization parameters

Usage:
    from evaluation.config import get_cfg
    cfg = get_cfg()  # Get default config
    cfg.DATASETS.BOMNI.ENABLED = True
    cfg.freeze()     # Prevent accidental modifications
"""

from yacs.config import CfgNode as CN
from pathlib import Path

# ============================================================================
# Create config object
# ============================================================================

_C = CN()

# ============================================================================
# DATASETS: Dataset-specific configurations
# ============================================================================

_C.DATASETS = CN()

# ----------------------------------------------------------------------------
# BOMNI Dataset Configuration
# ----------------------------------------------------------------------------

_C.DATASETS.BOMNI = CN()

# Enable BOMNI dataset
_C.DATASETS.BOMNI.ENABLED = True

# Root directory of BOMNI dataset
_C.DATASETS.BOMNI.ROOT_DIR = "datasets/all-datasets/bomni-5841"

# Frames directory (extracted from videos)
_C.DATASETS.BOMNI.FRAMES_DIR = "datasets/all-datasets/bomni-5841/frames/scenario1"

# Annotations directory (rotated bboxes from omnidet-rotinv, manually corrected)
# Original: "datasets/all-datasets/omnidet-rotinv-master/omnidet-rotinv-master/rotate/bomni/rotate/scenario1"
# Corrected: Removed 86 incorrect annotations (25.5%) through manual review
_C.DATASETS.BOMNI.ANNOTATIONS_DIR = "datasets/all-datasets/BOMNI-corrected/Rotated-annotations/scenario1"

# Sequences to use (only scenario1 top cameras have rotated annotations)
# Available: ["top-0", "top-1", "top-2", "top-3"]
# Note: top-4 and side cameras are excluded (no rotated annotations)
_C.DATASETS.BOMNI.SEQUENCES = ["top-0", "top-1", "top-2", "top-3"]

# Image format
_C.DATASETS.BOMNI.IMAGE_EXT = ".jpg"

# Annotation format
_C.DATASETS.BOMNI.ANNOTATION_EXT = ".xml"

# Image dimensions (from annotations)
_C.DATASETS.BOMNI.IMAGE_WIDTH = 640
_C.DATASETS.BOMNI.IMAGE_HEIGHT = 480

# Fisheye center coordinates (for rotation angle calculation)
# Default: image center (width/2, height/2)
_C.DATASETS.BOMNI.FISHEYE_CENTER_X = 320.0
_C.DATASETS.BOMNI.FISHEYE_CENTER_Y = 240.0

# ----------------------------------------------------------------------------
# Future datasets can be added here
# ----------------------------------------------------------------------------

# _C.DATASETS.PIROPO = CN()
# _C.DATASETS.PIROPO.ENABLED = False
# ...

# ============================================================================
# METRICS: Evaluation Metrics Configuration
# ============================================================================

_C.METRICS = CN()

# Standard COCO/YOLO metrics to compute
_C.METRICS.COMPUTE_AP = True  # Average Precision at IoU=0.50:0.95
_C.METRICS.COMPUTE_AP50 = True  # Average Precision at IoU=0.50
_C.METRICS.COMPUTE_AP75 = True  # Average Precision at IoU=0.75
_C.METRICS.COMPUTE_MAP = True  # Mean Average Precision
_C.METRICS.COMPUTE_AR = True  # Average Recall
_C.METRICS.COMPUTE_F1 = True  # F1 score

# IoU thresholds for evaluation
_C.METRICS.IOU_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

# Primary metric for comparison (when comparing configurations)
_C.METRICS.PRIMARY_METRIC = "map"

# Secondary metrics to report
_C.METRICS.SECONDARY_METRICS = ["ap50", "ar", "f1"]

# Compute confidence intervals
_C.METRICS.COMPUTE_CONFIDENCE_INTERVALS = True
_C.METRICS.CONFIDENCE_LEVEL = 0.95

# ============================================================================
# BACKPROJECTION EVALUATION
# ============================================================================

_C.BACKPROJECTION = CN()

# Evaluation stage: "composite", "fisheye", or "both"
# "fisheye" = compare backprojected detections to fisheye ground truth
# "composite" = compare composite detections to composite ground truth (if available)
# "both" = evaluate at both stages
_C.BACKPROJECTION.EVALUATION_STAGE = "fisheye"

# Handle occluded/partial detections in fisheye
_C.BACKPROJECTION.HANDLE_PARTIAL_DETECTIONS = True

# Reduce score for partial detections (penalty factor)
_C.BACKPROJECTION.PARTIAL_DETECTION_PENALTY = 0.1

# ============================================================================
# OUTPUT: Result Storage
# ============================================================================

_C.OUTPUT = CN()

# Root directory for all evaluation results
_C.OUTPUT.ROOT_DIR = "evaluation/results"

# Store detailed per-image results
_C.OUTPUT.STORE_PER_IMAGE_RESULTS = True

# Store visualization of evaluations
_C.OUTPUT.STORE_EVALUATION_VISUALIZATIONS = True

# Store raw detections (not just metrics)
_C.OUTPUT.STORE_RAW_DETECTIONS = True

# ============================================================================
# VISUALIZATION: Annotation Visualization Settings
# ============================================================================

_C.VISUALIZATION = CN()

# Enable annotation visualization
_C.VISUALIZATION.ENABLED = True

# Output directory for annotation visualizations
# Subdirectories will be created per dataset and sequence
_C.VISUALIZATION.OUTPUT_DIR = "evaluation/results/bomni/annotation_visualization"

# Visualization parameters
_C.VISUALIZATION.BBOX_COLOR = (0, 255, 0)  # Green (BGR format)
_C.VISUALIZATION.BBOX_THICKNESS = 2
_C.VISUALIZATION.FONT_SCALE = 0.5
_C.VISUALIZATION.FONT_THICKNESS = 1
_C.VISUALIZATION.SHOW_LABELS = True  # Show class labels ("person")
_C.VISUALIZATION.SHOW_ROTATION_ANGLE = True  # Show rotation angle in degrees

# Draw rotated rectangle vs axis-aligned rectangle
_C.VISUALIZATION.DRAW_ROTATED = True  # True = draw rotated bbox, False = draw axis-aligned

# Draw image center and radial line (for debugging rotation angle)
_C.VISUALIZATION.DRAW_CENTER_AND_RADIAL = False

# Image format for saving visualizations
_C.VISUALIZATION.IMAGE_FORMAT = "jpg"
_C.VISUALIZATION.IMAGE_QUALITY = 95  # JPEG quality (0-100)

# Limit number of images to visualize per sequence
# Options: "all" = all images, integer N = up to N images per sequence
# If N exceeds available images, defaults to all
# Useful for quick testing (e.g., 10)
_C.VISUALIZATION.MAX_IMAGES_PER_SEQUENCE = "all"

# ============================================================================
# DEBUG / VERBOSE
# ============================================================================

_C.VERBOSE = False  # Print detailed information


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
