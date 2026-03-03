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
# Module-level constants
# ============================================================================

# Prediction bbox color presets (BGR format)
PRED_BBOX_COLOR_PRESETS = {
    "yellow": (0, 255, 255),      # High contrast on brown/dark scenes
    "light_red": (0, 100, 255),   # Orange-red
    "bright_red": (0, 0, 255)     # Classic red
}

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
# Use BOMNI-production (validated, 245 images) for evaluation, not bomni-5841 (raw data)
_C.DATASETS.BOMNI.ROOT_DIR = "datasets/all-datasets/BOMNI-production"

# Frames directory (extracted from videos)
_C.DATASETS.BOMNI.FRAMES_DIR = "datasets/all-datasets/bomni-5841/frames/scenario1"

# Annotation format to use
# Options:
#   "tamura" - Third-party annotations from Tamura et al. (omnidet-rotinv)
#              Format: Pascal VOC XML with repurposed fields (legacy, for reference only)
#   "standard" - Our standard JSON annotations (all values precomputed)
#              Format: Clean JSON with center_x, center_y, width, height, angle, class_name
# Default: "standard" (recommended for all datasets)
_C.DATASETS.BOMNI.ANNOTATION_FORMAT = "standard"

# Annotations directory - Tamura et al. format (Pascal VOC XML, manually corrected)
# Original: "datasets/all-datasets/omnidet-rotinv-master/omnidet-rotinv-master/rotate/bomni/rotate/scenario1"
# Corrected: Removed 86 incorrect annotations (25.5%) through manual review
# Note: This is kept for reference and re-conversion if needed
_C.DATASETS.BOMNI.TAMURA_ANNOTATIONS_DIR = "datasets/all-datasets/BOMNI-corrected/Rotated-annotations/scenario1"

# Annotations directory - Standard format (JSON with all values precomputed)
# Generated from Tamura annotations with precomputed center, angle, width, height
# This is our unified annotation format used by all datasets
_C.DATASETS.BOMNI.STANDARD_ANNOTATIONS_DIR = "datasets/all-datasets/BOMNI-corrected/Standard-annotations-ours/scenario1"

# Sequences to use (only scenario1 top cameras have rotated annotations)
# Available: ["top-0", "top-1", "top-2", "top-3"]
# Note: top-4 and side cameras are excluded (no rotated annotations)
_C.DATASETS.BOMNI.SEQUENCES = ["top-0", "top-1", "top-2", "top-3"]

# Image format
_C.DATASETS.BOMNI.IMAGE_EXT = ".jpg"

# Image dimensions (from annotations)
_C.DATASETS.BOMNI.IMAGE_WIDTH = 640
_C.DATASETS.BOMNI.IMAGE_HEIGHT = 480

# Fisheye center coordinates (for rotation angle calculation)
# Default: image center (width/2, height/2)
_C.DATASETS.BOMNI.FISHEYE_CENTER_X = 320.0
_C.DATASETS.BOMNI.FISHEYE_CENTER_Y = 240.0

# ----------------------------------------------------------------------------
# PIROPO Dataset Configuration
# ----------------------------------------------------------------------------

_C.DATASETS.PIROPO = CN()
_C.DATASETS.PIROPO.ENABLED = True

# Root directory of PIROPO-production dataset (validated, 3,004 images)
_C.DATASETS.PIROPO.ROOT_DIR = "datasets/all-datasets/PIROPO-production"

# Standard JSON annotations directory (3-level hierarchy: Room/Camera/CameraSeq/)
_C.DATASETS.PIROPO.STANDARD_ANNOTATIONS_DIR = "datasets/all-datasets/PIROPO-production/standard-annotations"

# Rooms to include in evaluation (discovered cameras/sequences within each room are auto-found)
# Room_A has 3 cameras (omni_1A, omni_2A, omni_3A); Room_B has 1 camera (omni_1B)
_C.DATASETS.PIROPO.ROOMS = ["Room_A", "Room_B"]

# Image format
_C.DATASETS.PIROPO.IMAGE_EXT = ".jpg"

# Image dimensions (all PIROPO cameras produce 800x600 images)
_C.DATASETS.PIROPO.IMAGE_WIDTH = 800
_C.DATASETS.PIROPO.IMAGE_HEIGHT = 600

# Fisheye center coordinates (image center for 800x600)
_C.DATASETS.PIROPO.FISHEYE_CENTER_X = 400.0
_C.DATASETS.PIROPO.FISHEYE_CENTER_Y = 300.0

# ----------------------------------------------------------------------------
# CEPDOF Dataset Configuration
# ----------------------------------------------------------------------------

_C.DATASETS.CEPDOF = CN()
_C.DATASETS.CEPDOF.ENABLED = True

# Root directory of CEPDOF-production dataset
_C.DATASETS.CEPDOF.ROOT_DIR = "datasets/all-datasets/CEPDOF-production"

# Standard JSON annotations directory (flat: {Sequence}/{frame_id}.json)
_C.DATASETS.CEPDOF.STANDARD_ANNOTATIONS_DIR = "datasets/all-datasets/CEPDOF-production/standard-annotations"

# Sequences to include (must match folder names under ROOT_DIR and standard-annotations/)
_C.DATASETS.CEPDOF.SEQUENCES = [
    "Lunch1", "Lunch2", "Lunch3", "Edge_cases",
    "High_activity", "All_off", "IRfilter", "IRill"
]

# Image format
_C.DATASETS.CEPDOF.IMAGE_EXT = ".jpg"

# Fisheye center: -1 = auto-compute per image (width/2, height/2)
# CEPDOF has mixed resolutions (2048x2048 and 1080x1080)
_C.DATASETS.CEPDOF.FISHEYE_CENTER_X = -1.0
_C.DATASETS.CEPDOF.FISHEYE_CENTER_Y = -1.0

# ----------------------------------------------------------------------------
# Future datasets can be added here
# ----------------------------------------------------------------------------

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

# Visualization parameters for ground truth bboxes
_C.VISUALIZATION.GT_BBOX_COLOR = (0, 255, 0)  # Green (BGR format)
_C.VISUALIZATION.GT_BBOX_THICKNESS = 2
_C.VISUALIZATION.FONT_SCALE = 0.5
_C.VISUALIZATION.FONT_THICKNESS = 1
_C.VISUALIZATION.SHOW_LABELS = True  # Show class labels ("person")
_C.VISUALIZATION.SHOW_ROTATION_ANGLE = True  # Show rotation angle in degrees

# Visualization parameters for predicted bboxes
# Color options:
#   - "yellow": (0, 255, 255) - High contrast on brown/dark scenes
#   - "light_red": (0, 100, 255) - Orange-red, good contrast
#   - "bright_red": (0, 0, 255) - Classic red (default)
_C.VISUALIZATION.PRED_BBOX_COLOR_PRESET = "yellow"  # Options: "yellow", "light_red", "bright_red"
_C.VISUALIZATION.PRED_BBOX_THICKNESS = 3  # Thicker than GT for better visibility

# Legacy parameter (kept for backward compatibility with dataset visualization)
_C.VISUALIZATION.BBOX_COLOR = (0, 255, 0)  # Green (BGR format)
_C.VISUALIZATION.BBOX_THICKNESS = 2

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
# IOU_VALIDATION: IoU Visual Validation Settings
# ============================================================================

_C.IOU_VALIDATION = CN()

# Projection configuration JSON file
# Set to None to auto-detect first JSON file in evaluation/projection_configs/
_C.IOU_VALIDATION.CONFIG_JSON = None

# Which configuration ID to use from the JSON file
# Set to None to use the first configuration
_C.IOU_VALIDATION.CONFIG_ID = None

# Number of images to process for validation
# Set to None to process all images in the dataset
_C.IOU_VALIDATION.NUM_SAMPLES = None

# Dataset to use for validation
_C.IOU_VALIDATION.DATASET_NAME = "bomni"

# Detection settings (shared across all datasets)
_C.IOU_VALIDATION.YOLO_MODEL_PATH = "models/yolo11x.pt"
_C.IOU_VALIDATION.CONF_THRESHOLD = 0.25

# Output directory for IoU validation visualizations
_C.IOU_VALIDATION.OUTPUT_DIR = "evaluation/iou-visual-validation"

# BOMNI-specific settings for IoU validation
_C.IOU_VALIDATION.BOMNI = CN()
_C.IOU_VALIDATION.BOMNI.DATASET_ROOT = "datasets/all-datasets/BOMNI-production"
_C.IOU_VALIDATION.BOMNI.FRAMES_DIR = "datasets/all-datasets/BOMNI-production/frames/scenario1"
_C.IOU_VALIDATION.BOMNI.ANNOTATIONS_DIR = "datasets/all-datasets/BOMNI-production/Standard-annotations/scenario1"
_C.IOU_VALIDATION.BOMNI.SEQUENCES = ["top-0", "top-1", "top-2", "top-3"]
_C.IOU_VALIDATION.BOMNI.IMAGE_CENTER_X = 320.0  # Image center (W/2, H/2), used as rotation reference
_C.IOU_VALIDATION.BOMNI.IMAGE_CENTER_Y = 240.0

# ============================================================================
# METRICS_EVALUATION: Configuration Comparison Metrics
# ============================================================================

_C.METRICS_EVALUATION = CN()

# Projection configuration module (Python file within evaluation/)
# Default: "evaluation.projection_configs_for_metrics"
# Can change to other config files like "evaluation.projection_configs_ablation"
_C.METRICS_EVALUATION.PROJECTION_CONFIG_MODULE = "evaluation.projection_configs_for_metrics"

# YOLO model to use for all configurations (fixed for fair comparison)
_C.METRICS_EVALUATION.YOLO_MODEL = "models/yolov8m.pt"

# Datasets to evaluate (list of dataset names)
_C.METRICS_EVALUATION.DATASETS = ["bomni", "piropo", "cepdof"]  # options : "bomni", "piropo" and "cepdof".

# Output directory for metrics evaluation results
_C.METRICS_EVALUATION.OUTPUT_DIR = "evaluation/proj-conf-comparison"

# IoU thresholds for evaluation (0.30 to 0.95 with 0.05 step)
# Starting from 0.30 to capture easier detections
_C.METRICS_EVALUATION.IOU_THRESHOLDS = [
    0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
    0.75, 0.80, 0.85, 0.90, 0.95
]

# IoU threshold used to classify predictions as TP/FP/FN in visual outputs.
# Purely for display — does not affect any metrics computation.
_C.METRICS_EVALUATION.VIS_IOU_THRESHOLD = 0.50

# Enable pipeline timing measurements
_C.METRICS_EVALUATION.ENABLE_TIMING = True

# Enable precision-recall curve generation
_C.METRICS_EVALUATION.ENABLE_PR_CURVES = True

# Save visual results (composite with detections + fisheye with GT and predictions)
# Output: {session}/visuals/{config_id}/ — flat folder, one image pair per evaluated frame
_C.METRICS_EVALUATION.ENABLE_VISUALS = True

# Limit number of images to process (None = all images)
# Set to integer N for quick testing (e.g., 5 for subset test)
# Edit this value in config.py to control how many images to process
_C.METRICS_EVALUATION.MAX_IMAGES = 3000  # Change to 5 for quick test, None for full dataset

# When True, spread the sampled frames evenly across the full dataset
# (step = total / max_images) instead of taking the first N consecutive frames.
# Ignored when MAX_IMAGES is None (all frames are used).
_C.METRICS_EVALUATION.SPREAD_SAMPLES = True

# ============================================================================
# SINGLE_CONFIG_RUN: Incremental per-configuration evaluation
# ============================================================================

_C.SINGLE_CONFIG_RUN = CN()

# ID of the projection configuration to evaluate (must match an 'id' field in
# evaluation/projection_configs_for_metrics.py)
# _C.SINGLE_CONFIG_RUN.CONFIG_ID = "grid-2x2-fov60"
_C.SINGLE_CONFIG_RUN.CONFIG_ID = "chiang-2021-baseline"

# Datasets to evaluate for this configuration run
# Options: "bomni", "piropo", "cepdof"
_C.SINGLE_CONFIG_RUN.DATASETS = ["bomni", "piropo", "cepdof"]

# Maximum number of images to process per dataset (None = all images)
# BOMNI has 245 frames (all used when max_images >= 245)
# PIROPO has 3,004 frames; CEPDOF has 25,358 frames
_C.SINGLE_CONFIG_RUN.MAX_IMAGES = 5000

# When True, distribute sampled frames evenly across the full dataset
# (step = total / max_images) instead of taking the first N consecutive frames.
_C.SINGLE_CONFIG_RUN.SPREAD_SAMPLES = True

# Enable pipeline timing measurements
_C.SINGLE_CONFIG_RUN.ENABLE_TIMING = True

# Enable precision-recall curve generation (individual config plots)
_C.SINGLE_CONFIG_RUN.ENABLE_PR_CURVES = True

# Save composite + fisheye visual image pairs
_C.SINGLE_CONFIG_RUN.ENABLE_VISUALS = True

# When False (default), skip evaluation if config folder already exists (safe).
# Set to True to overwrite a previously evaluated configuration.
_C.SINGLE_CONFIG_RUN.OVERWRITE_EXISTING = False

# Root output directory for per-config results
_C.SINGLE_CONFIG_RUN.OUTPUT_DIR = "evaluation/proj-conf-comparison/configs"

# YOLO model to use (fixed for fair comparison across all configs)
_C.SINGLE_CONFIG_RUN.YOLO_MODEL = "models/yolov8m.pt"

# IoU thresholds for evaluation (0.30 to 0.95 with 0.05 step, COCO-extended)
_C.SINGLE_CONFIG_RUN.IOU_THRESHOLDS = [
    0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
    0.75, 0.80, 0.85, 0.90, 0.95
]

# IoU threshold used to classify predictions as TP/FP/FN in visual outputs.
# Purely for display — does not affect any metrics computation.
_C.SINGLE_CONFIG_RUN.VIS_IOU_THRESHOLD = 0.50

# ============================================================================
# COMPARATOR: Compare pre-computed per-config results
# ============================================================================

_C.COMPARATOR = CN()

# Config IDs to compare. Empty list = compare ALL configs found in OUTPUT_DIR.
_C.COMPARATOR.CONFIG_IDS = []

# Datasets to include in the comparison
_C.COMPARATOR.DATASETS = ["bomni", "piropo", "cepdof"]

# Root directory where per-config results are stored (must match SINGLE_CONFIG_RUN.OUTPUT_DIR)
_C.COMPARATOR.OUTPUT_DIR = "evaluation/proj-conf-comparison/configs"

# Directory to write comparison outputs (tables, figures, winner file)
_C.COMPARATOR.COMPARISON_OUT_DIR = "evaluation/proj-conf-comparison/comparisons"

# Enable figure generation (bar charts, PR overlay plots)
_C.COMPARATOR.ENABLE_FIGURES = True

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


def get_pred_bbox_color(cfg):
    """
    Get prediction bbox color as BGR tuple from preset string.

    Args:
        cfg: YACS config object

    Returns:
        tuple: BGR color tuple (B, G, R)
    """
    preset = cfg.VISUALIZATION.PRED_BBOX_COLOR_PRESET.lower()

    if preset not in PRED_BBOX_COLOR_PRESETS:
        print(f"Warning: Unknown color preset '{preset}', defaulting to yellow")
        return PRED_BBOX_COLOR_PRESETS["yellow"]

    return PRED_BBOX_COLOR_PRESETS[preset]


if __name__ == "__main__":
    """Print default configuration when run as script."""
    cfg = get_cfg()
    print(cfg)
