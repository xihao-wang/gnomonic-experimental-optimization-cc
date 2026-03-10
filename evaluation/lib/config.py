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

# Valid YOLO inference resolutions (multiples of 32 accepted by ultralytics).
# Set METRICS_EVALUATION.YOLO_IMGSZ to one of these keys.
YOLO_IMGSZ_OPTIONS = {
    "320":  320,
    "416":  416,
    "512":  512,
    "640":  640,
    "768":  768,
    "1024": 1024,
    "1280": 1280,
}

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
_C.IOU_VALIDATION.YOLO_MODEL_PATH = "models/yolov8m.pt"
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
# METRICS_EVALUATION: Configuration comparison — shared run parameters
# ============================================================================

_C.METRICS_EVALUATION = CN()

# Projection configuration module (Python file within evaluation/)
# Can change to other config files like "evaluation.projection_configs_ablation"
_C.METRICS_EVALUATION.PROJECTION_CONFIG_MODULE = "evaluation.projection_configs_for_metrics"

# YOLO model filename for single-model runs (fixed for fair comparison).
# Specify the .pt filename only; full path resolved as <project_root>/models/<filename>.
# Also controls WHERE run_comparator.py reads/writes: subfolder = Path(YOLO_MODEL).stem.
_C.METRICS_EVALUATION.YOLO_MODEL = "yolov9e.pt"

# YOLO inference resolution key — must be one of YOLO_IMGSZ_OPTIONS.
# Fixed across all configs and models to ensure a fair comparison.
_C.METRICS_EVALUATION.YOLO_IMGSZ = "640"

# Datasets to evaluate — options: "bomni", "piropo", "cepdof"
_C.METRICS_EVALUATION.DATASETS = ["bomni", "piropo", "cepdof"]

# Root output directory (sessions and per-config result folders are created inside)
_C.METRICS_EVALUATION.OUTPUT_DIR = "evaluation/proj-conf-comparison"

# IoU thresholds for evaluation: AP@[0.50:0.75] (6 thresholds, step 0.05)
# Rationale: predictions are radially aligned by design; GT annotations on CEPDOF
# are free-orientation. High thresholds (0.80–0.95) penalise this systematic
# convention mismatch rather than localisation error. Truncating the ceiling at
# 0.75 removes thresholds where annotation convention dominates the score.
# The floor stays at 0.50 (standard Pascal VOC threshold) — lowering it would
# inflate AP with near-misses and is not justified here.
_C.METRICS_EVALUATION.IOU_THRESHOLDS = [
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75
]

# Enable pipeline timing measurements
_C.METRICS_EVALUATION.ENABLE_TIMING = True

# Enable precision-recall curve generation
_C.METRICS_EVALUATION.ENABLE_PR_CURVES = True

# Limit number of images to process per dataset (None = all images)
# Set to a small integer (e.g., 5) for quick smoke tests
_C.METRICS_EVALUATION.MAX_IMAGES = 3000

# When True, spread sampled frames evenly across the full dataset
# (step = total / max_images). Ignored when MAX_IMAGES is None.
_C.METRICS_EVALUATION.SPREAD_SAMPLES = True

# ----------------------------------------------------------------------------
# METRICS_EVALUATION.VIS: Visualization parameters (shared by all runners)
# ----------------------------------------------------------------------------

_C.METRICS_EVALUATION.VIS = CN()

# Enable saving of composite + fisheye visual image pairs
_C.METRICS_EVALUATION.VIS.ENABLE = True

# IoU threshold used to classify predictions as TP / FP / FN in visuals.
# Display only — does not affect any metrics computation.
_C.METRICS_EVALUATION.VIS.IOU_THRESHOLD = 0.50

# Alternating colors (RGB) for projection boundary overlays on fisheye visuals.
# Colors cycle across projections: proj 0 → colors[0], proj 1 → colors[1], proj 2 → colors[0], ...
# (255, 174, 201) = rose,  (115, 251, 253) = light cyan
_C.METRICS_EVALUATION.VIS.PROJ_BOUNDARY_COLORS = [(255, 174, 201), (115, 251, 253)]

# Maximum number of visual image pairs (composite + fisheye) saved per dataset.
# 0 = no limit (save all). Each dataset has its own handle.
_C.METRICS_EVALUATION.VIS.MAX_SAMPLES = CN()
_C.METRICS_EVALUATION.VIS.MAX_SAMPLES.BOMNI  = 100
_C.METRICS_EVALUATION.VIS.MAX_SAMPLES.PIROPO = 100
_C.METRICS_EVALUATION.VIS.MAX_SAMPLES.CEPDOF = 100

# When True, the N visual samples are spread uniformly across the evaluated
# frames (step = total / N), so the last saved visual is near the last frame.
# When False, the first N consecutive evaluated frames are saved.
_C.METRICS_EVALUATION.VIS.SPREAD_SAMPLES = True

# ----------------------------------------------------------------------------
# METRICS_EVALUATION.SINGLE_CONFIG_RUN: Incremental per-configuration runner
# ----------------------------------------------------------------------------
# All other run parameters (YOLO model, datasets, IoU thresholds, max images,
# output dir, timing, PR curves, visuals) are inherited from METRICS_EVALUATION.

_C.METRICS_EVALUATION.SINGLE_CONFIG_RUN = CN()

# Config IDs to evaluate (each must match an 'id' in
# evaluation/projection_configs_for_metrics.py).
# Empty list [] = evaluate ALL configs defined in that file.
# Already-evaluated configs are skipped automatically when OVERWRITE_EXISTING=False,
# so re-running with [] only processes configs that have no result folder yet.
# Config search study: effect of projection configuration parameters.
# Progression: 4 proj (wide FOV) → 6 proj (our family) → 8 proj (Chiang) → 9 proj.
# Already-evaluated configs (TEST#26, chiang) are skipped automatically.
_C.METRICS_EVALUATION.SINGLE_CONFIG_RUN.CONFIG_IDS = [
    # 4 projections — low count, high FOV baselines
    "TEST#05-h90-v90-g(2,2)",       # 4 proj, 90×90 FOV, lat45 — zero h-overlap baseline
    "TEST#07-h106-v80-g(2,2)",      # 4 proj, 106×80 FOV, lat50 — wider FOV, steeper look
    # 6 projections — parameter sweep within the winning family
    "TEST#04-h60-v90-g(2,3)",       # 6 proj, 60×90 FOV, lat45 — zero h-overlap reference
    "TEST#09-h70-v90-g(2,3)",       # 6 proj, 70×90 FOV, lat45 — 10° h-overlap (isolates overlap effect)
    "TEST#23-h60-v90-g(2,3)",       # 6 proj, 60×90 FOV, lat50 — isolates latitude vs TEST#04
    "TEST#26-h60-v90-g(2,3)",       # 6 proj, OUR PROPOSED — already run, skipped automatically
    # 8 projections — Chiang baseline
    "chiang-2021-baseline",         # 8 proj, 48×96 FOV, lat36 — already run, skipped automatically
    # 9 projections — finer angular sampling, overlap progression
    "TEST#15-h40-v90-g(3,3)",       # 9 proj, 40×90 FOV, lat45 — zero h-overlap baseline
    "TEST#16-h50-v90-g(3,3)",       # 9 proj, 50×90 FOV, lat45 — 10° h-overlap
    "TEST#17-h60-v90-g(3,3)",       # 9 proj, 60×90 FOV, lat45 — 20° h-overlap (same fov_h as our 6-proj)
]

# When False (default), skip configs whose result folder already exists.
# Set to True to force re-evaluation and overwrite existing results.
_C.METRICS_EVALUATION.SINGLE_CONFIG_RUN.OVERWRITE_EXISTING = False

# ----------------------------------------------------------------------------
# METRICS_EVALUATION.MULTI_MODEL_RUN: run all configs across several YOLO models
# ----------------------------------------------------------------------------

_C.METRICS_EVALUATION.MULTI_MODEL_RUN = CN()

# Model filenames to test — specify the .pt filename only.
# Full path resolved automatically as <project_root>/models/<filename>.
# _C.METRICS_EVALUATION.MULTI_MODEL_RUN.MODELS = [
#     # YOLOv8 (2023) — n/s/m/l/x
#     "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
#     # YOLOv9 (2024) — c (compact) / e (extended); no n/s/m/l/x naming
#     "yolov9c.pt", "yolov9e.pt",
#     # YOLOv10 (2024) — n/s/m/b/l/x (adds 'b' size)
#     "yolov10n.pt", "yolov10s.pt", "yolov10m.pt", "yolov10b.pt", "yolov10l.pt", "yolov10x.pt",
#     # YOLO11 (2024) — n/s/m/l/x
#     "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
#     # YOLO12 (2025) — n/s/m/l/x
#     "yolo12n.pt", "yolo12s.pt", "yolo12m.pt", "yolo12l.pt", "yolo12x.pt",
#     # YOLO26 (Jan 2026) — n/s/m/l/x
#     "yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt",
# ]
# Config search study uses YOLOv9e only (fixed backbone for fair parameter comparison).
# Restore the full list below once the config search study is complete and
# you want to re-run the cross-model study with the new configs.
_C.METRICS_EVALUATION.MULTI_MODEL_RUN.MODELS = [
    "yolov9e.pt",   # config search study: fixed backbone
]
# Full multi-model list (uncomment to restore for cross-model studies):
# _C.METRICS_EVALUATION.MULTI_MODEL_RUN.MODELS = [
#     "yolov8n.pt", "yolov8m.pt", "yolov9e.pt",
#     "yolov10m.pt", "yolo11m.pt", "yolo12m.pt", "yolo26m.pt",
# ]

# Config IDs to evaluate. Empty list = all configs in projection_configs_for_metrics.py.
_C.METRICS_EVALUATION.MULTI_MODEL_RUN.CONFIG_IDS = []

# When False, skip configs whose result folder already exists for a given model.
_C.METRICS_EVALUATION.MULTI_MODEL_RUN.OVERWRITE_EXISTING = False

# ============================================================================
# COMPARATOR: Compare pre-computed per-config results
# ============================================================================

_C.COMPARATOR = CN()

# Config IDs to compare. Empty list = compare ALL configs found automatically.
# _C.COMPARATOR.CONFIG_IDS = []
_C.COMPARATOR.CONFIG_IDS = [
    # 4 projections — low count, high FOV baselines
    "TEST#05-h90-v90-g(2,2)",       # 4 proj, 90×90 FOV, lat45 — zero h-overlap baseline
    "TEST#07-h106-v80-g(2,2)",      # 4 proj, 106×80 FOV, lat50 — wider FOV, steeper look
    # 6 projections — parameter sweep within the winning family
    "TEST#04-h60-v90-g(2,3)",       # 6 proj, 60×90 FOV, lat45 — zero h-overlap reference
    "TEST#09-h70-v90-g(2,3)",       # 6 proj, 70×90 FOV, lat45 — 10° h-overlap (isolates overlap effect)
    "TEST#23-h60-v90-g(2,3)",       # 6 proj, 60×90 FOV, lat50 — isolates latitude vs TEST#04
    "TEST#26-h60-v90-g(2,3)",       # 6 proj, OUR PROPOSED — already run, skipped automatically
    # 8 projections — Chiang baseline
    "chiang-2021-baseline",         # 8 proj, 48×96 FOV, lat36 — already run, skipped automatically
    # 9 projections — finer angular sampling, overlap progression
    "TEST#15-h40-v90-g(3,3)",       # 9 proj, 40×90 FOV, lat45 — zero h-overlap baseline
    "TEST#16-h50-v90-g(3,3)",       # 9 proj, 50×90 FOV, lat45 — 10° h-overlap
    "TEST#17-h60-v90-g(3,3)",       # 9 proj, 60×90 FOV, lat45 — 20° h-overlap (same fov_h as our 6-proj)
]

# Datasets to include in the comparison
_C.COMPARATOR.DATASETS = ["bomni", "piropo", "cepdof"]

# Enable figure generation (bar charts, PR overlay plots)
_C.COMPARATOR.ENABLE_FIGURES = True

# NOTE: result and comparison directories are NOT configured here.
# They are derived automatically in run_comparator.py from:
#   METRICS_EVALUATION.OUTPUT_DIR / <model_label> / "configs"     (input)
#   METRICS_EVALUATION.OUTPUT_DIR / <model_label> / "comparisons" (output)
# where <model_label> = Path(METRICS_EVALUATION.YOLO_MODEL).stem  (e.g. "yolov8m")

# ============================================================================
# CROSS_MODEL_COMPARATOR: Compare configs across all evaluated YOLO models
# ============================================================================

_C.CROSS_MODEL_COMPARATOR = CN()

# Models to include (folder names under METRICS_EVALUATION.OUTPUT_DIR,
# e.g. ["yolov8m", "yolo12x"]). Empty list = all model folders found.
_C.CROSS_MODEL_COMPARATOR.MODELS = []

# Original config IDs to include (without model suffix).
# Empty list = all configs found across all selected model folders.
_C.CROSS_MODEL_COMPARATOR.CONFIG_IDS = ["chiang-2021-baseline", "TEST#26-h60-v90-g(2,3)"]

# Datasets to include in the comparison.
_C.CROSS_MODEL_COMPARATOR.DATASETS = ["bomni", "piropo", "cepdof"]

# Enable figure generation (bar charts, PR overlay plots).
_C.CROSS_MODEL_COMPARATOR.ENABLE_FIGURES = True

# Results are written to:
#   METRICS_EVALUATION.OUTPUT_DIR / "cross-model" / comparison_{timestamp}/

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
