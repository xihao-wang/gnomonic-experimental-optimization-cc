"""
YACS configuration for dataset preparation (one-time operations).

This module defines all configurable parameters for dataset preparation workflow:
  - Frame extraction from videos
  - Annotation format conversion
  - Cleanup operations
  - Visualization for quality review

This is SEPARATE from evaluation/config.py which handles runtime evaluation.

Configuration hierarchy:
  - BOMNI: BOMNI dataset preparation settings
  - PIROPO: PIROPO dataset preparation settings
  - VISUALIZATION: Visualization settings for quality review

Usage:
    from datasets.config import get_cfg
    cfg = get_cfg()
    cfg.BOMNI.PREPARATION.TARGET_NAME = "BOMNI-test-1"
    cfg.freeze()
"""

from yacs.config import CfgNode as CN

# ============================================================================
# Create config object
# ============================================================================

_C = CN()

# ============================================================================
# BOMNI Dataset Preparation
# ============================================================================

_C.BOMNI = CN()

# ----------------------------------------------------------------------------
# INPUT: Raw data directories (from download)
# ----------------------------------------------------------------------------

# VIDEO_DIR must point to the scenario1 folder from the downloaded BOMNI dataset
# This folder contains video files: top-0.mp4, top-1.mp4, top-2.mp4, top-3.mp4
_C.BOMNI.VIDEO_DIR = "datasets/all-datasets/misc/bomni-5841/scenario1"
# Raw annotations directory (Tamura et al. format - Pascal VOC XML)
# Original annotations before manual correction
_C.BOMNI.RAW_ANNOTATIONS_DIR = "datasets/all-datasets/misc/omnidet-rotinv-master/rotate/bomni/rotate/scenario1"

# Input annotation format to convert from
# Options: "tamura" (Tamura et al. Pascal VOC XML)
_C.BOMNI.RAW_ANNOTATION_FORMAT = "tamura"

# ----------------------------------------------------------------------------
# OUTPUT: Target directory (change this for different preparation runs)
# ----------------------------------------------------------------------------

# Target directory name for this preparation run
# Examples: "BOMNI-test-1", "BOMNI-test-2", "BOMNI-corrected" (production)
# All outputs will be created under: datasets/all-datasets/{TARGET_NAME}/
_C.BOMNI.TARGET_NAME = "BOMNI-production"

# ----------------------------------------------------------------------------
# Processing settings
# ----------------------------------------------------------------------------

# Sequences to process (only these cameras have rotated annotations)
_C.BOMNI.SEQUENCES = ["top-0", "top-1", "top-2", "top-3"]

# Fisheye center coordinates (for rotation angle calculation)
# Default: image center for 640×480 images
_C.BOMNI.FISHEYE_CENTER_X = 320.0
_C.BOMNI.FISHEYE_CENTER_Y = 240.0

# Image dimensions
_C.BOMNI.IMAGE_WIDTH = 640
_C.BOMNI.IMAGE_HEIGHT = 480

# Folder structure (relative to TARGET_NAME)
_C.BOMNI.FRAMES_SUBPATH = "frames/scenario1"  # Where frames are stored/extracted
_C.BOMNI.ANNOTATIONS_SUBPATH = "standard-annotations/scenario1"  # Where JSON annotations are saved

# ============================================================================
# PIROPO Dataset Preparation
# ============================================================================

_C.PIROPO = CN()

# ----------------------------------------------------------------------------
# INPUT: Raw data directories
# ----------------------------------------------------------------------------

# Raw frames directory (downloaded PIROPO dataset with ALL frames - never modified)
_C.PIROPO.FRAMES_SOURCE_DIR = "datasets/all-datasets/misc/piropo-tamura"

# Raw annotations directory (Tamura et al. format - Pascal VOC XML)
_C.PIROPO.RAW_ANNOTATIONS_DIR = "datasets/all-datasets/taumra-annotations-omnidet-rotinv-master/rotate/piropo"

# Input annotation format to convert from
_C.PIROPO.RAW_ANNOTATION_FORMAT = "tamura"

# ----------------------------------------------------------------------------
# OUTPUT: Target directory (Step 1 output - cleaned data ready for review)
# ----------------------------------------------------------------------------

# Target directory name for this preparation run
_C.PIROPO.TARGET_NAME = "PIROPO-step1"  # datasets/all-datasets/_C.PIROPO.TARGET_NAME/

# ----------------------------------------------------------------------------
# Processing settings
# ----------------------------------------------------------------------------

# Rooms annotated by Tamura et al.
_C.PIROPO.ROOMS = ["Room_A", "Room_B"]

# Sequences annotated by Tamura et al. (same for all cameras)
_C.PIROPO.SEQUENCES = ["training", "test2", "test3"]

# Note: Cameras are discovered from annotations directory structure
# Room_A: omni_1A, omni_2A, omni_3A
# Room_B: omni_1B

# Fisheye center coordinates (for rotation angle calculation)
# PIROPO uses 800×600 images with center at (400, 300)
_C.PIROPO.FISHEYE_CENTER_X = 400.0
_C.PIROPO.FISHEYE_CENTER_Y = 300.0

# Image dimensions
_C.PIROPO.IMAGE_WIDTH = 800
_C.PIROPO.IMAGE_HEIGHT = 600

# Folder structure (relative to TARGET_NAME)
_C.PIROPO.FRAMES_SUBPATH = ""  # Frames at root: Room A/, Room B/
_C.PIROPO.ANNOTATIONS_SUBPATH = "standard-annotations"  # Annotations: standard-annotations/Room_A/, Room_B/

# ============================================================================
# CEPDOF Dataset Preparation
# ============================================================================

_C.CEPDOF = CN()

# Raw frames and annotations (from original dataset download - never modified)
_C.CEPDOF.FRAMES_SOURCE_DIR = "datasets/all-datasets/misc/CEPDOF/CEPDOF"
_C.CEPDOF.RAW_ANNOTATIONS_DIR = "datasets/all-datasets/misc/CEPDOF/CEPDOF/annotations"
_C.CEPDOF.RAW_ANNOTATION_FORMAT = "cepdof"

# Target directory
_C.CEPDOF.TARGET_NAME = "CEPDOF-production"

# Sequences (must match folder names in FRAMES_SOURCE_DIR)
_C.CEPDOF.SEQUENCES = ["Lunch1", "Lunch2", "Lunch3", "Edge_cases", "High_activity", "All_off", "IRfilter", "IRill"]

# Fisheye center: image center (images are square: 2048x2048 or 1080x1080)
# Set to None to auto-compute per image from its dimensions
_C.CEPDOF.FISHEYE_CENTER_X = -1.0   # -1 = auto (width / 2)
_C.CEPDOF.FISHEYE_CENTER_Y = -1.0   # -1 = auto (height / 2)

# Folder structure (relative to TARGET_NAME)
_C.CEPDOF.FRAMES_SUBPATH = ""         # Frames at root: {sequence}/
_C.CEPDOF.ANNOTATIONS_SUBPATH = "standard-annotations"

# Max frames to visualize per sequence for manual quality review
# Set to 0 to skip visualization entirely (annotations still fully converted)
# Set to -1 to visualize all (not recommended for CEPDOF: 25K+ frames)
_C.CEPDOF.MAX_VISUALIZATIONS_PER_SEQUENCE = 0

# When True, spread the sampled frames evenly across the full sequence
# (step = total / max) instead of taking the first N consecutive frames.
# Recommended for large datasets to get a representative sample.
_C.CEPDOF.SPREAD_VISUALIZATIONS = True

# ============================================================================
# VISUALIZATION: Quality Review Settings
# ============================================================================

_C.VISUALIZATION = CN()

# Visualization parameters for quality review
_C.VISUALIZATION.BBOX_COLOR = (0, 255, 0)  # Green (BGR format)
_C.VISUALIZATION.BBOX_THICKNESS = 3
_C.VISUALIZATION.FONT_SCALE = 0.5
_C.VISUALIZATION.FONT_THICKNESS = 1
_C.VISUALIZATION.SHOW_LABELS = True
_C.VISUALIZATION.SHOW_ROTATION_ANGLE = True

# Draw rotated rectangle vs axis-aligned rectangle
_C.VISUALIZATION.DRAW_ROTATED = True

# Draw image center and radial line (for debugging rotation angle)
_C.VISUALIZATION.DRAW_CENTER_AND_RADIAL = False

# Image format for saving visualizations
_C.VISUALIZATION.IMAGE_FORMAT = "jpg"
_C.VISUALIZATION.IMAGE_QUALITY = 100  # JPEG quality (0-100)

# ============================================================================
# DEBUG / VERBOSE
# ============================================================================

_C.VERBOSE = True  # Print detailed information during preparation

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
