"""
Configuration for the detection pipeline module.

This module:
1. Creates composite images using image-composer API
2. Runs YOLO detection on the composite image
3. Backprojects results from composite to fisheye coordinates
"""

from pathlib import Path

# ============================================================================
# YOLO DETECTOR SETTINGS
# ============================================================================

# YOLO confidence threshold for detections
YOLO_CONFIDENCE_THRESHOLD = 0.5

# YOLO IoU threshold for NMS
YOLO_IOU_THRESHOLD = 0.45

# Device to run YOLO on: "cpu", "cuda" (if NVIDIA GPU available)
YOLO_DEVICE = "cpu"

# ============================================================================
# PROJECTION SETTINGS
# ============================================================================

# Default projection configuration (can be overridden per-experiment)
# These are passed to image-composer's projection generator

DEFAULT_PROJECTIONS = {
    "count": 4,  # Number of projections (2x2 grid = 4)
    "fov_h": 48.0,  # Horizontal field of view
    "fov_v": 96.0,  # Vertical field of view
    "latitude": 36.0,  # Camera angle (0=nadir, 90=horizon)
}

# Composite image size (width, height) for YOLO input
COMPOSITE_SIZE = (640, 640)

# ============================================================================
# BACKPROJECTION SETTINGS
# ============================================================================

# When projecting detections back to fisheye, we may need to:
# - Account for projection uncertainty
# - Merge overlapping detections from different projections
# - Handle partial detections at projection boundaries

# Merge overlapping detections if IoU > threshold
MERGE_IoU_THRESHOLD = 0.5

# Mark detections as "partial" if they touch projection boundaries
# and may not be complete in the fisheye
MARK_PARTIAL_DETECTIONS = True

# ============================================================================
# OUTPUT & VISUALIZATION
# ============================================================================

# Save intermediate composite images
SAVE_COMPOSITE_IMAGES = False

# Save visualization of detections on composite
SAVE_COMPOSITE_DETECTIONS_VIZ = False

# Save visualization of backprojected detections on fisheye
SAVE_FISHEYE_DETECTIONS_VIZ = False

# ============================================================================
# DEBUGGING
# ============================================================================

VERBOSE = False  # Print detailed pipeline information
