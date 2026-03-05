"""
Configuration parameters for multi-persp.py

This file contains all configurable parameters for the multi-perspective
projection generation script. Edit these values as needed before running
the script.
"""

# ============================================================================
# PRESET SELECTION
# ============================================================================

# Set to None to use custom configuration below
# Or use a preset name: "default", "high_coverage", "horizon", "yolo_grid",
# "wide_angle", "high_res"

SELECTED_PRESET = None

# Override the preset's image path (set to None to use the preset's default image)
CUSTOM_IMAGE = None  # Example: "path/to/custom_fisheye.png"

# ============================================================================
# CUSTOM CONFIGURATION (used when SELECTED_PRESET is None)
# ============================================================================

# Input image path
IMG_PATH = "imgs/fisheye.png"

# Number of projections to generate (must be even or have integer sqrt)
PROJ_NBR = 8

# Field of view in degrees
FOV_H = 48.0  # Horizontal field of view
FOV_V = 96.0  # Vertical field of view

# Camera positioning
LATITUDE = 36.0  # 0=nadir (looking straight down), 90=horizon
LON_0 = 0.0  # Starting longitude
LON_STEP = 45  # Longitude step between projections

# Grid configuration - set to None for automatic determination
# Format: (rows, cols) or None
GRID = None  # e.g. (2, 3) for 2 rows and 3 columns

# Composite image size for YOLOv8 input
COMP_SZ = (640, 640)  # Width, height

# Target megapixels for each projection. If 'auto', projection width and height will match individual grid width and
# height after resize to COMP_SZ
TARGET_MP = 'auto'

# Output options
OUTPUT_DIR = "multi-persp-out"  # Base output directory

# ============================================================================
# PROJECTION PADDING
# ============================================================================
# Add black padding inside each projection cell to reduce aspect-ratio
# stretching (e.g. a 2×4 grid produces tall-narrow cells that compress
# pedestrians vertically; padding centres the content and avoids distortion).
#
# The projection content is rasterised at reduced resolution directly so that
# no extra remapping or rescaling is performed.  Requires TARGET_MP = 'auto'.
#
# PADDING_DIRECTION : "none" | "vertical" | "horizontal" | "both"
# PADDING_PCT       : fraction of the cell dimension to use as total black
#                     border (split evenly on both sides of that axis).
#                     Single float  for "vertical" or "horizontal".
#                     [v_pct, h_pct] list for "both".
#                     Example: 0.10 → 10% total → 5% black bar on each side.

PADDING_DIRECTION = "none"
PADDING_PCT = 0.10