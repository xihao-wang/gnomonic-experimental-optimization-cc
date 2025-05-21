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

SELECTED_PRESET = "nothing"

# Override the preset's image path (set to None to use the preset's default image)
CUSTOM_IMAGE = None  # Example: "path/to/custom_fisheye.png"

# ============================================================================
# CUSTOM CONFIGURATION (used when SELECTED_PRESET is None)
# ============================================================================

# Input image path
IMG_PATH = "imgs/fisheye.png"

# Number of projections to generate (must be even or have integer sqrt)
PROJ_NBR = 10

# Field of view in degrees
FOV_H = 48.0  # Horizontal field of view
FOV_V = 96.0  # Vertical field of view

# Camera positioning
LATITUDE = 45.0  # 0=nadir (looking straight down), 90=horizon
LON_0 = 0.0  # Starting longitude
LON_STEP = 36.6  # Longitude step between projections

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