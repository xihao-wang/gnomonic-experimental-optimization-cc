"""
Configuration for the tracker pipeline.

All user-editable parameters are defined here as module-level constants.
Application code imports from this file — values must not be hardcoded elsewhere.

Run from project root:
    python tracker_pipeline/process_video.py

Author: Yassir Zardoua — y.zardoua@caplogy.com | yassirzardoua@gmail.com
"""

# ============================================================================
# INPUT / OUTPUT
# ============================================================================

# Path to the input fisheye video (relative to project root)
INPUT_VIDEO = "detection_pipeline/videos/Meeting1.mp4"

# Base output directory (relative to project root)
# Session sub-directories are created automatically:
#   {OUTPUT_DIR}/{video_stem}_session_N/
OUTPUT_DIR = "tracker_pipeline/results"

# Save an additional composite-view video (YOLO detections, no tracking)
SAVE_COMPOSITE_VIDEO = False

# ============================================================================
# PROJECTION CONFIGURATION
# ============================================================================

# Preset name for gnomonic projection ("yolo_grid", "wide_angle", "default", etc.)
# Set to None to use the manual parameters below.
PROJECTION_PRESET = None  # using manual config below (TEST#26)

# ---- Manual projection parameters (only used when PROJECTION_PRESET = None) ----
PROJ_NBR   = 6
FOV_H      = 60.0         # degrees — horizontal field of view
FOV_V      = 85.0         # degrees — vertical field of view
LATITUDE   = 45.0         # degrees — 0 = nadir (straight down), 90 = horizon
LON_0      = 0.0          # degrees — starting longitude
LON_STEP   = 360.0 / 6    # degrees — longitude step between projections
GRID       = [2, 3]       # [rows, cols]
COMP_SIZE  = [640, 640]   # [width, height] of composite image
TARGET_MP  = "auto"       # megapixels per projection cell ("auto" recommended)

# ============================================================================
# YOLO DETECTION CONFIGURATION
# ============================================================================

# YOLO model file path relative to project root
YOLO_MODEL = "models/yolov9e.pt"

# Device: None = auto-detect, "cuda" = GPU, "cpu" = CPU
YOLO_DEVICE = None

# Confidence threshold for initial YOLO detection filter
YOLO_CONFIDENCE = 0.25

# ============================================================================
# NMS CONFIGURATION
# ============================================================================

# Stage 1: Standard NMS on composite image (before backprojection)
NMS_STAGE1_ENABLED       = True
NMS_STAGE1_IOU_THRESHOLD = 0.8    # high threshold: keeps overlapping boxes for stage 2

# Stage 2: Soft-NMS on fisheye image (after backprojection)
NMS_STAGE2_ENABLED         = True
NMS_STAGE2_SIGMA           = 0.2  # Gaussian decay: 0.1 = aggressive, 0.4 = gentle
NMS_STAGE2_SCORE_THRESHOLD = 0.3  # remove heavily penalized duplicates

# ============================================================================
# TRACKER CONFIGURATION (ByteTrack)
# ============================================================================

# Detections at or above this confidence threshold are used in stage 1 association
TRACK_HIGH_CONF = 0.5

# Detections below TRACK_HIGH_CONF but at or above this threshold go to stage 2
# Must be <= YOLO_CONFIDENCE (detections below YOLO_CONFIDENCE never reach tracker)
TRACK_LOW_CONF = 0.1

# Number of frames a lost track is kept alive using Kalman prediction
# before being permanently removed
MAX_AGE = 30

# Minimum number of consecutive matched frames before a track is displayed
MIN_HITS = 3

# IoU threshold for matching tracks to detections
IOU_THRESHOLD = 0.3

# Minimum axis-aligned bounding box area (pixels squared) to accept a detection
# Filters out very small spurious detections in the fisheye periphery
MIN_BOX_AREA = 100.0

# ============================================================================
# VISUALIZATION
# ============================================================================

# Color palette for track IDs — cycles automatically for IDs > len(TRACK_COLORS)
# Colors in BGR format
TRACK_COLORS = [
    (  0, 255,   0),  # green
    (255,   0,   0),  # blue
    (  0,   0, 255),  # red
    (  0, 255, 255),  # yellow
    (255,   0, 255),  # magenta
    (255, 128,   0),  # orange
    (128,   0, 255),  # purple
    (  0, 128, 255),  # sky blue
    (  0, 200, 100),  # teal
    (200, 200,   0),  # olive
]

# Bounding box line thickness
BBOX_THICKNESS = 2

# Show track ID label on each box
SHOW_IDS = True

# Show confidence score alongside the ID label
SHOW_CONFIDENCE = True

# Draw a centroid dot at the center of each track
SHOW_CENTROID = True

# Number of past centroid positions to draw as a motion trail (0 = disabled)
TRAIL_LENGTH = 30

# ============================================================================
# PROCESSING OPTIONS
# ============================================================================

# Process every Nth frame (1 = all frames)
PROCESS_EVERY_N_FRAMES = 1

# Maximum frames to process (None = entire video)
MAX_FRAMES = None

# Print progress every N processed frames (None = silent)
LOG_EVERY_N_FRAMES = 50

# Show a real-time preview window (press Q to quit early)
DISPLAY_FRAMES = True
