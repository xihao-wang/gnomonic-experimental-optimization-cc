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
_C.PROJECTION.FOV_H = 75.0  # Horizontal
_C.PROJECTION.FOV_V = 90.0  # Vertical

# Camera positioning
_C.PROJECTION.LATITUDE = 38.0  # 0=nadir (straight down), 90=horizon
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
# REDUNDANT_BBOX_FILTER: Suppress redundant (cross-tile fragment) detections
# ============================================================================
# Two-stage filter that targets the cross-tile fragment problem (a person
# seen fully in one tile and truncated in an adjacent tile yields two bboxes
# that IoU-based NMS cannot deduplicate because their areas differ too much):
#
#   1. Composite-side (geometric): flag a YOLO detection whose bbox has a
#      side flush against an active tile border. Flagged detections are NOT
#      dropped here — they continue through backprojection and Stage-2 NMS.
#
#   2. Fisheye-side (containment confirmation): after Stage-2 Soft-NMS, drop
#      a still-flagged bbox only if there exists another fisheye bbox that
#      is (a) sufficiently larger (area ratio >= MIN_AREA_RATIO_TO_LARGER)
#      and (b) partially contains it (IoS >= MIN_OVERLAP_IOS).
#
# A real near-edge detection passes (1) but fails (2) -> kept. A genuine
# cross-tile fragment passes both -> dropped.
#
# Currently exposes one method (border-based). The nested layout leaves a
# clean slot for a future alternative method without restructuring.

_C.REDUNDANT_BBOX_FILTER = CN()

# ---- Method: border-based (two stages, each with its own ENABLED) ----
_C.REDUNDANT_BBOX_FILTER.BORDER_BASED = CN()

# ---- Stage 1: composite-side border flagging ----
_C.REDUNDANT_BBOX_FILTER.BORDER_BASED.FLAGGING = CN()

# Stage 1 on/off. When False, no bbox is ever flagged, so Stage 2 is moot
# (its ENABLED is ignored in that case).
_C.REDUNDANT_BBOX_FILTER.BORDER_BASED.FLAGGING.ENABLED = True

# Default per-tile side activation, picked from:
#   "internal_only"     -> only sides shared with a grid neighbour (auto-computed
#                          from PROJECTION.GRID). External composite edges off.
#                          Assumes grid-adjacent tiles are also fisheye-adjacent;
#                          this holds for uniform longitude stepping but may not
#                          hold when image_composer extra_projections is used —
#                          a warning is printed in that case.
#   "all_but_tile_top"  -> every side enabled except the top of every tile.
#                          Use when each tile's top edge corresponds to the
#                          fisheye outer ring (no neighbour above in fisheye
#                          space, so a top-flush bbox is a real near-edge
#                          detection that must be kept).
#   "all_but_tile_top_and_bottom"
#                       -> only the vertical sides (left/right) of every tile
#                          participate. Use when the bottom edge of each tile
#                          maps to the fisheye centre (nadir/zenith) and the
#                          top edge maps to the outer ring — neither can be
#                          a cross-tile-fragment cut.
#   "all"               -> every side of every tile enabled, including outer
#                          composite edges (most aggressive).
_C.REDUNDANT_BBOX_FILTER.BORDER_BASED.FLAGGING.PRESET = "all_but_tile_top"

# Per-tile overrides applied on top of the preset. Each entry is a 4-element
# list [row, col, side, enabled] where side ∈ {"top","right","bottom","left"}
# and enabled ∈ {True, False}. Example:
#   [[0, 1, "top", False],     # disable top edge of tile (row=0, col=1)
#    [1, 0, "left", False]]
_C.REDUNDANT_BBOX_FILTER.BORDER_BASED.FLAGGING.OVERRIDES = []

# Pixel tolerance for "bbox side coincides with tile border" check (composite
# pixels). A bbox is flagged if any of its sides is within this many pixels
# of an active tile border. Floats are allowed (e.g. 0.5, 1.0).
#
# Direction:
#   LARGER value -> wider acceptance window -> flags MORE bboxes (catches
#     duplicates whose side sits a few pixels off the cut, but also risks
#     flagging real near-edge detections).
#   SMALLER value -> narrower window -> flags FEWER bboxes (only those whose
#     side is almost exactly on the cut; conservative, may miss genuine
#     fragments whose side landed a pixel or two short).
#
# YOLO box sides are rarely exactly on the integer cut, so a tiny non-zero
# tolerance is needed. Practical range: 0.5–3 px depending on how often
# fragments survive vs. how often real detections get falsely flagged. The
# Stage-2 confirmation step is what catches the false flags introduced by a
# permissive tolerance — so leaning permissive here and relying on Stage 2
# is usually safe. When Stage 2 is disabled, lean conservative (small value).
_C.REDUNDANT_BBOX_FILTER.BORDER_BASED.FLAGGING.TOLERANCE_PX = 1.2

# Spatial gate restricting where a side may fire, expressed as the flagged
# side's vertical position inside the tile (0.0 = tile top, 1.0 = tile bottom).
# The flag fires only when the flagged side's Y fraction is >= this value.
# Use 0.5 to restrict dedup flagging to the lower half of a projection tile.
_C.REDUNDANT_BBOX_FILTER.BORDER_BASED.FLAGGING.MIN_SIDE_Y_FRACTION_IN_TILE = 0.0

# ---- Stage 2: fisheye-side containment confirmation (after Stage-2 Soft-NMS) ----
_C.REDUNDANT_BBOX_FILTER.BORDER_BASED.CONFIRMATION = CN()

# Stage 2 on/off. Only consulted when Stage 1 is enabled.
#   True  -> confirmation runs: a flagged bbox is dropped only if some other
#            fisheye bbox passes the area-ratio + IoS conditions below.
#   False -> confirmation is skipped and every flagged bbox is dropped
#            directly. Safe only when the projection setup guarantees that
#            every real person appears fully in at least one tile (e.g.
#            wide-enough FOV); otherwise real near-edge detections will be
#            lost. In this mode, the Stage-1 TOLERANCE_PX should be
#            conservative.
_C.REDUNDANT_BBOX_FILTER.BORDER_BASED.CONFIRMATION.ENABLED = False

# Confirmation parameters (ignored when CONFIRMATION.ENABLED is False).
# A flagged bbox A is dropped only when both of the following hold against
# *some other* fisheye bbox B:
#   area(B) / area(A) >= MIN_AREA_RATIO_TO_LARGER
#   IoS(B, A)         >= MIN_OVERLAP_IOS
#
# IoS stands for "Intersection over Smaller":
#   IoS(B, A) = |A ∩ B| / min(|A|, |B|).
# In short: IoS is just the raw intersection area normalised into [0, 1] by
# dividing by the smaller box's area — so the threshold is a scale-invariant
# fraction rather than a pixel count that would depend on bbox sizes and
# image resolution.
#
# Why IoS and not IoU here? The cross-tile fragment case is asymmetric by
# construction: A (the fragment) is much smaller than B (the full person).
# IoU = |A ∩ B| / |A ∪ B| underweights this overlap because the union is
# dominated by the large box, so IoU stays low even when A is fully inside
# B — which is exactly the failure mode that lets fragments survive vanilla
# NMS. IoS normalises by the smaller area instead, so a small box mostly
# contained in a larger one scores near 1.0 regardless of how mismatched
# the areas are. This is the same trick COCO uses for `iscrowd` matching.
#
# Tuning:
#   MIN_AREA_RATIO_TO_LARGER  ↑  -> stricter: B must be clearly bigger
#   MIN_OVERLAP_IOS           ↑  -> stricter: deeper containment required
_C.REDUNDANT_BBOX_FILTER.BORDER_BASED.CONFIRMATION.MIN_AREA_RATIO_TO_LARGER = 1.5
_C.REDUNDANT_BBOX_FILTER.BORDER_BASED.CONFIRMATION.MIN_OVERLAP_IOS = 0.6

# Save per-stage visualizations: composite (flagged vs unflagged) and fisheye
# (confirmed-drop vs kept) in the standalone pipeline.
_C.REDUNDANT_BBOX_FILTER.BORDER_BASED.SAVE_STAGE_VISUALS = True

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

# Save an additional fisheye visualization with each projection's FOV outline
# overlaid in a unique colour (thin line). Useful for diagnosing missed
# detections: by comparing the missed bbox location against projection
# boundaries you can see if the target falls in a low-coverage region.
_C.OUTPUT.SAVE_FISHEYE_PROJ_BORDERS = True
_C.OUTPUT.PROJ_BORDERS_THICKNESS = 1

# ============================================================================
# VIDEO: Video Processing Parameters
# ============================================================================

_C.VIDEO = CN()

# Input video path (relative to project root)
_C.VIDEO.INPUT_PATH = "detection_pipeline/videos/jewelry_store.mp4"

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
