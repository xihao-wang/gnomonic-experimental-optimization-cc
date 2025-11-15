"""
Configuration for evaluation metrics and performance analysis.

Defines which metrics to compute and how to compare results.
"""

# ============================================================================
# EVALUATION METRICS
# ============================================================================

# Standard COCO/YOLO metrics to compute
COMPUTE_METRICS = {
    "ap": True,  # Average Precision at IoU=0.50:0.95
    "ap50": True,  # Average Precision at IoU=0.50
    "ap75": True,  # Average Precision at IoU=0.75
    "map": True,  # Mean Average Precision (class-agnostic, single class in our case)
    "ar": True,  # Average Recall
    "f1": True,  # F1 score
}

# IoU thresholds for evaluation
IoU_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

# ============================================================================
# BACKPROJECTION EVALUATION
# ============================================================================

# When evaluating detections that have been backprojected to fisheye:
# - Fisheye ground truth should be available
# - Or we can evaluate on composite and estimate fisheye performance

# Compare at which stage?
EVALUATION_STAGE = "fisheye"  # "composite", "fisheye", or "both"

# Handling occluded/partial detections in fisheye
# (e.g., detections that go off the edge during projection)
HANDLE_PARTIAL_DETECTIONS = True
PARTIAL_DETECTION_PENALTY = 0.1  # Reduce score for partial detections

# ============================================================================
# RESULT STORAGE
# ============================================================================

# Store detailed per-image results
STORE_PER_IMAGE_RESULTS = True

# Store visualization of evaluations
STORE_EVALUATION_VISUALIZATIONS = True

# Store raw detections (not just metrics)
STORE_RAW_DETECTIONS = True

# ============================================================================
# COMPARISON METRICS
# ============================================================================

# When comparing configurations, which metric to prioritize?
PRIMARY_METRIC = "map"  # "map", "ap50", "f1", etc.

# Also report these secondary metrics
SECONDARY_METRICS = ["ap50", "ar", "f1"]

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

# Compute standard deviation/confidence intervals?
COMPUTE_CONFIDENCE_INTERVALS = True
CONFIDENCE_LEVEL = 0.95
