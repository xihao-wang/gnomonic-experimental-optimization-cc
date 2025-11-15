"""
Configuration for configuration search and comparison.

This module searches for the best projection configuration across:
- Different numbers of projections (2x2, 2x3, 3x3, 4x4, etc.)
- Different field of view settings
- Different camera angles/latitudes
"""

# ============================================================================
# SEARCH STRATEGY
# ============================================================================

# Search algorithm: "grid", "random", "bayesian", "genetic"
SEARCH_STRATEGY = "grid"

# ============================================================================
# GRID SEARCH PARAMETERS
# ============================================================================

# If using grid search, define the parameter space

# Number of projections to try (as factors: 4=2x2, 6=2x3, 8=2x4, 9=3x3, 12=3x4, 16=4x4)
PROJECTION_COUNTS = [4, 6, 8, 9, 12, 16]

# Horizontal field of view options (degrees)
FOV_H_OPTIONS = [40.0, 48.0, 60.0, 75.0, 90.0]

# Vertical field of view options (degrees)
FOV_V_OPTIONS = [60.0, 96.0, 120.0, 150.0, 180.0]

# Camera latitude (0=nadir/straight down, 90=horizon)
LATITUDE_OPTIONS = [0.0, 18.0, 36.0, 45.0, 60.0, 90.0]

# ============================================================================
# CONSTRAINTS
# ============================================================================

# Optional: Relationships between parameters
# E.g., ensure FOV_V >= FOV_H for proper coverage
CONSTRAINT_FOV_RATIO_MIN = 0.5  # FOV_V >= FOV_H * ratio

# ============================================================================
# SEARCH OUTPUT
# ============================================================================

# How many top configurations to keep
TOP_N_CONFIGS = 10

# Save all configurations tried and their scores
SAVE_SEARCH_LOG = True

# ============================================================================
# PARALLELIZATION
# ============================================================================

# Number of parallel jobs for evaluation
# -1 = use all CPU cores, 1 = sequential
N_JOBS = 1

# ============================================================================
# DATASETS FOR SEARCH
# ============================================================================

# Which datasets to evaluate on during search
# (may be different from full evaluation)
SEARCH_DATASETS = ["piropo"]  # Can be ["bomni"], ["piropo"], or ["bomni", "piropo"]

# Use subset of each dataset for faster search
USE_SUBSET = True
SUBSET_SIZE = 50  # Number of images per dataset

# ============================================================================
# METRIC FOR RANKING
# ============================================================================

# Which metric to optimize for
OPTIMIZATION_METRIC = "map"  # "map", "ap50", "f1"

# Minimize or maximize?
OPTIMIZATION_MODE = "maximize"  # "maximize" or "minimize"
