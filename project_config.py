"""
Global project configuration for pedestrian detection in fisheye images.

This file contains project-wide settings that apply across all modules:
- Dataset paths and default dataset
- Model paths
- Output directories
- Global parameters

Component-specific configurations are in their respective config.py files:
- detection_pipeline/config.py - Pipeline parameters
- datasets/config.py - Dataset paths
- evaluation/config.py - Evaluation metrics
- config_search/config.py - Search parameters
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results"
DATASETS_DIR = PROJECT_ROOT / "datasets" / "data"
MODELS_DIR = PROJECT_ROOT / "detection-pipeline" / "models"

# Create directories if they don't exist
RESULTS_DIR.mkdir(exist_ok=True)
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DEFAULT DATASET & MODELS
# ============================================================================

# Default dataset to use for experiments: "bomni", "piropo", "custom"
DEFAULT_DATASET = "piropo"

# YOLO model to use (should be in detection-pipeline/models/)
YOLO_MODEL = "yolov8n.pt"  # nano, small, medium, large

# ============================================================================
# SUPPORTED DATASETS
# ============================================================================

SUPPORTED_DATASETS = ["bomni", "piropo"]

# Dataset configuration will be loaded from datasets/config.py

# ============================================================================
# RESULTS OUTPUT
# ============================================================================

# Metrics will be saved in results/metrics/
METRICS_DIR = RESULTS_DIR / "metrics"
METRICS_DIR.mkdir(exist_ok=True)

# Visualizations in results/visualizations/
VIZ_DIR = RESULTS_DIR / "visualizations"
VIZ_DIR.mkdir(exist_ok=True)

# Comparison reports in results/reports/
REPORTS_DIR = RESULTS_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE = PROJECT_ROOT / "project.log"

# ============================================================================
# DEBUG MODE
# ============================================================================

DEBUG = False  # Set to True for verbose output and intermediate visualizations
