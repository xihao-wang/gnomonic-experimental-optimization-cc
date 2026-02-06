"""
Backprojection visual validation test.

This script runs visual testing to verify that backprojected bounding boxes from
composite images to fisheye images are correctly aligned with ground truth.

PURPOSE: Visual validation only - generates images with GT and predicted bboxes overlaid.
This is NOT for computing evaluation metrics (precision, recall, mAP).

Edit this file to specify which projection configurations and datasets to test visually.
Then run: python evaluation/run_backprojection_visual_test.py

Author: Yassir Zardoua
Date: 2025-12-04
Updated: 2026-01-22 (Separated from evaluation metrics computation)
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configuration JSON file to use
CONFIG_JSON = "evaluation/projection_configs/test_configs_v1.json"

# Datasets to evaluate on (extensible list - designed to support multiple datasets)
DATASETS = ["bomni"]  # Only BOMNI for now, will extend later

# Dataset root paths (must specify root for each dataset in DATASETS list)
DATASET_ROOTS = {
    ""
    "bomni": "datasets/all-datasets/BOMNI-production"
    # Add other dataset roots when ready
}

# Verbosity
VERBOSE = True

# Maximum images to process per dataset (None = all images)
# Use a small number for testing, None for full evaluation
MAX_IMAGES = None  # Set to 5 for quick testing

if __name__ == "__main__":
    from evaluation.run_projection_evaluation import ProjectionEvaluator

    print("="*80)
    print("Backprojection Visual Validation Test")
    print("="*80)
    print("\nPURPOSE: Visual validation of backprojected bounding boxes")
    print("OUTPUT: Images with GT (green) and predicted (yellow) bboxes overlaid")
    print("NOTE: This does NOT compute evaluation metrics")
    print("\n" + "-"*80)
    print(f"Configuration file: {CONFIG_JSON}")
    print(f"Datasets: {', '.join(DATASETS)}")
    print(f"Verbose: {VERBOSE}")
    print(f"Max images per dataset: {MAX_IMAGES if MAX_IMAGES else 'All'}")
    print("="*80)
    print()

    # Create evaluator
    evaluator = ProjectionEvaluator(
        config_json_path=CONFIG_JSON,
        dataset_roots=DATASET_ROOTS,
        verbose=VERBOSE,
        max_images=MAX_IMAGES
    )

    # Run visual validation
    evaluator.run_evaluation(dataset_names=DATASETS)

    print("\n" + "="*80)
    print("Visual validation complete!")
    print("="*80)
