"""
Main entry point for the pedestrian detection pipeline.

This script orchestrates the entire workflow:
1. Load dataset (BOMNI, PIROPO, etc.)
2. For each image & projection configuration:
   - Create composite image using image-composer
   - Run YOLO detection on composite
   - Backproject results to fisheye coordinates
3. Evaluate and store metrics
4. Compare configurations to find best one

Usage:
    python main.py

Configuration:
    Edit project_config.py and component-specific config files (in each module folder)
    before running.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import project_config


def main():
    """Main execution entry point."""

    print("=" * 80)
    print("Pedestrian Detection in Fisheye Images")
    print("Configuration Search & Projection-Based Detection Pipeline")
    print("=" * 80)
    print()

    print("Project Structure:")
    print(f"  Root: {project_config.PROJECT_ROOT}")
    print(f"  Datasets: {project_config.DATASETS_DIR}")
    print(f"  Results: {project_config.RESULTS_DIR}")
    print(f"  Models: {project_config.MODELS_DIR}")
    print()

    print("Next steps:")
    print("  1. Configure project_config.py (dataset paths, default dataset)")
    print("  2. Configure detection-pipeline/config.py (pipeline parameters)")
    print("  3. Configure datasets/config.py (dataset-specific paths)")
    print("  4. Configure evaluation/config.py (evaluation metrics)")
    print("  5. Configure config_search/config.py (search parameters)")
    print()
    print("Then implement the pipeline modules and run experiments.")
    print()


if __name__ == "__main__":
    main()
