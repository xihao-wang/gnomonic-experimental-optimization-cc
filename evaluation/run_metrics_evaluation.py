"""
Entry point for running metrics evaluation on projection configurations.

This script evaluates multiple projection configurations against ground truth
datasets and generates comprehensive metrics (precision, recall, F1, AP) with
timing measurements for fair comparison.

Usage:
    python evaluation/run_metrics_evaluation.py

Configuration:
    Edit evaluation/lib/config.py METRICS_EVALUATION section to configure all parameters.

Author: Yassir Zardoua
Email: y.zardoua@caplogy.com | yassirzardoua@gmail.com
Date: 2026-02-12
"""

import sys
from pathlib import Path

# Add project root to sys.path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.lib.config import get_cfg
from evaluation.lib.metrics_evaluator_runner import MetricsEvaluatorRunner


def main():
    """
    Main entry point for metrics evaluation.

    All configuration is loaded from evaluation/lib/config.py METRICS_EVALUATION section.
    """
    # Load configuration
    cfg = get_cfg()

    # Dataset roots mapping
    dataset_roots = {
        "bomni": cfg.DATASETS.BOMNI.ROOT_DIR
    }

    print("=" * 80)
    print("PROJECTION CONFIGURATION METRICS EVALUATION")
    print("=" * 80)
    print(f"Projection configs: {cfg.METRICS_EVALUATION.PROJECTION_CONFIG_MODULE}")
    print(f"YOLO model: {cfg.METRICS_EVALUATION.YOLO_MODEL}")
    print(f"Datasets: {cfg.METRICS_EVALUATION.DATASETS}")
    print(f"Max images: {cfg.METRICS_EVALUATION.MAX_IMAGES or 'All'}")
    print(f"Output: {cfg.METRICS_EVALUATION.OUTPUT_DIR}")
    print("=" * 80)

    # Create metrics evaluator runner
    runner = MetricsEvaluatorRunner(
        projection_config_module=cfg.METRICS_EVALUATION.PROJECTION_CONFIG_MODULE,
        yolo_model=cfg.METRICS_EVALUATION.YOLO_MODEL,
        datasets=cfg.METRICS_EVALUATION.DATASETS,
        dataset_roots=dataset_roots,
        iou_thresholds=cfg.METRICS_EVALUATION.IOU_THRESHOLDS,
        output_dir=cfg.METRICS_EVALUATION.OUTPUT_DIR,
        enable_timing=cfg.METRICS_EVALUATION.ENABLE_TIMING,
        enable_pr_curves=cfg.METRICS_EVALUATION.ENABLE_PR_CURVES,
        max_images=cfg.METRICS_EVALUATION.MAX_IMAGES
    )

    # Run evaluation
    runner.run_evaluation()


if __name__ == "__main__":
    main()
