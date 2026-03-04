"""
Entry point for running metrics evaluation on projection configurations.

This script evaluates multiple projection configurations against ground truth
datasets and generates comprehensive metrics (precision, recall, F1, AP) with
timing measurements for fair comparison.

Usage:
    python evaluation/run_metrics_evaluation.py

Configuration:
    - Run parameters (datasets, model, max images, output dir):
        Edit evaluation/lib/config.py  ->  METRICS_EVALUATION section

    - Projection configurations to compare (the actual configs under test):
        Edit evaluation/projection_configs_for_metrics.py

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
        "bomni": cfg.DATASETS.BOMNI.ROOT_DIR,
        "piropo": cfg.DATASETS.PIROPO.ROOT_DIR,
        "cepdof": cfg.DATASETS.CEPDOF.ROOT_DIR
    }

    print("=" * 80)
    print("PROJECTION CONFIGURATION METRICS EVALUATION")
    print("=" * 80)
    print("Config file : evaluation/lib/config.py  (METRICS_EVALUATION section)")
    print(f"Proj configs: {cfg.METRICS_EVALUATION.PROJECTION_CONFIG_MODULE}")
    print(f"YOLO model  : {cfg.METRICS_EVALUATION.YOLO_MODEL}")
    print(f"Datasets    : {cfg.METRICS_EVALUATION.DATASETS}")
    print(f"Max images  : {cfg.METRICS_EVALUATION.MAX_IMAGES or 'All'}")
    print(f"Output      : {cfg.METRICS_EVALUATION.OUTPUT_DIR}")
    print("=" * 80)

    vis = cfg.METRICS_EVALUATION.VIS

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
        enable_visuals=vis.ENABLE,
        max_images=cfg.METRICS_EVALUATION.MAX_IMAGES,
        spread_samples=cfg.METRICS_EVALUATION.SPREAD_SAMPLES,
        vis_iou_threshold=vis.IOU_THRESHOLD,
        proj_boundary_colors=[tuple(c) for c in vis.PROJ_BOUNDARY_COLORS]
    )

    # Run evaluation
    runner.run_evaluation()


if __name__ == "__main__":
    main()
