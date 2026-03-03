"""
Entry point for incremental per-configuration evaluation.

Evaluates ONE projection configuration on the requested datasets and saves
all results to evaluation/proj-conf-comparison/configs/{config_id}/.

Each configuration run is fully independent: existing results for other
configurations are never touched. Re-running the same config ID is safe
(skips by default; set OVERWRITE_EXISTING=True to force re-evaluation).

Configuration:
    - Which config to run:
        Edit evaluation/lib/config.py  ->  SINGLE_CONFIG_RUN.CONFIG_ID

    - Which configs exist (id, name, projection params):
        Edit evaluation/projection_configs_for_metrics.py

    - All other run parameters (datasets, max images, output dir, etc.):
        Edit evaluation/lib/config.py  ->  SINGLE_CONFIG_RUN section

Usage:
    python evaluation/run_single_config.py

Author: Yassir Zardoua
Email: y.zardoua@caplogy.com | yassirzardoua@gmail.com
Date: 2026-03-03
"""

import sys
import importlib
from pathlib import Path

# Add project root to sys.path before any local imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.lib.config import get_cfg
from evaluation.lib.single_config_runner import SingleConfigRunner


def main():
    """Load config, locate the requested projection config, and run evaluation."""
    cfg = get_cfg()
    scr = cfg.SINGLE_CONFIG_RUN

    # --- Resolve dataset roots -----------------------------------------------
    dataset_roots = {
        "bomni": cfg.DATASETS.BOMNI.ROOT_DIR,
        "piropo": cfg.DATASETS.PIROPO.ROOT_DIR,
        "cepdof": cfg.DATASETS.CEPDOF.ROOT_DIR
    }

    # --- Load projection configurations from module --------------------------
    proj_module = importlib.import_module(
        cfg.METRICS_EVALUATION.PROJECTION_CONFIG_MODULE
    )
    all_configs = proj_module.ProjectionConfigs.CONFIGS

    # Locate the requested config ID
    config_id = scr.CONFIG_ID
    matched = [c for c in all_configs if c["id"] == config_id]
    if not matched:
        available = [c["id"] for c in all_configs]
        print(
            f"ERROR: CONFIG_ID '{config_id}' not found in "
            f"{cfg.METRICS_EVALUATION.PROJECTION_CONFIG_MODULE}.\n"
            f"Available IDs: {available}"
        )
        sys.exit(1)

    config = matched[0]

    # --- Print run summary ---------------------------------------------------
    print("=" * 70)
    print("INCREMENTAL SINGLE-CONFIG EVALUATION")
    print("=" * 70)
    print(f"Config ID   : {config['id']}")
    print(f"Config name : {config['name']}")
    print(f"YOLO model  : {scr.YOLO_MODEL}")
    print(f"Datasets    : {list(scr.DATASETS)}")
    print(f"Max images  : {scr.MAX_IMAGES or 'All'} per dataset")
    print(f"Spread smpl : {scr.SPREAD_SAMPLES}")
    print(f"Overwrite   : {scr.OVERWRITE_EXISTING}")
    print(f"Output dir  : {scr.OUTPUT_DIR}")
    print("=" * 70)

    # --- Run evaluation -------------------------------------------------------
    runner = SingleConfigRunner(
        config=config,
        yolo_model=scr.YOLO_MODEL,
        datasets=list(scr.DATASETS),
        dataset_roots=dataset_roots,
        iou_thresholds=list(scr.IOU_THRESHOLDS),
        output_dir=scr.OUTPUT_DIR,
        enable_timing=scr.ENABLE_TIMING,
        enable_pr_curves=scr.ENABLE_PR_CURVES,
        enable_visuals=scr.ENABLE_VISUALS,
        max_images=scr.MAX_IMAGES,
        spread_samples=scr.SPREAD_SAMPLES,
        overwrite_existing=scr.OVERWRITE_EXISTING,
        vis_iou_threshold=scr.VIS_IOU_THRESHOLD
    )

    runner.run()


if __name__ == "__main__":
    main()
