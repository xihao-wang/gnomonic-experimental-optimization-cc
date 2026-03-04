"""
Entry point for incremental per-configuration evaluation.

Evaluates each configuration listed in SINGLE_CONFIG_RUN.CONFIG_IDS and saves
results to evaluation/proj-conf-comparison/configs/{config_id}/ independently.

Each configuration run is fully independent: existing results for other
configurations are never touched. Configs whose folder already exists are
skipped automatically (set OVERWRITE_EXISTING=True to force re-evaluation).

Configuration:
    - Which configs to run:
        Edit evaluation/lib/config.py  ->  SINGLE_CONFIG_RUN.CONFIG_IDS  (list)

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
    """Load config, resolve each requested config ID, and run evaluations."""
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
    all_config_ids = [c["id"] for c in all_configs]

    # --- Validate all requested IDs up front ---------------------------------
    requested_ids = list(scr.CONFIG_IDS)
    if not requested_ids:
        print("ERROR: SINGLE_CONFIG_RUN.CONFIG_IDS is empty. Add at least one config ID.")
        sys.exit(1)

    missing = [cid for cid in requested_ids if cid not in all_config_ids]
    if missing:
        print(
            f"ERROR: The following CONFIG_IDS were not found in "
            f"{cfg.METRICS_EVALUATION.PROJECTION_CONFIG_MODULE}:\n"
            f"  {missing}\n"
            f"Available IDs: {all_config_ids}"
        )
        sys.exit(1)

    # --- Print run summary ---------------------------------------------------
    print("=" * 70)
    print("INCREMENTAL MULTI-CONFIG EVALUATION")
    print("=" * 70)
    print(f"Configs     : {requested_ids}")
    print(f"YOLO model  : {scr.YOLO_MODEL}")
    print(f"Datasets    : {list(scr.DATASETS)}")
    print(f"Max images  : {scr.MAX_IMAGES or 'All'} per dataset")
    print(f"Spread smpl : {scr.SPREAD_SAMPLES}")
    print(f"Overwrite   : {scr.OVERWRITE_EXISTING}")
    print(f"Output dir  : {scr.OUTPUT_DIR}")
    print("=" * 70)

    # --- Run each config in sequence -----------------------------------------
    configs_to_run = [c for c in all_configs if c["id"] in requested_ids]
    # Preserve the order from CONFIG_IDS
    configs_to_run.sort(key=lambda c: requested_ids.index(c["id"]))

    ran, skipped = 0, 0
    for config in configs_to_run:
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
        did_run = runner.run()
        if did_run:
            ran += 1
        else:
            skipped += 1

    print("\n" + "=" * 70)
    print(f"DONE  —  {ran} evaluated, {skipped} skipped (already exist).")
    print("=" * 70)


if __name__ == "__main__":
    main()
