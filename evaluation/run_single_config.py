"""
Entry point for incremental per-configuration evaluation.

Evaluates each configuration listed in
METRICS_EVALUATION.SINGLE_CONFIG_RUN.CONFIG_IDS and saves results to
evaluation/proj-conf-comparison/configs/{config_id}/ independently.

Configs whose folder already exists are skipped automatically
(set OVERWRITE_EXISTING=True to force re-evaluation).

Configuration:
    - Which configs to run:
        evaluation/lib/config.py  ->  METRICS_EVALUATION.SINGLE_CONFIG_RUN.CONFIG_IDS

    - Which configs exist (id, name, projection params):
        evaluation/projection_configs_for_metrics.py

    - All other run parameters (datasets, model, max images, visuals, etc.):
        evaluation/lib/config.py  ->  METRICS_EVALUATION section

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
    me  = cfg.METRICS_EVALUATION
    scr = cfg.METRICS_EVALUATION.SINGLE_CONFIG_RUN
    vis = cfg.METRICS_EVALUATION.VIS

    # --- Resolve dataset roots -----------------------------------------------
    dataset_roots = {
        "bomni": cfg.DATASETS.BOMNI.ROOT_DIR,
        "piropo": cfg.DATASETS.PIROPO.ROOT_DIR,
        "cepdof": cfg.DATASETS.CEPDOF.ROOT_DIR
    }

    # --- Load projection configurations from module --------------------------
    proj_module = importlib.import_module(me.PROJECTION_CONFIG_MODULE)
    all_configs = proj_module.ProjectionConfigs.CONFIGS
    all_config_ids = [c["id"] for c in all_configs]

    # --- Resolve requested IDs (empty list = run all) ------------------------
    requested_ids = list(scr.CONFIG_IDS) if scr.CONFIG_IDS else all_config_ids

    missing = [cid for cid in requested_ids if cid not in all_config_ids]
    if missing:
        print(
            f"ERROR: The following CONFIG_IDS were not found in "
            f"{me.PROJECTION_CONFIG_MODULE}:\n"
            f"  {missing}\n"
            f"Available IDs: {all_config_ids}"
        )
        sys.exit(1)

    # Per-config results are stored under OUTPUT_DIR/<model_label>/configs/
    model_label = Path(me.YOLO_MODEL).stem
    configs_output_dir = str(Path(me.OUTPUT_DIR) / model_label / "configs")

    # --- Print run summary ---------------------------------------------------
    print("=" * 70)
    print("INCREMENTAL MULTI-CONFIG EVALUATION")
    print("=" * 70)
    print(f"Configs     : {'ALL' if not scr.CONFIG_IDS else requested_ids}")
    print(f"YOLO model  : {me.YOLO_MODEL}")
    print(f"Datasets    : {list(me.DATASETS)}")
    print(f"Max images  : {me.MAX_IMAGES or 'All'} per dataset")
    print(f"Spread smpl : {me.SPREAD_SAMPLES}")
    print(f"Overwrite   : {scr.OVERWRITE_EXISTING}")
    print(f"Visuals     : {vis.ENABLE}  (IoU threshold: {vis.IOU_THRESHOLD})")
    print(f"Vis samples : BOMNI={vis.MAX_SAMPLES.BOMNI or 'all'}  "
          f"PIROPO={vis.MAX_SAMPLES.PIROPO or 'all'}  "
          f"CEPDOF={vis.MAX_SAMPLES.CEPDOF or 'all'}")
    print(f"Model label : {model_label}")
    print(f"Output dir  : {configs_output_dir}")
    print("=" * 70)

    # --- Run each config in sequence -----------------------------------------
    configs_to_run = [c for c in all_configs if c["id"] in requested_ids]
    configs_to_run.sort(key=lambda c: requested_ids.index(c["id"]))

    ran, skipped = 0, 0
    for config in configs_to_run:
        runner = SingleConfigRunner(
            config=config,
            yolo_model=me.YOLO_MODEL,
            datasets=list(me.DATASETS),
            dataset_roots=dataset_roots,
            iou_thresholds=list(me.IOU_THRESHOLDS),
            output_dir=configs_output_dir,
            enable_timing=me.ENABLE_TIMING,
            enable_pr_curves=me.ENABLE_PR_CURVES,
            enable_visuals=vis.ENABLE,
            max_images=me.MAX_IMAGES,
            spread_samples=me.SPREAD_SAMPLES,
            overwrite_existing=scr.OVERWRITE_EXISTING,
            vis_iou_threshold=vis.IOU_THRESHOLD,
            proj_boundary_colors=[tuple(c) for c in vis.PROJ_BOUNDARY_COLORS],
            vis_max_samples={
                "bomni":  vis.MAX_SAMPLES.BOMNI,
                "piropo": vis.MAX_SAMPLES.PIROPO,
                "cepdof": vis.MAX_SAMPLES.CEPDOF
            },
            vis_spread_samples=vis.SPREAD_SAMPLES
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
