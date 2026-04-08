"""
Entry point for incremental per-configuration evaluation.

Evaluates each configuration listed in SINGLE_CONFIG_RUN.CONFIG_IDS across
one or more YOLO models. If MULTI_MODEL_RUN.MODELS is non-empty, iterates
over those models; otherwise uses METRICS_EVALUATION.YOLO_MODEL.

Results are stored under OUTPUT_DIR/<model_label>/configs/<config_id>/.
Configs whose folder already exists are skipped (set OVERWRITE_EXISTING=True
to force re-evaluation).

Configuration:
    evaluation/lib/config.py  ->  METRICS_EVALUATION section

Author: Yassir Zardoua
Email: y.zardoua@caplogy.com | yassirzardoua@gmail.com
Date: 2026-03-03
"""

import sys
import shutil
import importlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.lib.config import get_cfg, YOLO_IMGSZ_OPTIONS
from evaluation.lib.single_config_runner import SingleConfigRunner


def _ensure_model(model_path: Path) -> bool:
    """
    Ensure a YOLO model .pt file exists at model_path.
    If missing, downloads it via ultralytics (using the filename as the hub key)
    and copies the result into model_path.

    Returns True if the model is ready, False if download failed.
    """
    if model_path.exists():
        return True

    model_filename = model_path.name
    print(f"\n[DOWNLOAD] {model_filename} not found — downloading via ultralytics hub...")
    try:
        from ultralytics import YOLO as _YOLO
        tmp = _YOLO(model_filename)   # downloads to ultralytics cache
        cached = Path(tmp.ckpt_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, model_path)
        print(f"[DOWNLOAD] Saved to {model_path}")
        return True
    except Exception as e:
        print(f"[SKIP] Could not download {model_filename}: {e}")
        return False


def main():
    cfg = get_cfg()
    me  = cfg.METRICS_EVALUATION
    scr = cfg.METRICS_EVALUATION.SINGLE_CONFIG_RUN
    mmr = cfg.METRICS_EVALUATION.MULTI_MODEL_RUN
    vis = cfg.METRICS_EVALUATION.VIS
    yolo_imgsz = YOLO_IMGSZ_OPTIONS[me.YOLO_IMGSZ]
    project_root = Path(__file__).parent.parent

    dataset_roots = {
        "bomni":  cfg.DATASETS.BOMNI.ROOT_DIR,
        "piropo": cfg.DATASETS.PIROPO.ROOT_DIR,
        "cepdof": cfg.DATASETS.CEPDOF.ROOT_DIR,
    }

    proj_module    = importlib.import_module(me.PROJECTION_CONFIG_MODULE)
    all_configs    = proj_module.ProjectionConfigs.CONFIGS
    all_config_ids = [c["id"] for c in all_configs]

    requested_ids = list(scr.CONFIG_IDS) if scr.CONFIG_IDS else all_config_ids
    missing = [cid for cid in requested_ids if cid not in all_config_ids]
    if missing:
        print(
            f"ERROR: CONFIG_IDS not found in {me.PROJECTION_CONFIG_MODULE}:\n"
            f"  {missing}\nAvailable: {all_config_ids}"
        )
        sys.exit(1)

    configs_to_run = [c for c in all_configs if c["id"] in requested_ids]
    configs_to_run.sort(key=lambda c: requested_ids.index(c["id"]))
    n_configs = len(configs_to_run)

    # Multi-model: use MULTI_MODEL_RUN.MODELS if set, else fall back to YOLO_MODEL
    model_filenames = list(mmr.MODELS) if mmr.MODELS else [me.YOLO_MODEL]
    n_models = len(model_filenames)

    print("=" * 70)
    print("INCREMENTAL MULTI-CONFIG EVALUATION")
    print("=" * 70)
    print(f"Models      : {model_filenames}")
    print(f"Configs     : {'ALL' if not scr.CONFIG_IDS else requested_ids}")
    print(f"Datasets    : {list(me.DATASETS)}")
    print(f"Max images  : {me.MAX_IMAGES or 'All'} per dataset")
    print(f"Imgsz       : {me.YOLO_IMGSZ} -> {yolo_imgsz}px")
    print(f"Overwrite   : {scr.OVERWRITE_EXISTING}")
    print(f"Output root : {me.OUTPUT_DIR}")
    print("=" * 70)

    total_ran = 0
    total_skipped = 0

    for m_idx, model_filename in enumerate(model_filenames, start=1):
        model_path = project_root / "models" / model_filename
        if not _ensure_model(model_path):
            continue
        model_path  = str(model_path)
        model_label = Path(model_filename).stem
        configs_output_dir = str(Path(me.OUTPUT_DIR) / model_label / "configs")

        print(f"\n{'=' * 70}")
        print(f"MODEL {m_idx}/{n_models}: {model_label}")
        print(f"{'=' * 70}")

        for c_idx, config in enumerate(configs_to_run, start=1):
            runner = SingleConfigRunner(
                config=config,
                yolo_model=model_path,
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
                    "cepdof": vis.MAX_SAMPLES.CEPDOF,
                },
                vis_spread_samples=vis.SPREAD_SAMPLES,
                yolo_imgsz=yolo_imgsz,
                model_idx=m_idx,
                model_total=n_models,
                config_idx=c_idx,
                config_total=n_configs,
            )
            did_run = runner.run()
            if did_run:
                total_ran += 1
            else:
                total_skipped += 1

    print("\n" + "=" * 70)
    print(f"DONE  —  {total_ran} evaluated, {total_skipped} skipped.")
    print("=" * 70)


if __name__ == "__main__":
    main()
