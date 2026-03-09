"""
Entry point for cross-model configuration comparison.

Reads pre-computed per-config results from every model subfolder under
METRICS_EVALUATION.OUTPUT_DIR and compares all configurations side-by-side.
Each entry is labelled "{config_id} [{model_label}]" in all outputs.
No detection pipeline is ever re-executed.

Directory structure read:
    METRICS_EVALUATION.OUTPUT_DIR/
    ├── yolov8m/configs/{config_id}/{dataset}/metrics.json
    └── yolo12x/configs/{config_id}/{dataset}/metrics.json

Comparison outputs written to:
    METRICS_EVALUATION.OUTPUT_DIR/cross-model/comparison_{timestamp}/

Configuration:
    evaluation/lib/config.py  ->  CROSS_MODEL_COMPARATOR section
      - MODELS     : model labels to include (empty = all found)
      - CONFIG_IDS : original config IDs to include (empty = all found)
      - DATASETS   : datasets to compare
      - ENABLE_FIGURES

Usage:
    python evaluation/run_cross_model_comparator.py

Author: Yassir Zardoua
Email: y.zardoua@caplogy.com | yassirzardoua@gmail.com
Date: 2026-03-06
"""

import sys
from pathlib import Path

# Add project root to sys.path before any local imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.lib.config import get_cfg
from evaluation.lib.cross_model_comparator import CrossModelComparator


def main():
    """Load config and run the cross-model comparator."""
    cfg = get_cfg()
    me  = cfg.METRICS_EVALUATION
    cmc = cfg.CROSS_MODEL_COMPARATOR

    output_root        = Path(me.OUTPUT_DIR)
    comparison_out_dir = output_root / "cross-model"

    models     = list(cmc.MODELS)     or None
    config_ids = list(cmc.CONFIG_IDS) or None
    datasets   = list(cmc.DATASETS)   or None

    print("=" * 70)
    print("CROSS-MODEL CONFIGURATION COMPARATOR")
    print("=" * 70)
    print(f"Output root : {output_root}")
    print(f"Models      : {models or 'ALL found'}")
    print(f"Configs     : {config_ids or 'ALL found'}")
    print(f"Datasets    : {datasets}")
    print(f"Results dir : {comparison_out_dir}")
    print(f"Figures     : {cmc.ENABLE_FIGURES}")
    print("=" * 70)

    comparator = CrossModelComparator(
        output_root=str(output_root),
        comparison_out_dir=str(comparison_out_dir),
        models=models,
        config_ids=config_ids,
        datasets=datasets,
        enable_figures=cmc.ENABLE_FIGURES,
    )

    comparator.run()


if __name__ == "__main__":
    main()
