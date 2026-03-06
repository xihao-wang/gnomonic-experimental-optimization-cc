"""
Entry point for comparing pre-computed per-configuration evaluation results.

Reads metrics.json / timing.json files written by run_single_config.py and
produces comparison tables, PR overlay figures, and a winner determination.
No detection pipeline is ever re-executed.

Configuration:
    - Which configs to compare (empty = all found automatically):
        Edit evaluation/lib/config.py  ->  COMPARATOR.CONFIG_IDS

    - Where pre-computed results live and where comparisons are written:
        Derived automatically from:
          METRICS_EVALUATION.OUTPUT_DIR / <model_label> / "configs"
          METRICS_EVALUATION.OUTPUT_DIR / <model_label> / "comparisons"
        where <model_label> = Path(METRICS_EVALUATION.YOLO_MODEL).stem

Usage:
    python evaluation/run_comparator.py

Author: Yassir Zardoua
Email: y.zardoua@caplogy.com | yassirzardoua@gmail.com
Date: 2026-03-03
"""

import sys
from pathlib import Path

# Add project root to sys.path before any local imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.lib.config import get_cfg
from evaluation.lib.comparator import ConfigComparator


def main():
    """Load config and run the comparator."""
    cfg = get_cfg()
    me  = cfg.METRICS_EVALUATION
    cmp = cfg.COMPARATOR

    # Derive directories from METRICS_EVALUATION so they always stay in sync
    # with run_single_config.py (same model label, same root).
    model_label = Path(me.YOLO_MODEL).stem
    configs_dir = Path(me.OUTPUT_DIR) / model_label / "configs"
    comparison_out_dir = Path(me.OUTPUT_DIR) / model_label / "comparisons"

    config_ids = list(cmp.CONFIG_IDS)  # empty list = compare all found

    print("=" * 70)
    print("CONFIGURATION COMPARATOR")
    print("=" * 70)
    if config_ids:
        print(f"Comparing   : {config_ids}")
    else:
        print("Comparing   : ALL configs found in results dir (auto-discover)")
    print(f"Datasets    : {list(cmp.DATASETS)}")
    print(f"Model label : {model_label}")
    print(f"Results dir : {configs_dir}")
    print(f"Output dir  : {comparison_out_dir}")
    print(f"Figures     : {cmp.ENABLE_FIGURES}")
    print("=" * 70)

    comparator = ConfigComparator(
        configs_dir=str(configs_dir),
        comparison_out_dir=str(comparison_out_dir),
        config_ids=config_ids if config_ids else None,
        datasets=list(cmp.DATASETS) if cmp.DATASETS else None,
        enable_figures=cmp.ENABLE_FIGURES
    )

    comparator.run()


if __name__ == "__main__":
    main()
