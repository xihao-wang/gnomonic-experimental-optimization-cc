"""
Entry point for comparing pre-computed per-configuration evaluation results.

Reads metrics.json / timing.json files written by run_single_config.py and
produces comparison tables, PR overlay figures, and a winner determination.
No detection pipeline is ever re-executed.

Configuration:
    - Which configs to compare (empty = all found automatically):
        Edit evaluation/lib/config.py  ->  COMPARATOR.CONFIG_IDS

    - Where pre-computed results live:
        evaluation/lib/config.py  ->  COMPARATOR.OUTPUT_DIR
        (must match SINGLE_CONFIG_RUN.OUTPUT_DIR)

    - Comparison outputs are written to:
        evaluation/lib/config.py  ->  COMPARATOR.COMPARISON_OUT_DIR

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
    cmp = cfg.COMPARATOR

    config_ids = list(cmp.CONFIG_IDS)  # empty list = compare all found

    print("=" * 70)
    print("CONFIGURATION COMPARATOR")
    print("=" * 70)
    if config_ids:
        print(f"Comparing   : {config_ids}")
    else:
        print("Comparing   : ALL configs found in output dir (auto-discover)")
    print(f"Datasets    : {list(cmp.DATASETS)}")
    print(f"Results dir : {cmp.OUTPUT_DIR}")
    print(f"Output dir  : {cmp.COMPARISON_OUT_DIR}")
    print(f"Figures     : {cmp.ENABLE_FIGURES}")
    print("=" * 70)

    comparator = ConfigComparator(
        configs_dir=cmp.OUTPUT_DIR,
        comparison_out_dir=cmp.COMPARISON_OUT_DIR,
        config_ids=config_ids if config_ids else None,
        datasets=list(cmp.DATASETS) if cmp.DATASETS else None,
        enable_figures=cmp.ENABLE_FIGURES
    )

    comparator.run()


if __name__ == "__main__":
    main()
