"""
Configuration comparator: reads pre-computed per-config results and produces
comparison tables, PR overlay figures, and a scientific winner determination.

No detection re-execution occurs — this module only reads metrics.json /
timing.json files written by SingleConfigRunner.

Author: Yassir Zardoua
Email: y.zardoua@caplogy.com | yassirzardoua@gmail.com
Date: 2026-03-03
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Statistical confidence flag threshold (frames)
_LOW_CONFIDENCE_THRESHOLD = 500  # datasets with fewer frames get a low-confidence note

# IoU thresholds used by each AP style
_COCO_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

# Human-readable notes appended to every comparison table and winner file
_METRIC_NOTES = """\
METRIC NOTES
------------
AP@0.50          Pascal VOC style. A detection is correct if rotated IoU with
                 the GT box is >= 0.50. Single threshold, widely used baseline.

AP@[0.50:0.95]   COCO style. Mean AP across IoU thresholds 0.50, 0.55, ..., 0.95
                 (10 thresholds, step 0.05). Industry standard for detection
                 benchmarks. Stricter than Pascal VOC.

AP@[0.30:0.95]   Custom (this project). Same as COCO but extended down to 0.30
                 (14 thresholds). Rationale: all IoU values here are computed on
                 ROTATED bounding boxes in fisheye space (after backprojection).
                 Rotated-box IoU is geometrically stricter than axis-aligned IoU
                 at the same threshold value — even a small angle or center error
                 can drop overlap significantly. Starting from 0.30 accounts for
                 this. Especially relevant for CEPDOF, where ground-truth boxes
                 are NOT radially aligned (free body orientation), making the
                 backprojected box shape less predictable than in BOMNI/PIROPO.

All IoU values are computed between rotated bounding boxes in fisheye image
space (i.e. after backprojection + Soft-NMS). These IoU thresholds are
entirely independent of the NMS IoU parameters used inside the pipeline.
"""


class ConfigComparator:
    """
    Reads pre-computed per-config evaluation results and generates:
    - Per-dataset comparison tables (text)
    - PR curve overlay plots (one figure per dataset, all configs overlaid)
    - Bar charts (mAP per config per dataset)
    - Scientific winner determination (overall_winner.txt)

    All outputs are written to a timestamped folder inside comparisons/.
    No detection pipeline is ever invoked.
    """

    def __init__(
        self,
        configs_dir: str,
        comparison_out_dir: str,
        config_ids: Optional[List[str]] = None,
        datasets: Optional[List[str]] = None,
        enable_figures: bool = True
    ):
        """
        Initialize the comparator.

        Args:
            configs_dir:        Root folder containing per-config result folders
                                (evaluation/proj-conf-comparison/configs).
            comparison_out_dir: Root folder for comparison outputs
                                (evaluation/proj-conf-comparison/comparisons).
            config_ids:         Config IDs to compare. None / empty = all found.
            datasets:           Dataset names to include. None = auto-discover.
            enable_figures:     Generate bar charts and PR overlay PNGs.
        """
        self.configs_dir = Path(configs_dir)
        self.comparison_out_dir = Path(comparison_out_dir)
        self.config_ids = config_ids if config_ids else []
        self.datasets = datasets if datasets else []
        self.enable_figures = enable_figures

    def run(self):
        """
        Execute the full comparison workflow.

        1. Discover available config IDs and datasets.
        2. Load metrics + timing from JSON files.
        3. Sampling consistency check.
        4. Write per-dataset comparison tables.
        5. Generate figures (if enabled).
        6. Determine and write the overall winner.
        """
        # Create timestamped output folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = self.comparison_out_dir / f"comparison_{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nComparison output: {out_dir}")

        # Resolve which config IDs to compare
        resolved_ids = self._resolve_config_ids()
        if not resolved_ids:
            print("ERROR: No evaluated configurations found. Run run_single_config.py first.")
            return

        # Resolve which datasets to include
        resolved_datasets = self._resolve_datasets(resolved_ids)
        if not resolved_datasets:
            print("ERROR: No datasets found across the resolved configurations.")
            return

        print(f"Comparing configs: {resolved_ids}")
        print(f"Datasets         : {resolved_datasets}")

        # Load all results
        data = self._load_all_results(resolved_ids, resolved_datasets)

        if not data:
            print("ERROR: No valid results could be loaded.")
            return

        # Save comparison_config.json (which configs/datasets were compared)
        self._save_comparison_config(out_dir, resolved_ids, resolved_datasets)

        # Sampling consistency check
        self._check_sampling_consistency(resolved_ids, resolved_datasets, out_dir)

        # Per-dataset tables and figures
        for dataset_name in resolved_datasets:
            dataset_data = {
                cid: data[cid][dataset_name]
                for cid in resolved_ids
                if cid in data and dataset_name in data[cid]
            }
            if not dataset_data:
                continue

            self._write_comparison_table(
                dataset_name=dataset_name,
                dataset_data=dataset_data,
                out_dir=out_dir
            )

            if self.enable_figures:
                figures_dir = out_dir / "figures"
                figures_dir.mkdir(parents=True, exist_ok=True)
                self._plot_map_bar_chart(dataset_name, dataset_data, figures_dir)
                self._plot_pr_overlay(dataset_name, dataset_data, figures_dir)

        # Overall winner determination
        self._write_winner(
            resolved_ids=resolved_ids,
            resolved_datasets=resolved_datasets,
            data=data,
            out_dir=out_dir
        )

        print(f"\nComparison complete. Results at: {out_dir}")

    # -------------------------------------------------------------------------
    # Discovery
    # -------------------------------------------------------------------------

    def _resolve_config_ids(self) -> List[str]:
        """Return the list of config IDs to compare (explicit list or all found)."""
        if self.config_ids:
            # Validate that each requested ID has a result folder
            valid = []
            for cid in self.config_ids:
                folder = self.configs_dir / cid
                if not folder.is_dir():
                    print(f"  WARNING: Config '{cid}' has no result folder at {folder}")
                else:
                    valid.append(cid)
            return valid
        else:
            # Discover all config folders
            if not self.configs_dir.is_dir():
                return []
            found = sorted(
                p.name for p in self.configs_dir.iterdir() if p.is_dir()
            )
            return found

    def _resolve_datasets(self, config_ids: List[str]) -> List[str]:
        """
        Return datasets to compare. Uses self.datasets if provided; otherwise
        discovers dataset names that appear in ALL resolved config folders.
        """
        if self.datasets:
            return self.datasets

        # Auto-discover: find datasets present in at least one config
        seen = set()
        for cid in config_ids:
            config_dir = self.configs_dir / cid
            for item in config_dir.iterdir():
                if item.is_dir() and (item / "metrics.json").exists():
                    seen.add(item.name)
        return sorted(seen)

    # -------------------------------------------------------------------------
    # Data loading
    # -------------------------------------------------------------------------

    def _load_all_results(
        self, config_ids: List[str], datasets: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Load metrics.json and timing.json for every config × dataset pair.

        Returns:
            Nested dict: data[config_id][dataset_name] = {metrics, timing, ...}
        """
        data: Dict[str, Dict[str, Any]] = {}

        for cid in config_ids:
            config_dir = self.configs_dir / cid

            # Load config.json (metadata)
            config_json_path = config_dir / "config.json"
            config_meta = {}
            if config_json_path.exists():
                with open(config_json_path) as f:
                    config_meta = json.load(f)

            data[cid] = {}

            for dataset_name in datasets:
                dataset_dir = config_dir / dataset_name
                metrics_path = dataset_dir / "metrics.json"

                if not metrics_path.exists():
                    print(
                        f"  WARNING: Missing {metrics_path} — "
                        f"'{cid}' / '{dataset_name}' skipped."
                    )
                    continue

                with open(metrics_path) as f:
                    metrics = json.load(f)

                # Rehydrate float keys for metrics_per_iou and pr_curves_data
                metrics["metrics_per_iou"] = {
                    float(k): v
                    for k, v in metrics.get("metrics_per_iou", {}).items()
                }
                metrics["pr_curves_data"] = {
                    float(k): v
                    for k, v in metrics.get("pr_curves_data", {}).items()
                }

                # Load timing (optional)
                timing = None
                timing_path = dataset_dir / "timing.json"
                if timing_path.exists():
                    with open(timing_path) as f:
                        timing = json.load(f)

                data[cid][dataset_name] = {
                    "metrics": metrics,
                    "timing": timing,
                    "config_meta": config_meta,
                    "config_name": config_meta.get(
                        "config_name", metrics.get("config_name", cid)
                    )
                }

        return data

    # -------------------------------------------------------------------------
    # Sampling consistency check
    # -------------------------------------------------------------------------

    def _check_sampling_consistency(
        self,
        config_ids: List[str],
        datasets: List[str],
        out_dir: Path
    ):
        """
        Compare sample_indices across configs for each dataset.

        Warns (and logs to consistency_check.txt) if different configs used
        different frame subsets on the same dataset — this affects comparability.
        """
        report_lines = ["SAMPLING CONSISTENCY CHECK", "=" * 60, ""]
        any_mismatch = False

        for dataset_name in datasets:
            report_lines.append(f"Dataset: {dataset_name.upper()}")
            indices_per_config = {}

            for cid in config_ids:
                config_dir = self.configs_dir / cid
                config_json_path = config_dir / "config.json"
                if not config_json_path.exists():
                    continue
                with open(config_json_path) as f:
                    meta = json.load(f)

                dataset_meta = meta.get("datasets_evaluated", {}).get(dataset_name)
                if dataset_meta is not None:
                    indices_per_config[cid] = dataset_meta.get("sample_indices", [])

            if len(indices_per_config) < 2:
                report_lines.append("  Only one config has data — nothing to compare.")
                report_lines.append("")
                continue

            # Compare all pairs
            cids = list(indices_per_config.keys())
            ref_cid = cids[0]
            ref_indices = indices_per_config[ref_cid]
            mismatch = False

            for cid in cids[1:]:
                if indices_per_config[cid] != ref_indices:
                    msg = (
                        f"  MISMATCH: '{cid}' used different frames than '{ref_cid}' "
                        f"on {dataset_name}. Metrics may not be directly comparable."
                    )
                    print(f"  WARNING: {msg}")
                    report_lines.append(msg)
                    mismatch = True
                    any_mismatch = True

            if not mismatch:
                report_lines.append("  OK — all configs used the same frame indices.")
            report_lines.append("")

        report_path = out_dir / "consistency_check.txt"
        with open(report_path, "w") as f:
            f.write("\n".join(report_lines))

        if any_mismatch:
            print(
                f"\n  WARNING: Sampling mismatch detected. See {report_path} for details.\n"
            )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _compute_coco_ap(metrics_data: Dict) -> Optional[float]:
        """
        Compute COCO-style AP@[0.50:0.95] from a metrics.json metrics block.

        Averages AP over the 10 COCO thresholds (0.50 to 0.95, step 0.05).
        All these thresholds are a subset of our stored range (0.30:0.95),
        so no re-running is needed — the values are read directly from
        metrics_per_iou (keys are floats after rehydration).
        """
        metrics_per_iou = metrics_data.get("metrics_per_iou", {})
        ap_values = [
            metrics_per_iou[t]["ap"]
            for t in _COCO_THRESHOLDS
            if t in metrics_per_iou and "ap" in metrics_per_iou[t]
        ]
        return float(np.mean(ap_values)) if ap_values else None

    # -------------------------------------------------------------------------
    # Comparison table
    # -------------------------------------------------------------------------

    def _write_comparison_table(
        self,
        dataset_name: str,
        dataset_data: Dict[str, Any],
        out_dir: Path
    ):
        """
        Write a side-by-side comparison table for one dataset.

        Rows: AP@[0.30:0.95], AP@0.50, AP@0.75, Precision@0.5,
              Recall@0.5, F1@0.5, FPS (1 / total_end_to_end_mean).
        Note: single class (person) — AP is reported, not mAP.
        """
        table_path = out_dir / f"{dataset_name}_comparison_table.txt"

        config_ids = list(dataset_data.keys())
        col_w = 18

        with open(table_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write(f"Configuration Comparison: {dataset_name.upper()} Dataset\n")
            f.write("=" * 80 + "\n\n")

            # Header
            header = f"{'Metric':<22}"
            for cid in config_ids:
                header += f"| {cid[:col_w]:^{col_w}} "
            f.write(header + "\n")
            f.write("-" * 22 + ("|" + "-" * (col_w + 2)) * len(config_ids) + "\n")

            # Metric rows
            rows = [
                ("AP@[0.30:0.95] *", lambda d: d["metrics"]["summary"]["ap"]),
                ("AP@[0.50:0.95] **", lambda d: self._compute_coco_ap(d["metrics"])),
                ("AP@0.50 ***",      lambda d: d["metrics"]["summary"]["ap50"]),
                ("AP@0.75",          lambda d: d["metrics"]["summary"]["ap75"]),
                ("Precision@0.5",    lambda d: d["metrics"]["summary"]["precision@0.5"]),
                ("Recall@0.5",       lambda d: d["metrics"]["summary"]["recall@0.5"]),
                ("F1@0.5",           lambda d: d["metrics"]["summary"]["f1@0.5"]),
            ]

            for metric_name, getter in rows:
                row = f"{metric_name:<22}"
                for cid in config_ids:
                    try:
                        val = getter(dataset_data[cid])
                        if val is None:
                            row += f"| {'N/A':^{col_w}} "
                        else:
                            row += f"| {val:^{col_w}.3f} "
                    except (KeyError, TypeError):
                        row += f"| {'N/A':^{col_w}} "
                f.write(row + "\n")

            # FPS row (if timing available for at least one config)
            has_timing = any(
                dataset_data[cid]["timing"] is not None for cid in config_ids
            )
            if has_timing:
                row = f"{'FPS':<20}"
                for cid in config_ids:
                    timing = dataset_data[cid]["timing"]
                    if timing is not None:
                        mean_t = timing.get("total_end_to_end", {}).get("mean", None)
                        fps = 1.0 / mean_t if mean_t and mean_t > 0 else float("nan")
                        row += f"| {fps:^{col_w}.2f} "
                    else:
                        row += f"| {'N/A':^{col_w}} "
                f.write(row + "\n")

            # Best config per metric
            f.write("\n" + "=" * 80 + "\n")
            f.write("BEST PER METRIC (this dataset):\n")

            metric_bests = [
                ("AP@[0.30:0.95] *",  lambda d: d["metrics"]["summary"]["ap"]),
                ("AP@[0.50:0.95] **", lambda d: self._compute_coco_ap(d["metrics"])),
                ("AP@0.50 ***",       lambda d: d["metrics"]["summary"]["ap50"]),
                ("AP@0.75",           lambda d: d["metrics"]["summary"]["ap75"]),
                ("Precision@0.5",     lambda d: d["metrics"]["summary"]["precision@0.5"]),
                ("Recall@0.5",        lambda d: d["metrics"]["summary"]["recall@0.5"]),
                ("F1@0.5",            lambda d: d["metrics"]["summary"]["f1@0.5"]),
            ]

            for metric_name, getter in metric_bests:
                scores = {}
                for cid in config_ids:
                    try:
                        val = getter(dataset_data[cid])
                        if val is not None:
                            scores[cid] = val
                    except (KeyError, TypeError):
                        pass
                if scores:
                    best_cid = max(scores, key=scores.__getitem__)
                    f.write(f"  {metric_name:<22}: {best_cid} ({scores[best_cid]:.3f})\n")

            if has_timing:
                fps_scores = {}
                for cid in config_ids:
                    timing = dataset_data[cid]["timing"]
                    if timing is not None:
                        mean_t = timing.get("total_end_to_end", {}).get("mean", None)
                        if mean_t and mean_t > 0:
                            fps_scores[cid] = 1.0 / mean_t
                if fps_scores:
                    best_fps_cid = max(fps_scores, key=fps_scores.__getitem__)
                    f.write(
                        f"  {'Fastest (FPS)':<22}: {best_fps_cid} "
                        f"({fps_scores[best_fps_cid]:.2f} FPS)\n"
                    )

            f.write("=" * 80 + "\n")
            f.write("\n" + _METRIC_NOTES)
            f.write(
                "  * Custom (this project)  "
                "** COCO style  "
                "*** Pascal VOC style\n"
            )

        print(f"Saved: {table_path}")

    # -------------------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------------------

    def _plot_map_bar_chart(
        self,
        dataset_name: str,
        dataset_data: Dict[str, Any],
        figures_dir: Path
    ):
        """Bar chart: AP@[0.30:0.95] per configuration for one dataset."""
        config_ids = list(dataset_data.keys())
        config_names = [dataset_data[cid]["config_name"] for cid in config_ids]
        ap_values = [
            dataset_data[cid]["metrics"]["summary"].get("ap", 0.0)
            for cid in config_ids
        ]

        fig, ax = plt.subplots(figsize=(max(6, len(config_ids) * 2), 5))
        bars = ax.bar(range(len(config_ids)), ap_values, color="steelblue", width=0.6)
        ax.set_xticks(range(len(config_ids)))
        ax.set_xticklabels(config_names, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("AP@[0.30:0.95]", fontsize=11)
        ax.set_title(
            f"AP Comparison — {dataset_name.upper()} Dataset", fontsize=13
        )
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, ap_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9
            )

        plt.tight_layout()
        out_path = figures_dir / f"{dataset_name}_ap_bar.png"
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

    def _plot_pr_overlay(
        self,
        dataset_name: str,
        dataset_data: Dict[str, Any],
        figures_dir: Path,
        iou_threshold: float = 0.50
    ):
        """
        PR curve overlay: all configurations on one figure for one dataset.

        Uses IoU=0.50 by default (the most common reporting threshold).
        """
        colors = plt.cm.tab10.colors
        fig, ax = plt.subplots(figsize=(9, 7))

        plotted = 0
        for idx, (cid, d) in enumerate(dataset_data.items()):
            pr_data = d["metrics"].get("pr_curves_data", {}).get(iou_threshold)
            if pr_data is None:
                print(
                    f"  WARNING: No PR curve data at IoU={iou_threshold} "
                    f"for '{cid}' / '{dataset_name}' — skipped in overlay."
                )
                continue

            precision = pr_data.get("precision", [])
            recall = pr_data.get("recall", [])
            ap = pr_data.get("ap", None)

            if not precision or not recall:
                continue

            label = d["config_name"]
            if ap is not None:
                label += f" (AP={ap:.3f})"

            ax.plot(
                recall, precision,
                linewidth=2,
                color=colors[idx % len(colors)],
                label=label
            )
            plotted += 1

        if plotted == 0:
            plt.close(fig)
            return

        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        ax.set_title(
            f"PR Curves (IoU={iou_threshold:.2f}) — {dataset_name.upper()} Dataset",
            fontsize=13
        )

        plt.tight_layout()
        out_path = figures_dir / f"{dataset_name}_pr_curves_overlay.png"
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

    # -------------------------------------------------------------------------
    # Winner determination
    # -------------------------------------------------------------------------

    def _write_winner(
        self,
        resolved_ids: List[str],
        resolved_datasets: List[str],
        data: Dict[str, Dict[str, Any]],
        out_dir: Path
    ):
        """
        Determine and write two independent winners:
          - Robustness winner: highest mean AP@[0.30:0.95] across datasets
          - Speed winner:      highest mean FPS across datasets

        Aggregation: equal weight per dataset (each environment counts equally).
        No tie-breaking between the two winners — they are reported separately.
        """
        # Collect per-dataset AP (3 styles) and FPS per config
        per_dataset_ap_custom: Dict[str, Dict[str, float]] = {}  # AP@[0.30:0.95]
        per_dataset_ap_coco:   Dict[str, Dict[str, float]] = {}  # AP@[0.50:0.95]
        per_dataset_ap50:      Dict[str, Dict[str, float]] = {}  # AP@0.50
        per_dataset_fps:       Dict[str, Dict[str, float]] = {}

        for dataset_name in resolved_datasets:
            per_dataset_ap_custom[dataset_name] = {}
            per_dataset_ap_coco[dataset_name] = {}
            per_dataset_ap50[dataset_name] = {}
            per_dataset_fps[dataset_name] = {}
            for cid in resolved_ids:
                d = data.get(cid, {}).get(dataset_name)
                if d is None:
                    continue

                metrics = d["metrics"]
                ap_custom = metrics["summary"].get("ap", None)
                ap_coco   = self._compute_coco_ap(metrics)
                ap50      = metrics["summary"].get("ap50", None)

                if ap_custom is not None:
                    per_dataset_ap_custom[dataset_name][cid] = ap_custom
                if ap_coco is not None:
                    per_dataset_ap_coco[dataset_name][cid] = ap_coco
                if ap50 is not None:
                    per_dataset_ap50[dataset_name][cid] = ap50

                timing = d.get("timing")
                if timing is not None:
                    mean_t = timing.get("total_end_to_end", {}).get("mean", None)
                    if mean_t and mean_t > 0:
                        per_dataset_fps[dataset_name][cid] = 1.0 / mean_t

        def _per_dataset_winner(per_ds):
            return {
                ds: (max(scores, key=scores.__getitem__) if scores else None)
                for ds, scores in per_ds.items()
            }

        def _overall_mean(per_ds):
            result = {}
            for cid in resolved_ids:
                vals = [
                    per_ds[ds][cid]
                    for ds in resolved_datasets
                    if cid in per_ds.get(ds, {})
                ]
                if vals:
                    result[cid] = float(np.mean(vals))
            return result

        overall_ap_custom = _overall_mean(per_dataset_ap_custom)
        overall_ap_coco   = _overall_mean(per_dataset_ap_coco)
        overall_ap50      = _overall_mean(per_dataset_ap50)
        overall_fps       = _overall_mean(per_dataset_fps)

        if not overall_ap_custom:
            print("WARNING: Could not determine winner — no AP data available.")
            return

        def _ranking(scores):
            return sorted(scores.keys(), key=scores.__getitem__, reverse=True)

        ranking_custom = _ranking(overall_ap_custom)
        ranking_coco   = _ranking(overall_ap_coco)
        ranking_ap50   = _ranking(overall_ap50)
        ranking_fps    = _ranking(overall_fps) if overall_fps else []

        winner_custom = ranking_custom[0]
        winner_coco   = ranking_coco[0] if ranking_coco else None
        winner_ap50   = ranking_ap50[0] if ranking_ap50 else None
        winner_fps    = ranking_fps[0]  if ranking_fps  else None

        # Low-confidence dataset flags
        low_conf_datasets = []
        for dataset_name in resolved_datasets:
            for cid in resolved_ids:
                d = data.get(cid, {}).get(dataset_name)
                if d is not None:
                    n = d["metrics"].get("num_images", 0)
                    if n < _LOW_CONFIDENCE_THRESHOLD:
                        low_conf_datasets.append((dataset_name, n))
                    break

        def _write_robustness_section(f, label, per_ds, overall, ranking, winner):
            f.write("-" * 60 + "\n")
            f.write(f"ROBUSTNESS ({label})\n")
            f.write("-" * 60 + "\n")
            f.write("Per-dataset winners:\n")
            winners = _per_dataset_winner(per_ds)
            for ds in resolved_datasets:
                w = winners.get(ds)
                if w is not None:
                    score = per_ds[ds][w]
                    n = data[w][ds]["metrics"].get("num_images", "?")
                    f.write(f"  {ds.upper():<10} -> {w}  (AP={score:.3f}, n={n})\n")
                else:
                    f.write(f"  {ds.upper():<10} -> N/A\n")
            f.write(f"\nOverall ranking (equal-weight mean across datasets):\n")
            for rank, cid in enumerate(ranking, 1):
                f.write(f"  {rank}. {cid:<35} mean AP={overall[cid]:.3f}\n")
            f.write(f"\nWINNER: {winner}  (mean AP={overall[winner]:.3f})\n\n")

        # Write overall_winner.txt
        winner_path = out_dir / "overall_winner.txt"
        with open(winner_path, "w") as f:
            f.write("WINNER DETERMINATION\n")
            f.write("=" * 60 + "\n")
            f.write("Class       : person (single class — AP reported, not mAP)\n")
            f.write(
                "Aggregation : equal-weight mean across all evaluated datasets\n"
                "              (each environment contributes equally)\n\n"
            )

            _write_robustness_section(
                f, "Custom — AP@[0.30:0.95]  *",
                per_dataset_ap_custom, overall_ap_custom, ranking_custom, winner_custom
            )
            if winner_coco:
                _write_robustness_section(
                    f, "COCO   — AP@[0.50:0.95]  **",
                    per_dataset_ap_coco, overall_ap_coco, ranking_coco, winner_coco
                )
            if winner_ap50:
                _write_robustness_section(
                    f, "Pascal VOC — AP@0.50     ***",
                    per_dataset_ap50, overall_ap50, ranking_ap50, winner_ap50
                )

            # Speed
            f.write("-" * 60 + "\n")
            f.write("SPEED  —  FPS (1 / mean total pipeline time per image)\n")
            f.write("-" * 60 + "\n")
            if winner_fps is not None:
                fps_winners = _per_dataset_winner(per_dataset_fps)
                f.write("Per-dataset winners:\n")
                for ds in resolved_datasets:
                    w = fps_winners.get(ds)
                    if w is not None:
                        fps = per_dataset_fps[ds][w]
                        n = data[w][ds]["metrics"].get("num_images", "?")
                        f.write(
                            f"  {ds.upper():<10} -> {w}  (FPS={fps:.2f}, n={n})\n"
                        )
                    else:
                        f.write(f"  {ds.upper():<10} -> N/A\n")
                f.write("\nOverall ranking (equal-weight mean FPS across datasets):\n")
                for rank, cid in enumerate(ranking_fps, 1):
                    f.write(
                        f"  {rank}. {cid:<35} mean FPS={overall_fps[cid]:.2f}\n"
                    )
                f.write(f"\nSPEED WINNER: {winner_fps}"
                        f"  (mean FPS={overall_fps[winner_fps]:.2f})\n")
            else:
                f.write("No timing data available.\n")

            # Statistical notes
            if low_conf_datasets:
                f.write("\n" + "-" * 60 + "\n")
                f.write("Statistical note:\n")
                for ds, n in low_conf_datasets:
                    f.write(
                        f"  {ds.upper()} results carry lower statistical confidence "
                        f"({n} frames < {_LOW_CONFIDENCE_THRESHOLD} threshold).\n"
                    )

            f.write("\n" + "=" * 60 + "\n")
            f.write(_METRIC_NOTES)
            f.write(
                "  * Custom (this project)  "
                "** COCO style  "
                "*** Pascal VOC style\n"
            )

        print(f"Saved: {winner_path}")
        print(f"\nROBUSTNESS WINNER (Custom AP@[0.30:0.95]) : {winner_custom}"
              f"  ({overall_ap_custom[winner_custom]:.3f})")
        if winner_coco:
            print(f"ROBUSTNESS WINNER (COCO   AP@[0.50:0.95]) : {winner_coco}"
                  f"  ({overall_ap_coco[winner_coco]:.3f})")
        if winner_ap50:
            print(f"ROBUSTNESS WINNER (PascalVOC AP@0.50)     : {winner_ap50}"
                  f"  ({overall_ap50[winner_ap50]:.3f})")
        if winner_fps:
            print(f"SPEED WINNER                               : {winner_fps}"
                  f"  ({overall_fps[winner_fps]:.2f} FPS)")

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    def _save_comparison_config(
        self,
        out_dir: Path,
        config_ids: List[str],
        datasets: List[str]
    ):
        """Save a JSON record of which configs and datasets were compared."""
        comparison_cfg = {
            "date": datetime.now().isoformat(timespec="seconds"),
            "config_ids_compared": config_ids,
            "datasets": datasets,
            "configs_dir": str(self.configs_dir)
        }
        path = out_dir / "comparison_config.json"
        with open(path, "w") as f:
            json.dump(comparison_cfg, f, indent=2)
        print(f"Saved: {path}")
