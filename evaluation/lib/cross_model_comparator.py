"""
Cross-model configuration comparator.

Reads pre-computed per-config results from every model subfolder under
METRICS_EVALUATION.OUTPUT_DIR and compares them side-by-side.

Each configuration is labelled "{config_id} [{model_label}]" in all outputs
(tables, figures, winner file) so that the origin of every result is
unambiguous.

Expected directory structure (read):
    output_root/
    ├── yolov8m/
    │   └── configs/{config_id}/{dataset}/metrics.json
    └── yolo12x/
        └── configs/{config_id}/{dataset}/metrics.json

Comparison outputs are written to:
    output_root/cross-model/comparison_{timestamp}/

Author: Yassir Zardoua
Email: y.zardoua@caplogy.com | yassirzardoua@gmail.com
Date: 2026-03-06
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from evaluation.lib.comparator import ConfigComparator


class CrossModelComparator(ConfigComparator):
    """
    Compares projection configurations across multiple YOLO model runs.

    Subclasses ConfigComparator, overriding only the discovery and data-loading
    methods. All comparison, table-writing, figure-generation, and winner-
    determination logic is inherited unchanged.

    Config labels use the format  "{config_id} [{model_label}]".
    Sampling consistency is checked per original config ID (same config run
    under different models should have used the same frame indices).
    """

    # Regex that parses "some config id [model_label]"
    _LABEL_RE = re.compile(r"^(.*) \[([^\[\]]+)\]$")

    def __init__(
        self,
        output_root: str,
        comparison_out_dir: str,
        models: Optional[List[str]] = None,
        config_ids: Optional[List[str]] = None,
        datasets: Optional[List[str]] = None,
        enable_figures: bool = True,
    ):
        """
        Args:
            output_root:        Root directory that contains per-model subfolders
                                (e.g. evaluation/proj-conf-comparison/).
            comparison_out_dir: Where to write comparison outputs
                                (e.g. output_root/cross-model/).
            models:             Model labels to include. None/[] = all found.
            config_ids:         Original config IDs to include (without model
                                suffix). None/[] = all found.
            datasets:           Dataset names to include. None/[] = auto-discover.
            enable_figures:     Generate bar charts and PR overlay PNGs.
        """
        self.output_root = Path(output_root)
        self._models_filter = models or []
        self._config_ids_filter = config_ids or []

        # Pass output_root as configs_dir placeholder; actual loading is
        # fully overridden so this value is never used for path construction.
        super().__init__(
            configs_dir=str(self.output_root),
            comparison_out_dir=comparison_out_dir,
            config_ids=None,
            datasets=datasets,
            enable_figures=enable_figures,
        )

    # -------------------------------------------------------------------------
    # Override: config ID discovery
    # -------------------------------------------------------------------------

    def _resolve_config_ids(self) -> List[str]:
        """
        Return labels '{config_id} [{model}]' for every available combination,
        filtered by self._models_filter and self._config_ids_filter.
        """
        labels = []
        for model_dir in sorted(self.output_root.iterdir()):
            if not model_dir.is_dir() or model_dir.name == "cross-model":
                continue
            if self._models_filter and model_dir.name not in self._models_filter:
                continue
            configs_dir = model_dir / "configs"
            if not configs_dir.is_dir():
                continue
            for config_dir in sorted(configs_dir.iterdir()):
                if not config_dir.is_dir():
                    continue
                original_id = config_dir.name
                if self._config_ids_filter and original_id not in self._config_ids_filter:
                    continue
                labels.append(f"{original_id} [{model_dir.name}]")

        if not labels:
            print("  WARNING: No config/model combinations found.")
        return labels

    # -------------------------------------------------------------------------
    # Override: data loading
    # -------------------------------------------------------------------------

    def _load_all_results(
        self, config_ids: List[str], datasets: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Load metrics/timing for every '{config_id} [{model}]' × dataset pair.

        The config_name stored in each result entry is also suffixed with
        '[model_label]' so that plot legends and table headers are unambiguous.
        """
        data: Dict[str, Dict[str, Any]] = {}

        for label in config_ids:
            m = self._LABEL_RE.match(label)
            if not m:
                print(f"  WARNING: Cannot parse label '{label}' — skipped.")
                continue
            original_id, model_label = m.group(1), m.group(2)
            config_dir = self.output_root / model_label / "configs" / original_id

            config_json_path = config_dir / "config.json"
            config_meta = {}
            if config_json_path.exists():
                with open(config_json_path) as f:
                    config_meta = json.load(f)

            data[label] = {}

            for dataset_name in datasets:
                dataset_dir = config_dir / dataset_name
                metrics_path = dataset_dir / "metrics.json"

                if not metrics_path.exists():
                    print(
                        f"  WARNING: Missing {metrics_path} — "
                        f"'{label}' / '{dataset_name}' skipped."
                    )
                    continue

                with open(metrics_path) as f:
                    metrics = json.load(f)

                metrics["metrics_per_iou"] = {
                    float(k): v
                    for k, v in metrics.get("metrics_per_iou", {}).items()
                }
                metrics["pr_curves_data"] = {
                    float(k): v
                    for k, v in metrics.get("pr_curves_data", {}).items()
                }

                timing = None
                timing_path = dataset_dir / "timing.json"
                if timing_path.exists():
                    with open(timing_path) as f:
                        timing = json.load(f)

                base_name = config_meta.get(
                    "config_name", metrics.get("config_name", original_id)
                )
                data[label][dataset_name] = {
                    "metrics": metrics,
                    "timing": timing,
                    "config_meta": config_meta,
                    "config_name": f"{base_name} [{model_label}]",
                }

        return data

    # -------------------------------------------------------------------------
    # Override: sampling consistency check
    # -------------------------------------------------------------------------

    def _check_sampling_consistency(
        self,
        config_ids: List[str],
        datasets: List[str],
        out_dir: Path
    ):
        """
        Check that the same original config used the same frame indices on each
        dataset regardless of which model ran it. Cross-model differences in
        frame selection would make metric comparisons unreliable.
        """
        report_lines = ["SAMPLING CONSISTENCY CHECK (cross-model)", "=" * 60, ""]
        any_mismatch = False

        for dataset_name in datasets:
            report_lines.append(f"Dataset: {dataset_name.upper()}")

            # Collect indices per label
            indices_per_label: Dict[str, list] = {}
            for label in config_ids:
                m = self._LABEL_RE.match(label)
                if not m:
                    continue
                original_id, model_label = m.group(1), m.group(2)
                config_json_path = (
                    self.output_root / model_label / "configs" / original_id / "config.json"
                )
                if not config_json_path.exists():
                    continue
                with open(config_json_path) as f:
                    meta = json.load(f)
                ds_meta = meta.get("datasets_evaluated", {}).get(dataset_name)
                if ds_meta is not None:
                    indices_per_label[label] = ds_meta.get("sample_indices", [])

            # Group by original_id; compare same config across models
            groups: Dict[str, Dict[str, list]] = {}
            for label, indices in indices_per_label.items():
                m = self._LABEL_RE.match(label)
                if not m:
                    continue
                orig, model = m.group(1), m.group(2)
                groups.setdefault(orig, {})[model] = indices

            dataset_mismatch = False
            for orig_id, model_indices in groups.items():
                models = list(model_indices.keys())
                if len(models) < 2:
                    continue
                ref = models[0]
                for other in models[1:]:
                    if model_indices[other] != model_indices[ref]:
                        msg = (
                            f"  MISMATCH: '{orig_id}' used different frames on "
                            f"{dataset_name} between [{ref}] and [{other}]."
                        )
                        print(f"  WARNING: {msg}")
                        report_lines.append(msg)
                        dataset_mismatch = True
                        any_mismatch = True

            if not dataset_mismatch:
                report_lines.append("  OK — same-config runs used consistent frame indices.")
            report_lines.append("")

        report_path = out_dir / "consistency_check.txt"
        with open(report_path, "w") as f:
            f.write("\n".join(report_lines))

        if any_mismatch:
            print(f"\n  WARNING: Sampling mismatch detected. See {report_path}\n")

    # -------------------------------------------------------------------------
    # Override: winner determination (adds per-model head-to-head)
    # -------------------------------------------------------------------------

    def _write_winner(
        self,
        resolved_ids: List[str],
        resolved_datasets: List[str],
        data: Dict[str, Any],
        out_dir: Path
    ):
        """Call parent global ranking, then write per-model head-to-head file."""
        super()._write_winner(resolved_ids, resolved_datasets, data, out_dir)
        self._write_per_model_head_to_head(resolved_ids, resolved_datasets, data, out_dir)

    def _write_per_model_head_to_head(
        self,
        resolved_ids: List[str],
        resolved_datasets: List[str],
        data: Dict[str, Any],
        out_dir: Path
    ):
        """
        Write per_model_head_to_head.txt.

        For each YOLO model, for each dataset, for each metric: rank all
        original config IDs best-to-worst with value and delta from rank-1.

        Metrics: AP@[0.30:0.95], AP@[0.50:0.95], AP@0.50, AP@0.75,
                 Precision@0.50, Recall@0.50, F1@0.50, FPS.

        Bottom section: consistency summary matrix (WIN/LOSS per model per
        dataset) based on the primary metric AP@[0.30:0.95].
        """
        # Metrics to rank: (label, getter(d) -> float|None)
        # d is data[label][dataset_name]
        METRICS = [
            ("AP@[0.30:0.95]",  lambda d: d["metrics"]["summary"].get("ap")),
            ("AP@[0.50:0.95]",  lambda d: self._compute_coco_ap(d["metrics"])),
            ("AP@0.50",         lambda d: d["metrics"]["summary"].get("ap50")),
            ("AP@0.75",         lambda d: d["metrics"]["summary"].get("ap75")),
            ("Precision@0.50",  lambda d: d["metrics"]["summary"].get("precision@0.5")),
            ("Recall@0.50",     lambda d: d["metrics"]["summary"].get("recall@0.5")),
            ("F1@0.50",         lambda d: d["metrics"]["summary"].get("f1@0.5")),
            ("FPS",             lambda d: (
                1.0 / d["timing"]["total_end_to_end"]["mean"]
                if d.get("timing") and d["timing"].get("total_end_to_end", {}).get("mean")
                else None
            )),
        ]

        # --- collect original config IDs (ordered by first appearance) ---
        all_orig_ids: List[str] = []
        seen_orig: set = set()
        for label in resolved_ids:
            m = self._LABEL_RE.match(label)
            if m and m.group(1) not in seen_orig:
                seen_orig.add(m.group(1))
                all_orig_ids.append(m.group(1))

        # --- group labels by model ---
        models_ordered: List[str] = []
        by_model: Dict[str, Dict[str, str]] = {}   # model -> {orig_id -> label}
        for label in resolved_ids:
            m = self._LABEL_RE.match(label)
            if not m:
                continue
            orig_id, model = m.group(1), m.group(2)
            if model not in by_model:
                by_model[model] = {}
                models_ordered.append(model)
            by_model[model][orig_id] = label

        # --- helper: rank orig_ids by score dict, best first ---
        def _rank(scores: Dict[str, Optional[float]]) -> List[str]:
            valid = [(oid, v) for oid, v in scores.items() if v is not None]
            return [oid for oid, _ in sorted(valid, key=lambda x: x[1], reverse=True)]

        # --- extract primary metric for consistency matrix ---
        primary_getter = METRICS[0][1]

        report_path = out_dir / "per_model_head_to_head.txt"

        with open(report_path, "w") as f:
            f.write("PER-MODEL HEAD-TO-HEAD RANKING\n")
            f.write("=" * 70 + "\n")
            f.write("Configs ranked best-to-worst per metric, per dataset, per model.\n\n")

            # Track primary-metric winner per (model, dataset) for consistency matrix
            winner_matrix: Dict[str, Dict[str, Optional[str]]] = {}

            for model in models_ordered:
                f.write("=" * 70 + "\n")
                f.write(f"MODEL: {model}\n")
                f.write("=" * 70 + "\n")
                winner_matrix[model] = {}

                for ds in resolved_datasets:
                    f.write(f"\n  --- Dataset: {ds.upper()} ---\n")

                    for metric_label, getter in METRICS:
                        scores: Dict[str, Optional[float]] = {}
                        for orig_id in all_orig_ids:
                            label = by_model[model].get(orig_id)
                            val = None
                            if label:
                                d = data.get(label, {}).get(ds)
                                if d is not None:
                                    try:
                                        val = getter(d)
                                    except Exception:
                                        val = None
                            scores[orig_id] = val

                        ranked = _rank(scores)
                        if not ranked:
                            continue
                        best_val = scores[ranked[0]]

                        f.write(f"\n  {metric_label}\n")
                        f.write(f"  {'Rank':<6}{'Config':<38}{'Value':>8}{'Delta':>10}\n")
                        f.write("  " + "-" * 62 + "\n")
                        for rank, oid in enumerate(ranked, 1):
                            val = scores[oid]
                            delta = (val - best_val) if rank > 1 else 0.0
                            delta_str = f"{delta:+.3f}" if rank > 1 else "—"
                            f.write(f"  {rank:<6}{oid:<38}{val:>8.3f}{delta_str:>10}\n")

                        # Record primary metric winner
                        if metric_label == METRICS[0][0]:
                            winner_matrix[model][ds] = ranked[0] if ranked else None

                f.write("\n")

            # ---- Consistency summary matrix (primary metric: AP@[0.30:0.95]) ----
            win_count: Dict[str, int] = {oid: 0 for oid in all_orig_ids}
            for model in models_ordered:
                for ds in resolved_datasets:
                    w = winner_matrix.get(model, {}).get(ds)
                    if w:
                        win_count[w] += 1

            expected_winner = max(win_count, key=win_count.__getitem__) if win_count else None

            f.write("=" * 70 + "\n")
            f.write(f"CONSISTENCY SUMMARY  —  primary metric: {METRICS[0][0]}\n")
            f.write(f"Expected winner: {expected_winner}\n")
            f.write("WIN = ranks #1 on that dataset for this model\n")
            f.write("=" * 70 + "\n")

            col_ds_w = 12
            header = f"  {'Model':<18}"
            for ds in resolved_datasets:
                header += f"  {ds.upper():^{col_ds_w}}"
            header += f"  {'All datasets?':^14}"
            f.write(header + "\n")
            f.write("  " + "-" * (18 + (col_ds_w + 2) * len(resolved_datasets) + 16) + "\n")

            n_consistent = 0
            for model in models_ordered:
                row = f"  {model:<18}"
                model_consistent = True
                for ds in resolved_datasets:
                    w = winner_matrix.get(model, {}).get(ds)
                    won = (w == expected_winner)
                    if not won:
                        model_consistent = False
                    row += f"  {'WIN' if won else 'LOSS':^{col_ds_w}}"
                row += f"  {'YES' if model_consistent else 'NO':^14}"
                f.write(row + "\n")
                if model_consistent:
                    n_consistent += 1

            f.write("\n")
            f.write(
                f"  {expected_winner} is #1 on ALL datasets "
                f"in {n_consistent}/{len(models_ordered)} models.\n"
            )
            f.write("=" * 70 + "\n")

        print(f"Saved: {report_path}")

    # -------------------------------------------------------------------------
    # Override: comparison config metadata
    # -------------------------------------------------------------------------

    def _save_comparison_config(
        self,
        out_dir: Path,
        config_ids: List[str],
        datasets: List[str]
    ):
        """Save a JSON record of which configs/models/datasets were compared."""
        comparison_cfg = {
            "date": datetime.now().isoformat(timespec="seconds"),
            "comparison_mode": "cross-model",
            "output_root": str(self.output_root),
            "models_included": sorted({
                self._LABEL_RE.match(l).group(2)
                for l in config_ids
                if self._LABEL_RE.match(l)
            }),
            "config_ids_compared": config_ids,
            "datasets": datasets,
        }
        path = out_dir / "comparison_config.json"
        with open(path, "w") as f:
            json.dump(comparison_cfg, f, indent=2)
        print(f"Saved: {path}")
