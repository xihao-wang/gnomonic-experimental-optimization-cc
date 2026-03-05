"""
Metrics evaluation orchestrator for comparing projection configurations.

This module orchestrates the full metrics evaluation workflow:
1. Load projection configurations
2. Load dataset(s)
3. Run detection pipeline with timing
4. Evaluate predictions against ground truth
5. Aggregate results across dataset
6. Generate comparison tables and PR curves

Author: Yassir Zardoua
Email: y.zardoua@caplogy.com | yassirzardoua@gmail.com
Date: 2026-02-12
"""

import sys
import json
import importlib
import cv2
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import numpy as np

from evaluation.lib.config import get_cfg
from evaluation.lib.bomni_dataset import BOMNIDataset
from evaluation.lib.piropo_dataset import PIROPODataset
from evaluation.lib.cepdof_dataset import CEPDOFDataset
from evaluation.lib.timing import PipelineTimer
from evaluation.lib.evaluator import DetectionEvaluator
from evaluation.lib.aggregator import ResultsAggregator
from evaluation.lib.metrics import match_predictions_to_ground_truth
from detection_pipeline.pipeline import DetectionPipeline
from image_composer.multi_persp import draw_fov_on_fisheye

# Visual color/thickness constants (BGR).  Drive the fisheye output only.
_CLR_GT   = (0, 255, 0)      # Green  — detected GT box (thick background)
_CLR_FN   = (0, 0, 255)      # Red    — missed GT / false negative (thin)
_CLR_TP   = (255, 0, 0)      # Blue   — true positive prediction (thin)
_CLR_FP   = (0, 255, 255)    # Yellow — false positive prediction (thin)
_CLR_COMP = (255, 255, 255)  # White  — raw YOLO boxes on composite image
_THICK_GT   = 3
_THICK_THIN = 2


class MetricsEvaluatorRunner:
    """
    Main orchestrator for metrics evaluation workflow.

    Evaluates multiple projection configurations against ground truth dataset(s)
    and generates comprehensive metrics with timing measurements.
    """

    def __init__(
        self,
        projection_config_module: str,
        yolo_model: str,
        datasets: List[str],
        dataset_roots: Dict[str, str],
        iou_thresholds: List[float],
        output_dir: str,
        enable_timing: bool = True,
        enable_pr_curves: bool = True,
        enable_visuals: bool = False,
        max_images: int = None,
        spread_samples: bool = True,
        vis_iou_threshold: float = 0.50,
        proj_boundary_colors: List[tuple] = None,
        vis_max_samples: Dict[str, int] = None
    ):
        """
        Initialize metrics evaluator runner.

        Args:
            projection_config_module: Python module path to projection configs
                                     (e.g., "evaluation.projection_configs_for_metrics")
            yolo_model: Path to YOLO model file
            datasets: List of dataset names to evaluate (e.g., ["bomni"])
            dataset_roots: Dict mapping dataset name to root directory
            iou_thresholds: List of IoU thresholds for evaluation
            output_dir: Base directory for metrics results
            enable_timing: Whether to measure pipeline timing
            enable_pr_curves: Whether to generate PR curve plots
            max_images: Maximum number of images to process (None = all)
            spread_samples: When True, spread sampled frames evenly across the
                            full dataset instead of taking the first N consecutive
            vis_iou_threshold: IoU threshold used to classify TP/FP/FN in
                               visual outputs (display only, no effect on metrics)
            proj_boundary_colors: RGB color tuples cycling across projections for
                                  boundary overlays on fisheye visuals. None = no boundaries.
            vis_max_samples: Per-dataset cap on visual image pairs saved.
                             Keys: "bomni", "piropo", "cepdof". Value 0 = no limit.
                             None = no limit for all datasets.
        """
        self.projection_config_module = projection_config_module
        self.yolo_model = yolo_model
        self.datasets = datasets
        self.dataset_roots = dataset_roots
        self.iou_thresholds = sorted(iou_thresholds)
        self.output_dir = Path(output_dir)
        self.enable_timing = enable_timing
        self.enable_pr_curves = enable_pr_curves
        self.enable_visuals = enable_visuals
        self.max_images = max_images
        self.spread_samples = spread_samples
        self.vis_iou_threshold = vis_iou_threshold
        self.proj_boundary_colors = proj_boundary_colors or []
        self.vis_max_samples = vis_max_samples or {}

        # Load projection configurations
        self.configs = self._load_projection_configs()

        # Create versioned session folder
        self.session_name = self._create_session_folder()
        self.session_path = self.output_dir / self.session_name

        print(f"Session: {self.session_name}")
        print(f"Output: {self.session_path}")

    def _load_projection_configs(self) -> List[Dict]:
        """
        Load projection configurations from Python module.

        Returns:
            List of configuration dicts
        """
        try:
            # Import the module dynamically
            config_module = importlib.import_module(self.projection_config_module)

            # Get CONFIGS list from ProjectionConfigs class
            configs = config_module.ProjectionConfigs.CONFIGS

            print(f"Loaded {len(configs)} projection configurations")
            for cfg in configs:
                print(f"  - {cfg['id']}: {cfg['name']}")

            return configs

        except Exception as e:
            raise RuntimeError(
                f"Failed to load projection configs from '{self.projection_config_module}': {e}"
            )

    def _create_session_folder(self) -> str:
        """
        Create versioned session folder.

        Returns:
            Session folder name (e.g., "metrics_eval_session_1")
        """
        session_number = 1
        while True:
            session_name = f"metrics_eval_session_{session_number}"
            session_path = self.output_dir / session_name
            if not session_path.exists():
                break
            session_number += 1

        # Create session folder
        session_path.mkdir(parents=True, exist_ok=True)

        return session_name

    def _compute_sample_indices(self, total: int, max_images, spread: bool) -> List[int]:
        """
        Compute which dataset indices to evaluate.

        Args:
            total:      Total number of images in the dataset
            max_images: Maximum images to sample (None = all)
            spread:     When True, distribute samples evenly across the full
                        dataset (step = total / max_images). When False, take
                        the first max_images consecutive frames.

        Returns:
            List of integer indices into the dataset
        """
        if max_images is None or max_images >= total:
            return list(range(total))

        if spread:
            step = total / max_images
            return [int(i * step) for i in range(max_images)]
        else:
            return list(range(max_images))

    def run_evaluation(self):
        """
        Run full metrics evaluation workflow.

        For each dataset:
        1. Load dataset
        2. For each projection configuration:
           - Run detection pipeline with timing
           - Evaluate predictions against ground truth
           - Aggregate results
        3. Generate comparison tables and PR curves
        4. Save all results
        """
        print("\n" + "=" * 80)
        print("METRICS EVALUATION")
        print("=" * 80)

        # Evaluate each dataset
        for dataset_name in self.datasets:
            print(f"\nDataset: {dataset_name.upper()}")
            print("-" * 80)

            # Load dataset
            dataset = self._load_dataset(dataset_name)

            if dataset is None:
                print(f"Skipping dataset: {dataset_name}")
                continue

            # Compute sample indices (spread or consecutive)
            total_images = len(dataset)
            indices = self._compute_sample_indices(total_images, self.max_images, self.spread_samples)
            num_images = len(indices)

            if self.max_images is not None and num_images < total_images:
                spread_label = "spread" if self.spread_samples else "consecutive"
                print(f"Processing {num_images} / {total_images} images ({spread_label}, MAX_IMAGES={self.max_images})")
            else:
                print(f"Processing all {num_images} images")

            # Evaluate each configuration
            config_results = []

            for config in self.configs:
                print(f"\nConfiguration: {config['name']}")
                print(f"  ID: {config['id']}")

                result = self._evaluate_configuration(
                    config=config,
                    dataset=dataset,
                    dataset_name=dataset_name,
                    indices=indices
                )

                config_results.append(result)

            # Save results for this dataset
            self._save_results(
                config_results=config_results,
                dataset_name=dataset_name,
                num_images=num_images,
                total_images=total_images
            )

        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print(f"Results saved to: {self.session_path}")
        print("=" * 80)

    def _load_dataset(self, dataset_name: str):
        """
        Load dataset by name.

        Args:
            dataset_name: Name of dataset ("bomni", "piropo", etc.)

        Returns:
            Dataset object or None if loading fails
        """
        if dataset_name.lower() == "bomni":
            return self._load_bomni_dataset()
        elif dataset_name.lower() == "piropo":
            return self._load_piropo_dataset()
        elif dataset_name.lower() == "cepdof":
            return self._load_cepdof_dataset()
        else:
            print(f"ERROR: Unknown dataset '{dataset_name}'")
            return None

    def _load_bomni_dataset(self):
        """
        Load BOMNI dataset.

        Returns:
            BOMNIDataset object
        """
        # Create config for BOMNI
        cfg = get_cfg()

        # Update with dataset root
        dataset_root = self.dataset_roots.get("bomni", "datasets/all-datasets/BOMNI-production")
        cfg.DATASETS.BOMNI.ROOT_DIR = dataset_root
        cfg.DATASETS.BOMNI.FRAMES_DIR = str(Path(dataset_root) / "frames" / "scenario1")
        cfg.DATASETS.BOMNI.STANDARD_ANNOTATIONS_DIR = str(Path(dataset_root) / "Standard-annotations" / "scenario1")
        cfg.DATASETS.BOMNI.ANNOTATION_FORMAT = "standard"
        cfg.DATASETS.BOMNI.SEQUENCES = ["top-0", "top-1", "top-2", "top-3"]

        dataset = BOMNIDataset(cfg)
        print(f"  Loaded BOMNI: {len(dataset)} images")

        return dataset

    def _load_piropo_dataset(self):
        """
        Load PIROPO dataset.

        Returns:
            PIROPODataset object
        """
        cfg = get_cfg()

        dataset_root = self.dataset_roots.get("piropo", "datasets/all-datasets/PIROPO-production")
        cfg.DATASETS.PIROPO.ROOT_DIR = dataset_root
        cfg.DATASETS.PIROPO.STANDARD_ANNOTATIONS_DIR = str(
            Path(dataset_root) / "standard-annotations"
        )

        dataset = PIROPODataset(cfg)
        print(f"  Loaded PIROPO: {len(dataset)} images")

        return dataset

    def _load_cepdof_dataset(self):
        """
        Load CEPDOF dataset.

        Returns:
            CEPDOFDataset object
        """
        cfg = get_cfg()

        dataset_root = self.dataset_roots.get("cepdof", "datasets/all-datasets/CEPDOF-production")
        cfg.DATASETS.CEPDOF.ROOT_DIR = dataset_root
        cfg.DATASETS.CEPDOF.STANDARD_ANNOTATIONS_DIR = str(
            Path(dataset_root) / "standard-annotations"
        )

        dataset = CEPDOFDataset(cfg)
        print(f"  Loaded CEPDOF: {len(dataset)} images")

        return dataset

    def _evaluate_configuration(
        self,
        config: Dict,
        dataset,
        dataset_name: str,
        indices: List[int]
    ) -> Dict[str, Any]:
        """
        Evaluate a single projection configuration.

        Args:
            config:      Projection configuration dict
            dataset:     Dataset object
            dataset_name: Name of dataset
            indices:     List of dataset indices to evaluate (supports spread sampling)

        Returns:
            Results dict with metrics and timing
        """
        config_id = config["id"]
        config_name = config["name"]
        num_images = len(indices)

        # Initialize detection pipeline
        pipeline = DetectionPipeline(
            model_path=self.yolo_model,
            conf_threshold=0.25
        )

        # Initialize timer (if enabled)
        timer = PipelineTimer() if self.enable_timing else None

        # Initialize evaluator and aggregator
        evaluator = DetectionEvaluator(iou_thresholds=self.iou_thresholds)
        aggregator = ResultsAggregator(
            iou_thresholds=self.iou_thresholds,
            enable_pr_curves=self.enable_pr_curves
        )

        # Process each image
        per_image_results = []
        all_predictions_with_confidence = []
        vis_limit = self.vis_max_samples.get(dataset_name, 0)
        visuals_saved = 0

        for i, idx in enumerate(indices):
            # Load image and ground truth
            data = dataset[idx]
            fisheye_image = data["image"]
            ground_truth = data["annotations"]

            save_visual = (
                self.enable_visuals
                and (vis_limit == 0 or visuals_saved < vis_limit)
            )

            # Run detection pipeline with timing
            pipeline_result = pipeline.run(
                fisheye_image=fisheye_image,
                projection_config=config,
                timer=timer,
                return_visuals=save_visual
            )

            if save_visual:
                detections, composite_image, raw_detections = pipeline_result
                self._save_visuals(
                    composite_image=composite_image,
                    raw_detections=raw_detections,
                    fisheye_image=fisheye_image,
                    ground_truth=ground_truth,
                    predictions=detections,
                    config_id=config_id,
                    idx=i,
                    dataset_name=dataset_name,
                    projection_config=config
                )
                visuals_saved += 1
            else:
                detections = pipeline_result

            # Evaluate predictions against ground truth
            result = evaluator.evaluate_image(
                predictions=detections,
                ground_truth=ground_truth
            )

            per_image_results.append(result)
            all_predictions_with_confidence.append(detections)

            # Progress indicator
            if (i + 1) % 50 == 0 or i == num_images - 1:
                print(f"  Processed: {i + 1} / {num_images}")

        # Aggregate results across dataset
        aggregated_results = aggregator.aggregate(
            per_image_results=per_image_results,
            predictions_with_confidence=all_predictions_with_confidence
        )

        # Get timing statistics (if enabled)
        timing_stats = None
        if timer is not None:
            timing_stats = timer.get_all_stats()

        # Generate PR curves (if enabled)
        if self.enable_pr_curves and "pr_curves_data" in aggregated_results:
            self._generate_pr_curves(
                pr_curves_data=aggregated_results["pr_curves_data"],
                config_id=config_id,
                config_name=config_name,
                dataset_name=dataset_name
            )

        return {
            "config_id": config_id,
            "config_name": config_name,
            "num_images": num_images,
            "metrics": aggregated_results["metrics_per_iou"],
            "summary": aggregated_results["summary"],
            "timing": timing_stats
        }

    def _save_visuals(
        self,
        composite_image,
        raw_detections: List[Dict],
        fisheye_image,
        ground_truth: List[Dict],
        predictions: List[Dict],
        config_id: str,
        idx: int,
        dataset_name: str,
        projection_config: Dict = None
    ):
        """
        Save composite and fisheye visual results for one image.

        Composite  : raw YOLO axis-aligned boxes in white.
        Fisheye    : TP/FP/FN colour-coded rotated boxes with confidence labels.
                     Green (thick) = detected GT drawn first (background reference).
                     Red   (thin)  = missed GT (false negatives), labelled "FN".
                     Blue  (thin)  = true positive predictions with confidence score.
                     Yellow(thin)  = false positive predictions with confidence score.
        Legend     : saved once as visuals/{config_id}/legend.png (on the first frame).

        Outputs:
            {session}/{dataset_name}/visuals/{config_id}/{idx:05d}_composite.jpg
            {session}/{dataset_name}/visuals/{config_id}/{idx:05d}_fisheye.jpg
            {session}/{dataset_name}/visuals/{config_id}/legend.png (first frame only)
        """
        visuals_dir = self.session_path / dataset_name / "visuals" / config_id
        visuals_dir.mkdir(parents=True, exist_ok=True)

        if idx == 0:
            self._save_legend(visuals_dir)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # ── Composite: raw YOLO detections in white ───────────────────────────
        comp_viz = composite_image.copy()
        H_c, W_c = comp_viz.shape[:2]
        for det in raw_detections:
            cx = int(det["x"] * W_c)
            cy = int(det["y"] * H_c)
            bw = int(det["w"] * W_c)
            bh = int(det["h"] * H_c)
            x1, y1 = cx - bw // 2, cy - bh // 2
            x2, y2 = cx + bw // 2, cy + bh // 2
            cv2.rectangle(comp_viz, (x1, y1), (x2, y2), _CLR_COMP, _THICK_THIN)
            cv2.putText(
                comp_viz, f"{det['confidence']:.2f}",
                (x1, max(0, y1 - 4)),
                font, 0.4, _CLR_COMP, 1
            )
        cv2.imwrite(str(visuals_dir / f"{idx:05d}_composite.jpg"), comp_viz)

        # ── Fisheye: classify predictions against GT ──────────────────────────
        true_positives, false_positives, false_negatives = \
            match_predictions_to_ground_truth(
                predictions, ground_truth, self.vis_iou_threshold
            )
        tp_gt_indices = {tp[1] for tp in true_positives}

        fish_viz = fisheye_image.copy()
        fish_viz = self._draw_projection_boundaries(fish_viz, projection_config)

        # 1. Green (thick): detected GT boxes — drawn first as background layer
        for gt_idx in tp_gt_indices:
            ann = ground_truth[gt_idx]
            box = np.intp(cv2.boxPoints((
                (ann["center_x"], ann["center_y"]),
                (ann["width"], ann["height"]),
                ann["angle"]
            )))
            cv2.drawContours(fish_viz, [box], 0, _CLR_GT, _THICK_GT)

        # 2. Red (thin): missed GT boxes (false negatives), labelled "FN"
        for gt_idx in false_negatives:
            ann = ground_truth[gt_idx]
            box = np.intp(cv2.boxPoints((
                (ann["center_x"], ann["center_y"]),
                (ann["width"], ann["height"]),
                ann["angle"]
            )))
            cv2.drawContours(fish_viz, [box], 0, _CLR_FN, _THICK_THIN)
            tl = self._top_left_corner(box.astype(np.float32))
            cv2.putText(fish_viz, "FN", tl, font, 0.4, _CLR_FN, 1)

        # 3. Blue (thin): true positive predictions with confidence score
        for pred_idx, gt_idx, _iou in true_positives:
            pred = predictions[pred_idx]
            box = np.intp(cv2.boxPoints((
                (pred["center_x"], pred["center_y"]),
                (pred["width"], pred["height"]),
                pred["angle"]
            )))
            cv2.drawContours(fish_viz, [box], 0, _CLR_TP, _THICK_THIN)
            tl = self._top_left_corner(box.astype(np.float32))
            score = pred.get("confidence", pred.get("score", 0.0))
            cv2.putText(fish_viz, f"{score:.2f}", tl, font, 0.4, _CLR_TP, 1)

        # 4. Yellow (thin): false positive predictions with confidence score
        for pred_idx in false_positives:
            pred = predictions[pred_idx]
            box = np.intp(cv2.boxPoints((
                (pred["center_x"], pred["center_y"]),
                (pred["width"], pred["height"]),
                pred["angle"]
            )))
            cv2.drawContours(fish_viz, [box], 0, _CLR_FP, _THICK_THIN)
            tl = self._top_left_corner(box.astype(np.float32))
            score = pred.get("confidence", pred.get("score", 0.0))
            cv2.putText(fish_viz, f"{score:.2f}", tl, font, 0.4, _CLR_FP, 1)

        cv2.imwrite(str(visuals_dir / f"{idx:05d}_fisheye.jpg"), fish_viz)

    def _draw_projection_boundaries(
        self, fisheye_img: np.ndarray, projection_config: Dict
    ) -> np.ndarray:
        """
        Overlay gnomonic projection boundary outlines on a fisheye image.

        Colors cycle through self.proj_boundary_colors (RGB → converted to BGR).
        Returns the image unchanged if proj_boundary_colors is empty or
        projection_config is None.
        """
        if not self.proj_boundary_colors or projection_config is None:
            return fisheye_img

        H, W = fisheye_img.shape[:2]
        cx, cy = W / 2.0, H / 2.0
        r = min(W, H) / 2.0

        proj_nbr = projection_config.get("proj_nbr", 4)
        lon_0    = projection_config.get("lon_0", 0.0)
        lon_step = projection_config.get("lon_step", 90.0)
        latitude = projection_config.get("latitude", 0.0)
        fov_h    = projection_config.get("fov_h", 90.0)
        fov_v    = projection_config.get("fov_v", 90.0)
        n_colors = len(self.proj_boundary_colors)

        img = fisheye_img
        for i in range(proj_nbr):
            longitude = lon_0 + i * lon_step
            rgb = self.proj_boundary_colors[i % n_colors]
            bgr = (rgb[2], rgb[1], rgb[0])
            img = draw_fov_on_fisheye(img, cx, cy, r, longitude, latitude, fov_h, fov_v, color=bgr)
        return img

    @staticmethod
    def _top_left_corner(pts: np.ndarray):
        """
        Return the top-left pixel of a rotated box for consistent text placement.

        Strategy: pick the two topmost corners (smallest y), then take the
        leftmost of those two.

        Args:
            pts: (4, 2) float32 array from cv2.boxPoints.

        Returns:
            (int, int) pixel coordinate for cv2.putText.
        """
        sorted_by_y = pts[np.argsort(pts[:, 1])]
        top_two = sorted_by_y[:2]
        corner = top_two[np.argmin(top_two[:, 0])]
        return (int(corner[0]), int(corner[1]))

    @staticmethod
    def _save_legend(visuals_dir: Path):
        """
        Save a standalone legend.png on a white background explaining the
        colour coding used in the fisheye visual outputs.
        """
        W, H = 420, 188
        img = np.full((H, W, 3), 255, dtype=np.uint8)

        cv2.putText(
            img, "Fisheye Visual Legend",
            (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1
        )

        items = [
            (_CLR_GT,   _THICK_GT,   "GT detected  (green, thick)"),
            (_CLR_FN,   _THICK_THIN, "FN  missed GT (red, thin)"),
            (_CLR_TP,   _THICK_THIN, "TP  prediction (blue, thin)"),
            (_CLR_FP,   _THICK_THIN, "FP  prediction (yellow, thin)"),
        ]

        x0, y0, sw, sh, rh = 12, 32, 44, 20, 38
        font = cv2.FONT_HERSHEY_SIMPLEX

        for i, (color, thick, label) in enumerate(items):
            y = y0 + i * rh
            cv2.rectangle(img, (x0, y), (x0 + sw, y + sh), color, thick)
            cv2.putText(
                img, label,
                (x0 + sw + 10, y + sh - 3),
                font, 0.45, (30, 30, 30), 1
            )

        cv2.imwrite(str(visuals_dir / "legend.png"), img)

    def _generate_pr_curves(
        self,
        pr_curves_data: Dict,
        config_id: str,
        config_name: str,
        dataset_name: str
    ):
        """
        Generate and save PR curve plots.

        Args:
            pr_curves_data: PR curve data for each IoU threshold
            config_id: Configuration ID
            config_name: Configuration name
            dataset_name: Dataset name
        """
        pr_curves_dir = self.session_path / dataset_name / "pr_curves"
        pr_curves_dir.mkdir(parents=True, exist_ok=True)

        aggregator = ResultsAggregator(
            iou_thresholds=self.iou_thresholds,
            enable_pr_curves=True
        )

        for iou_threshold, pr_data in pr_curves_data.items():
            output_path = pr_curves_dir / f"{config_id}_pr_curve_iou{iou_threshold:.2f}.png"

            aggregator.generate_pr_curve_plot(
                pr_curve_data=pr_data,
                iou_threshold=iou_threshold,
                output_path=output_path,
                config_name=config_name
            )

    def _save_results(
        self,
        config_results: List[Dict],
        dataset_name: str,
        num_images: int,
        total_images: int
    ):
        """
        Save all results to disk.

        Args:
            config_results: List of results dicts (one per configuration)
            dataset_name: Dataset name
            num_images: Number of images processed
            total_images: Total images in dataset
        """
        dataset_dir = self.session_path / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Save individual config results as JSON
        metrics_json_path = dataset_dir / "metrics.json"
        with open(metrics_json_path, "w") as f:
            json.dump(config_results, f, indent=2)
        print(f"\nSaved: {metrics_json_path}")

        # Save timing stats as JSON (if available)
        if config_results[0].get("timing") is not None:
            timing_json_path = dataset_dir / "timing.json"
            timing_data = {
                result["config_id"]: result["timing"]
                for result in config_results
            }
            with open(timing_json_path, "w") as f:
                json.dump(timing_data, f, indent=2)
            print(f"Saved: {timing_json_path}")

        # Save human-readable summary
        summary_path = dataset_dir / "summary.txt"
        self._write_summary(config_results, summary_path, dataset_name, num_images)
        print(f"Saved: {summary_path}")

        # Save comparison table
        comparison_path = dataset_dir / "comparison_table.txt"
        self._write_comparison_table(config_results, comparison_path, dataset_name)
        print(f"Saved: {comparison_path}")

        # Save metadata
        metadata_path = self.session_path / "metadata.txt"
        self._write_metadata(dataset_name, num_images, total_images, metadata_path)
        print(f"Saved: {metadata_path}")

        # Copy projection configs snapshot
        self._copy_projection_config_snapshot()

    def _write_summary(
        self,
        config_results: List[Dict],
        output_path: Path,
        dataset_name: str,
        num_images: int
    ):
        """Write human-readable summary file."""
        with open(output_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write(f"Metrics Evaluation Summary: {dataset_name.upper()}\n")
            f.write("=" * 80 + "\n\n")

            for result in config_results:
                f.write(f"Configuration: {result['config_name']}\n")
                f.write(f"  ID: {result['config_id']}\n")
                f.write(f"  Images: {result['num_images']}\n")
                f.write(f"\n")

                # Summary metrics
                summary = result["summary"]
                f.write(f"  Summary Metrics:\n")
                f.write(f"    AP@[0.30:0.95]: {summary['ap']:.3f}\n")
                f.write(f"    AP@0.50:       {summary['ap50']:.3f}\n")
                f.write(f"    AP@0.75:       {summary['ap75']:.3f}\n")
                f.write(f"    Precision@0.5: {summary['precision@0.5']:.3f}\n")
                f.write(f"    Recall@0.5:    {summary['recall@0.5']:.3f}\n")
                f.write(f"    F1@0.5:        {summary['f1@0.5']:.3f}\n")
                f.write(f"\n")

                # Timing (if available)
                if result.get("timing") is not None:
                    timing = result["timing"]
                    f.write(f"  Timing (mean per image):\n")
                    f.write(f"    Total:          {timing['total_end_to_end']['mean']:.3f}s\n")
                    f.write(f"    Composite Gen:  {timing['composite_generation']['mean']:.3f}s\n")
                    f.write(f"    YOLO Detection: {timing['yolo_detection']['mean']:.3f}s\n")
                    f.write(f"    Backprojection: {timing['backprojection']['mean']:.3f}s\n")
                    f.write(f"    Soft-NMS:       {timing['soft_nms']['mean']:.3f}s\n")
                    f.write(f"\n")

                f.write("-" * 80 + "\n\n")

    def _write_comparison_table(
        self,
        config_results: List[Dict],
        output_path: Path,
        dataset_name: str
    ):
        """Write side-by-side comparison table."""
        with open(output_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write(f"Configuration Comparison: {dataset_name.upper()} Dataset\n")
            f.write("=" * 80 + "\n\n")

            # Header row
            header = "Metric          "
            for result in config_results:
                config_id = result["config_id"]
                header += f"| {config_id[:18]:^18} "
            f.write(header + "\n")

            # Separator
            separator = "-" * 16
            for _ in config_results:
                separator += "|" + "-" * 20
            f.write(separator + "\n")

            # Metric rows
            metrics_to_show = [
                ("AP@[0.30:0.95]", "summary", "ap"),
                ("AP@0.50", "summary", "ap50"),
                ("AP@0.75", "summary", "ap75"),
                ("Precision@0.5", "summary", "precision@0.5"),
                ("Recall@0.5", "summary", "recall@0.5"),
                ("F1@0.5", "summary", "f1@0.5")
            ]

            for metric_name, source, key in metrics_to_show:
                row = f"{metric_name:16}"
                for result in config_results:
                    value = result[source][key]
                    row += f"| {value:^18.3f} "
                f.write(row + "\n")

            # Timing rows (if available)
            if config_results[0].get("timing") is not None:
                f.write("\nTiming (mean per image):\n")

                timing_metrics = [
                    ("Total Time", "total_end_to_end"),
                    ("Composite Gen", "composite_generation"),
                    ("YOLO Detection", "yolo_detection"),
                    ("Backprojection", "backprojection"),
                    ("Soft-NMS", "soft_nms")
                ]

                for metric_name, stage in timing_metrics:
                    row = f"{metric_name:16}"
                    for result in config_results:
                        value = result["timing"][stage]["mean"]
                        row += f"| {value:^16.3f}s "
                    f.write(row + "\n")

            # Find best configuration for each metric
            f.write("\n" + "=" * 80 + "\n")
            f.write("BEST CONFIGURATION PER METRIC:\n")

            # Best AP@[0.30:0.95]
            best_ap_idx = max(range(len(config_results)), key=lambda i: config_results[i]["summary"]["ap"])
            best_ap_config = config_results[best_ap_idx]
            f.write(f"  Best AP@[0.30:0.95]: {best_ap_config['config_id']} ({best_ap_config['summary']['ap']:.3f})\n")

            # Best AP@0.50
            best_ap50_idx = max(range(len(config_results)), key=lambda i: config_results[i]["summary"]["ap50"])
            best_ap50_config = config_results[best_ap50_idx]
            f.write(f"  Best AP@0.50:       {best_ap50_config['config_id']} ({best_ap50_config['summary']['ap50']:.3f})\n")

            # Best AP@0.75
            best_ap75_idx = max(range(len(config_results)), key=lambda i: config_results[i]["summary"]["ap75"])
            best_ap75_config = config_results[best_ap75_idx]
            f.write(f"  Best AP@0.75:       {best_ap75_config['config_id']} ({best_ap75_config['summary']['ap75']:.3f})\n")

            # Best Precision@0.5
            best_prec_idx = max(range(len(config_results)), key=lambda i: config_results[i]["summary"]["precision@0.5"])
            best_prec_config = config_results[best_prec_idx]
            f.write(f"  Best Precision@0.5: {best_prec_config['config_id']} ({best_prec_config['summary']['precision@0.5']:.3f})\n")

            # Best Recall@0.5
            best_recall_idx = max(range(len(config_results)), key=lambda i: config_results[i]["summary"]["recall@0.5"])
            best_recall_config = config_results[best_recall_idx]
            f.write(f"  Best Recall@0.5:    {best_recall_config['config_id']} ({best_recall_config['summary']['recall@0.5']:.3f})\n")

            # Best F1@0.5
            best_f1_idx = max(range(len(config_results)), key=lambda i: config_results[i]["summary"]["f1@0.5"])
            best_f1_config = config_results[best_f1_idx]
            f.write(f"  Best F1@0.5:        {best_f1_config['config_id']} ({best_f1_config['summary']['f1@0.5']:.3f})\n")

            # Fastest (if timing available)
            if config_results[0].get("timing") is not None:
                fastest_idx = min(range(len(config_results)), key=lambda i: config_results[i]["timing"]["total_end_to_end"]["mean"])
                fastest_config = config_results[fastest_idx]
                fastest_time = fastest_config["timing"]["total_end_to_end"]["mean"]
                f.write(f"  Fastest:            {fastest_config['config_id']} ({fastest_time:.3f}s/image)\n")

            f.write("=" * 80 + "\n")

    def _write_metadata(
        self,
        dataset_name: str,
        num_images: int,
        total_images: int,
        output_path: Path
    ):
        """Write session metadata file."""
        with open(output_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("Metrics Evaluation Session Metadata\n")
            f.write("=" * 80 + "\n")
            f.write(f"Session ID: {self.session_name}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write(f"YOLO Model: {self.yolo_model}\n")
            f.write(f"Dataset: {dataset_name.upper()} ({num_images} images processed")
            if num_images < total_images:
                f.write(f" of {total_images} total")
            f.write(")\n\n")

            f.write("Projection Configurations Evaluated:\n")
            for i, config in enumerate(self.configs, 1):
                f.write(f"  {i}. {config['id']} ({config['name']})\n")
            f.write("\n")

            f.write(f"IoU Thresholds: {self.iou_thresholds}\n\n")

            f.write("=" * 80 + "\n")

    def _copy_projection_config_snapshot(self):
        """Copy projection config file to session folder for reference."""
        try:
            # Import module and get file path
            config_module = importlib.import_module(self.projection_config_module)
            source_path = Path(config_module.__file__)

            # Copy to session folder
            dest_path = self.session_path / "projection_configs_snapshot.py"
            with open(source_path, "r") as src, open(dest_path, "w") as dst:
                dst.write(src.read())

            print(f"Saved: {dest_path}")

        except Exception as e:
            print(f"Warning: Could not copy projection config snapshot: {e}")
