"""
Incremental single-configuration evaluation runner.

Runs one projection configuration on all requested datasets and saves results
to evaluation/proj-conf-comparison/configs/{config_id}/. Each configuration
is fully independent — re-running one config never touches others.

Key outputs per config:
  configs/{config_id}/
    config.json               — projection params + run metadata + sample indices
    {dataset}/metrics.json    — aggregated metrics + raw PR curve data
    {dataset}/timing.json     — per-stage timing statistics
    {dataset}/pr_curves/      — individual PR curve PNGs
    {dataset}/visuals/        — composite + fisheye image pairs

Author: Yassir Zardoua
Email: y.zardoua@caplogy.com | yassirzardoua@gmail.com
Date: 2026-03-03
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from evaluation.lib.config import get_cfg
from evaluation.lib.bomni_dataset import BOMNIDataset
from evaluation.lib.piropo_dataset import PIROPODataset
from evaluation.lib.cepdof_dataset import CEPDOFDataset
from evaluation.lib.timing import PipelineTimer
from evaluation.lib.evaluator import DetectionEvaluator
from evaluation.lib.aggregator import ResultsAggregator
from evaluation.lib.metrics import match_predictions_to_ground_truth
from detection_pipeline.pipeline import DetectionPipeline
from image_composer.multi_persp import draw_fov_on_fisheye, resolve_proj_list

# Known pipeline NMS constants (documented for reproducibility in config.json)
_NMS_PARAMS = {
    "stage1_iou": 0.8,
    "stage2_sigma": 0.2,
    "stage2_score_thresh": 0.3
}

# Visual color/thickness constants (BGR).  These drive the fisheye output only.
_CLR_GT   = (0, 255, 0)      # Green  — detected GT box (thick background)
_CLR_FN   = (0, 0, 255)      # Red    — missed GT / false negative (thin)
_CLR_TP   = (255, 0, 0)      # Blue   — true positive prediction (thin)
_CLR_FP   = (0, 255, 255)    # Yellow — false positive prediction (thin)
_CLR_COMP = (255, 255, 255)  # White  — raw YOLO boxes on composite image
_THICK_GT   = 3
_THICK_THIN = 2


class SingleConfigRunner:
    """
    Evaluates a single projection configuration on one or more datasets.

    Each invocation is fully independent: results are stored under
    evaluation/proj-conf-comparison/configs/{config_id}/ and existing
    results from other configs are never touched.

    Overwrite policy (OVERWRITE_EXISTING):
      False (default) — if {config_id}/ already exists, skip and warn.
      True            — overwrite any existing results for this config.
    """

    def __init__(
        self,
        config: Dict,
        yolo_model: str,
        datasets: List[str],
        dataset_roots: Dict[str, str],
        iou_thresholds: List[float],
        output_dir: str,
        enable_timing: bool = True,
        enable_pr_curves: bool = True,
        enable_visuals: bool = False,
        max_images: Optional[int] = None,
        spread_samples: bool = True,
        overwrite_existing: bool = False,
        vis_iou_threshold: float = 0.50,
        proj_boundary_colors: Optional[List[tuple]] = None,
        vis_max_samples: Optional[Dict[str, int]] = None,
        vis_spread_samples: bool = True
    ):
        """
        Initialize the runner.

        Args:
            config:            Projection configuration dict (id, name, proj params)
            yolo_model:        Path to YOLO model file
            datasets:          Dataset names to evaluate (e.g. ["bomni", "piropo"])
            dataset_roots:     Dict mapping dataset name → root directory path
            iou_thresholds:    List of IoU thresholds for evaluation
            output_dir:        Root directory for per-config result folders
            enable_timing:     Whether to measure per-stage pipeline timing
            enable_pr_curves:  Whether to generate individual PR curve PNGs
            enable_visuals:    Whether to save composite + fisheye image pairs
            max_images:        Maximum images per dataset (None = all)
            spread_samples:    Distribute samples evenly vs. taking first N
            overwrite_existing: Overwrite if this config has already been evaluated
            vis_iou_threshold: IoU threshold used to classify TP/FP/FN in visuals
                               (display only — does not affect metrics)
            proj_boundary_colors: RGB color tuples cycling across projections for
                                  boundary overlays on fisheye visuals. Converted
                                  internally to BGR for OpenCV. None = no boundaries.
            vis_max_samples: Per-dataset cap on visual image pairs saved.
                             Keys: "bomni", "piropo", "cepdof". Value 0 = no limit.
                             None = no limit for all datasets.
            vis_spread_samples: When True, spread the N visual saves uniformly
                                across all evaluated frames (step = total / N).
                                When False, save the first N consecutive frames.
        """
        self.config = config
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
        self.overwrite_existing = overwrite_existing
        self.vis_iou_threshold = vis_iou_threshold
        self.proj_boundary_colors = proj_boundary_colors or []
        self.vis_max_samples = vis_max_samples or {}
        self.vis_spread_samples = vis_spread_samples

    def run(self) -> bool:
        """
        Run evaluation for this configuration on all requested datasets.

        Returns:
            True if evaluation ran; False if skipped due to overwrite policy.
        """
        config_id = self.config["id"]
        config_dir = self.output_dir / config_id

        # Overwrite check
        if config_dir.exists() and not self.overwrite_existing:
            print(
                f"[SKIP] Config '{config_id}' already has results at:\n"
                f"       {config_dir}\n"
                f"       Set OVERWRITE_EXISTING=True in config.py to re-run."
            )
            return False

        config_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nConfig  : {self.config['name']} ({config_id})")
        print(f"Output  : {config_dir}")
        print(f"Datasets: {self.datasets}")
        print(f"YOLO    : {self.yolo_model}")
        print(f"Max imgs: {self.max_images or 'All'} per dataset")
        print("=" * 70)

        datasets_meta: Dict[str, Any] = {}

        for dataset_name in self.datasets:
            print(f"\nDataset: {dataset_name.upper()}")
            print("-" * 50)

            dataset = self._load_dataset(dataset_name)
            if dataset is None:
                continue

            total_images = len(dataset)
            indices = self._compute_sample_indices(
                total_images, self.max_images, self.spread_samples
            )
            num_images = len(indices)

            if self.max_images is not None and num_images < total_images:
                mode = "spread" if self.spread_samples else "consecutive"
                print(f"  Sampling {num_images} / {total_images} frames ({mode})")
            else:
                print(f"  Processing all {num_images} frames")

            result = self._evaluate_on_dataset(
                dataset=dataset,
                dataset_name=dataset_name,
                indices=indices
            )

            dataset_out_dir = config_dir / dataset_name
            dataset_out_dir.mkdir(parents=True, exist_ok=True)

            self._save_dataset_results(result, dataset_out_dir, dataset_name)

            datasets_meta[dataset_name] = {
                "num_images_total": total_images,
                "num_images_used": num_images,
                "max_images": self.max_images,
                "spread_samples": self.spread_samples,
                "sample_indices": indices
            }

        # Save config.json with full run metadata
        self._save_config_json(config_dir, datasets_meta)

        print(f"\nDone. Results at: {config_dir}")
        return True

    # -------------------------------------------------------------------------
    # Dataset loading (mirrors metrics_evaluator_runner.py)
    # -------------------------------------------------------------------------

    def _load_dataset(self, dataset_name: str):
        """Load dataset by name. Returns None on unknown name."""
        loaders = {
            "bomni": self._load_bomni_dataset,
            "piropo": self._load_piropo_dataset,
            "cepdof": self._load_cepdof_dataset
        }
        loader = loaders.get(dataset_name.lower())
        if loader is None:
            print(f"  ERROR: Unknown dataset '{dataset_name}'")
            return None
        return loader()

    def _load_bomni_dataset(self):
        cfg = get_cfg()
        root = self.dataset_roots.get("bomni", "datasets/all-datasets/BOMNI-production")
        cfg.DATASETS.BOMNI.ROOT_DIR = root
        cfg.DATASETS.BOMNI.FRAMES_DIR = str(Path(root) / "frames" / "scenario1")
        cfg.DATASETS.BOMNI.STANDARD_ANNOTATIONS_DIR = str(
            Path(root) / "Standard-annotations" / "scenario1"
        )
        cfg.DATASETS.BOMNI.ANNOTATION_FORMAT = "standard"
        cfg.DATASETS.BOMNI.SEQUENCES = ["top-0", "top-1", "top-2", "top-3"]
        dataset = BOMNIDataset(cfg)
        print(f"  Loaded BOMNI: {len(dataset)} images")
        return dataset

    def _load_piropo_dataset(self):
        cfg = get_cfg()
        root = self.dataset_roots.get("piropo", "datasets/all-datasets/PIROPO-production")
        cfg.DATASETS.PIROPO.ROOT_DIR = root
        cfg.DATASETS.PIROPO.STANDARD_ANNOTATIONS_DIR = str(
            Path(root) / "standard-annotations"
        )
        dataset = PIROPODataset(cfg)
        print(f"  Loaded PIROPO: {len(dataset)} images")
        return dataset

    def _load_cepdof_dataset(self):
        cfg = get_cfg()
        root = self.dataset_roots.get("cepdof", "datasets/all-datasets/CEPDOF-production")
        cfg.DATASETS.CEPDOF.ROOT_DIR = root
        cfg.DATASETS.CEPDOF.STANDARD_ANNOTATIONS_DIR = str(
            Path(root) / "standard-annotations"
        )
        dataset = CEPDOFDataset(cfg)
        print(f"  Loaded CEPDOF: {len(dataset)} images")
        return dataset

    # -------------------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------------------

    def _compute_sample_indices(
        self, total: int, max_images: Optional[int], spread: bool
    ) -> List[int]:
        """
        Compute which dataset indices to evaluate.

        Args:
            total:      Total number of images in the dataset.
            max_images: Maximum images to sample (None = all).
            spread:     Distribute evenly (True) or take first N (False).

        Returns:
            Sorted list of integer indices.
        """
        if max_images is None or max_images >= total:
            return list(range(total))
        if spread:
            step = total / max_images
            return [int(i * step) for i in range(max_images)]
        return list(range(max_images))

    # -------------------------------------------------------------------------
    # Core evaluation loop
    # -------------------------------------------------------------------------

    def _evaluate_on_dataset(
        self,
        dataset,
        dataset_name: str,
        indices: List[int]
    ) -> Dict[str, Any]:
        """
        Run detection + evaluation on a dataset subset.

        Args:
            dataset:      Dataset object (BOMNI / PIROPO / CEPDOF).
            dataset_name: Dataset name string.
            indices:      List of frame indices to process.

        Returns:
            Result dict with keys: config_id, config_name, num_images,
            metrics_per_iou, summary, pr_curves_data, timing.
        """
        num_images = len(indices)
        config_id = self.config["id"]
        config_name = self.config["name"]

        pipeline = DetectionPipeline(
            model_path=self.yolo_model,
            conf_threshold=0.25
        )
        timer = PipelineTimer() if self.enable_timing else None
        evaluator = DetectionEvaluator(iou_thresholds=self.iou_thresholds)
        aggregator = ResultsAggregator(
            iou_thresholds=self.iou_thresholds,
            enable_pr_curves=True  # Always collect; needed for comparator overlays
        )

        per_image_results = []
        all_predictions_with_confidence = []
        vis_limit = self.vis_max_samples.get(dataset_name, 0)
        total_frames = len(indices)
        if vis_limit == 0:
            vis_save_set = None  # save all
        elif self.vis_spread_samples:
            step = total_frames / vis_limit
            vis_save_set = set(int(j * step) for j in range(min(vis_limit, total_frames)))
        else:
            vis_save_set = set(range(min(vis_limit, total_frames)))

        for i, idx in enumerate(indices):
            data = dataset[idx]
            fisheye_image = data["image"]
            ground_truth = data["annotations"]

            save_visual = (
                self.enable_visuals
                and (vis_save_set is None or i in vis_save_set)
            )

            pipeline_result = pipeline.run(
                fisheye_image=fisheye_image,
                projection_config=self.config,
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
                    dataset_out_dir=self.output_dir / config_id / dataset_name
                )
            else:
                detections = pipeline_result

            result = evaluator.evaluate_image(
                predictions=detections,
                ground_truth=ground_truth
            )
            per_image_results.append(result)
            all_predictions_with_confidence.append(detections)

            if (i + 1) % 50 == 0 or i == num_images - 1:
                print(f"  Processed: {i + 1} / {num_images}")

        aggregated = aggregator.aggregate(
            per_image_results=per_image_results,
            predictions_with_confidence=all_predictions_with_confidence
        )

        timing_stats = timer.get_all_stats() if timer is not None else None

        # Generate individual PR curve PNGs (per IoU threshold)
        if self.enable_pr_curves and "pr_curves_data" in aggregated:
            self._generate_pr_curves(
                pr_curves_data=aggregated["pr_curves_data"],
                config_id=config_id,
                config_name=config_name,
                dataset_out_dir=self.output_dir / config_id / dataset_name
            )

        return {
            "config_id": config_id,
            "config_name": config_name,
            "num_images": num_images,
            "metrics_per_iou": aggregated["metrics_per_iou"],
            "summary": aggregated["summary"],
            "pr_curves_data": aggregated.get("pr_curves_data", {}),
            "timing": timing_stats
        }

    # -------------------------------------------------------------------------
    # Visual output
    # -------------------------------------------------------------------------

    def _save_visuals(
        self,
        composite_image,
        raw_detections: List[Dict],
        fisheye_image,
        ground_truth: List[Dict],
        predictions: List[Dict],
        config_id: str,
        idx: int,
        dataset_out_dir: Path
    ):
        """
        Save composite + fisheye image pair for one frame.

        Composite  : raw YOLO axis-aligned boxes in white.
        Fisheye    : TP/FP/FN colour-coded rotated boxes with confidence labels.
                     Green (thick) = detected GT drawn first (background reference).
                     Red   (thin)  = missed GT (false negatives), labelled "FN".
                     Blue  (thin)  = true positive predictions with confidence score.
                     Yellow(thin)  = false positive predictions with confidence score.
        Legend     : saved once as visuals/legend.png (on the first frame).

        Output:
            {dataset_out_dir}/visuals/{idx:05d}_composite.jpg
            {dataset_out_dir}/visuals/{idx:05d}_fisheye.jpg
            {dataset_out_dir}/visuals/legend.png  (first frame only)
        """
        visuals_dir = dataset_out_dir / "visuals"
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
        fish_viz = self._draw_projection_boundaries(fish_viz)

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

    def _draw_projection_boundaries(self, fisheye_img: np.ndarray) -> np.ndarray:
        """
        Overlay gnomonic projection boundary outlines on a fisheye image.

        Colors cycle through self.proj_boundary_colors (RGB → converted to BGR).
        Supports both uniform configs and configs with extra_projections.
        Returns the image unchanged if proj_boundary_colors is empty.
        """
        if not self.proj_boundary_colors:
            return fisheye_img

        H, W = fisheye_img.shape[:2]
        cx, cy = W / 2.0, H / 2.0
        r = min(W, H) / 2.0
        n_colors = len(self.proj_boundary_colors)

        img = fisheye_img
        for i, proj in enumerate(resolve_proj_list(self.config)):
            rgb = self.proj_boundary_colors[i % n_colors]
            bgr = (rgb[2], rgb[1], rgb[0])
            img = draw_fov_on_fisheye(
                img, cx, cy, r,
                proj["longitude"], proj["latitude"],
                proj["fov_h"], proj["fov_v"],
                color=bgr
            )
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

    # -------------------------------------------------------------------------
    # PR curves (individual per-config plots)
    # -------------------------------------------------------------------------

    def _generate_pr_curves(
        self,
        pr_curves_data: Dict,
        config_id: str,
        config_name: str,
        dataset_out_dir: Path
    ):
        """Generate individual PR curve PNGs for each IoU threshold."""
        pr_curves_dir = dataset_out_dir / "pr_curves"
        pr_curves_dir.mkdir(parents=True, exist_ok=True)

        aggregator = ResultsAggregator(
            iou_thresholds=self.iou_thresholds,
            enable_pr_curves=True
        )

        for iou_threshold, pr_data in pr_curves_data.items():
            out_path = pr_curves_dir / f"{config_id}_pr_curve_iou{iou_threshold:.2f}.png"
            aggregator.generate_pr_curve_plot(
                pr_curve_data=pr_data,
                iou_threshold=iou_threshold,
                output_path=out_path,
                config_name=config_name
            )

    # -------------------------------------------------------------------------
    # Result serialization
    # -------------------------------------------------------------------------

    def _save_dataset_results(
        self, result: Dict[str, Any], dataset_out_dir: Path, dataset_name: str
    ):
        """
        Save metrics.json and timing.json for one dataset.

        metrics.json contains aggregated metrics + raw PR curve data (needed
        by the comparator for overlay plots without re-running detection).
        """
        # JSON requires string keys — float IoU thresholds → str
        metrics_per_iou_str = {
            str(k): v for k, v in result["metrics_per_iou"].items()
        }
        pr_curves_str = {
            str(k): v for k, v in result["pr_curves_data"].items()
        }

        metrics_data = {
            "config_id": result["config_id"],
            "config_name": result["config_name"],
            "dataset": dataset_name,
            "num_images": result["num_images"],
            "metrics_per_iou": metrics_per_iou_str,
            "summary": result["summary"],
            "pr_curves_data": pr_curves_str
        }
        metrics_path = dataset_out_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=2)
        print(f"  Saved: {metrics_path}")

        if result["timing"] is not None:
            timing_path = dataset_out_dir / "timing.json"
            with open(timing_path, "w") as f:
                json.dump(result["timing"], f, indent=2)
            print(f"  Saved: {timing_path}")

    def _save_config_json(
        self, config_dir: Path, datasets_meta: Dict[str, Any]
    ):
        """
        Save config.json with full run metadata, projection params, and
        per-dataset sampling indices.
        """
        # Extract pure projection params (exclude id and name)
        proj_param_keys = [
            "proj_nbr", "fov_h", "fov_v", "latitude",
            "lon_0", "lon_step", "grid", "comp_sz", "target_mp"
        ]
        projection_params = {
            k: self.config[k] for k in proj_param_keys if k in self.config
        }

        config_data = {
            "config_id": self.config["id"],
            "config_name": self.config["name"],
            "description": self.config.get("description", ""),
            "date": datetime.now().isoformat(timespec="seconds"),
            "yolo_model": self.yolo_model,
            "projection_params": projection_params,
            "datasets_evaluated": datasets_meta,
            "iou_thresholds": self.iou_thresholds,
            "nms_params": _NMS_PARAMS
        }

        config_path = config_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=4)
        print(f"  Saved: {config_path}")
