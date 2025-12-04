"""
Main evaluation runner for projection configuration testing.

This module orchestrates the evaluation of multiple projection configurations
on multiple datasets, generating bbox predictions and visualizations.

Author: Generated for gnomonic projection pedestrian detection project
Date: 2025-12-04
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from evaluation.dataset_registry import get_dataset_structure, list_available_datasets
from evaluation.output_manager import OutputPathManager
from evaluation.visualization import visualize_gt_and_predictions, visualize_composite_detections
from evaluation.bomni_dataset import BOMNIDataset
from evaluation.config import get_cfg
from detection_pipeline.config import get_cfg as get_detection_cfg
from detection_pipeline.pipeline import DetectionPipeline
from image_composer import generate_composite_from_config


class ProjectionEvaluator:
    """Main evaluator for testing projection configurations on datasets."""

    def __init__(
        self,
        config_json_path: str,
        dataset_roots: Dict[str, str],
        verbose: bool = True,
        max_images: Optional[int] = None
    ):
        """
        Initialize projection evaluator.

        Args:
            config_json_path: Path to JSON configuration file
            dataset_roots: Dictionary mapping dataset names to root paths
            verbose: Whether to print progress
            max_images: Maximum images to process per dataset (None = all)
        """
        self.config_json_path = Path(config_json_path)
        self.dataset_roots = dataset_roots
        self.verbose = verbose
        self.max_images = max_images

        # Load projection configurations
        self.load_config()

        # Initialize output manager
        self.output_manager = OutputPathManager()

        # Extract JSON filename (without extension) for output folders
        self.json_name = self.config_json_path.stem

    def load_config(self):
        """Load projection configurations from JSON file."""
        if not self.config_json_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_json_path}")

        with open(self.config_json_path, 'r') as f:
            config_data = json.load(f)

        self.yolo_model = config_data.get("yolo_model", "detection_pipeline/models/yolov8n.pt")
        self.configurations = config_data.get("configurations", [])

        if self.verbose:
            print(f"Loaded {len(self.configurations)} projection configurations")
            print(f"YOLO model: {self.yolo_model}")

    def run_evaluation(self, dataset_names: List[str]):
        """
        Run evaluation on specified datasets.

        Args:
            dataset_names: List of dataset names to evaluate
        """
        # Validate datasets
        for dataset_name in dataset_names:
            if dataset_name not in get_dataset_structure.__globals__['DATASET_STRUCTURES']:
                raise ValueError(f"Dataset '{dataset_name}' not found in registry")
            if dataset_name not in self.dataset_roots:
                raise ValueError(f"Dataset root not specified for '{dataset_name}'")

        # Run evaluation for each configuration
        for config in self.configurations:
            config_id = config["id"]
            config_name = config["name"]

            if self.verbose:
                print(f"\n{'='*80}")
                print(f"Configuration: {config_name} (ID: {config_id})")
                print(f"{'='*80}")

            # Evaluate on each dataset
            for dataset_name in dataset_names:
                if self.verbose:
                    print(f"\n[{dataset_name.upper()}] Starting evaluation...")

                self._evaluate_dataset(config, dataset_name)

    def _evaluate_dataset(self, config: Dict[str, Any], dataset_name: str):
        """
        Evaluate a single configuration on a single dataset.

        Args:
            config: Projection configuration dictionary
            dataset_name: Name of dataset
        """
        config_id = config["id"]
        dataset_root = Path(self.dataset_roots[dataset_name])
        dataset_structure = get_dataset_structure(dataset_name)

        # Load dataset
        if dataset_name == "bomni":
            dataset = self._load_bomni_dataset(dataset_root, dataset_structure)
        else:
            raise NotImplementedError(f"Dataset loader for '{dataset_name}' not implemented")

        # Limit images if specified
        total_images = len(dataset)
        if self.max_images:
            total_images = min(self.max_images, total_images)

        if self.verbose:
            print(f"Processing {total_images} images...")

        # Process each image
        for idx in range(total_images):
            item = dataset[idx]
            self._process_image(item, config, dataset_name, dataset_structure, idx, total_images)

        if self.verbose:
            print(f"\n[{dataset_name.upper()}] Evaluation complete for config '{config_id}'")

    def _load_bomni_dataset(self, dataset_root: Path, dataset_structure: Dict) -> BOMNIDataset:
        """Load BOMNI dataset."""
        cfg = get_cfg()
        cfg.DATASETS.BOMNI.STANDARD_ANNOTATIONS_DIR = str(
            dataset_root / dataset_structure["annotation_root"] / "scenario1"
        )
        cfg.DATASETS.BOMNI.FRAMES_DIR = str(
            dataset_root / dataset_structure["frames_root"] / "scenario1"
        )
        cfg.DATASETS.BOMNI.SEQUENCES = dataset_structure["sequences"]["scenario1"]
        cfg.VERBOSE = False  # Suppress dataset loading messages

        return BOMNIDataset(cfg)

    def _process_image(
        self,
        item: Dict,
        config: Dict[str, Any],
        dataset_name: str,
        dataset_structure: Dict,
        idx: int,
        total: int
    ):
        """
        Process a single image: detect, backproject, save predictions, visualize.

        Args:
            item: Dataset item (image, annotations, metadata)
            config: Projection configuration
            dataset_name: Dataset name
            dataset_structure: Dataset structure from registry
            idx: Current image index
            total: Total images
        """
        config_id = config["id"]
        image = item["image"]
        gt_annotations = item["annotations"]
        image_path = item["image_path"]
        image_name = Path(image_path).stem

        if self.verbose and (idx + 1) % 10 == 0:
            print(f"  Processing {idx+1}/{total}...")

        # Compute relative path for this image (dataset-specific structure)
        relative_path = self._compute_relative_path(item, dataset_name)

        # Run detection with this projection configuration
        composite, metadata, detections, fisheye_bboxes = self._run_detection(image_path, config)

        # Convert fisheye bboxes to standard JSON format
        predictions = self._convert_to_standard_format(fisheye_bboxes)

        # Save predictions (numeric bbox coordinates)
        self._save_predictions(predictions, config_id, dataset_name, relative_path)

        # Generate composite visualization
        self._save_composite_visualization(composite, detections, config_id, dataset_name, image_name)

        # Generate fisheye visualization (GT + predictions)
        self._save_fisheye_visualization(
            image, gt_annotations, predictions, config_id, dataset_name,
            relative_path, dataset_structure["fisheye_center"]
        )

    def _compute_relative_path(self, item: Dict, dataset_name: str) -> str:
        """
        Compute relative path for an image based on dataset structure.

        Args:
            item: Dataset item
            dataset_name: Dataset name

        Returns:
            Relative path string (e.g., "scenario1/top-0/0001.json")
        """
        if dataset_name == "bomni":
            sequence = item["sequence"]
            image_name = Path(item["image_path"]).stem
            return f"scenario1/{sequence}/{image_name}.json"
        else:
            raise NotImplementedError(f"Relative path computation for '{dataset_name}' not implemented")

    def _run_detection(self, image_path: str, config: Dict[str, Any]):
        """
        Run detection pipeline with specified projection configuration.

        Args:
            image_path: Path to fisheye image
            config: Projection configuration

        Returns:
            Tuple of (composite, metadata, detections, fisheye_bboxes)
        """
        # Configure detection pipeline
        det_cfg = get_detection_cfg()
        det_cfg.INPUT.IMAGE_PATH = image_path
        det_cfg.PROJECTION.PRESET = None  # Use custom config
        det_cfg.PROJECTION.PROJ_NBR = config["proj_nbr"]
        det_cfg.PROJECTION.FOV_H = config["fov_h"]
        det_cfg.PROJECTION.FOV_V = config["fov_v"]
        det_cfg.PROJECTION.LATITUDE = config["latitude"]
        det_cfg.PROJECTION.LON_0 = config["lon_0"]
        det_cfg.PROJECTION.LON_STEP = config["lon_step"]
        det_cfg.PROJECTION.GRID = tuple(config["grid"])
        det_cfg.PROJECTION.COMP_SIZE = tuple(config["comp_sz"])
        det_cfg.PROJECTION.TARGET_MP = config["target_mp"]
        det_cfg.YOLO.MODEL = self.yolo_model
        det_cfg.OUTPUT.SAVE_COMPOSITE = False  # Don't save to pipeline results
        det_cfg.OUTPUT.SAVE_COMPOSITE_VIZ = False
        det_cfg.OUTPUT.SAVE_FISHEYE_VIZ = False
        det_cfg.OUTPUT.SAVE_LATTICE_VIZ = False
        det_cfg.VERBOSE = False

        # Run detection
        pipeline = DetectionPipeline(det_cfg)
        detections, composite, metadata, results_dir, fisheye_bboxes = pipeline.run()

        return composite, metadata, detections, fisheye_bboxes

    def _convert_to_standard_format(self, fisheye_bboxes: List[Dict]) -> List[Dict]:
        """
        Convert detection output to standard JSON format.

        Args:
            fisheye_bboxes: List of fisheye bbox dicts from backprojection

        Returns:
            List of bbox dicts in standard format
        """
        standard_bboxes = []
        for bbox in fisheye_bboxes:
            standard_bboxes.append({
                "center_x": float(bbox["center"][0]),
                "center_y": float(bbox["center"][1]),
                "width": float(bbox["size"][0]),
                "height": float(bbox["size"][1]),
                "angle": float(bbox["angle"]),
                "class_name": bbox["class_name"]
            })
        return standard_bboxes

    def _save_predictions(
        self,
        predictions: List[Dict],
        config_id: str,
        dataset_name: str,
        relative_path: str
    ):
        """Save predicted bboxes in standard JSON format."""
        output_path = self.output_manager.get_bboxes_numeric_path(
            self.json_name, config_id, dataset_name, relative_path
        )
        self.output_manager.ensure_directories_exist(output_path)

        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=2)

    def _save_composite_visualization(
        self,
        composite: np.ndarray,
        detections: List[Dict],
        config_id: str,
        dataset_name: str,
        image_name: str
    ):
        """Save composite image with YOLO detections."""
        vis_image = visualize_composite_detections(composite, detections, show_labels=True)

        output_path = self.output_manager.get_bboxes_visuals_composite_path(
            self.json_name, config_id, dataset_name, f"{image_name}.jpg"
        )
        self.output_manager.ensure_directories_exist(output_path)

        cv2.imwrite(str(output_path), vis_image)

    def _save_fisheye_visualization(
        self,
        image: np.ndarray,
        gt_annotations: List[Dict],
        pred_annotations: List[Dict],
        config_id: str,
        dataset_name: str,
        relative_path: str,
        fisheye_center: tuple
    ):
        """Save fisheye image with GT + predicted bboxes."""
        # Convert relative path from .json to .jpg
        vis_relative_path = relative_path.replace('.json', '.jpg')

        vis_image = visualize_gt_and_predictions(
            image, gt_annotations, pred_annotations,
            fisheye_center=fisheye_center,
            show_legend=True
        )

        output_path = self.output_manager.get_bboxes_visuals_fisheye_path(
            self.json_name, config_id, dataset_name, vis_relative_path
        )
        self.output_manager.ensure_directories_exist(output_path)

        cv2.imwrite(str(output_path), vis_image)
