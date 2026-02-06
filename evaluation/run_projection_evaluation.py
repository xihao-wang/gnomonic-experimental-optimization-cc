"""
Main evaluation runner for projection configuration testing.

This module orchestrates the evaluation of multiple projection configurations
on multiple datasets, generating bbox predictions and visualizations.

Author: Yassir Zardoua
Date: 2025-12-04
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
import gc

from evaluation.dataset_registry import get_dataset_structure, list_available_datasets
from evaluation.output_manager import OutputPathManager
from evaluation.visualization import visualize_gt_and_predictions, visualize_composite_detections
from evaluation.bomni_dataset import BOMNIDataset
from evaluation.config import get_cfg, get_pred_bbox_color
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

        # Load evaluation config for visualization parameters
        self.eval_cfg = get_cfg()

        # Load projection configurations
        self.load_config()

        # Initialize output manager with config JSON path for versioned sessions
        self.output_manager = OutputPathManager(config_json_path=self.config_json_path)

    def load_config(self):
        """Load projection configurations from JSON file."""
        if not self.config_json_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_json_path}")

        with open(self.config_json_path, 'r') as f:
            config_data = json.load(f)

        # Validate and load YOLO model path
        self.yolo_model = self._validate_yolo_model(config_data)
        self.configurations = config_data.get("configurations", [])

        if self.verbose:
            print(f"Loaded {len(self.configurations)} projection configurations")
            print(f"YOLO model: {self.yolo_model}")

    def _validate_yolo_model(self, config_data: Dict[str, Any]) -> str:
        """
        Validate YOLO model specification in JSON config and check file existence.
        If model file doesn't exist, ask user if they want to download it automatically.

        Args:
            config_data: Parsed JSON configuration dictionary

        Returns:
            str: Path to YOLO model file

        Raises:
            ValueError: If yolo_model not specified in JSON or user declines download
            FileNotFoundError: If download fails
        """
        # Check if yolo_model is specified in JSON
        if "yolo_model" not in config_data:
            print("\n" + "="*80)
            print("ERROR: 'yolo_model' not specified in JSON configuration")
            print("="*80)
            print(f"\nConfiguration file: {self.config_json_path}")
            print("\nPlease add the YOLO model path to your JSON config:")
            print('  "yolo_model": "models/yolov8n.pt"')
            print("\n" + "="*80)
            raise ValueError("YOLO model path must be specified in JSON configuration")

        yolo_model_path = config_data["yolo_model"]
        model_path = Path(yolo_model_path)

        # Check if model file exists
        if not model_path.exists():
            # Extract model name from path
            model_name = model_path.name

            print("\n" + "="*80)
            print(f"WARNING: YOLO model file not found at: {yolo_model_path}")
            print("="*80)

            # Ask user if they want to download
            response = input(f"\nDo you want to download '{model_name}' automatically? (yes/no): ").strip().lower()

            if response in ['yes', 'y']:
                print(f"\nDownloading {model_name}...")
                try:
                    self._download_yolo_model(model_name, model_path)
                    print(f"Successfully downloaded {model_name} to {yolo_model_path}")
                except Exception as e:
                    print(f"\nERROR: Failed to download model: {e}")
                    print("\nPlease download manually or use your own model.")
                    print("Update your JSON config to point to the correct path:")
                    print(f'  "yolo_model": "path/to/your/model.pt"')
                    print("="*80)
                    raise FileNotFoundError(f"Failed to download model: {e}")
            else:
                print("\nDownload declined.")
                print("\nTo proceed, either:")
                print("1. Place your own .pt model file somewhere in the project")
                print("2. Update your JSON config to point to that path:")
                print(f'   "yolo_model": "path/to/your/model.pt"')
                print("\n" + "="*80)
                raise ValueError(f"Model file not found and download declined: {yolo_model_path}")

        return yolo_model_path

    def _download_yolo_model(self, model_name: str, destination_path: Path):
        """
        Download YOLO model from ultralytics repository.

        Args:
            model_name: Name of the model file (e.g., 'yolov8n.pt', 'yolo11x.pt')
            destination_path: Path where to save the model

        Raises:
            Exception: If download fails
        """
        from ultralytics import YOLO

        # Create parent directory if it doesn't exist
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        # Ultralytics will automatically download the model when instantiated
        # and we can then copy it to our desired location
        try:
            model = YOLO(model_name)
            # The model is now downloaded by ultralytics to its cache
            # We need to move/copy it to our specified path
            import shutil

            # Find where ultralytics cached it
            cache_path = Path.home() / '.cache' / 'ultralytics' / model_name
            if cache_path.exists():
                shutil.copy(cache_path, destination_path)
            else:
                # If not in standard cache, the model object has it loaded
                # Save it to our destination
                model.save(destination_path)
        except Exception as e:
            raise Exception(f"Failed to download {model_name}: {e}")

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

        # Create pipeline ONCE for this configuration (YOLO model loaded here)
        pipeline = self._create_detection_pipeline(config)

        # Process each image with REUSED pipeline
        for idx in range(total_images):
            item = dataset[idx]
            self._process_image(item, config, dataset_name, dataset_structure, idx, total_images, pipeline)

        # Cleanup after configuration completes
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

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

    def _create_detection_pipeline(self, config: Dict[str, Any]) -> DetectionPipeline:
        """
        Create a DetectionPipeline for a specific projection configuration.
        Pipeline will be reused across all images in this configuration.

        Args:
            config: Projection configuration dictionary

        Returns:
            DetectionPipeline: Initialized pipeline with YOLO model loaded
        """
        det_cfg = get_detection_cfg()
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
        det_cfg.OUTPUT.SAVE_COMPOSITE = False
        det_cfg.OUTPUT.SAVE_COMPOSITE_VIZ = False
        det_cfg.OUTPUT.SAVE_FISHEYE_VIZ = False
        det_cfg.OUTPUT.SAVE_LATTICE_VIZ = False
        det_cfg.VERBOSE = False

        # Create pipeline ONCE (YOLO model loaded here)
        pipeline = DetectionPipeline(det_cfg)
        return pipeline

    def _process_image(
        self,
        item: Dict,
        config: Dict[str, Any],
        dataset_name: str,
        dataset_structure: Dict,
        idx: int,
        total: int,
        pipeline: DetectionPipeline
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
            pipeline: REUSED DetectionPipeline object
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

        # Run detection with REUSED pipeline
        composite, metadata, detections, fisheye_bboxes = self._run_detection(image_path, config, pipeline)

        # Convert fisheye bboxes to standard JSON format
        predictions = self._convert_to_standard_format(fisheye_bboxes)

        # Save predictions (numeric bbox coordinates)
        self._save_predictions(predictions, config_id, dataset_name, relative_path)

        # Generate composite visualization
        composite_relative_path = relative_path.replace('.json', '.jpg')
        self._save_composite_visualization(composite, detections, config_id, dataset_name, composite_relative_path)

        # Generate fisheye visualization (GT + predictions)
        self._save_fisheye_visualization(
            image, gt_annotations, predictions, config_id, dataset_name,
            relative_path, dataset_structure["fisheye_center"]
        )

        # Explicit cleanup of large objects
        del composite
        del metadata

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

    def _run_detection(self, image_path: str, config: Dict[str, Any], pipeline: DetectionPipeline):
        """
        Run detection using REUSED pipeline.
        Only updates image path, keeps same YOLO model and projection config.

        Args:
            image_path: Path to fisheye image
            config: Projection configuration
            pipeline: REUSED DetectionPipeline object

        Returns:
            Tuple of (composite, metadata, detections, fisheye_bboxes)
        """
        # Update ONLY the image path (everything else stays the same)
        pipeline.cfg.INPUT.IMAGE_PATH = image_path

        # Run detection (YOLO model already loaded, just runs inference)
        detections, composite, metadata, results_dir, fisheye_bboxes = pipeline.run()

        return composite, metadata, detections, fisheye_bboxes

    def _convert_to_standard_format(self, fisheye_bboxes: Optional[List[Dict]]) -> List[Dict]:
        """
        Convert detection output to standard JSON format.

        Args:
            fisheye_bboxes: List of fisheye bbox dicts from backprojection (None if no detections)

        Returns:
            List of bbox dicts in standard format (empty list if no detections)
        """
        if fisheye_bboxes is None or len(fisheye_bboxes) == 0:
            return []

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
        """Save predicted bboxes in standard JSON format to both hierarchical and flattened structures."""
        # Save to hierarchical structure (existing)
        output_path_hierarchical = self.output_manager.get_bboxes_numeric_path(
            config_id, dataset_name, relative_path
        )
        self.output_manager.ensure_directories_exist(output_path_hierarchical)

        with open(output_path_hierarchical, 'w') as f:
            json.dump(predictions, f, indent=2)

        # Save to flattened structure (new)
        output_path_flattened = self.output_manager.get_all_annotations_path(
            config_id, dataset_name, relative_path
        )
        self.output_manager.ensure_directories_exist(output_path_flattened)

        with open(output_path_flattened, 'w') as f:
            json.dump(predictions, f, indent=2)

    def _save_composite_visualization(
        self,
        composite: np.ndarray,
        detections: List[Dict],
        config_id: str,
        dataset_name: str,
        relative_path: str
    ):
        """Save composite image with YOLO detections to both hierarchical and flattened structures."""
        vis_image = visualize_composite_detections(composite, detections, show_labels=True)

        # Save to hierarchical structure (existing)
        output_path_hierarchical = self.output_manager.get_bboxes_visuals_composite_path(
            config_id, dataset_name, relative_path
        )
        self.output_manager.ensure_directories_exist(output_path_hierarchical)
        cv2.imwrite(str(output_path_hierarchical), vis_image)

        # Save to flattened structure (new)
        output_path_flattened = self.output_manager.get_all_images_composite_path(
            config_id, dataset_name, relative_path
        )
        self.output_manager.ensure_directories_exist(output_path_flattened)
        cv2.imwrite(str(output_path_flattened), vis_image)

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
        """Save fisheye image with GT + predicted bboxes to both hierarchical and flattened structures."""
        # Convert relative path from .json to .jpg
        vis_relative_path = relative_path.replace('.json', '.jpg')

        # Get visualization parameters from config
        gt_color = tuple(self.eval_cfg.VISUALIZATION.GT_BBOX_COLOR)
        gt_thickness = self.eval_cfg.VISUALIZATION.GT_BBOX_THICKNESS
        pred_color = get_pred_bbox_color(self.eval_cfg)
        pred_thickness = self.eval_cfg.VISUALIZATION.PRED_BBOX_THICKNESS

        vis_image = visualize_gt_and_predictions(
            image, gt_annotations, pred_annotations,
            fisheye_center=fisheye_center,
            show_legend=True,
            gt_color=gt_color,
            gt_thickness=gt_thickness,
            pred_color=pred_color,
            pred_thickness=pred_thickness
        )

        # Save to hierarchical structure (existing)
        output_path_hierarchical = self.output_manager.get_bboxes_visuals_fisheye_path(
            config_id, dataset_name, vis_relative_path
        )
        self.output_manager.ensure_directories_exist(output_path_hierarchical)
        cv2.imwrite(str(output_path_hierarchical), vis_image)

        # Save to flattened structure (new)
        output_path_flattened = self.output_manager.get_all_images_fisheye_path(
            config_id, dataset_name, vis_relative_path
        )
        self.output_manager.ensure_directories_exist(output_path_flattened)
        cv2.imwrite(str(output_path_flattened), vis_image)
