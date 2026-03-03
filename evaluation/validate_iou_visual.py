"""
IoU Visual Validation Script

This script validates the rotated rectangle IoU implementation by:
1. Running detection pipeline on fisheye images
2. Backprojecting detections to fisheye coordinates (rotated bboxes)
3. Computing IoU between predictions and ground truth
4. Generating debug visualizations:
   - All GT bboxes on one fisheye image
   - Each prediction on its own fisheye image with IoU value

Output: evaluation/iou-visual-validation/

Author: Generated for gnomonic projection pedestrian detection project
Date: 2026-02-06
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import json
from typing import List, Dict, Tuple
from tqdm import tqdm

from evaluation.lib.config import get_cfg
from evaluation.lib.dataset_registry import get_dataset_structure
from evaluation.lib.bomni_dataset import BOMNIDataset
from evaluation.lib.metrics import compute_rotated_iou
from datasets.utils.visualization import draw_rotated_bbox_on_image
from detection_pipeline.pipeline import DetectionPipeline


# =============================================================================
# CONFIGURATION
# =============================================================================
# All configuration is now in evaluation/lib/config.py under IOU_VALIDATION node
# Edit that file to change:
#   - CONFIG_JSON: Path to projection config JSON (None = auto-detect)
#   - CONFIG_ID: Which config to use (None = first)
#   - NUM_SAMPLES: Number of images to process (None = all)
#   - DATASET_NAME: Which dataset to use
#   - OUTPUT_DIR: Where to save validation visualizations
# =============================================================================


def create_output_folders(base_dir: Path) -> Tuple[Path, Path]:
    """Create output folders for GT and predictions."""
    gt_dir = base_dir / "ground_truth"
    pred_dir = base_dir / "predictions"

    gt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    return gt_dir, pred_dir


def visualize_ground_truth(
    fisheye_image: np.ndarray,
    gt_annotations: List[Dict],
    fisheye_center: Tuple[float, float]
) -> np.ndarray:
    """
    Visualize all ground truth bboxes on fisheye image.

    Args:
        fisheye_image: Original fisheye image
        gt_annotations: List of GT annotation dicts
        fisheye_center: Fisheye center coordinates

    Returns:
        Image with all GT bboxes drawn (green)
    """
    vis_image = fisheye_image.copy()

    if len(gt_annotations) > 0:
        vis_image = draw_rotated_bbox_on_image(
            image=vis_image,
            annotations=gt_annotations,
            bbox_color=(0, 255, 0),  # Green for GT
            bbox_thickness=2,
            show_labels=True,
            show_rotation_angle=False,
            draw_rotated=True,
            draw_center_and_radial=False,
            fisheye_center=fisheye_center
        )

    # Add label
    cv2.putText(
        vis_image,
        f"Ground Truth ({len(gt_annotations)} bboxes)",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )

    return vis_image


def visualize_prediction_with_iou(
    fisheye_image: np.ndarray,
    gt_annotations: List[Dict],
    pred_annotation: Dict,
    pred_idx: int,
    ious: List[float],
    fisheye_center: Tuple[float, float]
) -> np.ndarray:
    """
    Visualize single prediction with all GT bboxes and IoU values.

    Args:
        fisheye_image: Original fisheye image
        gt_annotations: List of GT annotation dicts
        pred_annotation: Single prediction annotation dict
        pred_idx: Index of this prediction
        ious: List of IoU values for this prediction against all GT bboxes
        fisheye_center: Fisheye center coordinates

    Returns:
        Image with GT (green) + prediction (yellow) + IoU values
    """
    vis_image = fisheye_image.copy()

    # Draw all GT bboxes first (green, thin)
    if len(gt_annotations) > 0:
        vis_image = draw_rotated_bbox_on_image(
            image=vis_image,
            annotations=gt_annotations,
            bbox_color=(0, 255, 0),  # Green for GT
            bbox_thickness=1,
            show_labels=False,
            show_rotation_angle=False,
            draw_rotated=True,
            draw_center_and_radial=False,
            fisheye_center=fisheye_center
        )

    # Draw prediction bbox (yellow, thick)
    vis_image = draw_rotated_bbox_on_image(
        image=vis_image,
        annotations=[pred_annotation],
        bbox_color=(0, 255, 255),  # Yellow for prediction
        bbox_thickness=3,
        show_labels=True,
        show_rotation_angle=False,
        draw_rotated=True,
        draw_center_and_radial=False,
        fisheye_center=fisheye_center
    )

    # Add IoU values for each GT bbox
    max_iou = max(ious) if len(ious) > 0 else 0.0
    max_iou_idx = ious.index(max_iou) if len(ious) > 0 else -1

    # Label with prediction index and max IoU
    label = f"Prediction {pred_idx} | Max IoU: {max_iou:.3f}"
    if max_iou_idx >= 0:
        label += f" (GT #{max_iou_idx})"

    cv2.putText(
        vis_image,
        label,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2
    )

    # Add IoU values for all GT bboxes (in text overlay)
    y_offset = 70
    for i, iou in enumerate(ious):
        iou_text = f"GT #{i}: IoU = {iou:.3f}"
        color = (0, 255, 255) if i == max_iou_idx else (200, 200, 200)
        cv2.putText(
            vis_image,
            iou_text,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )
        y_offset += 35

    return vis_image


def validate_iou_on_dataset(
    dataset: BOMNIDataset,
    pipeline: DetectionPipeline,
    cfg,
    num_samples: int = None
) -> None:
    """
    Run IoU validation on dataset.

    Args:
        dataset: Dataset to process
        pipeline: Detection pipeline
        cfg: Configuration
        num_samples: Number of images to process (None = all)
    """
    # Create output folders
    output_dir = Path(cfg.IOU_VALIDATION.OUTPUT_DIR)
    gt_dir, pred_dir = create_output_folders(output_dir)

    print(f"\n{'='*70}")
    print(f"IoU Visual Validation")
    print(f"{'='*70}")
    print(f"Dataset: {cfg.IOU_VALIDATION.DATASET_NAME.upper()}")
    print(f"Total images: {len(dataset)}")
    print(f"Processing: {num_samples if num_samples else 'ALL'} images")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")

    # Determine number of images to process
    total_images = len(dataset)
    if num_samples is None or num_samples > total_images:
        num_images = total_images
    else:
        num_images = num_samples

    # Fisheye center
    fisheye_center = (cfg.DATASETS.BOMNI.FISHEYE_CENTER_X, cfg.DATASETS.BOMNI.FISHEYE_CENTER_Y)

    # Process images
    for img_idx in tqdm(range(num_images), desc="Processing images"):
        # Load image and annotations
        item = dataset[img_idx]
        fisheye_image = item['image']
        gt_annotations = item['annotations']
        image_name = item['image_name']
        image_path = item['image_path']

        # Run detection pipeline
        pipeline.cfg.INPUT.IMAGE_PATH = str(image_path)
        detections, composite, metadata, results_dir, fisheye_bboxes = pipeline.run()

        # Convert fisheye_bboxes to standard format
        pred_annotations = []
        if fisheye_bboxes is not None:
            for bbox in fisheye_bboxes:
                pred_annotations.append({
                    "center_x": float(bbox["center"][0]),
                    "center_y": float(bbox["center"][1]),
                    "width": float(bbox["size"][0]),
                    "height": float(bbox["size"][1]),
                    "angle": float(bbox["angle"]),
                    "class_name": bbox["class_name"]
                })

        # Save GT visualization (all GT bboxes on one image)
        gt_vis = visualize_ground_truth(fisheye_image, gt_annotations, fisheye_center)
        gt_output_path = gt_dir / f"{image_name}_gt.jpg"
        cv2.imwrite(str(gt_output_path), gt_vis)

        # Process each prediction
        for pred_idx, pred_bbox in enumerate(pred_annotations):
            # Compute IoU with all GT bboxes
            ious = [compute_rotated_iou(pred_bbox, gt_bbox) for gt_bbox in gt_annotations]

            # Visualize prediction with IoU values
            pred_vis = visualize_prediction_with_iou(
                fisheye_image,
                gt_annotations,
                pred_bbox,
                pred_idx,
                ious,
                fisheye_center
            )

            # Save prediction visualization
            pred_output_path = pred_dir / f"{image_name}_pred{pred_idx:02d}.jpg"
            cv2.imwrite(str(pred_output_path), pred_vis)

        # If no predictions, save a note
        if len(pred_annotations) == 0:
            no_pred_vis = fisheye_image.copy()
            cv2.putText(
                no_pred_vis,
                "No predictions",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2
            )
            pred_output_path = pred_dir / f"{image_name}_no_predictions.jpg"
            cv2.imwrite(str(pred_output_path), no_pred_vis)

    print(f"\n{'='*70}")
    print(f"Validation complete!")
    print(f"GT visualizations: {gt_dir}")
    print(f"Prediction visualizations: {pred_dir}")
    print(f"{'='*70}\n")


def load_projection_config(config_json_path: str = None, config_id: str = None) -> Tuple[str, Dict]:
    """
    Load projection configuration from JSON file.

    Args:
        config_json_path: Path to JSON file, or None to auto-detect
        config_id: Config ID to select, or None for first

    Returns:
        Tuple of (config_id, projection_config_dict)

    Raises:
        FileNotFoundError: If no JSON files found or specified path doesn't exist
        ValueError: If JSON is invalid or config not found
    """
    # Auto-detect first JSON file if not specified
    if config_json_path is None:
        config_dir = Path("evaluation/projection_configs")
        if not config_dir.exists():
            raise FileNotFoundError(
                f"Configuration directory not found: {config_dir}\n"
                f"Please create the directory and add projection config JSON files."
            )

        json_files = list(config_dir.glob("*.json"))
        if len(json_files) == 0:
            raise FileNotFoundError(
                f"No JSON files found in {config_dir}\n"
                f"Please add at least one projection configuration JSON file.\n"
                f"To specify a custom path, edit cfg.IOU_VALIDATION.CONFIG_JSON in evaluation/lib/config.py"
            )

        config_json_path = str(json_files[0])
        print(f"Auto-detected config file: {config_json_path}")

    # Load JSON file
    config_path = Path(config_json_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_json_path}\n"
            f"Please check cfg.IOU_VALIDATION.CONFIG_JSON in evaluation/lib/config.py"
        )

    with open(config_path, 'r') as f:
        data = json.load(f)

    # Handle JSON structure: {"configurations": [...]}
    if isinstance(data, dict) and "configurations" in data:
        configs = data["configurations"]
    elif isinstance(data, list):
        configs = data
    else:
        raise ValueError(
            f"Invalid JSON format in {config_json_path}\n"
            f"Expected either a list of configs or a dict with 'configurations' key."
        )

    if len(configs) == 0:
        raise ValueError(f"No configurations found in {config_json_path}")

    # Select configuration
    if config_id is None:
        # Use first configuration
        selected_config = configs[0]
        selected_id = selected_config.get('id', 'config_0')
        print(f"Using first configuration: {selected_id}")
    else:
        # Find configuration by ID
        selected_config = None
        for config in configs:
            if config.get('id') == config_id:
                selected_config = config
                selected_id = config_id
                break

        if selected_config is None:
            available_ids = [c.get('id', 'unknown') for c in configs]
            raise ValueError(
                f"Configuration ID '{config_id}' not found in {config_json_path}\n"
                f"Available IDs: {', '.join(available_ids)}\n"
                f"To use the first config, set cfg.IOU_VALIDATION.CONFIG_ID = None"
            )

    # Map JSON fields to DetectionPipeline format
    # JSON format: fov_h, fov_v, grid, comp_sz, etc.
    # Pipeline format: fov_h_deg, fov_v_deg, grid_rows, grid_cols, composite_width, composite_height, etc.
    grid = selected_config.get('grid', [3, 3])
    comp_sz = selected_config.get('comp_sz', [640, 640])

    projection_config = {
        'num_projections': selected_config.get('proj_nbr', 9),
        'grid_rows': grid[0],
        'grid_cols': grid[1],
        'fov_h_deg': selected_config.get('fov_h', 70.0),
        'fov_v_deg': selected_config.get('fov_v', 70.0),
        'lat_offset_deg': selected_config.get('latitude', 45.0),
        'lon_offset_deg': selected_config.get('lon_0', 0.0),
        'composite_width': comp_sz[0],
        'composite_height': comp_sz[1]
    }

    return selected_id, projection_config


def main():
    """Main entry point."""
    # Load config
    cfg = get_cfg()
    cfg.VERBOSE = False

    # Get dataset name
    dataset_name = cfg.IOU_VALIDATION.DATASET_NAME.lower()

    # Override dataset paths from IOU_VALIDATION config
    if dataset_name == "bomni":
        cfg.DATASETS.BOMNI.ROOT_DIR = cfg.IOU_VALIDATION.BOMNI.DATASET_ROOT
        cfg.DATASETS.BOMNI.FRAMES_DIR = cfg.IOU_VALIDATION.BOMNI.FRAMES_DIR
        cfg.DATASETS.BOMNI.STANDARD_ANNOTATIONS_DIR = cfg.IOU_VALIDATION.BOMNI.ANNOTATIONS_DIR
        cfg.DATASETS.BOMNI.SEQUENCES = cfg.IOU_VALIDATION.BOMNI.SEQUENCES
        cfg.DATASETS.BOMNI.FISHEYE_CENTER_X = cfg.IOU_VALIDATION.BOMNI.IMAGE_CENTER_X
        cfg.DATASETS.BOMNI.FISHEYE_CENTER_Y = cfg.IOU_VALIDATION.BOMNI.IMAGE_CENTER_Y

    # Load projection configuration from JSON file
    try:
        config_id, projection_config = load_projection_config(
            cfg.IOU_VALIDATION.CONFIG_JSON,
            cfg.IOU_VALIDATION.CONFIG_ID
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"\n{'='*70}")
        print("ERROR: Failed to load projection configuration")
        print(f"{'='*70}")
        print(f"\n{e}\n")
        print(f"{'='*70}\n")
        return

    # Create dataset
    print(f"\nLoading {dataset_name.upper()} dataset...")
    dataset = BOMNIDataset(cfg)
    print(f"Dataset loaded: {len(dataset)} images")

    # Create detection pipeline with IOU_VALIDATION settings
    from detection_pipeline.config import get_cfg as get_detection_cfg
    detection_cfg = get_detection_cfg()

    # Set projection config
    detection_cfg.PROJECTION.PRESET = None
    detection_cfg.PROJECTION.PROJ_NBR = projection_config['num_projections']
    detection_cfg.PROJECTION.FOV_H = projection_config['fov_h_deg']
    detection_cfg.PROJECTION.FOV_V = projection_config['fov_v_deg']
    detection_cfg.PROJECTION.LATITUDE = projection_config['lat_offset_deg']
    detection_cfg.PROJECTION.LON_0 = projection_config['lon_offset_deg']
    detection_cfg.PROJECTION.LON_STEP = 0.0  # Calculated by image_composer
    detection_cfg.PROJECTION.GRID = (projection_config['grid_rows'], projection_config['grid_cols'])
    detection_cfg.PROJECTION.COMP_SIZE = (projection_config['composite_width'], projection_config['composite_height'])
    detection_cfg.PROJECTION.TARGET_MP = "auto"

    # Set YOLO config
    detection_cfg.YOLO.MODEL = cfg.IOU_VALIDATION.YOLO_MODEL_PATH
    detection_cfg.YOLO.CONF = cfg.IOU_VALIDATION.CONF_THRESHOLD

    # Disable saving outputs
    detection_cfg.OUTPUT.SAVE_COMPOSITE = False
    detection_cfg.OUTPUT.SAVE_COMPOSITE_VIZ = False
    detection_cfg.OUTPUT.SAVE_FISHEYE_VIZ = False
    detection_cfg.OUTPUT.SAVE_LATTICE_VIZ = False
    detection_cfg.VERBOSE = False

    print(f"\nInitializing detection pipeline...")
    print(f"  Config ID: {config_id}")
    print(f"  Model: {Path(cfg.IOU_VALIDATION.YOLO_MODEL_PATH).name}")
    print(f"  Confidence threshold: {cfg.IOU_VALIDATION.CONF_THRESHOLD}")
    print(f"  Projection: {projection_config.get('grid_rows')}x{projection_config.get('grid_cols')} grid, "
          f"{projection_config.get('fov_h_deg')}deg FOV")

    pipeline = DetectionPipeline(detection_cfg)

    # Run validation
    num_samples = cfg.IOU_VALIDATION.NUM_SAMPLES
    validate_iou_on_dataset(dataset, pipeline, cfg, num_samples)


if __name__ == "__main__":
    main()
