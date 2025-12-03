"""
Base dataset manager class for pedestrian detection datasets.

This abstract base class defines the interface for dataset preparation operations
including format conversion, visualization, and validation. Each dataset should
subclass this and implement dataset-specific operations.
"""

import json
import cv2
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

from datasets.utils.visualization import draw_rotated_bbox_on_image, save_annotated_image


class BaseDatasetManager(ABC):
    """
    Abstract base class for dataset management.

    All datasets should convert their native annotation format to our standard JSON format:
    {
        "center_x": float,
        "center_y": float,
        "width": float,      # Tight-fit rotated bbox width
        "height": float,     # Tight-fit rotated bbox height
        "angle": float,      # Rotation angle in degrees (0=vertical up, clockwise positive)
        "class_name": string
    }

    Subclasses must implement:
    - convert_to_standard_format(): Convert native annotations to standard JSON
    - get_dataset_name(): Return lowercase dataset identifier (e.g., "bomni", "piropo")

    Subclasses may implement dataset-specific operations (e.g., frame extraction, cleanup).
    """

    def __init__(self, cfg):
        """
        Initialize dataset manager.

        Args:
            cfg: YACS config object with dataset-specific settings
        """
        self.cfg = cfg
        self.verbose = cfg.VERBOSE

    @abstractmethod
    def convert_to_standard_format(self, **kwargs):
        """
        Convert dataset's native annotation format to our standard JSON format.

        This is the PRIMARY method each dataset must implement. After conversion,
        all datasets share the same format and can use shared visualization/validation logic.

        Args:
            **kwargs: Dataset-specific parameters (e.g., input_format, sequences, etc.)

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement convert_to_standard_format()")

    @abstractmethod
    def get_dataset_name(self) -> str:
        """
        Return dataset identifier (lowercase).

        Returns:
            Dataset name string (e.g., "bomni", "piropo", "fisheye8k")

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement get_dataset_name()")

    def visualize_annotations(
        self,
        annotations_dir: str,
        frames_dir: str,
        output_dir: Optional[str] = None,
        sequences: Optional[List[str]] = None,
        max_images: Optional[int] = None,
        fisheye_center: Optional[tuple] = None,
        image_ext: str = ".jpg"
    ):
        """
        Visualize annotations from standard JSON format (shared implementation).

        Reads standard JSON annotations, draws rotated bboxes, saves to:
        {output_dir}/{dataset_name}/annotation_visualization/{sequence}/

        Args:
            annotations_dir: Directory containing standard JSON annotations
            frames_dir: Directory containing image frames
            output_dir: Output directory for visualizations (default: from config)
            sequences: List of sequence names to visualize (default: all sequences)
            max_images: Max images per sequence ("all" or integer, default: from config)
            fisheye_center: Image center coordinates (default: from config)
            image_ext: Image file extension (default: ".jpg")
        """
        annotations_dir = Path(annotations_dir)
        frames_dir = Path(frames_dir)

        # Set defaults from config
        if output_dir is None:
            output_dir = Path(self.cfg.OUTPUT.ROOT_DIR) / self.get_dataset_name() / "annotation_visualization"
        else:
            output_dir = Path(output_dir) / self.get_dataset_name() / "annotation_visualization"

        if max_images is None:
            max_images = self.cfg.VISUALIZATION.MAX_IMAGES_PER_SEQUENCE

        output_dir.mkdir(parents=True, exist_ok=True)

        # Get visualization parameters from config
        bbox_color = tuple(self.cfg.VISUALIZATION.BBOX_COLOR)
        bbox_thickness = self.cfg.VISUALIZATION.BBOX_THICKNESS
        font_scale = self.cfg.VISUALIZATION.FONT_SCALE
        font_thickness = self.cfg.VISUALIZATION.FONT_THICKNESS
        show_labels = self.cfg.VISUALIZATION.SHOW_LABELS
        show_rotation_angle = self.cfg.VISUALIZATION.SHOW_ROTATION_ANGLE
        draw_rotated = self.cfg.VISUALIZATION.DRAW_ROTATED
        draw_center_and_radial = self.cfg.VISUALIZATION.DRAW_CENTER_AND_RADIAL
        image_format = self.cfg.VISUALIZATION.IMAGE_FORMAT
        image_quality = self.cfg.VISUALIZATION.IMAGE_QUALITY

        print(f"\nVisualizing {self.get_dataset_name().upper()} annotations...")
        print(f"Output directory: {output_dir}")

        # Find sequences if not specified
        if sequences is None:
            sequences = [d.name for d in annotations_dir.iterdir() if d.is_dir()]

        for sequence in sequences:
            sequence_annotations_dir = annotations_dir / sequence
            sequence_frames_dir = frames_dir / sequence
            sequence_output_dir = output_dir / sequence

            if not sequence_annotations_dir.exists():
                print(f"  [WARNING] Annotations not found: {sequence_annotations_dir}")
                continue

            if not sequence_frames_dir.exists():
                print(f"  [WARNING] Frames not found: {sequence_frames_dir}")
                continue

            sequence_output_dir.mkdir(parents=True, exist_ok=True)

            # Get annotation files
            annotation_files = sorted(sequence_annotations_dir.glob("*.json"))

            # Determine how many images to visualize
            total_available = len(annotation_files)
            if max_images is None or max_images == "all":
                num_to_visualize = total_available
                viz_mode = "all"
            elif isinstance(max_images, int):
                if max_images >= total_available:
                    num_to_visualize = total_available
                    viz_mode = f"all (requested {max_images}, only {total_available} available)"
                else:
                    num_to_visualize = max_images
                    viz_mode = f"{num_to_visualize}/{total_available}"
                    annotation_files = annotation_files[:num_to_visualize]
            else:
                num_to_visualize = total_available
                viz_mode = "all"

            print(f"  {sequence}: Visualizing {viz_mode} images...")

            for annotation_file in tqdm(annotation_files, desc=f"    {sequence}"):
                # Load annotation
                with open(annotation_file, 'r') as f:
                    annotations = json.load(f)

                # Get corresponding image
                image_name = annotation_file.stem + image_ext
                image_path = sequence_frames_dir / image_name

                if not image_path.exists():
                    if self.verbose:
                        print(f"    [WARNING] Image not found: {image_path}")
                    continue

                # Load image
                image = cv2.imread(str(image_path))
                if image is None:
                    if self.verbose:
                        print(f"    [WARNING] Could not load image: {image_path}")
                    continue

                # Draw annotations
                image = draw_rotated_bbox_on_image(
                    image=image,
                    annotations=annotations,
                    bbox_color=bbox_color,
                    bbox_thickness=bbox_thickness,
                    font_scale=font_scale,
                    font_thickness=font_thickness,
                    show_labels=show_labels,
                    show_rotation_angle=show_rotation_angle,
                    draw_rotated=draw_rotated,
                    draw_center_and_radial=draw_center_and_radial,
                    fisheye_center=fisheye_center
                )

                # Save visualization
                output_filename = f"annotated_{image_name}"
                if image_format != "jpg":
                    output_filename = output_filename.replace(".jpg", f".{image_format}")
                output_path = sequence_output_dir / output_filename

                save_annotated_image(image, str(output_path), image_format, image_quality)

        print(f"\nVisualization complete! Results saved to: {output_dir}")

    def validate(self, annotations_dir: str, frames_dir: str, sequences: List[str]):
        """
        Validate dataset integrity (shared implementation).

        Checks:
        - All annotation files have corresponding frame images
        - All annotations are in correct JSON format
        - All required fields present

        Args:
            annotations_dir: Directory containing standard JSON annotations
            frames_dir: Directory containing image frames
            sequences: List of sequence names to validate

        Returns:
            dict: Validation results with counts and errors
        """
        annotations_dir = Path(annotations_dir)
        frames_dir = Path(frames_dir)

        print(f"\n{'='*70}")
        print(f"Validating {self.get_dataset_name().upper()} Dataset")
        print(f"{'='*70}")

        total_annotations = 0
        total_images = 0
        errors = []

        for sequence in sequences:
            sequence_annotations_dir = annotations_dir / sequence
            sequence_frames_dir = frames_dir / sequence

            if not sequence_annotations_dir.exists():
                errors.append(f"Sequence annotations not found: {sequence}")
                continue

            if not sequence_frames_dir.exists():
                errors.append(f"Sequence frames not found: {sequence}")
                continue

            annotation_files = list(sequence_annotations_dir.glob("*.json"))
            print(f"\n{sequence}: {len(annotation_files)} annotation files")

            for annotation_file in annotation_files:
                # Check corresponding image exists
                image_name = annotation_file.stem + ".jpg"
                image_path = sequence_frames_dir / image_name

                if not image_path.exists():
                    errors.append(f"Missing image: {image_path}")
                    continue

                total_images += 1

                # Validate JSON format
                try:
                    with open(annotation_file, 'r') as f:
                        annotations = json.load(f)

                    if not isinstance(annotations, list):
                        errors.append(f"Invalid format (not list): {annotation_file}")
                        continue

                    for ann in annotations:
                        # Check required fields
                        required_fields = ["center_x", "center_y", "width", "height", "angle", "class_name"]
                        for field in required_fields:
                            if field not in ann:
                                errors.append(f"Missing field '{field}': {annotation_file}")

                        total_annotations += 1

                except Exception as e:
                    errors.append(f"Error reading {annotation_file}: {e}")

        print(f"\n{'-'*70}")
        print(f"Total images: {total_images}")
        print(f"Total annotations: {total_annotations}")
        print(f"Errors: {len(errors)}")

        if errors:
            print(f"\n{'='*70}")
            print("ERRORS FOUND:")
            for error in errors[:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors")
        else:
            print(f"\n[PASS] Dataset validation successful!")

        print(f"{'='*70}\n")

        return {
            "total_images": total_images,
            "total_annotations": total_annotations,
            "errors": errors
        }

    def get_statistics(self, annotations_dir: str, sequences: List[str]) -> Dict:
        """
        Get dataset statistics (shared implementation).

        Args:
            annotations_dir: Directory containing standard JSON annotations
            sequences: List of sequence names

        Returns:
            dict: Statistics including counts per sequence
        """
        annotations_dir = Path(annotations_dir)

        stats = {
            "dataset": self.get_dataset_name(),
            "sequences": {},
            "total_images": 0,
            "total_annotations": 0
        }

        for sequence in sequences:
            sequence_annotations_dir = annotations_dir / sequence

            if not sequence_annotations_dir.exists():
                continue

            annotation_files = list(sequence_annotations_dir.glob("*.json"))
            num_annotations = 0

            for annotation_file in annotation_files:
                try:
                    with open(annotation_file, 'r') as f:
                        annotations = json.load(f)
                    num_annotations += len(annotations)
                except:
                    pass

            stats["sequences"][sequence] = {
                "images": len(annotation_files),
                "annotations": num_annotations
            }
            stats["total_images"] += len(annotation_files)
            stats["total_annotations"] += num_annotations

        return stats
