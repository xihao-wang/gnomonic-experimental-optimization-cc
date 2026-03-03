"""
BOMNI Dataset handler for rotated bounding box annotations.

This module provides a class to load and process BOMNI dataset annotations at runtime
(for evaluation and testing).

Primary format: "standard" - Our unified JSON format with all values precomputed
Legacy format: "tamura" - Third-party XML format (for reference/re-conversion only)

Standard JSON format (recommended):
{
    "center_x": float,
    "center_y": float,
    "width": float,
    "height": float,
    "angle": float,
    "class_name": string
}
"""

import os
import json
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm


class BOMNIDataset:
    """
    BOMNI dataset handler for rotated bbox annotations.

    This class loads annotations from Pascal VOC XML files and provides
    methods to visualize rotated bounding boxes on fisheye images.
    """

    def __init__(self, cfg):
        """
        Initialize BOMNI dataset.

        Args:
            cfg: YACS config object with DATASETS.BOMNI settings
        """
        self.cfg = cfg
        self.root_dir = Path(cfg.DATASETS.BOMNI.ROOT_DIR)
        self.frames_dir = Path(cfg.DATASETS.BOMNI.FRAMES_DIR)

        # Annotation format and directory
        self.annotation_format = cfg.DATASETS.BOMNI.ANNOTATION_FORMAT
        if self.annotation_format == "tamura":
            # Legacy format (for reference only)
            self.annotations_dir = Path(cfg.DATASETS.BOMNI.TAMURA_ANNOTATIONS_DIR)
            self.annotation_ext = ".xml"
        elif self.annotation_format == "standard":
            # Standard JSON format (recommended)
            self.annotations_dir = Path(cfg.DATASETS.BOMNI.STANDARD_ANNOTATIONS_DIR)
            self.annotation_ext = ".json"
        else:
            raise ValueError(f"Invalid annotation format: {self.annotation_format}. "
                           f"Must be 'tamura' (legacy) or 'standard' (recommended).")

        self.sequences = cfg.DATASETS.BOMNI.SEQUENCES
        self.image_ext = cfg.DATASETS.BOMNI.IMAGE_EXT

        # Fisheye image dimensions and center
        self.image_width = cfg.DATASETS.BOMNI.IMAGE_WIDTH
        self.image_height = cfg.DATASETS.BOMNI.IMAGE_HEIGHT
        self.fisheye_center = (
            cfg.DATASETS.BOMNI.FISHEYE_CENTER_X,
            cfg.DATASETS.BOMNI.FISHEYE_CENTER_Y
        )

        # Verbose mode
        self.verbose = cfg.VERBOSE

        # Build image-annotation pairs
        self._build_dataset()

    def _build_dataset(self):
        """
        Build list of (image_path, annotation_path) pairs for all sequences.
        """
        self.data = []

        for sequence in self.sequences:
            sequence_frames_dir = self.frames_dir / sequence
            sequence_annotations_dir = self.annotations_dir / sequence

            if not sequence_frames_dir.exists():
                print(f"WARNING: Frames directory not found: {sequence_frames_dir}")
                continue

            if not sequence_annotations_dir.exists():
                print(f"WARNING: Annotations directory not found: {sequence_annotations_dir}")
                continue

            # Find all annotation files
            annotation_files = list(sequence_annotations_dir.glob(f"*{self.annotation_ext}"))

            for annotation_path in annotation_files:
                # Get corresponding image file
                image_name = annotation_path.stem + self.image_ext
                image_path = sequence_frames_dir / image_name

                if not image_path.exists():
                    if self.verbose:
                        print(f"WARNING: Image not found: {image_path}")
                    continue

                self.data.append({
                    "sequence": sequence,
                    "image_path": str(image_path),
                    "annotation_path": str(annotation_path),
                    "image_name": image_name
                })

        if self.verbose:
            print(f"BOMNI Dataset: Found {len(self.data)} image-annotation pairs")
            print(f"  Annotation format: {self.annotation_format}")
            for sequence in self.sequences:
                count = sum(1 for d in self.data if d["sequence"] == sequence)
                print(f"  {sequence}: {count} images")

    def __len__(self):
        """Return number of images in dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get image and annotations by index.

        Args:
            idx: Index of image-annotation pair

        Returns:
            dict with keys:
                - image: numpy array (H, W, 3)
                - annotations: list of rotated bbox dicts
                - image_path: str
                - annotation_path: str
                - sequence: str
        """
        data_item = self.data[idx]

        # Load image
        image = cv2.imread(data_item["image_path"])
        if image is None:
            raise ValueError(f"Could not load image: {data_item['image_path']}")

        # Load annotations
        annotations = self._load_annotations(data_item["annotation_path"])

        return {
            "image": image,
            "annotations": annotations,
            "image_path": data_item["image_path"],
            "annotation_path": data_item["annotation_path"],
            "sequence": data_item["sequence"],
            "image_name": data_item["image_name"]
        }

    def _load_annotations(self, annotation_path: str) -> List[Dict]:
        """
        Load annotations (dispatches to appropriate loader based on format).

        Args:
            annotation_path: Path to annotation file (XML or JSON)

        Returns:
            List of annotation dicts, each containing:
                - class_name: str (always "person")
                - center_x, center_y: float (bbox center)
                - width, height: float (bbox dimensions)
                - angle: float (rotation angle in degrees, 0=vertical up, clockwise)
        """
        if self.annotation_format == "tamura":
            return self._load_annotations_tamura(annotation_path)
        elif self.annotation_format == "standard":
            return self._load_annotations_standard(annotation_path)
        else:
            raise ValueError(f"Invalid annotation format: {self.annotation_format}")

    def _load_annotations_tamura(self, annotation_path: str) -> List[Dict]:
        """
        Load annotations from Tamura et al. Pascal VOC XML format.

        The XML format contains repurposed Pascal VOC fields:
        - xmin, ymin, xmax, ymax encode (center, dimensions) of rotated bbox
        - Rotation angle must be calculated from bbox center and image center

        Args:
            annotation_path: Path to XML annotation file

        Returns:
            List of annotation dicts, each containing:
                - class_name: str (always "person")
                - center_x, center_y: float (bbox center)
                - width, height: float (tight-fit rotated bbox dimensions)
                - angle: float (rotation angle in degrees, 0=vertical up, clockwise)
        """
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        annotations = []

        for obj in root.findall("object"):
            # Get class name
            class_name = obj.find("name").text

            # Get bounding box (repurposed Pascal VOC fields)
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            # Decode rotated bbox parameters from Pascal VOC fields
            center_x = (xmin + xmax) / 2.0
            center_y = (ymin + ymax) / 2.0
            width = xmax - xmin
            height = ymax - ymin

            # Calculate rotation angle
            # Angle between vertical line (pointing up) and line from image center to bbox center
            angle = self._calculate_rotation_angle(center_x, center_y)

            annotations.append({
                "class_name": class_name,
                "center_x": center_x,
                "center_y": center_y,
                "width": width,
                "height": height,
                "angle": angle
            })

        return annotations

    def _load_annotations_standard(self, annotation_path: str) -> List[Dict]:
        """
        Load annotations from our standard JSON format.

        The JSON format contains all rotated bbox parameters precomputed:
        - center_x, center_y: bbox center coordinates
        - width, height: tight-fit rotated bbox dimensions
        - angle: rotation angle in degrees (precomputed)
        - class_name: object class

        This is our unified annotation format used across all datasets.

        Args:
            annotation_path: Path to JSON annotation file

        Returns:
            List of annotation dicts, each containing:
                - class_name: str (always "person")
                - center_x, center_y: float (bbox center)
                - width, height: float (tight-fit rotated bbox dimensions)
                - angle: float (rotation angle in degrees, 0=vertical up, clockwise)
        """
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)

        # Annotations are already in the correct format
        return annotations

    def _calculate_rotation_angle(self, bbox_center_x: float, bbox_center_y: float) -> float:
        """
        Calculate rotation angle for a bounding box.

        From omnidet-rotinv README:
        "Rotation angle for each bounding box is the angle between a vertical line and
        a line connecting a image center and bounding box center."

        Args:
            bbox_center_x: X coordinate of bbox center
            bbox_center_y: Y coordinate of bbox center

        Returns:
            Rotation angle in degrees (0 = vertical up, clockwise positive)
        """
        img_center_x, img_center_y = self.fisheye_center

        # Vector from image center to bbox center
        dx = bbox_center_x - img_center_x
        dy = bbox_center_y - img_center_y

        # Angle from vertical line (0, -1) to (dx, dy)
        # Using atan2: angle = atan2(dx, -dy) to get angle from vertical (up direction)
        # Note: OpenCV uses (0, 0) at top-left, so up is negative y
        angle_rad = np.arctan2(dx, -dy)
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    def get_sequence_data(self, sequence: str) -> List[Dict]:
        """
        Get all data items for a specific sequence.

        Args:
            sequence: Sequence name (e.g., "top-0")

        Returns:
            List of data items for the sequence
        """
        return [d for d in self.data if d["sequence"] == sequence]

    def visualize_annotations(self, output_dir: Optional[str] = None, max_images: Optional[int] = None):
        """
        Visualize all annotations and save to output directory.

        Args:
            output_dir: Output directory for visualizations (default: from config)
            max_images: Maximum number of images to visualize per sequence (default: all)
        """
        if output_dir is None:
            output_dir = self.cfg.VISUALIZATION.OUTPUT_DIR

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Visualization settings from config
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

        print(f"\nVisualizing BOMNI annotations...")
        print(f"Output directory: {output_dir}")

        # Process each sequence
        for sequence in self.sequences:
            sequence_data = self.get_sequence_data(sequence)

            if not sequence_data:
                print(f"  {sequence}: No data found, skipping")
                continue

            total_available = len(sequence_data)

            # Determine how many images to visualize
            # Handle "all", None, or integer values
            # If max_images exceeds available, default to all
            if max_images is None or max_images == "all":
                num_to_visualize = total_available
                viz_mode = "all"
            elif isinstance(max_images, int):
                if max_images >= total_available:
                    num_to_visualize = total_available
                    viz_mode = f"all (requested {max_images}, but only {total_available} available)"
                else:
                    num_to_visualize = max_images
                    viz_mode = f"{num_to_visualize}/{total_available}"
                sequence_data = sequence_data[:num_to_visualize]
            else:
                # Fallback to all if invalid type
                num_to_visualize = total_available
                viz_mode = "all"

            # Create sequence output directory
            sequence_output_dir = output_dir / sequence
            sequence_output_dir.mkdir(parents=True, exist_ok=True)

            print(f"  {sequence}: Visualizing {viz_mode} images...")

            for data_item in tqdm(sequence_data, desc=f"    {sequence}"):
                # Load image and annotations
                item = self.__getitem__(self.data.index(data_item))
                image = item["image"].copy()
                annotations = item["annotations"]

                # Draw image center if requested
                if draw_center_and_radial:
                    img_center = (int(self.fisheye_center[0]), int(self.fisheye_center[1]))
                    cv2.circle(image, img_center, 5, (255, 0, 0), -1)  # Blue circle

                # Draw each annotation
                for ann in annotations:
                    center = (int(ann["center_x"]), int(ann["center_y"]))
                    size = (ann["width"], ann["height"])
                    angle = ann["angle"]

                    # Calculate axis-aligned bbox bounds (for label positioning and axis-aligned drawing)
                    half_width = ann["width"] / 2.0
                    half_height = ann["height"] / 2.0
                    xmin = int(ann["center_x"] - half_width)
                    ymin = int(ann["center_y"] - half_height)
                    xmax = int(ann["center_x"] + half_width)
                    ymax = int(ann["center_y"] + half_height)

                    if draw_rotated:
                        # Draw rotated rectangle
                        box = cv2.boxPoints(((ann["center_x"], ann["center_y"]), size, angle))
                        box = np.intp(box)  # Use intp instead of deprecated int0
                        cv2.drawContours(image, [box], 0, bbox_color, bbox_thickness)
                    else:
                        # Draw axis-aligned rectangle
                        cv2.rectangle(
                            image,
                            (xmin, ymin),
                            (xmax, ymax),
                            bbox_color,
                            bbox_thickness
                        )

                    # Draw radial line if requested
                    if draw_center_and_radial:
                        img_center = (int(self.fisheye_center[0]), int(self.fisheye_center[1]))
                        cv2.line(image, img_center, center, (0, 255, 255), 1)  # Yellow line

                    # Draw label
                    if show_labels or show_rotation_angle:
                        label_parts = []
                        if show_labels:
                            label_parts.append(ann["class_name"])
                        if show_rotation_angle:
                            label_parts.append(f"{angle:.1f}deg")
                        label = " ".join(label_parts)

                        # Position label above bbox (using top-left of axis-aligned approximation)
                        label_pos = (xmin, ymin - 5)
                        cv2.putText(
                            image,
                            label,
                            label_pos,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale,
                            bbox_color,
                            font_thickness
                        )

                # Save visualization
                output_filename = f"annotated_{item['image_name']}"
                if image_format != "jpg":
                    output_filename = output_filename.replace(".jpg", f".{image_format}")
                output_path = sequence_output_dir / output_filename

                if image_format == "jpg":
                    cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, image_quality])
                else:
                    cv2.imwrite(str(output_path), image)

        print(f"\nVisualization complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    """Test BOMNI dataset loading and visualization."""
    from evaluation.lib.config import get_cfg

    print("=" * 70)
    print("BOMNI Dataset Test")
    print("=" * 70)

    # Load config
    cfg = get_cfg()
    cfg.VERBOSE = True

    print(f"\nAnnotation format: {cfg.DATASETS.BOMNI.ANNOTATION_FORMAT}")
    print(f"  - 'tamura': Third-party XML (Tamura et al., legacy format)")
    print(f"  - 'standard': Our unified JSON format (all values precomputed)")

    # Create dataset
    dataset = BOMNIDataset(cfg)

    print(f"\nDataset size: {len(dataset)}")

    # Test loading one item
    if len(dataset) > 0:
        item = dataset[0]
        print(f"\nSample item:")
        print(f"  Image shape: {item['image'].shape}")
        print(f"  Number of annotations: {len(item['annotations'])}")
        print(f"  Sequence: {item['sequence']}")
        if item['annotations']:
            print(f"  First annotation: {item['annotations'][0]}")

    # Run visualization (limit to 2 images per sequence for testing)
    cfg.VISUALIZATION.MAX_IMAGES_PER_SEQUENCE = 2
    dataset.visualize_annotations(max_images=2)

    print("\n" + "=" * 70)
