"""
BOMNI dataset manager for pedestrian detection.

This manager handles all BOMNI-specific dataset preparation operations including:
- Frame extraction from videos
- Annotation format conversion (Tamura XML → our standard JSON)
- Cleanup operations for incorrect/unannotated frames
- Visualization for quality review
- Validation and statistics

Author: Generated for gnomonic projection pedestrian detection project
Date: 2025-12-03
"""

import json
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

from datasets.base_manager import BaseDatasetManager


class BOMNIManager(BaseDatasetManager):
    """
    BOMNI dataset manager.

    Handles dataset preparation workflow:
    1. extract_frames() - Extract frames from BOMNI videos
    2. cleanup_unannotated_frames() - Remove frames without annotations
    3. convert_to_standard_format() - Convert Tamura XML to our standard JSON
    4. visualize_annotations() - Generate visualizations for manual review
    5. cleanup_incorrect_annotations() - Remove bad annotations after manual review
    6. validate() - Final integrity check
    """

    def __init__(self, cfg):
        """
        Initialize BOMNI dataset manager.

        Args:
            cfg: YACS config object with BOMNI settings
                 Can be from datasets.config (cfg.BOMNI) or evaluation.config (cfg.DATASETS.BOMNI)
        """
        super().__init__(cfg)
        # Handle both config structures
        if hasattr(cfg, 'BOMNI'):
            # datasets/config.py structure
            self.bomni_cfg = cfg.BOMNI
        elif hasattr(cfg, 'DATASETS') and hasattr(cfg.DATASETS, 'BOMNI'):
            # evaluation/config.py structure
            self.bomni_cfg = cfg.DATASETS.BOMNI
        else:
            raise ValueError("Config must have either cfg.BOMNI or cfg.DATASETS.BOMNI")

    def get_dataset_name(self) -> str:
        """Return dataset identifier."""
        return "bomni"

    def extract_frames(
        self,
        video_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        sequences: Optional[List[str]] = None,
        start_index: int = 1
    ):
        """
        Extract frames from BOMNI video files.

        Args:
            video_dir: Directory containing video files (default: from config)
            output_dir: Directory to save extracted frames (default: from config)
            sequences: List of sequence names to extract (default: from config)
            start_index: Starting frame index (default: 1)
        """
        if video_dir is None:
            # Try to get video dir from config (dataset config structure)
            if hasattr(self.bomni_cfg, 'VIDEO_DIR'):
                video_dir = Path(self.bomni_cfg.VIDEO_DIR)
            elif hasattr(self.bomni_cfg, 'ROOT_DIR'):
                video_dir = Path(self.bomni_cfg.ROOT_DIR) / "scenario1"
            else:
                raise ValueError("Config must specify VIDEO_DIR or ROOT_DIR")
        else:
            video_dir = Path(video_dir)

        if output_dir is None:
            # Must be specified as argument or in config
            if hasattr(self.bomni_cfg, 'FRAMES_DIR'):
                output_dir = Path(self.bomni_cfg.FRAMES_DIR)
            else:
                raise ValueError("output_dir must be specified or FRAMES_DIR must be in config")
        else:
            output_dir = Path(output_dir)

        if sequences is None:
            sequences = self.bomni_cfg.SEQUENCES

        print("=" * 70)
        print("BOMNI Frame Extraction")
        print("=" * 70)
        print(f"\nVideo directory: {video_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Sequences: {sequences}")
        print(f"Start index: {start_index}")
        print("\n" + "-" * 70)

        for sequence in sequences:
            video_path = video_dir / f"{sequence}.mp4"

            if not video_path.exists():
                print(f"\n[WARNING] Video not found: {video_path}, skipping...")
                continue

            sequence_output_dir = output_dir / sequence
            sequence_output_dir.mkdir(parents=True, exist_ok=True)

            print(f"\nExtracting {sequence}...")

            cap = cv2.VideoCapture(str(video_path))
            frame_count = 0
            frame_index = start_index

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Save frame with 4-digit number (0001.jpg, 0002.jpg, etc.)
                frame_filename = f"{frame_index:04d}.jpg"
                frame_path = sequence_output_dir / frame_filename
                cv2.imwrite(str(frame_path), frame)

                frame_count += 1
                frame_index += 1

            cap.release()

            print(f"  -> Extracted {frame_count} frames")

        print("\n" + "-" * 70)
        print(f"\n[DONE] Frame extraction complete!")
        print(f"Output directory: {output_dir}")
        print("=" * 70)

    def cleanup_unannotated_frames(
        self,
        frames_dir: Optional[str] = None,
        annotations_dir: Optional[str] = None,
        sequences: Optional[List[str]] = None
    ):
        """
        Delete frames that don't have corresponding annotation files.

        This reduces disk space by removing frames that aren't annotated.
        BOMNI annotations are sparse (~10% of frames), so this significantly
        reduces the number of files.

        Args:
            frames_dir: Directory containing extracted frames (default: from config)
            annotations_dir: Directory containing annotations (default: from config)
            sequences: List of sequence names to clean (default: from config)
        """
        if frames_dir is None:
            if hasattr(self.bomni_cfg, 'FRAMES_DIR'):
                frames_dir = Path(self.bomni_cfg.FRAMES_DIR)
            else:
                raise ValueError("frames_dir must be specified or FRAMES_DIR must be in config")
        else:
            frames_dir = Path(frames_dir)

        if annotations_dir is None:
            # Try RAW_ANNOTATIONS_DIR first (dataset config), then TAMURA_ANNOTATIONS_DIR (eval config)
            if hasattr(self.bomni_cfg, 'RAW_ANNOTATIONS_DIR'):
                annotations_dir = Path(self.bomni_cfg.RAW_ANNOTATIONS_DIR)
            elif hasattr(self.bomni_cfg, 'TAMURA_ANNOTATIONS_DIR'):
                annotations_dir = Path(self.bomni_cfg.TAMURA_ANNOTATIONS_DIR)
            else:
                raise ValueError("annotations_dir must be specified or RAW_ANNOTATIONS_DIR/TAMURA_ANNOTATIONS_DIR must be in config")
        else:
            annotations_dir = Path(annotations_dir)

        if sequences is None:
            sequences = self.bomni_cfg.SEQUENCES

        print("=" * 70)
        print("BOMNI Unannotated Frame Cleanup")
        print("=" * 70)
        print(f"\nFrames directory: {frames_dir}")
        print(f"Annotations directory: {annotations_dir}")
        print(f"Sequences: {sequences}")
        print("\n" + "-" * 70)

        total_deleted = 0
        total_kept = 0

        for sequence in sequences:
            sequence_frames_dir = frames_dir / sequence
            sequence_annotations_dir = annotations_dir / sequence

            if not sequence_frames_dir.exists():
                print(f"\n[WARNING] Frames directory not found: {sequence_frames_dir}")
                continue

            if not sequence_annotations_dir.exists():
                print(f"\n[WARNING] Annotations directory not found: {sequence_annotations_dir}")
                continue

            # Get annotation file stems (without extension)
            annotation_files = list(sequence_annotations_dir.glob("*.xml"))
            annotation_stems = {ann_file.stem for ann_file in annotation_files}

            # Check each frame
            frame_files = list(sequence_frames_dir.glob("*.jpg"))
            deleted_count = 0
            kept_count = 0

            for frame_file in frame_files:
                frame_stem = frame_file.stem
                if frame_stem not in annotation_stems:
                    frame_file.unlink()  # Delete the file
                    deleted_count += 1
                else:
                    kept_count += 1

            print(f"\n{sequence}:")
            print(f"  Kept: {kept_count} frames")
            print(f"  Deleted: {deleted_count} frames")

            total_deleted += deleted_count
            total_kept += kept_count

        print("\n" + "-" * 70)
        print(f"\nTotal kept: {total_kept} frames")
        print(f"Total deleted: {total_deleted} frames")
        print(f"\n[DONE] Cleanup complete!")
        print("=" * 70)

    def convert_to_standard_format(
        self,
        input_format: str = "tamura",
        input_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        sequences: Optional[List[str]] = None,
        fisheye_center: Optional[tuple] = None
    ):
        """
        Convert BOMNI annotations to our standard JSON format.

        Supports converting from:
        - "tamura": Tamura et al. Pascal VOC XML format (omnidet-rotinv)

        Standard JSON format:
        {
            "center_x": float,
            "center_y": float,
            "width": float,
            "height": float,
            "angle": float,
            "class_name": string
        }

        Args:
            input_format: Source annotation format ("tamura")
            input_dir: Input annotations directory (default: from config based on format)
            output_dir: Output directory for standard JSON (default: from config)
            sequences: List of sequence names to convert (default: from config)
            fisheye_center: Image center for angle calculation (default: from config)
        """
        if input_format not in ["tamura"]:
            raise ValueError(f"Invalid input_format: {input_format}. Must be 'tamura'.")

        if input_dir is None:
            if input_format == "tamura":
                # Try RAW_ANNOTATIONS_DIR first (dataset config), then TAMURA_ANNOTATIONS_DIR (eval config)
                if hasattr(self.bomni_cfg, 'RAW_ANNOTATIONS_DIR'):
                    input_dir = Path(self.bomni_cfg.RAW_ANNOTATIONS_DIR)
                elif hasattr(self.bomni_cfg, 'TAMURA_ANNOTATIONS_DIR'):
                    input_dir = Path(self.bomni_cfg.TAMURA_ANNOTATIONS_DIR)
                else:
                    raise ValueError("Config must specify RAW_ANNOTATIONS_DIR or TAMURA_ANNOTATIONS_DIR")
        else:
            input_dir = Path(input_dir)

        if output_dir is None:
            if hasattr(self.bomni_cfg, 'STANDARD_ANNOTATIONS_DIR'):
                output_dir = Path(self.bomni_cfg.STANDARD_ANNOTATIONS_DIR)
            else:
                raise ValueError("output_dir must be specified or STANDARD_ANNOTATIONS_DIR must be in config")
        else:
            output_dir = Path(output_dir)

        if sequences is None:
            sequences = self.bomni_cfg.SEQUENCES

        if fisheye_center is None:
            fisheye_center = (
                self.bomni_cfg.FISHEYE_CENTER_X,
                self.bomni_cfg.FISHEYE_CENTER_Y
            )

        print("=" * 70)
        print("BOMNI Annotation Format Conversion")
        print("=" * 70)
        print(f"\nInput format: {input_format}")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Sequences: {sequences}")
        print(f"Fisheye center: {fisheye_center}")
        print("\n" + "-" * 70)

        total_files = 0
        total_annotations = 0

        for sequence in sequences:
            sequence_input_dir = input_dir / sequence
            sequence_output_dir = output_dir / sequence

            if not sequence_input_dir.exists():
                print(f"\n[WARNING] Sequence '{sequence}' not found in input directory, skipping...")
                continue

            # Create output directory
            sequence_output_dir.mkdir(parents=True, exist_ok=True)

            # Get all annotation files
            if input_format == "tamura":
                annotation_files = sorted(sequence_input_dir.glob("*.xml"))
            else:
                annotation_files = []

            print(f"\nProcessing {sequence}: {len(annotation_files)} files")

            sequence_file_count = 0
            sequence_annotation_count = 0

            for annotation_file in annotation_files:
                # Convert based on input format
                if input_format == "tamura":
                    annotations = self._convert_tamura_xml(annotation_file, fisheye_center)

                # Create output JSON file with same name
                json_file = sequence_output_dir / annotation_file.with_suffix('.json').name

                # Save as JSON
                with open(json_file, 'w') as f:
                    json.dump(annotations, f, indent=2)

                sequence_file_count += 1
                sequence_annotation_count += len(annotations)

            print(f"  -> Created {sequence_file_count} JSON files ({sequence_annotation_count} annotations)")

            total_files += sequence_file_count
            total_annotations += sequence_annotation_count

        print("\n" + "-" * 70)
        print(f"\n[DONE] Conversion complete!")
        print(f"  Total files: {total_files}")
        print(f"  Total annotations: {total_annotations}")
        print(f"\nOutput directory: {output_dir}")
        print("=" * 70)

    def _convert_tamura_xml(self, xml_path: Path, fisheye_center: tuple) -> List[Dict]:
        """
        Convert Tamura et al. Pascal VOC XML to our standard JSON format.

        Tamura XML format uses repurposed Pascal VOC fields:
        - xmin, ymin, xmax, ymax encode (center, dimensions) of rotated bbox
        - Angle must be calculated from bbox center and image center

        Args:
            xml_path: Path to Tamura XML file
            fisheye_center: Image center coordinates for angle calculation

        Returns:
            List of annotations in standard format
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()

        annotations = []

        for obj in root.findall('object'):
            class_name = obj.find('name').text

            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            # Decode rotated bbox parameters from Pascal VOC fields
            center_x = (xmin + xmax) / 2.0
            center_y = (ymin + ymax) / 2.0
            width = xmax - xmin
            height = ymax - ymin

            # Calculate rotation angle
            angle = self._calculate_rotation_angle(center_x, center_y, fisheye_center)

            annotations.append({
                "center_x": center_x,
                "center_y": center_y,
                "width": width,
                "height": height,
                "angle": angle,
                "class_name": class_name
            })

        return annotations

    def _calculate_rotation_angle(
        self,
        bbox_center_x: float,
        bbox_center_y: float,
        fisheye_center: tuple
    ) -> float:
        """
        Calculate rotation angle for a bounding box.

        From Tamura et al. (omnidet-rotinv):
        "Rotation angle for each bounding box is the angle between a vertical line and
        a line connecting the image center and bounding box center."

        Args:
            bbox_center_x: X coordinate of bbox center
            bbox_center_y: Y coordinate of bbox center
            fisheye_center: Image center coordinates (x, y)

        Returns:
            Rotation angle in degrees (0 = vertical up, clockwise positive)
        """
        img_center_x, img_center_y = fisheye_center
        dx = bbox_center_x - img_center_x
        dy = bbox_center_y - img_center_y
        angle_rad = np.arctan2(dx, -dy)  # -dy because y-axis points down
        angle_deg = np.degrees(angle_rad)
        return angle_deg

    def cleanup_incorrect_annotations(
        self,
        visualization_dir: str,
        annotations_dir: Optional[str] = None,
        frames_dir: Optional[str] = None,
        sequences: Optional[List[str]] = None
    ):
        """
        Remove annotations based on deleted visualization files (after manual review).

        Workflow:
        1. User generates visualizations with visualize_annotations()
        2. User manually reviews and DELETES visualization images with incorrect annotations
        3. This method scans remaining visualizations to determine which annotations to keep
        4. Deletes annotation files and frame images that don't match remaining visualizations

        Args:
            visualization_dir: Directory containing visualizations (after user deleted bad ones)
            annotations_dir: Directory containing standard JSON annotations (default: from config)
            frames_dir: Directory containing frame images (default: from config)
            sequences: List of sequence names to clean (default: from config)
        """
        visualization_dir = Path(visualization_dir)

        if annotations_dir is None:
            if hasattr(self.bomni_cfg, 'STANDARD_ANNOTATIONS_DIR'):
                annotations_dir = Path(self.bomni_cfg.STANDARD_ANNOTATIONS_DIR)
            else:
                raise ValueError("annotations_dir must be specified or STANDARD_ANNOTATIONS_DIR must be in config")
        else:
            annotations_dir = Path(annotations_dir)

        if frames_dir is None:
            if hasattr(self.bomni_cfg, 'FRAMES_DIR'):
                frames_dir = Path(self.bomni_cfg.FRAMES_DIR)
            else:
                raise ValueError("frames_dir must be specified or FRAMES_DIR must be in config")
        else:
            frames_dir = Path(frames_dir)

        if sequences is None:
            sequences = self.bomni_cfg.SEQUENCES

        print("=" * 70)
        print("BOMNI Incorrect Annotation Cleanup")
        print("=" * 70)
        print(f"\nVisualization directory: {visualization_dir}")
        print(f"Annotations directory: {annotations_dir}")
        print(f"Frames directory: {frames_dir}")
        print(f"Sequences: {sequences}")
        print("\n" + "-" * 70)

        total_kept = 0
        total_deleted = 0

        for sequence in sequences:
            viz_sequence_dir = visualization_dir / sequence
            annotations_sequence_dir = annotations_dir / sequence
            frames_sequence_dir = frames_dir / sequence

            if not viz_sequence_dir.exists():
                print(f"\n[WARNING] Visualization directory not found: {viz_sequence_dir}")
                continue

            # Get list of remaining visualization files (user kept only good ones)
            viz_files = list(viz_sequence_dir.glob("annotated_*.jpg"))
            kept_ids = set()
            for viz_file in viz_files:
                # Extract frame ID: "annotated_0001.jpg" -> "0001"
                frame_id = viz_file.stem.replace("annotated_", "")
                kept_ids.add(frame_id)

            print(f"\n{sequence}: {len(kept_ids)} visualizations kept by user")

            # Delete annotation files not in kept list
            deleted_annotations = 0
            if annotations_sequence_dir.exists():
                annotation_files = list(annotations_sequence_dir.glob("*.json"))
                for annotation_file in annotation_files:
                    frame_id = annotation_file.stem
                    if frame_id not in kept_ids:
                        annotation_file.unlink()
                        deleted_annotations += 1

            # Delete frame images not in kept list
            deleted_frames = 0
            if frames_sequence_dir.exists():
                frame_files = list(frames_sequence_dir.glob("*.jpg"))
                for frame_file in frame_files:
                    frame_id = frame_file.stem
                    if frame_id not in kept_ids:
                        frame_file.unlink()
                        deleted_frames += 1

            print(f"  Deleted {deleted_annotations} annotation files")
            print(f"  Deleted {deleted_frames} frame images")

            total_kept += len(kept_ids)
            total_deleted += deleted_annotations

        print("\n" + "-" * 70)
        print(f"\nTotal kept: {total_kept} annotations")
        print(f"Total deleted: {total_deleted} annotations")
        print(f"\n[DONE] Cleanup complete!")
        print("=" * 70)
