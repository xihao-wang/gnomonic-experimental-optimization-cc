"""
PIROPO dataset manager for pedestrian detection.

This manager handles PIROPO-specific dataset preparation operations:
- Converting Tamura annotations to standard JSON format
- Preserving PIROPO folder structure: Room → Camera → Sequence

PIROPO Dataset: Indoor omnidirectional pedestrian tracking dataset
- 2 rooms (Room_A, Room_B)
- 4 cameras (omni_1A, omni_2A, omni_3A, conv_4A-8A in Room_A; omni_1B, conv_2B in Room_B)
- 12+ sequences per camera (training, test1-12, etc.)
- 800×600 images, fisheye center at (400, 300)

Author: Yassir Zardoua
Email: y.zardoua@caplogy.com | yassirzardoua@gmail.com
Date: 2026-02-16
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from datasets.lib.base_manager import BaseDatasetManager


class PIROPOManager(BaseDatasetManager):
    """
    PIROPO dataset manager.

    Handles conversion from Tamura XML format to standard JSON format.
    Preserves PIROPO's hierarchical folder structure (Room/Camera/Sequence).
    """

    def __init__(self, cfg):
        """
        Initialize PIROPO dataset manager.

        Args:
            cfg: YACS config object (not currently used, reserved for future)
        """
        super().__init__(cfg)

    def get_dataset_name(self) -> str:
        """Return dataset identifier."""
        return "piropo"

    def extract_frames(
        self,
        video_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        sequences: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Frame extraction not needed for PIROPO.

        PIROPO frames are distributed pre-extracted, not as videos.
        This method is a no-op for consistency with the preparation pipeline.

        Args:
            video_dir: Ignored (PIROPO has no videos)
            output_dir: Ignored (frames already in place)
            sequences: Ignored (PIROPO uses Room/Camera/Sequence structure)
        """
        print("=" * 80)
        print("PIROPO Frame Extraction")
        print("=" * 80)
        print("\n[SKIP] PIROPO distributes pre-extracted frames.")
        print("Frames already exist in download folder.")
        print("Proceeding to next step...")
        print("=" * 80)

    def convert_to_standard_format(
        self,
        input_format: str = "tamura",
        input_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        rooms: Optional[List[str]] = None,
        fisheye_center: tuple = (400, 300)
    ):
        """
        Convert PIROPO Tamura annotations to standard JSON format.

        PIROPO structure: Room/Camera/Sequence/*.xml
        Output structure: Room/Camera/Sequence/*.json (same hierarchy)

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
            input_format: Source annotation format (only "tamura" supported)
            input_dir: Input Tamura annotations root directory
            output_dir: Output directory for standard JSON
            rooms: List of room names to process (default: ["Room_A", "Room_B"])
            fisheye_center: Image center for angle calculation (default: (400, 300))

        Raises:
            ValueError: If input_format is not "tamura" or directories are invalid
        """
        if input_format != "tamura":
            raise ValueError(f"Invalid input_format: {input_format}. Only 'tamura' is supported for PIROPO.")

        if input_dir is None:
            raise ValueError("input_dir must be specified")
        input_dir = Path(input_dir)

        if output_dir is None:
            raise ValueError("output_dir must be specified")
        output_dir = Path(output_dir)

        if rooms is None:
            rooms = ["Room_A", "Room_B"]

        if not input_dir.exists():
            raise ValueError(f"Input directory not found: {input_dir}")

        print("=" * 80)
        print("PIROPO Annotation Format Conversion")
        print("=" * 80)
        print(f"\nInput format: {input_format}")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Rooms: {rooms}")
        print(f"Fisheye center: {fisheye_center}")
        print("\n" + "-" * 80)

        total_files = 0
        total_annotations = 0

        for room in rooms:
            room_dir = input_dir / room
            if not room_dir.exists():
                print(f"\n[WARNING] Room '{room}' not found, skipping...")
                continue

            print(f"\nProcessing {room}...")

            # Get all camera folders
            camera_dirs = sorted([d for d in room_dir.iterdir() if d.is_dir()])

            for camera_dir in camera_dirs:
                camera_name = camera_dir.name
                print(f"\n  Camera: {camera_name}")

                # Get all sequence folders
                sequence_dirs = sorted([d for d in camera_dir.iterdir() if d.is_dir()])

                for sequence_dir in sequence_dirs:
                    sequence_name = sequence_dir.name

                    # Skip Ground_Truth_Annotations folders (as per user instruction)
                    if sequence_name == "Ground_Truth_Annotations":
                        continue

                    # Create output directory structure
                    output_sequence_dir = output_dir / room / camera_name / sequence_name
                    output_sequence_dir.mkdir(parents=True, exist_ok=True)

                    # Get all XML files
                    xml_files = sorted(sequence_dir.glob("*.xml"))

                    if not xml_files:
                        continue

                    sequence_file_count = 0
                    sequence_annotation_count = 0

                    for xml_file in xml_files:
                        # Convert XML to standard JSON
                        annotations = self._convert_tamura_xml(xml_file, fisheye_center)

                        # Create output JSON file with same name
                        json_file = output_sequence_dir / xml_file.with_suffix('.json').name

                        # Save as JSON
                        with open(json_file, 'w') as f:
                            json.dump(annotations, f, indent=2)

                        sequence_file_count += 1
                        sequence_annotation_count += len(annotations)

                    print(f"    {sequence_name}: {sequence_file_count} files ({sequence_annotation_count} annotations)")

                    total_files += sequence_file_count
                    total_annotations += sequence_annotation_count

        print("\n" + "-" * 80)
        print(f"\n[DONE] Conversion complete!")
        print(f"  Total files: {total_files}")
        print(f"  Total annotations: {total_annotations}")
        print(f"\nOutput directory: {output_dir}")
        print("=" * 80)

    def _convert_tamura_xml(self, xml_path: Path, fisheye_center: tuple) -> List[Dict]:
        """
        Convert Tamura et al. Pascal VOC XML to standard JSON format.

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

    def cleanup_unannotated_frames(
        self,
        frames_dir: str,
        annotations_dir: str,
        sequences: Optional[List[str]] = None,
        rooms: Optional[List[str]] = None,
        image_ext: str = ".jpg",
        source_dir: Optional[str] = None
    ):
        """
        Copy only annotated PIROPO frames from source to target directory.

        Tamura only annotated sparse keyframes (~10% of frames). This method
        copies ONLY frames that have corresponding Tamura annotations, leaving
        raw source data untouched.

        Args:
            frames_dir: Target directory for cleaned frames (output)
            annotations_dir: Root directory containing Tamura annotations (Room/Camera/Sequence/*.xml or *.json)
            sequences: Ignored for PIROPO (processes all sequences from config)
            rooms: List of rooms to process (default: from config)
            image_ext: Image file extension (default: ".jpg")
            source_dir: Source directory with ALL frames (default: from config FRAMES_SOURCE_DIR)
        """
        import os
        import shutil

        frames_dir = Path(frames_dir)
        annotations_dir = Path(annotations_dir)

        # Get source directory from config if not provided
        if source_dir is None:
            if hasattr(self.cfg, 'PIROPO') and hasattr(self.cfg.PIROPO, 'FRAMES_SOURCE_DIR'):
                source_dir = Path(self.cfg.PIROPO.FRAMES_SOURCE_DIR)
            else:
                raise ValueError("source_dir must be specified or PIROPO.FRAMES_SOURCE_DIR must be in config")
        else:
            source_dir = Path(source_dir)

        if rooms is None:
            if hasattr(self.cfg, 'PIROPO') and hasattr(self.cfg.PIROPO, 'ROOMS'):
                rooms = self.cfg.PIROPO.ROOMS
            else:
                rooms = ["Room_A", "Room_B"]

        print("=" * 80)
        print("PIROPO Annotated Frame Extraction")
        print("=" * 80)
        print(f"\nSource (all frames): {source_dir}")
        print(f"Target (annotated only): {frames_dir}")
        print(f"Annotations: {annotations_dir}")
        print(f"Rooms: {rooms}")
        print("\n" + "-" * 80)

        total_copied = 0
        total_skipped = 0

        for room in rooms:
            # Source: Handle both "Room A" (space) and "Room_A" (underscore)
            room_source_underscore = source_dir / room
            room_source_space = source_dir / room.replace("_", " ")

            if room_source_underscore.exists():
                room_source_dir = room_source_underscore
            elif room_source_space.exists():
                room_source_dir = room_source_space
            else:
                print(f"\n[WARNING] Source frames not found for {room}, skipping...")
                continue

            # Target: Use underscore naming (standardized)
            room_target_dir = frames_dir / room

            # Annotations: Use Tamura annotations to identify which frames to copy
            room_ann_dir = annotations_dir / room

            if not room_ann_dir.exists():
                print(f"\n[WARNING] Annotations not found for {room}, skipping...")
                continue

            print(f"\nProcessing {room}...")

            # Get all camera folders from annotations (authoritative source)
            camera_dirs = sorted([d for d in room_ann_dir.iterdir() if d.is_dir()])

            for camera_dir in camera_dirs:
                camera_name = camera_dir.name
                camera_source_dir = room_source_dir / camera_name
                camera_target_dir = room_target_dir / camera_name

                if not camera_source_dir.exists():
                    print(f"  [WARNING] Source frames not found for {room}/{camera_name}, skipping...")
                    continue

                # Get all sequence folders from annotations
                sequence_dirs = sorted([d for d in camera_dir.iterdir() if d.is_dir()])

                for sequence_dir in sequence_dirs:
                    sequence_name = sequence_dir.name

                    # Source sequence (may have different naming: omni1A_training vs omni_1A_training)
                    sequence_source_dir = camera_source_dir / sequence_name
                    # Also try without underscore between camera and sequence
                    sequence_source_alt = camera_source_dir / sequence_name.replace("_", "", 1)

                    if sequence_source_dir.exists():
                        source_seq_dir = sequence_source_dir
                    elif sequence_source_alt.exists():
                        source_seq_dir = sequence_source_alt
                    else:
                        print(f"    [WARNING] Source frames not found for {sequence_name}, skipping...")
                        continue

                    # Target sequence directory
                    sequence_target_dir = camera_target_dir / sequence_name
                    sequence_target_dir.mkdir(parents=True, exist_ok=True)

                    # Get annotated frame names from Tamura annotations
                    # Check both XML (raw Tamura) and JSON (if already converted)
                    xml_files = list(sequence_dir.glob("*.xml"))
                    json_files = list(sequence_dir.glob("*.json"))

                    if xml_files:
                        annotated_stems = set([f.stem for f in xml_files])
                    elif json_files:
                        annotated_stems = set([f.stem for f in json_files])
                    else:
                        print(f"    [WARNING] No annotations found for {sequence_name}, skipping...")
                        continue

                    copied_count = 0
                    skipped_count = 0

                    # Copy only annotated frames
                    for stem in annotated_stems:
                        source_frame = source_seq_dir / f"{stem}{image_ext}"
                        target_frame = sequence_target_dir / f"{stem}{image_ext}"

                        if source_frame.exists():
                            shutil.copy2(str(source_frame), str(target_frame))
                            copied_count += 1
                        else:
                            skipped_count += 1

                    if copied_count > 0 or skipped_count > 0:
                        print(f"    {sequence_name}: copied {copied_count}, skipped {skipped_count} (missing)")

                    total_copied += copied_count
                    total_skipped += skipped_count

        print("\n" + "-" * 80)
        print(f"\nTotal copied: {total_copied} frames")
        print(f"Total skipped: {total_skipped} frames (not found in source)")
        print(f"\n[DONE] Annotated frames extracted!")
        print(f"Target: {frames_dir}")
        print("=" * 80)

    def visualize_annotations(
        self,
        annotations_dir: str,
        frames_dir: str,
        output_dir: Optional[str] = None,
        sequences: Optional[List[str]] = None,
        max_images: Optional[int] = None,
        fisheye_center: tuple = (400, 300),
        image_ext: str = ".jpg",
        rooms: Optional[List[str]] = None
    ):
        """
        Visualize PIROPO annotations with hierarchical Room/Camera/Sequence structure.

        Overrides base class method to handle PIROPO's specific folder structure:
        Room/Camera/Sequence/*.json → Room/Camera/Sequence/*.jpg visualizations

        Args:
            annotations_dir: Root directory containing Standard-annotations (with Room/Camera/Sequence structure)
            frames_dir: Root directory containing frames (with same structure)
            output_dir: Output directory for visualizations
            sequences: Ignored for PIROPO (processes all sequences in all rooms/cameras)
            max_images: Max images per sequence ("all" or integer)
            fisheye_center: Image center coordinates (default: (400, 300))
            image_ext: Image file extension (default: ".jpg")
            rooms: List of rooms to process (default: ["Room_A", "Room_B"])
        """
        from datasets.utils.visualization import draw_rotated_bbox_on_image, save_annotated_image
        import cv2

        annotations_dir = Path(annotations_dir)
        frames_dir = Path(frames_dir)

        if output_dir is None:
            raise ValueError("output_dir must be specified for PIROPO")
        output_dir = Path(output_dir) / self.get_dataset_name() / "annotation_visualization"

        if rooms is None:
            rooms = ["Room_A", "Room_B"]

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

        print("=" * 80)
        print(f"PIROPO Annotation Visualization")
        print("=" * 80)
        print(f"\nAnnotations: {annotations_dir}")
        print(f"Frames: {frames_dir}")
        print(f"Output: {output_dir}")
        print(f"Max images per sequence: {max_images}")
        print(f"Fisheye center: {fisheye_center}")
        print("\n" + "-" * 80)

        total_visualized = 0

        for room in rooms:
            room_ann_dir = annotations_dir / room
            room_frames_dir = frames_dir / room

            if not room_ann_dir.exists():
                print(f"\n[WARNING] Room '{room}' not found in annotations, skipping...")
                continue

            print(f"\nProcessing {room}...")

            # Get all camera folders
            camera_dirs = sorted([d for d in room_ann_dir.iterdir() if d.is_dir()])

            for camera_dir in camera_dirs:
                camera_name = camera_dir.name
                camera_frames_dir = room_frames_dir / camera_name

                if not camera_frames_dir.exists():
                    print(f"  [WARNING] Frames not found for {room}/{camera_name}, skipping...")
                    continue

                # Get all sequence folders
                sequence_dirs = sorted([d for d in camera_dir.iterdir() if d.is_dir()])

                for sequence_dir in sequence_dirs:
                    sequence_name = sequence_dir.name
                    sequence_frames_dir = camera_frames_dir / sequence_name

                    if not sequence_frames_dir.exists():
                        print(f"    [WARNING] Frames not found for {sequence_name}, skipping...")
                        continue

                    # Create output directory for this sequence
                    sequence_output_dir = output_dir / room / camera_name / sequence_name
                    sequence_output_dir.mkdir(parents=True, exist_ok=True)

                    # Get all JSON annotation files
                    json_files = sorted(sequence_dir.glob("*.json"))

                    if not json_files:
                        continue

                    # Limit number of images if specified
                    if max_images != "all" and max_images is not None:
                        json_files = json_files[:max_images]

                    sequence_count = 0

                    for json_file in json_files:
                        # Find corresponding frame
                        frame_path = sequence_frames_dir / json_file.with_suffix(image_ext).name

                        if not frame_path.exists():
                            continue

                        # Load image
                        image = cv2.imread(str(frame_path))
                        if image is None:
                            continue

                        # Load annotations
                        with open(json_file, 'r') as f:
                            annotations = json.load(f)

                        # Draw bboxes
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
                        output_filename = frame_path.name
                        if image_format != "jpg":
                            output_filename = output_filename.replace(".jpg", f".{image_format}")
                        output_path = sequence_output_dir / output_filename
                        save_annotated_image(image, str(output_path), image_format, image_quality)
                        sequence_count += 1

                    if sequence_count > 0:
                        print(f"    {sequence_name}: {sequence_count} images")
                        total_visualized += sequence_count

        print("\n" + "-" * 80)
        print(f"\n[DONE] Visualization complete!")
        print(f"  Total images: {total_visualized}")
        print(f"\nOutput: {output_dir}")
        print("=" * 80)

    def cleanup_incorrect_annotations(
        self,
        visualization_dir: str,
        annotations_dir: str,
        frames_dir: str,
        sequences: Optional[List[str]] = None,
        rooms: Optional[List[str]] = None
    ):
        """
        Remove PIROPO annotations based on deleted visualizations (after manual review).

        Workflow:
        1. User generates visualizations with visualize_annotations()
        2. User manually reviews and DELETES visualization images with incorrect annotations
        3. This method removes annotation JSON files and frames that have no corresponding visualization

        Args:
            visualization_dir: Directory containing visualization images (after manual review)
            annotations_dir: Directory containing Standard-annotations
            frames_dir: Directory containing frames
            sequences: Ignored for PIROPO
            rooms: List of rooms to process (default: ["Room_A", "Room_B"])
        """
        import os

        visualization_dir = Path(visualization_dir)
        annotations_dir = Path(annotations_dir)
        frames_dir = Path(frames_dir)

        if rooms is None:
            rooms = ["Room_A", "Room_B"]

        print("=" * 80)
        print("PIROPO Incorrect Annotation Cleanup")
        print("=" * 80)
        print(f"\nVisualization dir: {visualization_dir}")
        print(f"Annotations dir: {annotations_dir}")
        print(f"Frames dir: {frames_dir}")
        print("\n" + "-" * 80)

        total_kept_annotations = 0
        total_deleted_annotations = 0
        total_kept_frames = 0
        total_deleted_frames = 0

        for room in rooms:
            room_viz_dir = visualization_dir / room
            room_ann_dir = annotations_dir / room
            room_frames_dir = frames_dir / room

            if not room_viz_dir.exists():
                print(f"\n[WARNING] No visualizations for {room}, skipping...")
                continue

            print(f"\nProcessing {room}...")

            # Get all camera folders
            camera_dirs = sorted([d for d in room_viz_dir.iterdir() if d.is_dir()])

            for camera_dir in camera_dirs:
                camera_name = camera_dir.name

                # Get all sequence folders
                sequence_dirs = sorted([d for d in camera_dir.iterdir() if d.is_dir()])

                for sequence_dir in sequence_dirs:
                    sequence_name = sequence_dir.name

                    # Get remaining visualization files (user has deleted incorrect ones)
                    viz_files = set([f.stem for f in sequence_dir.glob("*.jpg")])

                    # Annotation and frame directories
                    ann_sequence_dir = room_ann_dir / camera_name / sequence_name
                    frames_sequence_dir = room_frames_dir / camera_name / sequence_name

                    if not ann_sequence_dir.exists():
                        continue

                    # Get all annotation files
                    ann_files = list(ann_sequence_dir.glob("*.json"))

                    kept_ann = 0
                    deleted_ann = 0
                    kept_frm = 0
                    deleted_frm = 0

                    for ann_file in ann_files:
                        if ann_file.stem in viz_files:
                            # Keep this annotation (visualization still exists)
                            kept_ann += 1
                        else:
                            # Delete annotation (visualization was deleted by user)
                            os.remove(str(ann_file))
                            deleted_ann += 1

                            # Also delete corresponding frame if it exists
                            if frames_sequence_dir.exists():
                                frame_file = frames_sequence_dir / ann_file.with_suffix('.jpg').name
                                if frame_file.exists():
                                    os.remove(str(frame_file))
                                    deleted_frm += 1

                    # Count kept frames
                    if frames_sequence_dir.exists():
                        kept_frm = len(list(frames_sequence_dir.glob("*.jpg")))

                    if deleted_ann > 0 or deleted_frm > 0:
                        print(f"    {sequence_name}: kept {kept_ann} ann, deleted {deleted_ann} ann, deleted {deleted_frm} frames")

                    total_kept_annotations += kept_ann
                    total_deleted_annotations += deleted_ann
                    total_kept_frames += kept_frm
                    total_deleted_frames += deleted_frm

        print("\n" + "-" * 80)
        print(f"\nAnnotations: kept {total_kept_annotations}, deleted {total_deleted_annotations}")
        print(f"Frames: kept {total_kept_frames}, deleted {total_deleted_frames}")
        print(f"\n[DONE] Cleanup complete!")
        print("=" * 80)
