"""
Preprocess BOMNI rotated bbox annotations from Tamura et al. format to clean format.

This script converts the third-party annotations (omnidet-rotinv) from the inefficient
Pascal VOC XML format (where fields are repurposed) into a clean JSON format with
all rotated bbox parameters precomputed.

Input: datasets/all-datasets/BOMNI-corrected/Rotated-annotations/scenario1/
Output: datasets/all-datasets/BOMNI-corrected/Preprocessed-annotations/scenario1/

Author: Generated for gnomonic projection pedestrian detection project
Date: 2025-12-03
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np


def calculate_rotation_angle(bbox_center_x, bbox_center_y, img_center_x=320, img_center_y=240):
    """
    Calculate rotation angle from vertical (up) to vector from image center to bbox center.

    As specified by Tamura et al.: "Rotation angle for each bounding box is the angle
    between a vertical line and a line connecting the image center and bounding box center."

    Args:
        bbox_center_x: Bbox center X coordinate
        bbox_center_y: Bbox center Y coordinate
        img_center_x: Fisheye image center X (default 320 for 640×480 image)
        img_center_y: Fisheye image center Y (default 240 for 640×480 image)

    Returns:
        Rotation angle in degrees (0° = vertical up, clockwise positive)
    """
    dx = bbox_center_x - img_center_x
    dy = bbox_center_y - img_center_y
    angle_rad = np.arctan2(dx, -dy)  # -dy because y-axis points down in images
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def parse_tamura_xml(xml_path):
    """
    Parse Tamura et al. XML annotation (Pascal VOC format with repurposed fields).

    The XML stores rotated bbox parameters in Pascal VOC fields:
    - center_x = (xmin + xmax) / 2
    - center_y = (ymin + ymax) / 2
    - width = xmax - xmin (tight-fit rotated bbox width)
    - height = ymax - ymin (tight-fit rotated bbox height)
    - angle: NOT STORED, must be calculated

    Args:
        xml_path: Path to Pascal VOC XML file

    Returns:
        List of dicts, each containing:
            - center_x, center_y: Bbox center coordinates
            - width, height: Tight-fit rotated bbox dimensions
            - angle: Rotation angle in degrees
            - class_name: Object class (always "person")
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
        angle = calculate_rotation_angle(center_x, center_y)

        annotations.append({
            "center_x": center_x,
            "center_y": center_y,
            "width": width,
            "height": height,
            "angle": angle,
            "class_name": class_name
        })

    return annotations


def convert_annotations(input_dir, output_dir, sequences):
    """
    Convert all Tamura annotations to preprocessed JSON format.

    Args:
        input_dir: Path to BOMNI-corrected/Rotated-annotations/scenario1/
        output_dir: Path to BOMNI-corrected/Preprocessed-annotations/scenario1/
        sequences: List of sequence names (e.g., ["top-0", "top-1", ...])
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    total_files = 0
    total_annotations = 0

    print("=" * 70)
    print("BOMNI Annotation Preprocessing")
    print("=" * 70)
    print(f"\nInput:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"\nSequences: {sequences}")
    print("\n" + "-" * 70)

    for sequence in sequences:
        sequence_input_dir = input_dir / sequence
        sequence_output_dir = output_dir / sequence

        if not sequence_input_dir.exists():
            print(f"\n[WARNING] Sequence '{sequence}' not found in input directory, skipping...")
            continue

        # Create output directory
        sequence_output_dir.mkdir(parents=True, exist_ok=True)

        # Get all XML files
        xml_files = sorted(sequence_input_dir.glob("*.xml"))

        print(f"\nProcessing {sequence}: {len(xml_files)} files")

        sequence_file_count = 0
        sequence_annotation_count = 0

        for xml_file in xml_files:
            # Parse XML
            annotations = parse_tamura_xml(xml_file)

            # Create output JSON file with same name
            json_file = sequence_output_dir / xml_file.with_suffix('.json').name

            # Save as JSON
            with open(json_file, 'w') as f:
                json.dump(annotations, f, indent=2)

            sequence_file_count += 1
            sequence_annotation_count += len(annotations)

        print(f"  -> Created {sequence_file_count} JSON files ({sequence_annotation_count} annotations)")

        total_files += sequence_file_count
        total_annotations += sequence_annotation_count

    print("\n" + "-" * 70)
    print(f"\n[DONE] Preprocessing complete!")
    print(f"  Total files: {total_files}")
    print(f"  Total annotations: {total_annotations}")
    print(f"\nOutput directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    # Configuration
    input_dir = "datasets/all-datasets/BOMNI-corrected/Rotated-annotations/scenario1"
    output_dir = "datasets/all-datasets/BOMNI-corrected/Preprocessed-annotations/scenario1"
    sequences = ["top-0", "top-1", "top-2", "top-3"]

    # Convert annotations
    convert_annotations(input_dir, output_dir, sequences)
