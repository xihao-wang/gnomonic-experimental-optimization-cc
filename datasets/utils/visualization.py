"""
Shared visualization utilities for dataset annotations.

This module provides reusable visualization functions for drawing rotated bounding boxes
on fisheye images using our standard JSON annotation format.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional


def draw_rotated_bbox_on_image(
    image: np.ndarray,
    annotations: List[Dict],
    bbox_color: Tuple[int, int, int] = (0, 255, 0),
    bbox_thickness: int = 2,
    font_scale: float = 0.5,
    font_thickness: int = 1,
    show_labels: bool = True,
    show_rotation_angle: bool = True,
    draw_rotated: bool = True,
    draw_center_and_radial: bool = False,
    fisheye_center: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    Draw rotated bounding box annotations on an image.

    Expects annotations in our standard JSON format:
    {
        "center_x": float,
        "center_y": float,
        "width": float,
        "height": float,
        "angle": float,
        "class_name": string
    }

    Args:
        image: Input image (BGR format, will be modified in-place)
        annotations: List of annotation dicts in standard format
        bbox_color: BGR color tuple for bbox drawing
        bbox_thickness: Line thickness for bbox
        font_scale: Font scale for labels
        font_thickness: Font thickness for labels
        show_labels: Whether to show class labels
        show_rotation_angle: Whether to show rotation angle
        draw_rotated: If True, draw rotated bbox; if False, draw axis-aligned bbox
        draw_center_and_radial: Whether to draw image center and radial lines
        fisheye_center: Image center coordinates (for radial line drawing)

    Returns:
        Modified image with annotations drawn
    """
    # Draw image center if requested
    if draw_center_and_radial and fisheye_center:
        img_center = (int(fisheye_center[0]), int(fisheye_center[1]))
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
            box = np.intp(box)
            cv2.drawContours(image, [box], 0, bbox_color, bbox_thickness)
        else:
            # Draw axis-aligned rectangle
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), bbox_color, bbox_thickness)

        # Draw radial line if requested
        if draw_center_and_radial and fisheye_center:
            img_center = (int(fisheye_center[0]), int(fisheye_center[1]))
            cv2.line(image, img_center, center, (0, 255, 255), 1)  # Yellow line

        # Draw label
        if show_labels or show_rotation_angle:
            label_parts = []
            if show_labels:
                label_parts.append(ann["class_name"])
            if show_rotation_angle:
                label_parts.append(f"{angle:.1f}deg")
            label = " ".join(label_parts)

            # Position label above bbox
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

    return image


def save_annotated_image(
    image: np.ndarray,
    output_path: str,
    image_format: str = "jpg",
    image_quality: int = 95
) -> None:
    """
    Save annotated image to disk.

    Args:
        image: Image to save (BGR format)
        output_path: Output file path
        image_format: Image format ("jpg", "png", etc.)
        image_quality: JPEG quality (0-100)
    """
    if image_format == "jpg":
        cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, image_quality])
    else:
        cv2.imwrite(output_path, image)
