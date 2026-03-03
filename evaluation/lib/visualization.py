"""
Visualization module for evaluation framework.

This module provides functions to visualize ground truth and predicted bounding boxes
for evaluation purposes. Visualizations are generated for both composite images
(YOLO detections) and fisheye images (backprojected bboxes).

Key visual design:
- GT bboxes drawn FIRST: Green, thickness=2 (thicker)
- Predictions drawn on top: Red, thickness=1 (thinner)
- When aligned, both boxes are visible

Author: Generated for gnomonic projection pedestrian detection project
Date: 2025-12-04
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from datasets.utils.visualization import draw_rotated_bbox_on_image


def visualize_gt_and_predictions(
    image: np.ndarray,
    gt_annotations: List[Dict],
    pred_annotations: List[Dict],
    fisheye_center: Optional[Tuple[float, float]] = None,
    show_legend: bool = True,
    gt_color: Tuple[int, int, int] = (0, 255, 0),
    gt_thickness: int = 2,
    pred_color: Tuple[int, int, int] = (0, 255, 255),
    pred_thickness: int = 3
) -> np.ndarray:
    """
    Visualize ground truth and predicted bounding boxes on an image.

    GT boxes are drawn first (green, thick) and predictions on top (configurable color, thicker)
    so both are visible when they align.

    Args:
        image: Input image (BGR format, will be copied)
        gt_annotations: List of GT annotation dicts in standard format
        pred_annotations: List of prediction annotation dicts in standard format
        fisheye_center: Optional fisheye center for radial visualization
        show_legend: Whether to add color legend
        gt_color: BGR color tuple for ground truth bboxes (default: green)
        gt_thickness: Line thickness for ground truth bboxes (default: 2)
        pred_color: BGR color tuple for predicted bboxes (default: yellow)
        pred_thickness: Line thickness for predicted bboxes (default: 3)

    Returns:
        Annotated image with GT and predictions
    """
    # Make a copy to avoid modifying original
    img_vis = image.copy()

    # Draw GT bboxes FIRST
    if gt_annotations:
        img_vis = draw_rotated_bbox_on_image(
            image=img_vis,
            annotations=gt_annotations,
            bbox_color=gt_color,
            bbox_thickness=gt_thickness,
            show_labels=False,
            show_rotation_angle=False,
            draw_rotated=True,
            draw_center_and_radial=False,
            fisheye_center=fisheye_center
        )

    # Draw predicted bboxes on top
    if pred_annotations:
        img_vis = draw_rotated_bbox_on_image(
            image=img_vis,
            annotations=pred_annotations,
            bbox_color=pred_color,
            bbox_thickness=pred_thickness,
            show_labels=False,
            show_rotation_angle=False,
            draw_rotated=True,
            draw_center_and_radial=False,
            fisheye_center=fisheye_center
        )

    # Add legend if requested
    if show_legend:
        img_vis = add_legend(
            img_vis,
            gt_count=len(gt_annotations),
            pred_count=len(pred_annotations),
            gt_color=gt_color,
            pred_color=pred_color,
            gt_thickness=gt_thickness,
            pred_thickness=pred_thickness
        )

    return img_vis


def add_legend(
    image: np.ndarray,
    gt_count: int = 0,
    pred_count: int = 0,
    gt_color: Tuple[int, int, int] = (0, 255, 0),
    pred_color: Tuple[int, int, int] = (0, 255, 255),
    gt_thickness: int = 2,
    pred_thickness: int = 3
) -> np.ndarray:
    """
    Add a legend showing GT and predictions with counts and actual colors used.

    Args:
        image: Input image
        gt_count: Number of GT annotations
        pred_count: Number of predicted annotations
        gt_color: BGR color tuple for GT bboxes (default: green)
        pred_color: BGR color tuple for predicted bboxes (default: yellow)
        gt_thickness: Line thickness for GT bboxes (default: 2)
        pred_thickness: Line thickness for predicted bboxes (default: 3)

    Returns:
        Image with legend added
    """
    h, w = image.shape[:2]
    legend_height = 60
    legend_width = 250
    margin = 10

    # Legend background (semi-transparent black)
    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (margin, margin),
        (margin + legend_width, margin + legend_height),
        (0, 0, 0),
        -1
    )
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

    # GT legend (colored box + text)
    cv2.rectangle(
        image,
        (margin + 10, margin + 15),
        (margin + 30, margin + 25),
        gt_color,
        gt_thickness
    )
    cv2.putText(
        image,
        f"GT: {gt_count}",
        (margin + 40, margin + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    # Predictions legend (colored box + text)
    cv2.rectangle(
        image,
        (margin + 10, margin + 35),
        (margin + 30, margin + 45),
        pred_color,
        pred_thickness
    )
    cv2.putText(
        image,
        f"Pred: {pred_count}",
        (margin + 40, margin + 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    return image


def visualize_composite_detections(
    composite_image: np.ndarray,
    detections: List[Dict],
    bbox_color: Tuple[int, int, int] = (0, 255, 255),
    bbox_thickness: int = 3,
    show_labels: bool = True
) -> np.ndarray:
    """
    Visualize YOLO detections on composite image.

    Args:
        composite_image: Composite image (BGR format)
        detections: List of detection dicts with bbox in normalized coordinates (0-1)
                   Format: {'x': float, 'y': float, 'w': float, 'h': float,
                           'confidence': float, 'class_name': str}
        bbox_color: BGR color tuple for bboxes (default: yellow to match backprojection)
        bbox_thickness: Line thickness for bboxes (default: 3)
        show_labels: Whether to show class labels and confidence

    Returns:
        Annotated composite image
    """
    img_vis = composite_image.copy()
    h, w = img_vis.shape[:2]

    for det in detections:
        # Convert normalized coords to pixel coords
        x_center = int(det['x'] * w)
        y_center = int(det['y'] * h)
        width = int(det['w'] * w)
        height = int(det['h'] * h)

        # Calculate bbox corners
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        # Draw bbox with configurable color
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), bbox_color, bbox_thickness)

        # Draw label
        if show_labels:
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            cv2.putText(
                img_vis,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                bbox_color,
                1
            )

    return img_vis


def create_side_by_side_visualization(
    fisheye_image: np.ndarray,
    composite_image: np.ndarray,
    margin: int = 20
) -> np.ndarray:
    """
    Create side-by-side visualization with fisheye (left) and composite (right).

    Simply stacks two pre-rendered images horizontally with a white header bar.
    No bbox drawing is done here - the input images should already have their bboxes drawn:
    - fisheye_image: GT (green) + backprojected predictions (yellow)
    - composite_image: YOLO detections (yellow)

    Args:
        fisheye_image: Pre-rendered fisheye image with GT + predictions (BGR format)
        composite_image: Pre-rendered composite image with detections (BGR format)
        margin: Margin between the two images in pixels (default: 20)

    Returns:
        Combined side-by-side image with white header bar and labels
    """
    # Use images as-is (already rendered)
    fisheye_vis = fisheye_image
    composite_vis = composite_image

    # Get dimensions
    h_fish, w_fish = fisheye_vis.shape[:2]
    h_comp, w_comp = composite_vis.shape[:2]

    # Calculate header bar height (relative to image size, ~3% of max height)
    max_height = max(h_fish, h_comp)
    header_height = max(30, int(max_height * 0.03))  # Minimum 30px, or 3% of image height

    # Calculate font size relative to header height (50% smaller than before)
    font_scale = max(0.4, header_height / 60.0)  # Scale with header, min 0.4
    font_thickness = max(1, int(header_height / 25.0))  # Scale thickness too

    # Create combined canvas: header + images
    combined_width = w_fish + margin + w_comp
    total_height = header_height + max_height
    combined = np.ones((total_height, combined_width, 3), dtype=np.uint8) * 255

    # Place fisheye on left (centered vertically, below header)
    y_offset_fish = header_height + (max_height - h_fish) // 2
    combined[y_offset_fish:y_offset_fish + h_fish, 0:w_fish] = fisheye_vis

    # Place composite on right (centered vertically, below header)
    y_offset_comp = header_height + (max_height - h_comp) // 2
    x_offset_comp = w_fish + margin
    combined[y_offset_comp:y_offset_comp + h_comp, x_offset_comp:x_offset_comp + w_comp] = composite_vis

    # Add labels in white header bar
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (0, 0, 0)  # Black text on white background
    text_y = int(header_height * 0.7)  # Position text at 70% of header height

    # Label: Fisheye (left side)
    cv2.putText(
        combined,
        "Fisheye (Backprojected)",
        (10, text_y),
        font,
        font_scale,
        text_color,
        font_thickness
    )

    # Label: Composite (right side)
    cv2.putText(
        combined,
        "Composite (Original Detection)",
        (x_offset_comp + 10, text_y),
        font,
        font_scale,
        text_color,
        font_thickness
    )

    return combined
