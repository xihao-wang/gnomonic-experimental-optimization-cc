"""
Metrics module for evaluation framework.

This module provides metrics for evaluating rotated bounding box predictions
against ground truth annotations. All metrics operate on rotated rectangles
in fisheye image coordinates.

Key metrics:
- IoU (Intersection over Union) for rotated rectangles
- Precision, Recall, F1 score (to be added)
- Average Precision (to be added)

Author: Generated for gnomonic projection pedestrian detection project
Date: 2026-02-06
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional


def compute_rotated_iou(
    bbox1: Dict,
    bbox2: Dict
) -> float:
    """
    Compute IoU (Intersection over Union) between two rotated bounding boxes.

    Uses OpenCV's rotatedRectangleIntersection for accurate polygon intersection.
    Both bboxes must be in the same coordinate system (fisheye image coordinates).

    Args:
        bbox1: First bbox dict with keys:
               - center_x, center_y: Center coordinates
               - width, height: Bbox dimensions
               - angle: Rotation angle in degrees
        bbox2: Second bbox dict (same format as bbox1)

    Returns:
        IoU value between 0 and 1
        Returns 0.0 if no intersection or invalid input

    Example:
        >>> gt_bbox = {'center_x': 320, 'center_y': 240, 'width': 50, 'height': 100, 'angle': 45}
        >>> pred_bbox = {'center_x': 325, 'center_y': 245, 'width': 48, 'height': 95, 'angle': 42}
        >>> iou = compute_rotated_iou(gt_bbox, pred_bbox)
        >>> print(f"IoU: {iou:.3f}")
    """
    try:
        # Convert to OpenCV RotatedRect format: ((cx, cy), (w, h), angle)
        rect1 = (
            (bbox1['center_x'], bbox1['center_y']),
            (bbox1['width'], bbox1['height']),
            bbox1['angle']
        )

        rect2 = (
            (bbox2['center_x'], bbox2['center_y']),
            (bbox2['width'], bbox2['height']),
            bbox2['angle']
        )

        # Compute intersection using OpenCV
        intersection_type, intersection_points = cv2.rotatedRectangleIntersection(rect1, rect2)

        # Handle different intersection cases
        if intersection_type == cv2.INTERSECT_NONE:
            return 0.0

        if intersection_type == cv2.INTERSECT_FULL:
            # One rectangle fully inside the other
            # Area of intersection = area of smaller rectangle
            area1 = bbox1['width'] * bbox1['height']
            area2 = bbox2['width'] * bbox2['height']
            intersection_area = min(area1, area2)
            union_area = max(area1, area2)
            return intersection_area / union_area if union_area > 0 else 0.0

        # INTERSECT_PARTIAL: Compute polygon intersection area
        if intersection_points is None or len(intersection_points) < 3:
            return 0.0

        # Compute area of intersection polygon
        intersection_points = np.array(intersection_points, dtype=np.float32)
        intersection_area = cv2.contourArea(intersection_points)

        # Compute areas of both rectangles
        area1 = bbox1['width'] * bbox1['height']
        area2 = bbox2['width'] * bbox2['height']

        # Compute union area
        union_area = area1 + area2 - intersection_area

        # Compute IoU
        if union_area <= 0:
            return 0.0

        iou = intersection_area / union_area

        # Clamp to [0, 1] (numerical safety)
        return max(0.0, min(1.0, iou))

    except Exception as e:
        # Return 0 for any errors (invalid bbox format, numerical issues, etc.)
        print(f"Warning: Error computing IoU: {e}")
        return 0.0


def match_predictions_to_ground_truth(
    predictions: list,
    ground_truth: list,
    iou_threshold: float = 0.5
) -> Tuple[list, list, list]:
    """
    Match predicted bboxes to ground truth bboxes using IoU threshold.

    Greedy matching: Each prediction matched to GT with highest IoU (if above threshold).
    Each GT can only be matched once (first come, first served).

    Args:
        predictions: List of prediction dicts (rotated bbox format)
        ground_truth: List of GT dicts (rotated bbox format)
        iou_threshold: Minimum IoU to consider a match (default: 0.5)

    Returns:
        Tuple of (true_positives, false_positives, false_negatives)
        - true_positives: List of (pred_idx, gt_idx, iou) tuples
        - false_positives: List of pred_idx (unmatched predictions)
        - false_negatives: List of gt_idx (unmatched ground truth)
    """
    if len(predictions) == 0 and len(ground_truth) == 0:
        return [], [], []

    if len(predictions) == 0:
        # No predictions, all GT are false negatives
        return [], [], list(range(len(ground_truth)))

    if len(ground_truth) == 0:
        # No GT, all predictions are false positives
        return [], list(range(len(predictions))), []

    # Compute IoU matrix (predictions × ground_truth)
    iou_matrix = np.zeros((len(predictions), len(ground_truth)))
    for i, pred in enumerate(predictions):
        for j, gt in enumerate(ground_truth):
            iou_matrix[i, j] = compute_rotated_iou(pred, gt)

    # Greedy matching: Match each prediction to best GT (if above threshold)
    matched_gt = set()
    true_positives = []
    false_positives = []

    for pred_idx in range(len(predictions)):
        # Find GT with highest IoU for this prediction
        best_gt_idx = np.argmax(iou_matrix[pred_idx])
        best_iou = iou_matrix[pred_idx, best_gt_idx]

        # Check if IoU is above threshold and GT not already matched
        if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
            true_positives.append((pred_idx, best_gt_idx, best_iou))
            matched_gt.add(best_gt_idx)
        else:
            false_positives.append(pred_idx)

    # Unmatched GT are false negatives
    false_negatives = [i for i in range(len(ground_truth)) if i not in matched_gt]

    return true_positives, false_positives, false_negatives


def compute_precision_recall(
    true_positives: int,
    false_positives: int,
    false_negatives: int
) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 score.

    Args:
        true_positives: Number of correct predictions
        false_positives: Number of incorrect predictions
        false_negatives: Number of missed ground truth

    Returns:
        Tuple of (precision, recall, f1_score)
        Returns 0.0 for undefined cases (e.g., no predictions)
    """
    # Precision: TP / (TP + FP)
    if true_positives + false_positives > 0:
        precision = true_positives / (true_positives + false_positives)
    else:
        precision = 0.0

    # Recall: TP / (TP + FN)
    if true_positives + false_negatives > 0:
        recall = true_positives / (true_positives + false_negatives)
    else:
        recall = 0.0

    # F1 Score: 2 * (precision * recall) / (precision + recall)
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    return precision, recall, f1_score
