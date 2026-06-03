"""
Non-Maximum Suppression (NMS) for detection pipeline.

This module implements two-stage NMS:
- Stage 1: Standard NMS on composite image detections (before backprojection)
- Stage 2: Soft-NMS on fisheye detections (after backprojection)

Stage 2 uses Gaussian Soft-NMS to handle overlapping detections from multiple
projection views while preserving valid detections.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple


def compute_rotated_iou(bbox1: Dict, bbox2: Dict) -> float:
    """
    Compute IoU between two rotated bounding boxes.

    Args:
        bbox1, bbox2: Dicts with keys:
            - 'center': (x, y) center coordinates
            - 'size': (width, height) dimensions
            - 'angle': rotation angle in degrees

    Returns:
        IoU value in range [0, 1]
    """
    # Convert to OpenCV RotatedRect format: ((cx, cy), (w, h), angle)
    rect1 = (
        tuple(bbox1['center']),
        tuple(bbox1['size']),
        float(bbox1['angle'])
    )
    rect2 = (
        tuple(bbox2['center']),
        tuple(bbox2['size']),
        float(bbox2['angle'])
    )

    # Compute intersection
    intersection_type, intersection_points = cv2.rotatedRectangleIntersection(rect1, rect2)

    # Case 1: No intersection
    if intersection_type == cv2.INTERSECT_NONE:
        return 0.0

    # Case 2: Full containment (one rectangle fully inside the other)
    if intersection_type == cv2.INTERSECT_FULL:
        area1 = bbox1['size'][0] * bbox1['size'][1]
        area2 = bbox2['size'][0] * bbox2['size'][1]
        intersection_area = min(area1, area2)
        union_area = max(area1, area2)
        return intersection_area / union_area if union_area > 0 else 0.0

    # Case 3: Partial overlap
    # intersection_type == cv2.INTERSECT_PARTIAL
    if intersection_points is None or len(intersection_points) < 3:
        return 0.0

    # Compute intersection area using contour area (Shoelace formula)
    intersection_area = cv2.contourArea(intersection_points)

    # Compute union area
    area1 = bbox1['size'][0] * bbox1['size'][1]
    area2 = bbox2['size'][0] * bbox2['size'][1]
    union_area = area1 + area2 - intersection_area

    if union_area <= 0:
        return 0.0

    # Compute IoU and clamp to [0, 1]
    iou = intersection_area / union_area
    return max(0.0, min(1.0, iou))


def soft_nms_gaussian(detections: List[Dict], sigma: float = 0.2,
                      score_threshold: float = 0.005) -> List[Dict]:
    """
    Apply Gaussian Soft-NMS to rotated bounding boxes.

    Soft-NMS reduces the scores of overlapping detections instead of removing them,
    using a Gaussian penalty based on IoU:
        score ← score * exp((-IoU²) / sigma)

    Args:
        detections: List of detection dicts with keys:
            - 'center': (x, y) center coordinates
            - 'size': (width, height) dimensions
            - 'angle': rotation angle in degrees
            - 'confidence': detection confidence score
            - Additional keys preserved (e.g., 'class_name', 'lattice_points')
        sigma: Gaussian kernel width parameter
            - Lower values (0.1): more aggressive suppression
            - Higher values (0.4): gentler suppression
            - Literature suggests: 0.1, 0.2, 0.4
        score_threshold: Minimum score to keep after soft suppression
            - Suggested values: 0.001, 0.005, 0.01, 0.05

    Returns:
        List of detections with updated confidence scores, filtered by score_threshold
    """
    if len(detections) == 0:
        return []

    # Create working copy to avoid modifying input
    dets = [det.copy() for det in detections]

    # Sort by confidence (highest first)
    dets.sort(key=lambda x: x['confidence'], reverse=True)

    # Apply Soft-NMS
    for i in range(len(dets)):
        # Skip if already suppressed below threshold
        if dets[i]['confidence'] < score_threshold:
            continue

        # Compare with all lower-scoring detections
        for j in range(i + 1, len(dets)):
            if dets[j]['confidence'] < score_threshold:
                continue

            # Compute IoU between detections
            iou = compute_rotated_iou(dets[i], dets[j])

            # Apply Gaussian penalty: score ← score * exp((-IoU²) / sigma)
            penalty = np.exp(-(iou ** 2) / sigma)
            dets[j]['confidence'] *= penalty

    # Filter detections below threshold
    filtered = [det for det in dets if det['confidence'] >= score_threshold]

    return filtered


def apply_stage1_nms(detections: List[Dict], iou_threshold: float = 0.8) -> List[Dict]:
    """
    Apply Stage 1 standard NMS on composite image detections.

    Note: YOLO already applies internal NMS during inference, so this is typically
    a no-op unless you want additional filtering with a different threshold.

    Args:
        detections: List of detection dicts in composite coordinates (normalized 0-1)
            with keys: 'x', 'y', 'w', 'h', 'confidence', 'class_name'
        iou_threshold: IoU threshold for NMS (high threshold keeps more boxes)

    Returns:
        Filtered list of detections
    """
    # For axis-aligned boxes in normalized coordinates, we can use OpenCV's NMS
    # However, since YOLO already applies NMS, this is mainly for consistency

    if len(detections) == 0:
        return []

    # Convert to format for cv2.dnn.NMSBoxes: [x, y, w, h]
    boxes = [[det['x'], det['y'], det['w'], det['h']] for det in detections]
    confidences = [det['confidence'] for det in detections]

    # Apply NMS (returns indices of boxes to keep)
    # Note: cv2.dnn.NMSBoxes expects boxes in (x, y, w, h) format where (x, y) is top-left
    # Our detections have (x, y) as center, so we need to convert
    boxes_topleft = []
    for det in detections:
        x_topleft = det['x'] - det['w'] / 2
        y_topleft = det['y'] - det['h'] / 2
        boxes_topleft.append([x_topleft, y_topleft, det['w'], det['h']])

    indices = cv2.dnn.NMSBoxes(
        boxes_topleft,
        confidences,
        score_threshold=0.0,  # Keep all boxes (threshold already applied by YOLO)
        nms_threshold=iou_threshold
    )

    # Filter detections
    if len(indices) > 0:
        # cv2.dnn.NMSBoxes returns a 2D array in older versions, 1D in newer
        if isinstance(indices, np.ndarray):
            indices = indices.flatten()
        elif isinstance(indices, tuple):
            indices = list(indices)

        filtered = [detections[i] for i in indices]
        return filtered
    else:
        return []


def apply_stage2_nms(fisheye_bboxes: List[Dict], sigma: float = 0.2,
                     score_threshold: float = 0.005) -> List[Dict]:
    """
    Apply Stage 2 Soft-NMS on fisheye detections (after backprojection).

    Args:
        fisheye_bboxes: List of backprojected bbox dicts with keys:
            - 'center': (x, y) center in fisheye coordinates
            - 'size': (width, height) dimensions
            - 'angle': rotation angle in degrees
            - 'confidence': detection confidence
            - Additional keys preserved (e.g., 'class_name', 'lattice_points')
        sigma: Gaussian kernel width for Soft-NMS
            - 0.1: aggressive suppression
            - 0.2: moderate suppression (default)
            - 0.4: gentle suppression
        score_threshold: Minimum score to keep after soft suppression
            - Suggested: 0.001, 0.005, 0.01, 0.05

    Returns:
        Filtered list of fisheye bboxes with updated confidence scores
    """
    return soft_nms_gaussian(fisheye_bboxes, sigma, score_threshold)
