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


def _axis_aligned_bbox_from_corners(bbox: Dict) -> Tuple[List[float], List[float]]:
    """Return fisheye-space [x1, y1, x2, y2] and [x, y, w, h] boxes."""
    corners = bbox.get('corners')
    if corners is not None and len(corners) > 0:
        pts = np.asarray(corners, dtype=float)
        x1 = float(np.min(pts[:, 0]))
        y1 = float(np.min(pts[:, 1]))
        x2 = float(np.max(pts[:, 0]))
        y2 = float(np.max(pts[:, 1]))
    else:
        cx, cy = bbox['center']
        width, height = bbox['size']
        x1 = float(cx - width / 2.0)
        y1 = float(cy - height / 2.0)
        x2 = float(cx + width / 2.0)
        y2 = float(cy + height / 2.0)
    return [x1, y1, x2, y2], [x1, y1, x2 - x1, y2 - y1]


def compute_reid_source_quality(detection: Dict,
                                boundary_threshold_ratio: float = 0.03) -> float:
    """
    Score how suitable a source composite bbox is for ReID feature extraction.

    The first version is intentionally simple: prefer high-confidence, larger crops,
    and penalize boxes close to a projection-cell boundary because they are likely
    truncated by the gnomonic view.
    """
    conf = float(detection.get('source_confidence', detection.get('confidence', 0.0)))
    cell_bbox = detection.get('source_cell_bbox_xyxy')
    cell_size = detection.get('source_cell_size')

    if cell_bbox is None or cell_size is None:
        return conf

    x1, y1, x2, y2 = [float(v) for v in cell_bbox]
    cell_w, cell_h = [float(v) for v in cell_size]
    if cell_w <= 0 or cell_h <= 0:
        return conf

    bbox_w = max(0.0, x2 - x1)
    bbox_h = max(0.0, y2 - y1)
    area_ratio = (bbox_w * bbox_h) / max(cell_w * cell_h, 1e-12)
    area_score = min(1.0, float(np.sqrt(max(0.0, area_ratio))))

    margin = min(x1, y1, cell_w - x2, cell_h - y2)
    boundary_threshold = boundary_threshold_ratio * min(cell_w, cell_h)
    boundary_penalty = 1.0 if margin < boundary_threshold else 0.0

    return conf + area_score - boundary_penalty


def attach_reid_source_groups(filtered_detections: List[Dict],
                              original_detections: List[Dict],
                              duplicate_iou_threshold: float = 0.5) -> List[Dict]:
    """
    Attach duplicate-group and best-source metadata to Stage 2 kept detections.

    The kept fisheye bbox remains the representative detection. ReID should use the
    best source crop selected from all original detections that overlap this kept
    fisheye bbox in fisheye space.
    """
    enriched = []
    for kept in filtered_detections:
        group = []
        kept_source_id = kept.get('source_id')
        for candidate in original_detections:
            same_source = kept_source_id is not None and candidate.get('source_id') == kept_source_id
            overlaps = compute_rotated_iou(kept, candidate) >= duplicate_iou_threshold
            if same_source or overlaps:
                group.append(candidate)

        if not group:
            group = [kept]

        best_source = max(group, key=compute_reid_source_quality)
        out = kept.copy()
        out['duplicate_source_ids'] = [g.get('source_id') for g in group if g.get('source_id') is not None]
        out['duplicate_count'] = len(group)
        out['reid_source_id'] = best_source.get('source_id')
        out['reid_source_bbox_xyxy'] = best_source.get('source_bbox_xyxy')
        out['reid_source_bbox_norm'] = best_source.get('source_bbox_norm')
        out['reid_source_projection_id'] = best_source.get('source_projection_id')
        out['reid_source_cell'] = best_source.get('source_cell')
        out['reid_source_cell_bbox_xyxy'] = best_source.get('source_cell_bbox_xyxy')
        out['reid_source_confidence'] = best_source.get(
            'source_confidence', best_source.get('confidence', 0.0)
        )
        out['reid_source_quality'] = compute_reid_source_quality(best_source)
        out['tracking_bbox_xyxy'], out['tracking_bbox_tlwh'] = _axis_aligned_bbox_from_corners(out)
        enriched.append(out)

    return enriched


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
    original_bboxes = [bbox.copy() for bbox in fisheye_bboxes]
    filtered = soft_nms_gaussian(fisheye_bboxes, sigma, score_threshold)
    return attach_reid_source_groups(filtered, original_bboxes)
