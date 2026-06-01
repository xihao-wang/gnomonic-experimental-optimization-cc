"""Adapter from source-aware gnomonic detections to tracker detections.

The detection pipeline now returns two useful coordinate sources for each final
person candidate:

* a fisheye/tracking box, used by the tracker state and motion model;
* a perspective/composite source box, used to crop the clearest ReID image.

This module keeps that split explicit and exposes a DeepSORT-like detection
object: ``tlwh + confidence + feature``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import cv2
import numpy as np


FeatureExtractor = Callable[[np.ndarray, Sequence[np.ndarray]], np.ndarray]


@dataclass
class GnomonicTrackerDetection:
    """DeepSORT-compatible detection with extra gnomonic metadata.

    Attributes:
        tlwh: Axis-aligned fisheye tracking bbox as ``[top_left_x, top_left_y,
            width, height]``. This is the box that should feed Kalman/tracking.
        confidence: Detection confidence used by the tracker.
        feature: ReID embedding extracted from the configured crop source.
        reid_source_bbox_xyxy: Crop box used to compute ``feature``.
        metadata: Source-aware debug fields kept for inspection/logging.
    """

    tlwh: np.ndarray
    confidence: float
    feature: np.ndarray
    reid_source_bbox_xyxy: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)

    def to_tlbr(self) -> np.ndarray:
        """Convert ``tlwh`` to ``[min_x, min_y, max_x, max_y]``."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self) -> np.ndarray:
        """Convert ``tlwh`` to DeepSORT's ``[center_x, center_y, aspect, h]``."""
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        ret[2] /= ret[3]
        return ret


def xyxy_to_tlwh(xyxy: Sequence[float]) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    return np.asarray([x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)], dtype=np.float32)


def tlwh_to_xyxy(tlwh: Sequence[float]) -> np.ndarray:
    x, y, w, h = [float(v) for v in tlwh]
    return np.asarray([x, y, x + w, y + h], dtype=np.float32)


def clip_xyxy_to_image(xyxy: Sequence[float], image_shape: Sequence[int]) -> Optional[np.ndarray]:
    """Clip an xyxy bbox to an image. Return None if the clipped crop is empty."""
    h, w = image_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    x1 = max(0.0, min(float(w - 1), x1))
    y1 = max(0.0, min(float(h - 1), y1))
    x2 = max(0.0, min(float(w), x2))
    y2 = max(0.0, min(float(h), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def crop_xyxy(image: np.ndarray, xyxy: Sequence[float], pad: int = 0) -> Optional[np.ndarray]:
    """Crop an xyxy region from an image, optionally with pixel padding."""
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    if pad > 0:
        x1 -= pad
        y1 -= pad
        x2 += pad
        y2 += pad
    clipped = clip_xyxy_to_image([x1, y1, x2, y2], image.shape)
    if clipped is None:
        return None
    ix1, iy1, ix2, iy2 = [int(round(float(v))) for v in clipped]
    return image[iy1:iy2, ix1:ix2]


def _tracking_tlwh_from_bbox(bbox: Dict) -> np.ndarray:
    if bbox.get("tracking_bbox_tlwh") is not None:
        return np.asarray(bbox["tracking_bbox_tlwh"], dtype=np.float32)

    if bbox.get("tracking_bbox_xyxy") is not None:
        return xyxy_to_tlwh(bbox["tracking_bbox_xyxy"])

    if all(k in bbox for k in ("center_x", "center_y", "width", "height")):
        cx = float(bbox["center_x"])
        cy = float(bbox["center_y"])
        width = float(bbox["width"])
        height = float(bbox["height"])
        return np.asarray([cx - width / 2.0, cy - height / 2.0, width, height], dtype=np.float32)

    if all(k in bbox for k in ("center", "size")):
        cx, cy = bbox["center"]
        width, height = bbox["size"]
        return np.asarray([float(cx) - float(width) / 2.0, float(cy) - float(height) / 2.0,
                           float(width), float(height)], dtype=np.float32)

    raise ValueError("Cannot build tracking tlwh from bbox; missing tracking_bbox_tlwh/xyxy or center/size fields")


def _reid_source_xyxy_from_bbox(bbox: Dict) -> Optional[np.ndarray]:
    if bbox.get("reid_source_bbox_xyxy") is not None:
        return np.asarray(bbox["reid_source_bbox_xyxy"], dtype=np.float32)
    if bbox.get("source_bbox_xyxy") is not None:
        return np.asarray(bbox["source_bbox_xyxy"], dtype=np.float32)
    return None


def _metadata_from_bbox(bbox: Dict) -> Dict:
    keys = (
        "source_id",
        "source_bbox_xyxy",
        "source_bbox_norm",
        "source_projection_id",
        "source_cell",
        "source_confidence",
        "duplicate_source_ids",
        "duplicate_count",
        "reid_source_id",
        "reid_source_bbox_xyxy",
        "reid_source_bbox_norm",
        "reid_source_projection_id",
        "reid_source_cell",
        "reid_source_confidence",
        "reid_source_quality",
        "tracking_bbox_xyxy",
        "tracking_bbox_tlwh",
        "center_x",
        "center_y",
        "width",
        "height",
        "angle",
        "class_name",
    )
    return {key: bbox.get(key) for key in keys if key in bbox}


def build_tracker_detections(
    fisheye_bboxes: Iterable[Dict],
    reid_image: np.ndarray,
    feature_extractor: Optional[FeatureExtractor] = None,
    feature_dim: int = 0,
    require_features: bool = False,
    crop_pad: int = 0,
    reid_source: str = "fisheye",
) -> List[GnomonicTrackerDetection]:
    """Convert source-aware fisheye bboxes into tracker-ready detections.

    Args:
        fisheye_bboxes: Output of ``DetectionPipeline.run(..., return_visuals=True)``.
        reid_image: Image used for ReID crops. In the current track-fisheye
            experiment this is the fisheye frame; the legacy source-aware mode
            can still pass the perspective composite image.
        feature_extractor: Optional callable ``(reid_image, crop_boxes_xyxy)``
            returning an ``N x D`` feature matrix.
        feature_dim: Placeholder feature dimension when no extractor is provided.
        require_features: If True, raise when no extractor is supplied.
        crop_pad: Pixel padding applied only for validating/cropping ReID boxes.
        reid_source: ``"fisheye"`` to crop the final NMS fisheye bbox, or
            ``"composite"`` to crop the selected source-aware perspective bbox.

    Returns:
        List of ``GnomonicTrackerDetection`` objects.
    """
    if reid_source not in {"fisheye", "composite"}:
        raise ValueError(f"Unsupported reid_source={reid_source!r}; expected 'fisheye' or 'composite'")

    boxes = list(fisheye_bboxes or [])
    if not boxes:
        return []

    tracking_tlwhs: List[np.ndarray] = []
    reid_boxes: List[Optional[np.ndarray]] = []
    valid_reid_boxes: List[np.ndarray] = []

    for bbox in boxes:
        tracking_tlwh = _tracking_tlwh_from_bbox(bbox)
        tracking_tlwhs.append(tracking_tlwh)
        if reid_source == "fisheye":
            reid_box = tlwh_to_xyxy(tracking_tlwh)
        else:
            reid_box = _reid_source_xyxy_from_bbox(bbox)
        if reid_box is not None:
            reid_box = clip_xyxy_to_image(reid_box, reid_image.shape)
        reid_boxes.append(reid_box)
        valid_reid_boxes.append(reid_box if reid_box is not None else np.zeros(4, dtype=np.float32))

        # Touch the crop path here so invalid source boxes fail early in debug.
        if reid_box is not None and crop_pad:
            crop_xyxy(reid_image, reid_box, pad=crop_pad)

    if feature_extractor is None:
        if require_features:
            raise ValueError("feature_extractor is required when require_features=True")
        features = np.zeros((len(boxes), int(feature_dim)), dtype=np.float32)
    else:
        features = np.asarray(feature_extractor(reid_image, valid_reid_boxes), dtype=np.float32)
        if features.ndim != 2 or features.shape[0] != len(boxes):
            raise ValueError(
                "feature_extractor must return an N x D matrix matching the number of detections; "
                f"got shape {features.shape}, expected N={len(boxes)}"
            )

    detections: List[GnomonicTrackerDetection] = []
    for idx, bbox in enumerate(boxes):
        metadata = _metadata_from_bbox(bbox)
        metadata["reid_crop_source"] = reid_source
        metadata["reid_crop_bbox_xyxy"] = None if reid_boxes[idx] is None else reid_boxes[idx].tolist()
        detections.append(
            GnomonicTrackerDetection(
                tlwh=np.asarray(tracking_tlwhs[idx], dtype=np.float32),
                confidence=float(bbox.get("confidence", 0.0)),
                feature=np.asarray(features[idx], dtype=np.float32),
                reid_source_bbox_xyxy=reid_boxes[idx],
                metadata=metadata,
            )
        )

    return detections
