"""
YOLO detector wrapper for pedestrian detection.

This module wraps ultralytics YOLO to provide:
- Model loading with auto-download (works with any YOLO version: v8, v11, etc.)
- Inference on images
- Configurable detection parameters (confidence, IOU threshold, etc.)
"""

import numpy as np
from ultralytics import YOLO
import torch


class YOLODetector:
    """
    Generic YOLO detector for pedestrian detection.

    Works with any YOLO version supported by ultralytics (v8, v11, etc).
    The model version is determined by the model_name parameter.
    """

    def __init__(self, model_name="yolov8n.pt", device=None, confidence_threshold=0.05,
                 iou_threshold=0.8, max_detections=300):
        """
        Initialize YOLO detector.

        Args:
            model_name (str): YOLO model name (e.g., "yolov8n.pt", "yolov11n.pt", etc.)
                             Model will be auto-downloaded if not found.
                             Format: yolo{version}{size}.pt where:
                             - version: 8, 11, etc.
                             - size: n (nano), s (small), m (medium), l (large), x (extra-large)
            device (str or None): Device to run on: "cuda", "cpu", or None for auto-detect
                                 If None, uses GPU if available, else CPU
            confidence_threshold (float): Confidence threshold for detections (0-1)
                                         Low value (0.05) to catch all potential persons before NMS
            iou_threshold (float): IoU threshold for Stage 1 NMS (0-1)
                                  High value (0.8) keeps more boxes since Stage 2 NMS follows
            max_detections (int): Maximum number of detections to keep
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections

        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model (ultralytics handles any YOLO version)
        self.model = YOLO(model_name)
        self.model.to(self.device)

    def detect(self, image, class_filter=None):
        """
        Run YOLO detection on image.

        Args:
            image (np.ndarray): Input image (H, W, 3) in BGR format
            class_filter (str or list, optional): Filter detections to specific class(es).
                                                 If None, return all detections.
                                                 If str, return only that class (e.g., "person")
                                                 If list, return detections matching any class in the list

        Returns:
            list: List of detections, each with keys:
                {
                    'x': center x coordinate (0-1 normalized),
                    'y': center y coordinate (0-1 normalized),
                    'w': box width (0-1 normalized),
                    'h': box height (0-1 normalized),
                    'confidence': detection confidence (0-1),
                    'class_id': class ID,
                    'class_name': class name string
                }
        """
        # Normalize class_filter to list
        if class_filter is None:
            allowed_classes = None
        elif isinstance(class_filter, str):
            allowed_classes = [class_filter]
        elif isinstance(class_filter, list):
            allowed_classes = class_filter
        else:
            allowed_classes = None

        # Run inference
        results = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            max_det=self.max_detections,
            verbose=False
        )

        # Extract detections from results
        detections = []
        if len(results) > 0:
            result = results[0]  # Single image

            if result.boxes is not None:
                # Get image dimensions
                h, w = image.shape[:2]

                # Process each detection
                for box in result.boxes:
                    # Get confidence and class
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = result.names[class_id] if hasattr(result, 'names') else f"class_{class_id}"

                    # Filter by class if specified
                    if allowed_classes is not None and class_name not in allowed_classes:
                        continue

                    # Get box coordinates (x1, y1, x2, y2) in pixel space
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                    # Convert to center + size format, normalized (0-1)
                    center_x = ((x1 + x2) / 2) / w
                    center_y = ((y1 + y2) / 2) / h
                    box_w = (x2 - x1) / w
                    box_h = (y2 - y1) / h

                    detection = {
                        'x': center_x,
                        'y': center_y,
                        'w': box_w,
                        'h': box_h,
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name
                    }
                    detections.append(detection)

        return detections

    def __repr__(self):
        """String representation."""
        return (f"YOLODetector(model={self.model_name}, device={self.device}, "
                f"conf_thresh={self.confidence_threshold}, iou_thresh={self.iou_threshold})")
