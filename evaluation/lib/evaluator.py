"""
Per-image evaluation module for detection metrics.

This module evaluates predicted bounding boxes against ground truth for a single
image at multiple IoU thresholds. Used by metrics_evaluator_runner to aggregate
results across a dataset.

Author: Generated for gnomonic projection pedestrian detection project
Date: 2026-02-12
"""

from typing import Dict, List, Any
from evaluation.lib.metrics import (
    match_predictions_to_ground_truth,
    compute_precision_recall
)


class DetectionEvaluator:
    """
    Evaluator for per-image detection metrics.

    Evaluates predictions against ground truth at multiple IoU thresholds
    (e.g., 0.30, 0.35, ..., 0.95) in a single pass.
    """

    def __init__(self, iou_thresholds: List[float]):
        """
        Initialize evaluator with IoU thresholds.

        Args:
            iou_thresholds: List of IoU thresholds to evaluate at
                           (e.g., [0.30, 0.35, 0.40, ..., 0.95])
        """
        self.iou_thresholds = sorted(iou_thresholds)

    def evaluate_image(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict]
    ) -> Dict[str, Any]:
        """
        Evaluate predictions against ground truth for a single image.

        Args:
            predictions: List of prediction dicts with keys:
                        - center_x, center_y, width, height, angle, confidence
            ground_truth: List of GT dicts with keys:
                         - center_x, center_y, width, height, angle, class_name

        Returns:
            dict: Evaluation results with keys:
                - num_gt: Number of ground truth boxes
                - num_pred: Number of predictions
                - results_per_iou: Dict mapping IoU threshold → result dict
                  Each result dict contains:
                  - tp: Number of true positives
                  - fp: Number of false positives
                  - fn: Number of false negatives
                  - precision: Precision value
                  - recall: Recall value
                  - f1: F1 score
                  - tp_matches: List of (pred_idx, gt_idx, iou) tuples
        """
        num_gt = len(ground_truth)
        num_pred = len(predictions)

        results_per_iou = {}

        # Evaluate at each IoU threshold
        for iou_threshold in self.iou_thresholds:
            # Match predictions to ground truth
            tp_matches, fp_indices, fn_indices = match_predictions_to_ground_truth(
                predictions=predictions,
                ground_truth=ground_truth,
                iou_threshold=iou_threshold
            )

            # Count matches
            tp = len(tp_matches)
            fp = len(fp_indices)
            fn = len(fn_indices)

            # Compute precision, recall, F1
            precision, recall, f1 = compute_precision_recall(tp, fp, fn)

            # Store results for this IoU threshold
            results_per_iou[iou_threshold] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp_matches": tp_matches
            }

        return {
            "num_gt": num_gt,
            "num_pred": num_pred,
            "results_per_iou": results_per_iou
        }

    def evaluate_batch(
        self,
        predictions_batch: List[List[Dict]],
        ground_truth_batch: List[List[Dict]]
    ) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of images.

        Args:
            predictions_batch: List of prediction lists (one per image)
            ground_truth_batch: List of GT lists (one per image)

        Returns:
            List of evaluation result dicts (one per image)
        """
        if len(predictions_batch) != len(ground_truth_batch):
            raise ValueError(
                f"Batch size mismatch: {len(predictions_batch)} predictions "
                f"vs {len(ground_truth_batch)} ground truth"
            )

        results = []
        for preds, gts in zip(predictions_batch, ground_truth_batch):
            result = self.evaluate_image(preds, gts)
            results.append(result)

        return results
