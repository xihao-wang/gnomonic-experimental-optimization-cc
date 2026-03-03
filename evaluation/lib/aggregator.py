"""
Results aggregation module for computing dataset-level metrics.

This module aggregates per-image evaluation results to compute dataset-level
metrics including Average Precision (AP) using 11-point interpolation.

Author: Generated for gnomonic projection pedestrian detection project
Date: 2026-02-12
"""

from typing import Dict, List, Any, Tuple
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt


class ResultsAggregator:
    """
    Aggregates per-image results to compute dataset-level metrics.

    Computes:
    - Precision, Recall, F1 at each IoU threshold
    - Average Precision (AP) at each IoU threshold using 11-point interpolation
    - Summary metrics: AP@[0.30:0.95], AP@0.50, AP@0.75
    - Precision-Recall curve data for plotting
    """

    def __init__(self, iou_thresholds: List[float], enable_pr_curves: bool = True):
        """
        Initialize aggregator.

        Args:
            iou_thresholds: List of IoU thresholds to aggregate
            enable_pr_curves: Whether to generate PR curve data and plots
        """
        self.iou_thresholds = sorted(iou_thresholds)
        self.enable_pr_curves = enable_pr_curves

    def aggregate(
        self,
        per_image_results: List[Dict[str, Any]],
        predictions_with_confidence: List[List[Dict]]
    ) -> Dict[str, Any]:
        """
        Aggregate per-image results to dataset-level metrics.

        Args:
            per_image_results: List of per-image evaluation results from DetectionEvaluator
            predictions_with_confidence: List of predictions (one list per image)
                                        Each prediction must have 'confidence' field

        Returns:
            dict: Aggregated results with keys:
                - metrics_per_iou: Dict mapping IoU → {precision, recall, f1, ap, tp, fp, fn}
                - summary: Summary metrics (ap, ap50, ap75, precision@0.5, recall@0.5, f1@0.5)
                - pr_curves_data: PR curve data for each IoU (if enable_pr_curves=True)
        """
        metrics_per_iou = {}
        pr_curves_data = {}

        # Aggregate for each IoU threshold
        for iou_threshold in self.iou_thresholds:
            # Aggregate TP, FP, FN across all images
            total_tp = 0
            total_fp = 0
            total_fn = 0

            for result in per_image_results:
                iou_result = result["results_per_iou"][iou_threshold]
                total_tp += iou_result["tp"]
                total_fp += iou_result["fp"]
                total_fn += iou_result["fn"]

            # Compute precision, recall, F1 at dataset level
            if total_tp + total_fp > 0:
                precision = total_tp / (total_tp + total_fp)
            else:
                precision = 0.0

            if total_tp + total_fn > 0:
                recall = total_tp / (total_tp + total_fn)
            else:
                recall = 0.0

            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0

            # Compute Average Precision (AP) using 11-point interpolation
            ap, pr_curve_data = self._compute_ap_single_class(
                per_image_results=per_image_results,
                predictions_with_confidence=predictions_with_confidence,
                iou_threshold=iou_threshold
            )

            metrics_per_iou[iou_threshold] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "ap": ap,
                "tp": total_tp,
                "fp": total_fp,
                "fn": total_fn
            }

            if self.enable_pr_curves:
                # Add AP to pr_curve_data for plotting
                pr_curve_data['ap'] = ap
                pr_curves_data[iou_threshold] = pr_curve_data

        # Compute summary metrics
        summary = self._compute_summary(metrics_per_iou)

        result = {
            "metrics_per_iou": metrics_per_iou,
            "summary": summary
        }

        if self.enable_pr_curves:
            result["pr_curves_data"] = pr_curves_data

        return result

    def _compute_ap_single_class(
        self,
        per_image_results: List[Dict[str, Any]],
        predictions_with_confidence: List[List[Dict]],
        iou_threshold: float
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute Average Precision for single class using 11-point interpolation.

        PASCAL VOC style AP:
        1. Collect all predictions across images with confidence scores
        2. Sort by confidence (descending)
        3. Compute precision and recall at each prediction
        4. Apply 11-point interpolation (recall levels: 0, 0.1, 0.2, ..., 1.0)
        5. AP = mean of max precision at each recall level

        Args:
            per_image_results: Per-image evaluation results
            predictions_with_confidence: All predictions with confidence scores
            iou_threshold: IoU threshold to use for matching

        Returns:
            Tuple of (ap, pr_curve_data):
                - ap: Average Precision value
                - pr_curve_data: Dict with 'precision', 'recall', 'confidence' arrays
        """
        # Collect all predictions with their match status
        all_detections = []  # List of (confidence, is_true_positive, image_idx, pred_idx)

        for img_idx, (result, preds) in enumerate(zip(per_image_results, predictions_with_confidence)):
            iou_result = result["results_per_iou"][iou_threshold]
            tp_matches = iou_result["tp_matches"]  # List of (pred_idx, gt_idx, iou)
            fp_indices = iou_result["fp"]  # List of pred_idx

            # Mark true positives
            tp_pred_indices = {match[0] for match in tp_matches}

            for pred_idx, pred in enumerate(preds):
                confidence = pred.get("confidence", 0.0)
                is_tp = pred_idx in tp_pred_indices

                all_detections.append({
                    "confidence": confidence,
                    "is_tp": is_tp,
                    "image_idx": img_idx,
                    "pred_idx": pred_idx
                })

        # Sort by confidence (descending)
        all_detections.sort(key=lambda x: x["confidence"], reverse=True)

        # Count total ground truth boxes
        total_gt = sum(result["num_gt"] for result in per_image_results)

        if total_gt == 0:
            # No ground truth, AP is undefined (return 0)
            return 0.0, {"precision": [], "recall": [], "confidence": []}

        if len(all_detections) == 0:
            # No predictions, AP is 0
            return 0.0, {"precision": [0.0], "recall": [0.0], "confidence": [0.0]}

        # Compute precision and recall at each detection
        precision_curve = []
        recall_curve = []
        confidence_curve = []

        tp_cumsum = 0
        fp_cumsum = 0

        for det in all_detections:
            if det["is_tp"]:
                tp_cumsum += 1
            else:
                fp_cumsum += 1

            precision = tp_cumsum / (tp_cumsum + fp_cumsum)
            recall = tp_cumsum / total_gt

            precision_curve.append(precision)
            recall_curve.append(recall)
            confidence_curve.append(det["confidence"])

        # Convert to numpy arrays
        precision_curve = np.array(precision_curve)
        recall_curve = np.array(recall_curve)
        confidence_curve = np.array(confidence_curve)

        # Apply 11-point interpolation
        recall_levels = np.linspace(0, 1, 11)  # [0.0, 0.1, 0.2, ..., 1.0]
        interpolated_precisions = []

        for r_level in recall_levels:
            # Find all precisions at recall >= r_level
            precisions_at_recall = precision_curve[recall_curve >= r_level]

            if len(precisions_at_recall) > 0:
                # Max precision at this recall level
                interpolated_precisions.append(np.max(precisions_at_recall))
            else:
                # No predictions at this recall level
                interpolated_precisions.append(0.0)

        # AP = mean of interpolated precisions
        ap = np.mean(interpolated_precisions)

        # PR curve data for plotting
        pr_curve_data = {
            "precision": precision_curve.tolist(),
            "recall": recall_curve.tolist(),
            "confidence": confidence_curve.tolist(),
            "interpolated_recall": recall_levels.tolist(),
            "interpolated_precision": interpolated_precisions
        }

        return float(ap), pr_curve_data

    def _compute_summary(self, metrics_per_iou: Dict[float, Dict]) -> Dict[str, float]:
        """
        Compute summary metrics from per-IoU metrics.

        Summary metrics:
        - ap: Mean AP over all IoU thresholds (AP@[0.30:0.95])
        - ap50: AP at IoU=0.50
        - ap75: AP at IoU=0.75
        - precision@0.5: Precision at IoU=0.50
        - recall@0.5: Recall at IoU=0.50
        - f1@0.5: F1 score at IoU=0.50

        Args:
            metrics_per_iou: Dict mapping IoU threshold → metrics dict

        Returns:
            Summary metrics dict
        """
        # Compute mean AP over all IoU thresholds
        ap_values = [metrics["ap"] for metrics in metrics_per_iou.values()]
        mean_ap = np.mean(ap_values) if len(ap_values) > 0 else 0.0

        # Get AP at specific IoU thresholds
        ap50 = metrics_per_iou.get(0.50, {}).get("ap", 0.0)
        ap75 = metrics_per_iou.get(0.75, {}).get("ap", 0.0)

        # Get precision, recall, F1 at IoU=0.50
        metrics_at_50 = metrics_per_iou.get(0.50, {})
        precision_at_50 = metrics_at_50.get("precision", 0.0)
        recall_at_50 = metrics_at_50.get("recall", 0.0)
        f1_at_50 = metrics_at_50.get("f1", 0.0)

        return {
            "ap": float(mean_ap),
            "ap50": float(ap50),
            "ap75": float(ap75),
            "precision@0.5": float(precision_at_50),
            "recall@0.5": float(recall_at_50),
            "f1@0.5": float(f1_at_50)
        }

    def generate_pr_curve_plot(
        self,
        pr_curve_data: Dict[str, Any],
        iou_threshold: float,
        output_path: Path,
        config_name: str = ""
    ):
        """
        Generate and save precision-recall curve plot as PNG.

        Args:
            pr_curve_data: PR curve data from _compute_ap_single_class
            iou_threshold: IoU threshold used for this curve
            output_path: Path to save PNG file
            config_name: Configuration name for plot title
        """
        precision = pr_curve_data["precision"]
        recall = pr_curve_data["recall"]

        if len(precision) == 0 or len(recall) == 0:
            print(f"Warning: No PR curve data for IoU={iou_threshold}, skipping plot")
            return

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot PR curve
        ax.plot(recall, precision, linewidth=2, color='blue', label='PR Curve')

        # Plot 11-point interpolated curve (if available)
        if "interpolated_recall" in pr_curve_data and "interpolated_precision" in pr_curve_data:
            interp_recall = pr_curve_data["interpolated_recall"]
            interp_precision = pr_curve_data["interpolated_precision"]
            ax.plot(
                interp_recall, interp_precision,
                'ro-', linewidth=1, markersize=6,
                label='11-point Interpolation'
            )

        # Configure plot
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)

        # Title with config name and IoU threshold
        title = f"Precision-Recall Curve"
        if config_name:
            title += f"\n{config_name}"
        title += f"\nIoU Threshold: {iou_threshold:.2f}"

        # Add AP to title
        if "ap" in pr_curve_data:
            title += f" | AP: {pr_curve_data['ap']:.3f}"

        ax.set_title(title, fontsize=14)

        # Save figure
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"Saved PR curve: {output_path}")
