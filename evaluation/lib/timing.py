"""
Pipeline timing utilities for measuring execution time of detection stages.

This module provides utilities to measure the execution time of different stages
in the detection pipeline without modifying the core pipeline code.

Usage:
    from evaluation.lib.timing import PipelineTimer

    timer = PipelineTimer()

    with timer.time_composite_generation():
        # Generate composite image
        pass

    with timer.time_yolo_detection():
        # Run YOLO detection
        pass

    # Get statistics
    stats = timer.get_all_stats()
    print(f"Composite generation: {stats['composite_generation']['mean']:.3f}s")
"""

import time
from contextlib import contextmanager
from typing import Dict, Any, Optional
import numpy as np


class PipelineTimer:
    """
    Timer for measuring execution time of detection pipeline stages.

    Tracks timing for 5 stages:
    1. composite_generation - Generate composite image from fisheye
    2. yolo_detection - Run YOLO on composite
    3. backprojection - Backproject detections to fisheye
    4. soft_nms - Apply Soft-NMS Stage 2
    5. total_end_to_end - Full pipeline execution

    Accumulates measurements across multiple runs and provides statistics.
    """

    def __init__(self):
        """Initialize timer with empty stage measurements."""
        self.stage_times = {
            "composite_generation": [],
            "yolo_detection": [],
            "backprojection": [],
            "soft_nms": [],
            "total_end_to_end": []
        }

    @contextmanager
    def time_stage(self, stage_name: str):
        """
        Context manager to time a specific stage.

        Args:
            stage_name: Name of the stage to time

        Yields:
            None

        Example:
            with timer.time_stage("yolo_detection"):
                detections = detector.detect(image)
        """
        if stage_name not in self.stage_times:
            raise ValueError(f"Unknown stage: {stage_name}")

        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self.stage_times[stage_name].append(elapsed)

    @contextmanager
    def time_composite_generation(self):
        """Context manager to time composite generation."""
        with self.time_stage("composite_generation"):
            yield

    @contextmanager
    def time_yolo_detection(self):
        """Context manager to time YOLO detection."""
        with self.time_stage("yolo_detection"):
            yield

    @contextmanager
    def time_backprojection(self):
        """Context manager to time backprojection."""
        with self.time_stage("backprojection"):
            yield

    @contextmanager
    def time_soft_nms(self):
        """Context manager to time Soft-NMS."""
        with self.time_stage("soft_nms"):
            yield

    @contextmanager
    def time_total_end_to_end(self):
        """Context manager to time full end-to-end pipeline."""
        with self.time_stage("total_end_to_end"):
            yield

    def get_stage_stats(self, stage_name: str) -> Dict[str, float]:
        """
        Get statistics for a specific stage.

        Args:
            stage_name: Name of the stage

        Returns:
            dict: Statistics with keys: mean, std, min, max, total, count
        """
        times = self.stage_times[stage_name]

        if len(times) == 0:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "total": 0.0,
                "count": 0
            }

        return {
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "min": float(np.min(times)),
            "max": float(np.max(times)),
            "total": float(np.sum(times)),
            "count": len(times)
        }

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for all stages.

        Returns:
            dict: Mapping from stage name to statistics dict
        """
        return {
            stage: self.get_stage_stats(stage)
            for stage in self.stage_times.keys()
        }

    def reset(self):
        """Reset all timing measurements."""
        for stage in self.stage_times:
            self.stage_times[stage] = []

    def __repr__(self):
        """Return string representation showing all stage statistics."""
        lines = ["PipelineTimer:"]
        stats = self.get_all_stats()

        for stage, stage_stats in stats.items():
            if stage_stats["count"] > 0:
                lines.append(
                    f"  {stage}: {stage_stats['mean']:.3f}s "
                    f"(±{stage_stats['std']:.3f}s, n={stage_stats['count']})"
                )
            else:
                lines.append(f"  {stage}: no measurements")

        return "\n".join(lines)


# ============================================================================
# Context managers for use without PipelineTimer instance
# ============================================================================

@contextmanager
def time_composite_generation(timer: Optional[PipelineTimer]):
    """
    Context manager to time composite generation (works with optional timer).

    Args:
        timer: PipelineTimer instance or None. If None, does nothing.

    Yields:
        None

    Example:
        with time_composite_generation(timer):
            composite = generate_composite(fisheye)
    """
    if timer is not None:
        with timer.time_composite_generation():
            yield
    else:
        yield


@contextmanager
def time_yolo_detection(timer: Optional[PipelineTimer]):
    """Context manager to time YOLO detection (works with optional timer)."""
    if timer is not None:
        with timer.time_yolo_detection():
            yield
    else:
        yield


@contextmanager
def time_backprojection(timer: Optional[PipelineTimer]):
    """Context manager to time backprojection (works with optional timer)."""
    if timer is not None:
        with timer.time_backprojection():
            yield
    else:
        yield


@contextmanager
def time_soft_nms(timer: Optional[PipelineTimer]):
    """Context manager to time Soft-NMS (works with optional timer)."""
    if timer is not None:
        with timer.time_soft_nms():
            yield
    else:
        yield


@contextmanager
def time_total_end_to_end(timer: Optional[PipelineTimer]):
    """Context manager to time total end-to-end (works with optional timer)."""
    if timer is not None:
        with timer.time_total_end_to_end():
            yield
    else:
        yield
