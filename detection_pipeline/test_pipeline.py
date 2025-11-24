"""
Test script for detection pipeline end-to-end.

Tests:
1. Load fisheye image
2. Generate composite image using image_composer
3. Run YOLO detection on composite
4. Save visualization of detections on composite
"""

import sys
from pathlib import Path
import cv2

# Standard Python imports
from config import get_cfg
from pipeline import DetectionPipeline


def main():
    """Run detection pipeline test."""
    print("=" * 80)
    print("DETECTION PIPELINE TEST")
    print("=" * 80)

    # Get config and customize for test
    cfg = get_cfg()
    cfg.INPUT.IMAGE_PATH = str(Path(__file__).parent / "fisheye-sample.png")
    cfg.PROJECTION.PRESET = "yolo_grid"
    cfg.YOLO.MODEL = str(Path(__file__).parent / "models" / "yolov8n.pt")  # Use local model
    cfg.YOLO.DEVICE = None  # Auto-detect (GPU if available, else CPU)
    cfg.OUTPUT.SAVE_DIR = str(Path(__file__).parent / "results")
    cfg.VERBOSE = True

    print(f"\nConfiguration:")
    print(f"  Image: {cfg.INPUT.IMAGE_PATH}")
    print(f"  Preset: {cfg.PROJECTION.PRESET}")
    print(f"  Model: {cfg.YOLO.MODEL}")
    print(f"  Device: {cfg.YOLO.DEVICE} (will auto-detect)")
    print()

    # Run pipeline
    try:
        pipeline = DetectionPipeline(cfg)
        detections, composite, metadata, results_dir, fisheye_bboxes = pipeline.run()

        print("\n" + "=" * 80)
        print("TEST COMPLETE - SUCCESS")
        print("=" * 80)
        print(f"\nResults Summary:")
        print(f"  Detections: {len(detections)}")
        if fisheye_bboxes:
            print(f"  Backprojected bboxes: {len(fisheye_bboxes)}")
        print(f"  Results saved to: {results_dir}")
        print(f"\nFiles created:")
        print(f"  - fisheye-sample.png (original)")
        print(f"  - composite.png (projected composite)")
        print(f"  - detections.png (with bounding boxes on composite)")
        if fisheye_bboxes:
            print(f"  - fisheye_detections.png (backprojected bboxes on fisheye)")
        print(f"  - metadata.txt (configuration and results)")

        return 0

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
