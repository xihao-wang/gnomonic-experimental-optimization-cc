"""
Test script for detection pipeline end-to-end.

Tests:
1. Load fisheye image
2. Generate composite image using image-composer
3. Run YOLO detection on composite
4. Save visualization of detections on composite
"""

import sys
from pathlib import Path
import cv2
import importlib.util

# Helper function to import modules from directories with hyphens in their names
def _import_module_from_path(module_name, filepath):
    """Import a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import config and pipeline from detection-pipeline
_detection_pipeline_dir = Path(__file__).parent
config_mod = _import_module_from_path("pipeline_config", _detection_pipeline_dir / "config.py")
pipeline_mod = _import_module_from_path("detection_pipeline_mod", _detection_pipeline_dir / "pipeline.py")

get_cfg = config_mod.get_cfg
DetectionPipeline = pipeline_mod.DetectionPipeline


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
    cfg.YOLO.DEVICE = "cpu"  # Use CPU for testing
    cfg.OUTPUT.SAVE_COMPOSITE = True
    cfg.OUTPUT.SAVE_COMPOSITE_VIZ = True
    cfg.OUTPUT.SAVE_DIR = str(Path(__file__).parent)
    cfg.VERBOSE = True

    print(f"\nConfiguration:")
    print(f"  Image: {cfg.INPUT.IMAGE_PATH}")
    print(f"  Preset: {cfg.PROJECTION.PRESET}")
    print(f"  Model: {cfg.YOLO.MODEL}")
    print(f"  Device: {cfg.YOLO.DEVICE}")
    print()

    # Run pipeline
    try:
        pipeline = DetectionPipeline(cfg)
        detections, composite, metadata = pipeline.run()

        # Save detection visualization
        print("\n" + "=" * 80)
        print("SAVING RESULTS")
        print("=" * 80)

        # Create detection visualization on composite
        viz_image = composite.copy()
        h, w = viz_image.shape[:2]

        for i, det in enumerate(detections):
            # Convert normalized coords to pixel coords
            cx = int(det['x'] * w)
            cy = int(det['y'] * h)
            bw = int(det['w'] * w)
            bh = int(det['h'] * h)

            x1 = max(0, cx - bw // 2)
            y1 = max(0, cy - bh // 2)
            x2 = min(w, cx + bw // 2)
            y2 = min(h, cy + bh // 2)

            # Draw bounding box (green)
            cv2.rectangle(viz_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.putText(viz_image, label, (x1, max(0, y1 - 5)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Save as detection-composite-fisheye-sample.png
        output_path = Path(__file__).parent / "detection-composite-fisheye-sample.png"
        cv2.imwrite(str(output_path), viz_image)
        print(f"\n✓ Saved detection visualization: {output_path}")
        print(f"  Image size: {viz_image.shape}")
        print(f"  Detections: {len(detections)}")

        print("\n" + "=" * 80)
        print("TEST COMPLETE - SUCCESS")
        print("=" * 80)
        return 0

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
