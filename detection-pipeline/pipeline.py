"""
Detection pipeline orchestrator.

Coordinates the full workflow:
1. Load fisheye image from config
2. Create composite image using image-composer
3. Run YOLO detection on composite
4. Return detections in composite coordinate space

For backprojection to fisheye, use the metadata returned by image-composer.
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import importlib.util

# Helper function to import modules from directories with hyphens in their names
def _import_module_from_path(module_name, filepath, add_to_path=None):
    """Import a module from a file path."""
    if add_to_path and str(add_to_path) not in sys.path:
        sys.path.insert(0, str(add_to_path))
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import from image-composer (has hyphen in directory name)
_img_composer_dir = Path(__file__).parent.parent / "image-composer"
# Add image-composer to path so multi_persp.py can import presets and config
if str(_img_composer_dir) not in sys.path:
    sys.path.insert(0, str(_img_composer_dir))

multi_persp = _import_module_from_path("multi_persp", _img_composer_dir / "multi-persp.py", add_to_path=_img_composer_dir)
presets = _import_module_from_path("presets", _img_composer_dir / "presets.py", add_to_path=_img_composer_dir)

# Import from detection-pipeline (has hyphen in directory name)
_detection_pipeline_dir = Path(__file__).parent
config_mod = _import_module_from_path("pipeline_config", _detection_pipeline_dir / "config.py")
yolo_detector_mod = _import_module_from_path("yolo_detector", _detection_pipeline_dir / "yolo_detector.py")

# Alias the imports for convenience
generate_composite_from_config = multi_persp.generate_composite_from_config
get_preset = presets.get_preset
get_cfg = config_mod.get_cfg
get_cfg_as_dict = config_mod.get_cfg_as_dict
YOLODetector = yolo_detector_mod.YOLODetector


class DetectionPipeline:
    """
    Orchestrates the detection pipeline: fisheye -> composite -> YOLO detection.
    """

    def __init__(self, cfg=None):
        """
        Initialize detection pipeline.

        Args:
            cfg: YACS config object. If None, uses default config.
        """
        if cfg is None:
            cfg = get_cfg()

        self.cfg = cfg
        self.detector = None
        self._init_detector()

    def _init_detector(self):
        """Initialize YOLO detector from config."""
        self.detector = YOLODetector(
            model_name=self.cfg.YOLO.MODEL,
            device=self.cfg.YOLO.DEVICE,
            confidence_threshold=self.cfg.YOLO.CONFIDENCE_THRESHOLD,
            iou_threshold=self.cfg.YOLO.IOU_THRESHOLD,
            max_detections=self.cfg.YOLO.MAX_DETECTIONS
        )

        if self.cfg.VERBOSE:
            print(f"Initialized {self.detector}")

    def _build_projection_config(self):
        """
        Build projection configuration from YACS config.

        Returns:
            dict: Configuration for image-composer with keys:
                  img_path, proj_nbr, fov_h, fov_v, latitude, lon_0, lon_step,
                  grid, comp_sz, target_mp
        """
        # Check if using preset
        if self.cfg.PROJECTION.PRESET:
            proj_cfg = get_preset(self.cfg.PROJECTION.PRESET)
            if proj_cfg is None:
                raise ValueError(f"Preset '{self.cfg.PROJECTION.PRESET}' not found")

            # Override image path if custom image specified
            if self.cfg.PROJECTION.CUSTOM_IMAGE:
                proj_cfg['img_path'] = self.cfg.PROJECTION.CUSTOM_IMAGE
            else:
                proj_cfg['img_path'] = self.cfg.INPUT.IMAGE_PATH

            if self.cfg.VERBOSE:
                print(f"Using preset: {self.cfg.PROJECTION.PRESET}")
        else:
            # Build custom configuration
            proj_cfg = {
                'img_path': self.cfg.INPUT.IMAGE_PATH,
                'proj_nbr': self.cfg.PROJECTION.PROJ_NBR,
                'fov_h': self.cfg.PROJECTION.FOV_H,
                'fov_v': self.cfg.PROJECTION.FOV_V,
                'latitude': self.cfg.PROJECTION.LATITUDE,
                'lon_0': self.cfg.PROJECTION.LON_0,
                'lon_step': self.cfg.PROJECTION.LON_STEP,
                'grid': self.cfg.PROJECTION.GRID,
                'comp_sz': self.cfg.PROJECTION.COMP_SIZE,
                'target_mp': self.cfg.PROJECTION.TARGET_MP,
            }

            if self.cfg.VERBOSE:
                print("Using custom projection configuration")

        return proj_cfg

    def run(self):
        """
        Execute the full detection pipeline.

        Returns:
            tuple: (detections, composite_image, metadata) where:
                - detections: list of detection dicts in composite coordinate space
                - composite_image: numpy array of composite image (H, W, 3)
                - metadata: dict with projection info for backprojection
        """
        if self.cfg.VERBOSE:
            print("=" * 80)
            print("DETECTION PIPELINE START")
            print("=" * 80)

        # Step 1: Build projection configuration
        if self.cfg.VERBOSE:
            print("\n[1] Building projection configuration...")
        proj_cfg = self._build_projection_config()

        # Step 2: Create composite image
        if self.cfg.VERBOSE:
            print("[2] Generating composite image from fisheye...")
            print(f"    Image: {proj_cfg['img_path']}")
            print(f"    Projections: {proj_cfg['proj_nbr']}, Grid: {proj_cfg['grid']}")
            print(f"    Composite size: {proj_cfg['comp_sz']}")

        try:
            composite_image, metadata = generate_composite_from_config(proj_cfg)
        except Exception as e:
            print(f"ERROR: Failed to generate composite image: {e}")
            raise

        if self.cfg.VERBOSE:
            print(f"    ✓ Generated composite of size {composite_image.shape}")

        # Optionally save composite image
        if self.cfg.OUTPUT.SAVE_COMPOSITE:
            save_path = Path(self.cfg.OUTPUT.SAVE_DIR) / "composite.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), composite_image)
            if self.cfg.VERBOSE:
                print(f"    Saved composite to: {save_path}")

        # Step 3: Run YOLO detection
        if self.cfg.VERBOSE:
            print("[3] Running YOLO detection on composite...")
            print(f"    Model: {self.cfg.YOLO.MODEL}")
            print(f"    Device: {self.cfg.YOLO.DEVICE}")
            print(f"    Confidence threshold: {self.cfg.YOLO.CONFIDENCE_THRESHOLD}")
            print(f"    IoU threshold: {self.cfg.YOLO.IOU_THRESHOLD}")

        detections = self.detector.detect(composite_image, class_filter="person")

        if self.cfg.VERBOSE:
            print(f"    ✓ Found {len(detections)} detections")
            for i, det in enumerate(detections):
                print(f"      [{i}] {det['class_name']} at ({det['x']:.3f}, {det['y']:.3f}), "
                      f"conf={det['confidence']:.3f}")

        # Optionally save detection visualization on composite
        if self.cfg.OUTPUT.SAVE_COMPOSITE_VIZ:
            self._save_composite_viz(composite_image, detections)

        if self.cfg.VERBOSE:
            print("=" * 80)
            print("DETECTION PIPELINE COMPLETE")
            print("=" * 80)

        return detections, composite_image, metadata

    def _save_composite_viz(self, composite_image, detections):
        """
        Save visualization of detections on composite image.

        Args:
            composite_image: numpy array (H, W, 3)
            detections: list of detection dicts
        """
        viz_image = composite_image.copy()
        h, w = viz_image.shape[:2]

        for det in detections:
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
            cv2.putText(viz_image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        save_path = Path(self.cfg.OUTPUT.SAVE_DIR) / "composite_detections.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), viz_image)

        if self.cfg.VERBOSE:
            print(f"    Saved composite visualization to: {save_path}")


if __name__ == "__main__":
    """Run pipeline with default config."""
    cfg = get_cfg()

    # Optionally override config here
    # cfg.INPUT.IMAGE_PATH = "path/to/fisheye.png"
    # cfg.YOLO.DEVICE = "cuda"

    cfg.VERBOSE = True

    pipeline = DetectionPipeline(cfg)
    detections, composite, metadata = pipeline.run()

    print(f"\nFound {len(detections)} pedestrians")
    print(f"Composite size: {composite.shape}")
