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
import shutil
from datetime import datetime

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
            tuple: (detections, composite_image, metadata, results_dir) where:
                - detections: list of detection dicts in composite coordinate space
                - composite_image: numpy array of composite image (H, W, 3)
                - metadata: dict with projection info for backprojection
                - results_dir: path to results directory for this run
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

        # Step 3: Run YOLO detection
        if self.cfg.VERBOSE:
            print("[3] Running YOLO detection on composite...")
            print(f"    Model: {self.cfg.YOLO.MODEL}")
            print(f"    Device: {self.detector.device}")
            print(f"    Confidence threshold: {self.cfg.YOLO.CONFIDENCE_THRESHOLD}")
            print(f"    IoU threshold: {self.cfg.YOLO.IOU_THRESHOLD}")

        detections = self.detector.detect(composite_image, class_filter="person")

        if self.cfg.VERBOSE:
            print(f"    ✓ Found {len(detections)} detections")
            for i, det in enumerate(detections):
                print(f"      [{i}] {det['class_name']} at ({det['x']:.3f}, {det['y']:.3f}), "
                      f"conf={det['confidence']:.3f}")

        # Step 4: Save results
        results_dir = self._save_results(proj_cfg, composite_image, detections, metadata)

        if self.cfg.VERBOSE:
            print("=" * 80)
            print("DETECTION PIPELINE COMPLETE")
            print(f"Results saved to: {results_dir}")
            print("=" * 80)

        return detections, composite_image, metadata, results_dir

    def _save_results(self, proj_cfg, composite_image, detections, metadata):
        """
        Save detection results to organized directory structure.

        Creates: results/image_name/timestamp/
        Saves: fisheye_image.png, composite.png, detections.png, metadata.txt

        Args:
            proj_cfg: projection configuration dict
            composite_image: numpy array (H, W, 3)
            detections: list of detection dicts
            metadata: projection metadata dict

        Returns:
            Path to results directory
        """
        # Get input image name and create directory structure
        fisheye_path = Path(proj_cfg['img_path'])
        image_name = fisheye_path.stem  # e.g., "fisheye-sample" from "fisheye-sample.png"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        results_base = Path(self.cfg.OUTPUT.SAVE_DIR)
        results_dir = results_base / image_name / timestamp
        results_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Copy original fisheye image
        fisheye_out = results_dir / f"{image_name}.png"
        shutil.copy(str(fisheye_path), str(fisheye_out))

        # Step 2: Save composite image
        composite_out = results_dir / "composite.png"
        cv2.imwrite(str(composite_out), composite_image)

        # Step 3: Create and save detection visualization
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

        detections_out = results_dir / "detections.png"
        cv2.imwrite(str(detections_out), viz_image)

        # Step 4: Save metadata to text file
        metadata_out = results_dir / "metadata.txt"
        with open(str(metadata_out), 'w') as f:
            f.write("Detection Pipeline Results\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Input Image: {fisheye_path.name}\n\n")

            f.write("Projection Configuration:\n")
            f.write(f"  Preset: {self.cfg.PROJECTION.PRESET}\n")
            f.write(f"  Projections: {proj_cfg['proj_nbr']}\n")
            f.write(f"  Grid Layout: {proj_cfg['grid']}\n")
            f.write(f"  FOV H: {proj_cfg['fov_h']}°\n")
            f.write(f"  FOV V: {proj_cfg['fov_v']}°\n")
            f.write(f"  Latitude: {proj_cfg['latitude']}°\n")
            f.write(f"  Composite Size: {proj_cfg['comp_sz']}\n\n")

            f.write("YOLO Configuration:\n")
            f.write(f"  Model: {self.cfg.YOLO.MODEL}\n")
            f.write(f"  Device: {self.detector.device}\n")
            f.write(f"  Confidence Threshold: {self.cfg.YOLO.CONFIDENCE_THRESHOLD}\n")
            f.write(f"  IoU Threshold: {self.cfg.YOLO.IOU_THRESHOLD}\n\n")

            f.write("Detection Results:\n")
            f.write(f"  Total Pedestrians Detected: {len(detections)}\n\n")

            if len(detections) > 0:
                f.write("Detections:\n")
                for i, det in enumerate(detections, 1):
                    f.write(f"  [{i}] {det['class_name']} at ({det['x']:.3f}, {det['y']:.3f}), "
                           f"confidence={det['confidence']:.3f}\n")

        if self.cfg.VERBOSE:
            print(f"\n[4] Saving results...")
            print(f"    Fisheye image: {fisheye_out}")
            print(f"    Composite image: {composite_out}")
            print(f"    Detections image: {detections_out}")
            print(f"    Metadata: {metadata_out}")

        return results_dir


if __name__ == "__main__":
    """Run pipeline with default config."""
    cfg = get_cfg()

    # Optionally override config here
    # cfg.INPUT.IMAGE_PATH = "path/to/fisheye.png"
    # cfg.YOLO.DEVICE = "cuda"  # or None for auto-detect

    cfg.VERBOSE = True

    pipeline = DetectionPipeline(cfg)
    detections, composite, metadata, results_dir = pipeline.run()

    print(f"\nFound {len(detections)} pedestrians")
    print(f"Composite size: {composite.shape}")
    print(f"Results saved to: {results_dir}")
