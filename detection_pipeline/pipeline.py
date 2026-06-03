"""
Detection pipeline orchestrator.

Coordinates the full workflow:
1. Load fisheye image from config
2. Create composite image using image_composer
3. Run YOLO detection on composite
4. Return detections in composite coordinate space

For backprojection to fisheye, use the metadata returned by image_composer.
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import shutil
from datetime import datetime

# Add parent directory to path to allow imports from image_composer
_parent_dir = Path(__file__).parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

# Standard Python imports from image_composer
from image_composer.multi_persp import (
    generate_composite_from_config,
    draw_fov_on_fisheye,
    generate_rainbow_colors,
)
from image_composer.presets import get_preset  # presets.py imports from multi_persp

# Standard Python imports from detection_pipeline modules
from detection_pipeline.config import get_cfg, get_cfg_as_dict
from detection_pipeline.yolo_detector import YOLODetector
from detection_pipeline.backprojection import (
    backproject_detections, visualize_backprojection, visualize_bbox_lattice,
    draw_rotated_bbox,
)
from detection_pipeline.nms import apply_stage1_nms, apply_stage2_nms
from detection_pipeline.redundant_bbox_filter import (
    build_active_borders,
    flag_border_aligned_candidates,
    drop_confirmed_redundant_bboxes,
    visualize_composite_flags,
    visualize_detections_on_grid,
)


class DetectionPipeline:
    """
    Orchestrates the detection pipeline: fisheye -> composite -> YOLO detection.
    """

    def __init__(self, cfg=None, model_path=None, conf_threshold=None, imgsz=None):
        """
        Initialize detection pipeline.

        Args:
            cfg: YACS config object. If None, uses default config.
            model_path: Optional path to YOLO model (overrides cfg if provided)
            conf_threshold: Optional confidence threshold (overrides cfg if provided)
            imgsz: Optional YOLO inference resolution (overrides YOLODetector default if provided)
        """
        if cfg is None:
            cfg = get_cfg()

        self.cfg = cfg
        self.imgsz = imgsz  # None → YOLODetector uses its own default (640)

        # Override config with direct parameters if provided
        if model_path is not None:
            self.cfg.YOLO.MODEL = model_path
        if conf_threshold is not None:
            self.cfg.YOLO.CONFIDENCE_THRESHOLD = conf_threshold

        self.detector = None
        self._init_detector()

    def _init_detector(self):
        """Initialize YOLO detector from config."""
        # Use Stage 1 NMS threshold if enabled, otherwise use a default
        stage1_iou = self.cfg.NMS.STAGE1.IOU_THRESHOLD if self.cfg.NMS.STAGE1.ENABLED else 0.45

        detector_kwargs = dict(
            model_name=self.cfg.YOLO.MODEL,
            device=self.cfg.YOLO.DEVICE,
            confidence_threshold=self.cfg.YOLO.CONFIDENCE_THRESHOLD,
            iou_threshold=stage1_iou,
            max_detections=self.cfg.YOLO.MAX_DETECTIONS,
        )
        if self.imgsz is not None:
            detector_kwargs["imgsz"] = self.imgsz
        self.detector = YOLODetector(**detector_kwargs)

        if self.cfg.VERBOSE:
            print(f"Initialized {self.detector}")
            if self.cfg.NMS.STAGE1.ENABLED:
                print(f"Stage 1 NMS: Enabled (IoU threshold: {self.cfg.NMS.STAGE1.IOU_THRESHOLD})")
            else:
                print(f"Stage 1 NMS: Disabled")
            if self.cfg.NMS.STAGE2.ENABLED:
                print(f"Stage 2 Soft-NMS: Enabled (sigma: {self.cfg.NMS.STAGE2.SIGMA}, "
                      f"score threshold: {self.cfg.NMS.STAGE2.SCORE_THRESHOLD})")
            else:
                print(f"Stage 2 Soft-NMS: Disabled")

    def _build_projection_config(self):
        """
        Build projection configuration from YACS config.

        Returns:
            dict: Configuration for image_composer with keys:
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

    def run(self, fisheye_image=None, projection_config=None, timer=None, return_visuals=False):
        """
        Execute the full detection pipeline.

        Args:
            fisheye_image: Optional numpy array of fisheye image (H, W, 3).
                          If None, loads from config.
            projection_config: Optional projection configuration dict.
                             If None, builds from config.
            timer: Optional PipelineTimer for measuring stage execution times.
                  If None, no timing is performed.
            return_visuals: When True (metrics mode only), also return the composite
                           image and raw YOLO detections for visualization.

        Returns:
            If fisheye_image and projection_config are provided (metrics mode):
                return_visuals=False: list of converted fisheye bboxes
                return_visuals=True:  (fisheye_bboxes, composite_image, raw_detections)
            Otherwise (normal mode):
                tuple: (detections, composite_image, metadata, results_dir, fisheye_bboxes)
        """
        # Metrics mode: simplified pipeline without saving
        metrics_mode = (fisheye_image is not None and projection_config is not None)

        # Wrap in total timing context if timer provided
        if timer is not None:
            from evaluation.lib.timing import time_total_end_to_end
            timing_context = time_total_end_to_end(timer)
            timing_context.__enter__()
        else:
            timing_context = None
        if self.cfg.VERBOSE and not metrics_mode:
            print("=" * 80)
            print("DETECTION PIPELINE START")
            print("=" * 80)

        # Step 1: Build projection configuration
        if projection_config is not None:
            proj_cfg = projection_config
        else:
            if self.cfg.VERBOSE:
                print("\n[1] Building projection configuration...")
            proj_cfg = self._build_projection_config()

        # Step 2: Create composite image
        if self.cfg.VERBOSE and not metrics_mode:
            print("[2] Generating composite image from fisheye...")
            print(f"    Image: {proj_cfg.get('img_path', 'from memory')}")
            print(f"    Projections: {proj_cfg['proj_nbr']}, Grid: {proj_cfg['grid']}")
            print(f"    Composite size: {proj_cfg['comp_sz']}")

        try:
            # Wrap composite generation in timing if timer provided
            if timer is not None:
                from evaluation.lib.timing import time_composite_generation
                with time_composite_generation(timer):
                    composite_image, metadata = generate_composite_from_config(proj_cfg, fisheye_img=fisheye_image)
            else:
                # No timing
                composite_image, metadata = generate_composite_from_config(proj_cfg, fisheye_img=fisheye_image)
        except Exception as e:
            if timing_context is not None:
                timing_context.__exit__(None, None, None)
            print(f"ERROR: Failed to generate composite image: {e}")
            raise

        if self.cfg.VERBOSE and not metrics_mode:
            print(f"    ✓ Generated composite of size {composite_image.shape}")

        # Step 3: Run YOLO detection
        if self.cfg.VERBOSE and not metrics_mode:
            print("[3] Running YOLO detection on composite...")
            print(f"    Model: {self.cfg.YOLO.MODEL}")
            print(f"    Device: {self.detector.device}")
            print(f"    Confidence threshold: {self.cfg.YOLO.CONFIDENCE_THRESHOLD}")
            print(f"    IoU threshold: {self.cfg.YOLO.IOU_THRESHOLD}")

        # Wrap YOLO detection in timing if timer provided
        if timer is not None:
            from evaluation.lib.timing import time_yolo_detection
            with time_yolo_detection(timer):
                detections = self.detector.detect(composite_image, class_filter="person")
        else:
            detections = self.detector.detect(composite_image, class_filter="person")

        if self.cfg.VERBOSE and not metrics_mode:
            print(f"    ✓ Found {len(detections)} detections (after YOLO internal NMS)")
            if self.cfg.NMS.STAGE1.ENABLED:
                print(f"    (Stage 1 NMS applied by YOLO with IoU threshold: {self.cfg.NMS.STAGE1.IOU_THRESHOLD})")
            for i, det in enumerate(detections):
                print(f"      [{i}] {det['class_name']} at ({det['x']:.3f}, {det['y']:.3f}), "
                      f"conf={det['confidence']:.3f}")

        # Step 3b: Redundant-bbox filter — Stage 1 (composite-side flagging).
        # Tags border-aligned detections. When Stage 2 (confirmation) is
        # disabled, flagged detections are dropped HERE immediately, before
        # backprojection. This ordering matters:
        #
        # The drop MUST happen before Stage-2 Soft-NMS, never after. Soft-NMS
        # sorts by confidence and decays the score of lower-confidence
        # overlapping boxes. If the flagged duplicate happened to have a
        # higher YOLO confidence than the legitimate full-view detection,
        # Soft-NMS would decay the legitimate one below SCORE_THRESHOLD and
        # discard it. A drop-after-Soft-NMS step would then delete the
        # flagged box too — leaving the target with no detection at all
        # (false negative). Dropping before Soft-NMS sidesteps this: the
        # legitimate full-view box enters Soft-NMS without any flagged
        # competitor to bleed its score.
        rb_cfg = self.cfg.REDUNDANT_BBOX_FILTER.BORDER_BASED
        rb_flag_cfg = rb_cfg.FLAGGING
        rb_conf_cfg = rb_cfg.CONFIRMATION
        detections_raw = detections  # snapshot for visualization
        detections_post_flag = None  # set below when rb_active; full list with flag tags
        rb_active = rb_flag_cfg.ENABLED and len(detections) > 0
        dropped_immediate_composite = []
        if rb_active:
            grid_tuple = tuple(proj_cfg["grid"])
            has_extra = bool(proj_cfg.get("extra_projections"))
            active_borders = build_active_borders(
                grid=grid_tuple,
                preset=rb_flag_cfg.PRESET,
                overrides=rb_flag_cfg.OVERRIDES,
                has_extra_projections=has_extra,
            )
            detections = flag_border_aligned_candidates(
                detections=detections,
                composite_shape=composite_image.shape,
                grid=grid_tuple,
                active_borders=active_borders,
                tolerance_px=rb_flag_cfg.TOLERANCE_PX,
            )
            flagged_count = sum(1 for d in detections if "_flagged_border_side" in d)
            if self.cfg.VERBOSE and not metrics_mode:
                drop_mode = ("deferred to Stage-2 confirmation (post-backprojection, pre-Soft-NMS)"
                             if rb_conf_cfg.ENABLED
                             else "immediate on composite (pre-backprojection)")
                print(f"    Redundant-bbox filter — Stage 1 flagging "
                      f"(preset='{rb_flag_cfg.PRESET}'): "
                      f"{flagged_count}/{len(detections)} detections flagged "
                      f"(drop mode: {drop_mode})")
                for d in detections:
                    if "_flagged_border_side" in d:
                        print(f"      FLAG tile={d['_flagged_tile']} side={d['_flagged_border_side']} "
                              f"at ({d['x']:.3f}, {d['y']:.3f}), conf={d['confidence']:.3f}")

            # Snapshot of post-flag, pre-drop detections — used by the
            # composite visualization so the flagged-but-dropped boxes can
            # still be rendered for inspection even when they're about to be
            # removed below.
            detections_post_flag = list(detections)

            # If Stage 2 confirmation is OFF: drop flagged here on composite
            # right away. Backprojection and Soft-NMS will only see the
            # survivors, eliminating the false-negative risk described above.
            if not rb_conf_cfg.ENABLED:
                kept_composite = []
                for d in detections:
                    if "_flagged_border_side" in d:
                        d2 = dict(d)
                        d2["_drop_reason"] = "border_based_unconditional"
                        dropped_immediate_composite.append(d2)
                    else:
                        kept_composite.append(d)
                detections = kept_composite
                if self.cfg.VERBOSE and not metrics_mode:
                    print(f"    Redundant-bbox filter — composite drop (Stage 2 disabled): "
                          f"removed {len(dropped_immediate_composite)} flagged on composite, "
                          f"{len(detections)} survivors will be backprojected.")

        # Step 4: Backproject to fisheye coordinates (always enabled in metrics mode)
        fisheye_bboxes = None
        fisheye_dropped_confirmed = []  # populated by the confirmation pass below
        should_backproject = metrics_mode or (self.cfg.BACKPROJECTION.ENABLED and len(detections) > 0)

        if should_backproject and len(detections) > 0:
            if self.cfg.VERBOSE and not metrics_mode:
                print("[4] Backprojecting detections to fisheye coordinates...")

            # Get fisheye image
            if fisheye_image is not None:
                fisheye_img = fisheye_image
            else:
                fisheye_img = cv2.imread(proj_cfg['img_path'])

            if fisheye_img is None:
                print(f"    WARNING: Could not load fisheye image for backprojection")
            else:
                fisheye_shape = fisheye_img.shape[:2]  # (height, width)

                # Wrap backprojection in timing if timer provided
                if timer is not None:
                    from evaluation.lib.timing import time_backprojection
                    with time_backprojection(timer):
                        fisheye_bboxes = backproject_detections(
                            detections, metadata, fisheye_shape,
                            compute_lattice=self.cfg.OUTPUT.SAVE_LATTICE_VIZ if not metrics_mode else False,
                            lattice_height_samples=self.cfg.BACKPROJECTION.LATTICE_HEIGHT_SAMPLES
                        )
                else:
                    fisheye_bboxes = backproject_detections(
                        detections, metadata, fisheye_shape,
                        compute_lattice=self.cfg.OUTPUT.SAVE_LATTICE_VIZ if not metrics_mode else False,
                        lattice_height_samples=self.cfg.BACKPROJECTION.LATTICE_HEIGHT_SAMPLES
                    )

                if self.cfg.VERBOSE and not metrics_mode:
                    print(f"    ✓ Backprojected {len(fisheye_bboxes)} bboxes to fisheye")
                    for i, bbox in enumerate(fisheye_bboxes):
                        num_lattice = len(bbox.get('lattice_points', []))
                        print(f"      [{i}] {bbox['class_name']} at center ({bbox['center'][0]:.1f}, {bbox['center'][1]:.1f}), "
                              f"angle={bbox['angle']:.1f}°, conf={bbox.get('confidence', 0):.3f}, lattice: {num_lattice} points")

                # Step 4a: Redundant-bbox filter — Stage 2 (confirmation).
                # Runs BEFORE Soft-NMS so the flagged duplicates can't decay
                # the legitimate full-view box's score (see explanation in
                # Step 3b). Only relevant when confirmation is enabled; the
                # "Stage 2 disabled" case already dropped on composite.
                fisheye_dropped_confirmed = []
                if rb_active and rb_conf_cfg.ENABLED and len(fisheye_bboxes) > 0:
                    n_before_conf = len(fisheye_bboxes)
                    kept_confirmed, fisheye_dropped_confirmed = drop_confirmed_redundant_bboxes(
                        fisheye_bboxes,
                        min_area_ratio_to_larger=rb_conf_cfg.MIN_AREA_RATIO_TO_LARGER,
                        min_overlap_ios=rb_conf_cfg.MIN_OVERLAP_IOS,
                    )
                    fisheye_bboxes = kept_confirmed
                    if self.cfg.VERBOSE and not metrics_mode:
                        print(f"    Redundant-bbox filter — Stage 2 confirmation "
                              f"(area_ratio>={rb_conf_cfg.MIN_AREA_RATIO_TO_LARGER}, "
                              f"IoS>={rb_conf_cfg.MIN_OVERLAP_IOS}): "
                              f"{n_before_conf} → {len(fisheye_bboxes)} "
                              f"({len(fisheye_dropped_confirmed)} flagged & confirmed-dropped)")

                # Apply Stage 2 Soft-NMS if enabled
                if self.cfg.NMS.STAGE2.ENABLED and len(fisheye_bboxes) > 0:
                    if self.cfg.VERBOSE and not metrics_mode:
                        print(f"    Applying Stage 2 Soft-NMS (sigma={self.cfg.NMS.STAGE2.SIGMA}, "
                              f"threshold={self.cfg.NMS.STAGE2.SCORE_THRESHOLD})...")

                    fisheye_bboxes_before = len(fisheye_bboxes)

                    # Wrap Soft-NMS in timing if timer provided
                    if timer is not None:
                        from evaluation.lib.timing import time_soft_nms
                        with time_soft_nms(timer):
                            fisheye_bboxes = apply_stage2_nms(
                                fisheye_bboxes,
                                sigma=self.cfg.NMS.STAGE2.SIGMA,
                                score_threshold=self.cfg.NMS.STAGE2.SCORE_THRESHOLD
                            )
                    else:
                        fisheye_bboxes = apply_stage2_nms(
                            fisheye_bboxes,
                            sigma=self.cfg.NMS.STAGE2.SIGMA,
                            score_threshold=self.cfg.NMS.STAGE2.SCORE_THRESHOLD
                        )

                    if self.cfg.VERBOSE and not metrics_mode:
                        print(f"    ✓ Stage 2 Soft-NMS: {fisheye_bboxes_before} → {len(fisheye_bboxes)} detections")
                        for i, bbox in enumerate(fisheye_bboxes):
                            print(f"      [{i}] {bbox['class_name']} at center ({bbox['center'][0]:.1f}, {bbox['center'][1]:.1f}), "
                                  f"angle={bbox['angle']:.1f}°, conf={bbox.get('confidence', 0):.3f}")

        # Step 5: Save results (skip in metrics mode)
        if metrics_mode:
            # Metrics mode: exit timing context and return only fisheye bboxes
            if timing_context is not None:
                timing_context.__exit__(None, None, None)

            # Convert fisheye_bboxes to standard format expected by evaluator
            if fisheye_bboxes is None or len(fisheye_bboxes) == 0:
                if return_visuals:
                    return [], composite_image, detections
                return []

            # Convert format from backprojection output to evaluator expected format
            # Backprojection format: {center: (x, y), size: (w, h), angle, confidence, class_name}
            # Evaluator format: {center_x, center_y, width, height, angle, confidence, class_name}
            converted_bboxes = []
            for bbox in fisheye_bboxes:
                center_x, center_y = bbox['center']
                width, height = bbox['size']

                converted_bboxes.append({
                    'center_x': center_x,
                    'center_y': center_y,
                    'width': width,
                    'height': height,
                    'angle': bbox['angle'],
                    'confidence': bbox.get('confidence', 0.0),
                    'class_name': bbox.get('class_name', 'person')
                })

            if return_visuals:
                return converted_bboxes, composite_image, detections
            return converted_bboxes
        else:
            # Normal mode: save results and return full output
            results_dir = self._save_results(
                proj_cfg, composite_image, detections, metadata, fisheye_bboxes,
                detections_raw=detections_raw,
                detections_post_flag=detections_post_flag,
                fisheye_dropped_confirmed=fisheye_dropped_confirmed,
            )

            if self.cfg.VERBOSE:
                print("=" * 80)
                print("DETECTION PIPELINE COMPLETE")
                print(f"Results saved to: {results_dir}")
                print("=" * 80)

            # Exit timing context if active
            if timing_context is not None:
                timing_context.__exit__(None, None, None)

            return detections, composite_image, metadata, results_dir, fisheye_bboxes

    def _save_results(self, proj_cfg, composite_image, detections, metadata, fisheye_bboxes=None,
                      detections_raw=None, detections_post_flag=None,
                      fisheye_dropped_confirmed=None):
        """
        Save detection results to organized directory structure.

        Creates: results/image_name/timestamp/
        Saves: fisheye_image.png, composite.png, detections.png, metadata.txt, fisheye_detections.png (if backprojection enabled)

        Args:
            proj_cfg: projection configuration dict
            composite_image: numpy array (H, W, 3)
            detections: list of detection dicts
            metadata: projection metadata dict
            fisheye_bboxes: list of backprojected bbox dicts (optional)

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

        # Step 3b: Per-stage visualizations for the redundant-bbox filter
        # (border-based, two-stage). Only produced when filter is enabled and
        # SAVE_STAGE_VISUALS is on.
        rb_cfg = self.cfg.REDUNDANT_BBOX_FILTER.BORDER_BASED
        if rb_cfg.FLAGGING.ENABLED and rb_cfg.SAVE_STAGE_VISUALS and detections_raw is not None:
            grid_tuple = tuple(proj_cfg["grid"])

            # Composite: raw YOLO output (already post YOLO-internal Stage-1 NMS).
            raw_viz = visualize_detections_on_grid(
                composite_image, detections_raw, grid_tuple, color=(255, 255, 255)
            )
            cv2.imwrite(str(results_dir / "composite_01_yolo_post_stage1_nms.png"), raw_viz)

            # Composite: same detections but flagged ones highlighted (orange
            # bbox with magenta side, unflagged in green). Uses the post-flag
            # snapshot so flagged-but-dropped boxes are still visible here,
            # even when Stage 2 is disabled and the drop already happened on
            # composite.
            viz_input = detections_post_flag if detections_post_flag is not None else detections
            flag_viz = visualize_composite_flags(
                composite_image, viz_input, grid_tuple
            )
            cv2.imwrite(str(results_dir / "composite_02_flagged_for_review.png"),
                        flag_viz)

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
            f.write(f"  Confidence Threshold: {self.cfg.YOLO.CONFIDENCE_THRESHOLD}\n\n")

            f.write("NMS Configuration:\n")
            f.write(f"  Stage 1 (Composite):\n")
            f.write(f"    Enabled: {self.cfg.NMS.STAGE1.ENABLED}\n")
            if self.cfg.NMS.STAGE1.ENABLED:
                f.write(f"    IoU Threshold: {self.cfg.NMS.STAGE1.IOU_THRESHOLD}\n")
            f.write(f"  Stage 2 (Fisheye Soft-NMS):\n")
            f.write(f"    Enabled: {self.cfg.NMS.STAGE2.ENABLED}\n")
            if self.cfg.NMS.STAGE2.ENABLED:
                f.write(f"    Sigma: {self.cfg.NMS.STAGE2.SIGMA}\n")
                f.write(f"    Score Threshold: {self.cfg.NMS.STAGE2.SCORE_THRESHOLD}\n")
            f.write("\n")

            f.write("Detection Results:\n")
            f.write(f"  Total Pedestrians Detected: {len(detections)}\n\n")

            if len(detections) > 0:
                f.write("Detections:\n")
                for i, det in enumerate(detections, 1):
                    f.write(f"  [{i}] {det['class_name']} at ({det['x']:.3f}, {det['y']:.3f}), "
                           f"confidence={det['confidence']:.3f}\n")

        # Step 5: Save fisheye visualization if backprojection was performed
        if fisheye_bboxes is not None and len(fisheye_bboxes) > 0:
            fisheye_img = cv2.imread(str(fisheye_path))
            if fisheye_img is not None:
                # Save fisheye with bboxes
                fisheye_viz = visualize_backprojection(fisheye_img, fisheye_bboxes)
                fisheye_viz_out = results_dir / "fisheye_detections.png"
                cv2.imwrite(str(fisheye_viz_out), fisheye_viz)

                # Optional overlay: each projection's FOV outline in a unique colour.
                if self.cfg.OUTPUT.SAVE_FISHEYE_PROJ_BORDERS and metadata.get("proj_list"):
                    h, w = fisheye_img.shape[:2]
                    cx, cy = w // 2, h // 2
                    r = min(cx, cy)
                    proj_list = metadata["proj_list"]
                    colors = generate_rainbow_colors(len(proj_list))
                    viz_with_borders = fisheye_viz.copy()
                    for proj_params, color in zip(proj_list, colors):
                        viz_with_borders = draw_fov_on_fisheye(
                            viz_with_borders, cx, cy, r,
                            proj_params["longitude"], proj_params["latitude"],
                            proj_params["fov_h"], proj_params["fov_v"],
                            color=color,
                            thickness=self.cfg.OUTPUT.PROJ_BORDERS_THICKNESS,
                        )
                    proj_borders_out = results_dir / "fisheye_detections_with_proj_borders.png"
                    cv2.imwrite(str(proj_borders_out), viz_with_borders)
                    if self.cfg.VERBOSE:
                        print(f"    Fisheye with proj borders: {proj_borders_out}")

                # Optional: fisheye view showing redundant-bbox confirmation
                # result — kept boxes in blue, confirmation-dropped in red.
                if (rb_cfg.FLAGGING.ENABLED and rb_cfg.SAVE_STAGE_VISUALS and
                        fisheye_dropped_confirmed is not None and
                        len(fisheye_dropped_confirmed) > 0):
                    confirm_viz = fisheye_img.copy()
                    for bb in fisheye_bboxes:
                        draw_rotated_bbox(confirm_viz, bb, color=(255, 0, 0), thickness=2,
                                          draw_label=False)
                    for bb in fisheye_dropped_confirmed:
                        draw_rotated_bbox(confirm_viz, bb, color=(0, 0, 255), thickness=2,
                                          draw_label=False)
                    confirm_out = results_dir / "fisheye_03_confirmation_drops.png"
                    cv2.imwrite(str(confirm_out), confirm_viz)
                    if self.cfg.VERBOSE:
                        print(f"    Fisheye confirmation drops: {confirm_out} "
                              f"({len(fisheye_dropped_confirmed)} dropped, "
                              f"{len(fisheye_bboxes)} kept)")

                # Save individual lattice visualization for each bbox (if enabled)
                if self.cfg.VERBOSE:
                    print(f"    Fisheye detections: {fisheye_viz_out}")
                    if self.cfg.OUTPUT.SAVE_LATTICE_VIZ:
                        print(f"    Fisheye bbox lattices (individual):")

                if self.cfg.OUTPUT.SAVE_LATTICE_VIZ:
                    for i, bbox in enumerate(fisheye_bboxes):
                        # Visualize only this bbox's lattice
                        fisheye_lattice = visualize_bbox_lattice(fisheye_img, [bbox])
                        fisheye_lattice_out = results_dir / f"fisheye_bbox_lattice_{i}.png"
                        cv2.imwrite(str(fisheye_lattice_out), fisheye_lattice)

                        if self.cfg.VERBOSE:
                            print(f"      [{i}] {fisheye_lattice_out}")

        if self.cfg.VERBOSE:
            print(f"\n[5] Saving results...")
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
    detections, composite, metadata, results_dir, fisheye_bboxes = pipeline.run()

    print(f"\nFound {len(detections)} pedestrians")
    print(f"Composite size: {composite.shape}")
    if fisheye_bboxes:
        print(f"Backprojected {len(fisheye_bboxes)} bboxes to fisheye")
    print(f"Results saved to: {results_dir}")
