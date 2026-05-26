#!/usr/bin/env python3
"""Visualize source-aware duplicate grouping for one fisheye image."""

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

# Allow running as `python detection_pipeline/debug_source_aware.py`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detection_pipeline.config import get_cfg
from detection_pipeline.pipeline import DetectionPipeline
from image_composer.presets import get_preset


def _xyxy(det):
    return det.get("xyxy") or det.get("source_bbox_xyxy")


def _safe_int_box(xyxy):
    return [int(round(float(v))) for v in xyxy]


def _draw_label(image, text, origin, color, scale=0.45, thickness=1):
    x, y = origin
    y = max(14, min(image.shape[0] - 6, y))
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_composite_debug(composite, detections, fisheye_bboxes):
    vis = composite.copy()

    selected_ids = set()
    group_ids = set()
    for bbox in fisheye_bboxes or []:
        group_ids.update(bbox.get("duplicate_source_ids") or [])
        if bbox.get("reid_source_id") is not None:
            selected_ids.add(bbox.get("reid_source_id"))

    # First draw all source detections in gray.
    for det in detections:
        xyxy = _xyxy(det)
        if xyxy is None:
            continue
        x1, y1, x2, y2 = _safe_int_box(xyxy)
        sid = det.get("source_id")
        color = (160, 160, 160)
        thickness = 1
        if sid in group_ids:
            color = (0, 180, 255)  # orange for duplicate group members
            thickness = 2
        if sid in selected_ids:
            color = (0, 255, 0)  # green for selected ReID crop
            thickness = 3
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        label = f"S{sid} conf={float(det.get('confidence', 0.0)):.2f}"
        if sid in selected_ids:
            label = "SELECTED ReID " + label
        elif sid in group_ids:
            label = "DUP " + label
        _draw_label(vis, label, (x1, y1 - 5), color)

    # Draw group-level text near selected crops.
    for idx, bbox in enumerate(fisheye_bboxes or []):
        reid_xyxy = bbox.get("reid_source_bbox_xyxy")
        if reid_xyxy is None:
            continue
        x1, y1, x2, y2 = _safe_int_box(reid_xyxy)
        dup_ids = bbox.get("duplicate_source_ids") or []
        text = (
            f"final#{idx}: dup_count={bbox.get('duplicate_count')} "
            f"dup_ids={dup_ids} selected=S{bbox.get('reid_source_id')} "
            f"q={float(bbox.get('reid_source_quality') or 0.0):.3f}"
        )
        _draw_label(vis, text, (x1, y2 + 16), (0, 255, 0), scale=0.42)

    return vis


def draw_fisheye_debug(fisheye, fisheye_bboxes):
    vis = fisheye.copy()
    for idx, bbox in enumerate(fisheye_bboxes or []):
        color = (0, 255, 0)
        if bbox.get("corners") is not None:
            corners = cv2.convexHull(np.asarray(bbox["corners"], dtype=np.int32))
            cv2.polylines(vis, [corners], isClosed=True, color=color, thickness=2)
            x, y = corners[:, 0, :].min(axis=0)
            _draw_label(
                vis,
                f"final#{idx} conf={float(bbox.get('confidence', 0.0)):.2f} dup={bbox.get('duplicate_count')}",
                (int(x), int(y) - 5),
                color,
            )
            continue
        tracking_tlwh = bbox.get("tracking_bbox_tlwh")
        if tracking_tlwh is not None:
            x, y, w, h = [int(round(float(v))) for v in tracking_tlwh]
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            _draw_label(
                vis,
                f"final#{idx} conf={float(bbox.get('confidence', 0.0)):.2f} dup={bbox.get('duplicate_count')}",
                (x, y - 5),
                color,
            )
            continue

        # Fallback for converted metric-mode bboxes without corner points.
        if all(k in bbox for k in ("center_x", "center_y", "width", "height")):
            cx = float(bbox["center_x"])
            cy = float(bbox["center_y"])
            w = float(bbox["width"])
            h = float(bbox["height"])
            x1 = int(round(cx - w / 2))
            y1 = int(round(cy - h / 2))
            x2 = int(round(cx + w / 2))
            y2 = int(round(cy + h / 2))
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            _draw_label(
                vis,
                f"final#{idx} conf={float(bbox.get('confidence', 0.0)):.2f} dup={bbox.get('duplicate_count')}",
                (x1, y1 - 5),
                color,
            )
    return vis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="detection_pipeline/fisheye-sample.png", help="Input fisheye image path")
    parser.add_argument("--preset", default="yolo_grid", help="Projection preset")
    parser.add_argument("--model", default="detection_pipeline/models/yolov8n.pt", help="YOLO model path")
    parser.add_argument("--device", default="cpu", help="YOLO device, e.g. cpu/cuda/None")
    parser.add_argument("--conf", type=float, default=None, help="Optional YOLO confidence threshold")
    parser.add_argument("--stage1-iou", type=float, default=None, help="Optional Stage 1 NMS IoU threshold")
    parser.add_argument("--output-dir", default="detection_pipeline/results/source_aware_debug", help="Output directory")
    args = parser.parse_args()

    cfg = get_cfg()
    cfg.VERBOSE = False
    cfg.INPUT.IMAGE_PATH = args.image
    cfg.PROJECTION.PRESET = args.preset
    cfg.YOLO.MODEL = args.model
    cfg.YOLO.DEVICE = None if args.device == "None" else args.device
    if args.conf is not None:
        cfg.YOLO.CONFIDENCE_THRESHOLD = args.conf
    if args.stage1_iou is not None:
        cfg.NMS.STAGE1.IOU_THRESHOLD = args.stage1_iou

    cfg.OUTPUT.SAVE_COMPOSITE = False
    cfg.OUTPUT.SAVE_COMPOSITE_VIZ = False
    cfg.OUTPUT.SAVE_FISHEYE_VIZ = False
    cfg.OUTPUT.SAVE_LATTICE_VIZ = False
    cfg.OUTPUT.SAVE_DIR = args.output_dir

    fisheye = cv2.imread(args.image)
    if fisheye is None:
        raise FileNotFoundError(args.image)

    pipeline = DetectionPipeline(cfg)
    if args.preset:
        projection_config = get_preset(args.preset)
        if projection_config is None:
            raise ValueError(f"Preset '{args.preset}' not found")
        projection_config["img_path"] = args.image
    else:
        projection_config = pipeline._build_projection_config()

    fisheye_bboxes, composite, detections = pipeline.run(
        fisheye_image=fisheye,
        projection_config=projection_config,
        return_visuals=True,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_stem = Path(args.image).stem

    composite_debug = draw_composite_debug(composite, detections, fisheye_bboxes)
    composite_out = output_dir / f"{image_stem}_source_groups.png"
    cv2.imwrite(str(composite_out), composite_debug)

    fisheye_debug = draw_fisheye_debug(fisheye, fisheye_bboxes or [])
    fisheye_out = output_dir / f"{image_stem}_final_fisheye.png"
    cv2.imwrite(str(fisheye_out), fisheye_debug)

    print("Source-aware debug complete")
    print("image:", args.image)
    print("preset:", args.preset)
    print("raw composite detections:", len(detections))
    print("final fisheye bboxes:", len(fisheye_bboxes or []))
    for idx, bbox in enumerate(fisheye_bboxes or []):
        print(
            f"final#{idx}: source={bbox.get('source_id')} "
            f"dup_count={bbox.get('duplicate_count')} "
            f"dup_ids={bbox.get('duplicate_source_ids')} "
            f"selected_reid={bbox.get('reid_source_id')} "
            f"reid_quality={bbox.get('reid_source_quality')} "
            f"reid_bbox={bbox.get('reid_source_bbox_xyxy')}"
        )
    print("saved composite debug:", composite_out)
    print("saved fisheye debug:", fisheye_out)


if __name__ == "__main__":
    main()
