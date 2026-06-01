#!/usr/bin/env python3
"""Run one frame through the gnomonic detection-to-tracker adapter."""

import argparse
from pathlib import Path
import sys

import cv2


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detection_pipeline.config import get_cfg
from detection_pipeline.pipeline import DetectionPipeline
from image_composer.presets import get_preset
from tracker_pipeline.gnomonic_adapter import build_tracker_detections
from tracker_pipeline.reid import FastReIDFeatureExtractor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="detection_pipeline/fisheye-sample.png", help="Input fisheye image")
    parser.add_argument("--preset", default="yolo_grid", help="Projection preset")
    parser.add_argument("--model", default="detection_pipeline/models/yolov8n.pt", help="YOLO model path")
    parser.add_argument("--device", default="cpu", help="YOLO device")
    parser.add_argument("--conf", type=float, default=None, help="Optional YOLO confidence threshold")
    parser.add_argument("--feature-dim", type=int, default=8, help="Placeholder feature dimension for adapter smoke tests")
    parser.add_argument("--use-fastreid", action="store_true", help="Extract real FastReID/BoT features instead of placeholder features")
    parser.add_argument("--fastreid-root", default="tracker_pipeline/third_party/fast-reid", help="Project-relative or absolute FastReID source root")
    parser.add_argument("--fastreid-config", default="tracker_pipeline/reid/configs/bagtricks_S50.yml", help="FastReID config file")
    parser.add_argument("--fastreid-weights", default="tracker_pipeline/reid/weights/duke_bot_S50.pth", help="FastReID model weights")
    parser.add_argument("--fastreid-device", default="cpu", choices=["cpu", "cuda"], help="FastReID inference device")
    parser.add_argument("--fastreid-batch-size", type=int, default=32, help="FastReID crop batch size")
    parser.add_argument(
        "--reid-crop-source",
        choices=("fisheye", "composite"),
        default="fisheye",
        help="Use final fisheye NMS boxes or selected composite source boxes for ReID crops",
    )
    args = parser.parse_args()

    cfg = get_cfg()
    cfg.VERBOSE = False
    cfg.INPUT.IMAGE_PATH = args.image
    cfg.PROJECTION.PRESET = args.preset
    cfg.YOLO.MODEL = args.model
    cfg.YOLO.DEVICE = None if args.device == "None" else args.device
    if args.conf is not None:
        cfg.YOLO.CONFIDENCE_THRESHOLD = args.conf

    cfg.OUTPUT.SAVE_COMPOSITE = False
    cfg.OUTPUT.SAVE_COMPOSITE_VIZ = False
    cfg.OUTPUT.SAVE_FISHEYE_VIZ = False
    cfg.OUTPUT.SAVE_LATTICE_VIZ = False

    fisheye = cv2.imread(args.image)
    if fisheye is None:
        raise FileNotFoundError(args.image)

    projection_config = get_preset(args.preset)
    if projection_config is None:
        raise ValueError(f"Preset '{args.preset}' not found")
    projection_config["img_path"] = args.image

    pipeline = DetectionPipeline(cfg)
    fisheye_bboxes, composite, raw_detections = pipeline.run(
        fisheye_image=fisheye,
        projection_config=projection_config,
        return_visuals=True,
    )

    feature_extractor = None
    if args.use_fastreid:
        for label, path in (
            ("FastReID root", args.fastreid_root),
            ("FastReID config", args.fastreid_config),
            ("FastReID weights", args.fastreid_weights),
        ):
            if not Path(path).exists():
                raise FileNotFoundError(f"{label} not found: {path}")
        feature_extractor = FastReIDFeatureExtractor(
            fastreid_root=args.fastreid_root,
            config_file=args.fastreid_config,
            weights=args.fastreid_weights,
            device=args.fastreid_device,
            batch_size=args.fastreid_batch_size,
        )

    tracker_detections = build_tracker_detections(
        fisheye_bboxes,
        fisheye if args.reid_crop_source == "fisheye" else composite,
        feature_extractor=feature_extractor,
        feature_dim=args.feature_dim,
        require_features=args.use_fastreid,
        reid_source=args.reid_crop_source,
    )

    print("Adapter frame check")
    print("image:", args.image)
    print("raw composite detections:", len(raw_detections))
    print("final fisheye bboxes:", len(fisheye_bboxes or []))
    print("adapter detections:", len(tracker_detections))
    print("feature mode:", "FastReID" if args.use_fastreid else "placeholder")
    print("reid crop source:", args.reid_crop_source)

    for idx, det in enumerate(tracker_detections):
        md = det.metadata
        print(f"det#{idx}")
        print("  tracker tlwh:", det.tlwh.tolist())
        print("  confidence:", round(det.confidence, 4))
        print("  feature shape:", tuple(det.feature.shape))
        print("  actual reid crop source:", md.get("reid_crop_source"))
        print("  actual reid crop bbox xyxy:", md.get("reid_crop_bbox_xyxy"))
        print("  selected composite source id:", md.get("reid_source_id"))
        print("  selected composite source bbox xyxy:", md.get("reid_source_bbox_xyxy"))
        print("  duplicate source ids:", md.get("duplicate_source_ids"))
        print("  duplicate count:", md.get("duplicate_count"))
        print("  reid source quality:", md.get("reid_source_quality"))


if __name__ == "__main__":
    main()
