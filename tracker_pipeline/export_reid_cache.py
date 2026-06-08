#!/usr/bin/env python3
"""Export detection + ReID feature caches for fast tracker-only sweeps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from detection_pipeline.pipeline import DetectionPipeline
from tracker_pipeline.gnomonic_adapter import build_tracker_detections
from tracker_pipeline.reid import FastReIDFeatureExtractor
from tracker_pipeline.process_video_strongsort import (
    _build_detection_cfg,
    _build_projection_config,
    _collect_image_sequence,
    _resolve,
)


DEFAULT_SEQS = [
    "convenience_store",
    "empty_store",
    "exhibition",
    "it_office",
    "large_office",
    "large_office_2",
    "warehouse",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache WEPDTOF detections and ReID features.")
    parser.add_argument("--wepdtof-root", type=Path, default=Path("/home/wang/下载/WEPDTOF/WEPDTOF"))
    parser.add_argument("--output-root", type=Path, default=Path("tracker_pipeline/cache/dedup_reid_features"))
    parser.add_argument("--sequences", nargs="+", default=DEFAULT_SEQS)

    parser.add_argument("--preset", default="yolo_grid")
    parser.add_argument("--proj-nbr", type=int, default=6)
    parser.add_argument("--fov-h", type=float, default=90.0)
    parser.add_argument("--fov-v", type=float, default=90.0)
    parser.add_argument("--latitude", type=float, default=0.0)
    parser.add_argument("--lon-0", type=float, default=0.0)
    parser.add_argument("--lon-step", type=float, default=60.0)
    parser.add_argument("--grid-rows", type=int, default=2)
    parser.add_argument("--grid-cols", type=int, default=3)
    parser.add_argument("--comp-width", type=int, default=640)
    parser.add_argument("--comp-height", type=int, default=640)
    parser.add_argument("--target-mp", default="auto")
    parser.add_argument("--yolo-model", default="detection_pipeline/models/yolov9e.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--disable-stage1-nms", action="store_true")
    parser.add_argument("--stage1-iou", type=float, default=0.8)
    parser.add_argument("--disable-stage2-nms", action="store_true")
    parser.add_argument("--stage2-sigma", type=float, default=0.2)
    parser.add_argument("--stage2-score-threshold", type=float, default=0.3)

    parser.add_argument("--fastreid-root", default="tracker_pipeline/third_party/fast-reid")
    parser.add_argument("--fastreid-config", default="tracker_pipeline/reid/configs/bagtricks_S50.yml")
    parser.add_argument("--fastreid-weights", default="tracker_pipeline/reid/weights/duke_bot_S50.pth")
    parser.add_argument("--fastreid-device", default="cuda")
    parser.add_argument("--fastreid-batch-size", type=int, default=32)
    parser.add_argument("--reid-crop-pad", type=int, default=0)
    parser.add_argument("--reid-crop-source", choices=("fisheye", "composite"), default="fisheye")
    parser.add_argument("--sequence-fps", type=float, default=10.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def export_sequence(args: argparse.Namespace, seq: str) -> Path:
    frames_dir = args.wepdtof_root / "frames" / seq
    image_sequence = _collect_image_sequence(frames_dir)
    if not image_sequence:
        raise FileNotFoundError(f"No frames found for sequence: {frames_dir}")

    output_root = _resolve(str(args.output_root))
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{seq}.npz"

    det_cfg = _build_detection_cfg(args)
    proj_cfg = _build_projection_config(det_cfg, image_sequence[0])
    pipeline = DetectionPipeline(det_cfg)
    extractor = FastReIDFeatureExtractor(
        fastreid_root=str(_resolve(args.fastreid_root)),
        config_file=str(_resolve(args.fastreid_config)),
        weights=str(_resolve(args.fastreid_weights)),
        device=args.fastreid_device,
        batch_size=args.fastreid_batch_size,
    )

    frame_values = []
    tlwh_values = []
    confidence_values = []
    feature_values = []
    source_values = []
    processed = 0

    for frame_idx, frame_path in enumerate(image_sequence, start=1):
        if args.max_frames is not None and processed >= args.max_frames:
            break
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"WARNING: skipping unreadable frame: {frame_path}")
            continue

        fisheye_bboxes, composite_image, _ = pipeline.run(
            fisheye_image=frame,
            projection_config=proj_cfg,
            return_visuals=True,
        )
        adapter_dets = build_tracker_detections(
            fisheye_bboxes,
            frame if args.reid_crop_source == "fisheye" else composite_image,
            feature_extractor=extractor,
            require_features=True,
            crop_pad=args.reid_crop_pad,
            reid_source=args.reid_crop_source,
            fallback_reid_image=frame if args.reid_crop_source == "composite" else None,
        )

        for det in adapter_dets:
            feat = np.asarray(det.feature, dtype=np.float32)
            tlwh = np.asarray(det.tlwh, dtype=np.float32)
            if feat.size == 0 or tlwh[2] <= 0 or tlwh[3] <= 0:
                continue
            frame_values.append(frame_idx)
            tlwh_values.append(tlwh)
            confidence_values.append(float(det.confidence))
            feature_values.append(feat)
            source_values.append(str(det.metadata.get("reid_crop_source", args.reid_crop_source)))

        processed += 1
        if args.log_every > 0 and processed % args.log_every == 0:
            print(f"{seq}: frame={frame_idx} processed={processed} cached_dets={len(frame_values)}")

    if feature_values:
        features = np.stack(feature_values, axis=0).astype(np.float32)
        tlwhs = np.stack(tlwh_values, axis=0).astype(np.float32)
    else:
        features = np.zeros((0, 0), dtype=np.float32)
        tlwhs = np.zeros((0, 4), dtype=np.float32)

    metadata = {
        "sequence": seq,
        "frames_dir": str(frames_dir),
        "num_frames": len(image_sequence),
        "processed_frames": processed,
        "sequence_fps": float(args.sequence_fps),
        "reid_crop_source": args.reid_crop_source,
        "feature_dim": int(features.shape[1]) if features.ndim == 2 and features.shape[0] else 0,
        "detections": int(len(frame_values)),
    }

    np.savez(
        output_path,
        frames=np.asarray(frame_values, dtype=np.int32),
        tlwhs=tlwhs,
        confidences=np.asarray(confidence_values, dtype=np.float32),
        features=features,
        reid_sources=np.asarray(source_values, dtype=object),
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False), dtype=object),
    )
    print(f"saved cache: {output_path} detections={len(frame_values)}")
    return output_path


def main() -> None:
    args = parse_args()
    for seq in args.sequences:
        export_sequence(args, seq)


if __name__ == "__main__":
    main()
