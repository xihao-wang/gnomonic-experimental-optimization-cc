#!/usr/bin/env python3
"""Build temporal pair datasets from edited viewer GT.

This script is the gnomonic/fisheye counterpart of the old
``tools/build_temporal_pairs_from_gt.py``. It reads:

    viewer_sequence/
      img1/
      gt/gt.txt

and a MOT-like proposal file, normally ``viewer_sequence/result.txt`` generated
from ``tracks.csv``. For each proposal box it extracts a real FastReID feature,
matches proposals to GT identities by IoU, maintains one STM/LTM memory per GT
identity, and exports the ``.npz`` consumed by ``train_temporal_model.py``.

The important difference from ``viewer_sequence/detections.npy`` is that this
script does not use dummy zero features. It re-crops images and computes real
BoT/FastReID embeddings.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracker_pipeline.reid.fastreid_extractor import FastReIDFeatureExtractor
from tracker_pipeline.strongsort.detection import Detection
from tracker_pipeline.strongsort.track import Track, TrackState


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class Proposal:
    frame: int
    det_id: int
    tlwh: np.ndarray
    confidence: float
    feature: Optional[np.ndarray] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build learned temporal pair .npz from gnomonic viewer_sequence GT."
    )
    parser.add_argument("--sequence-dir", required=True, type=Path)
    parser.add_argument("--gt-txt", default=None, type=Path)
    parser.add_argument("--proposal-txt", default=None, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)

    parser.add_argument("--fastreid-root", default="tracker_pipeline/third_party/fast-reid")
    parser.add_argument("--fastreid-config", default="tracker_pipeline/reid/configs/bagtricks_S50.yml")
    parser.add_argument("--fastreid-weights", default="tracker_pipeline/reid/weights/duke_bot_S50.pth")
    parser.add_argument("--fastreid-device", default="cpu")
    parser.add_argument("--fastreid-batch-size", type=int, default=32)

    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--min-detection-height", type=float, default=0.0)
    parser.add_argument("--temporal-stride", type=int, default=2)
    parser.add_argument("--short-history-len", type=int, default=5)
    parser.add_argument("--long-history-len", type=int, default=15)
    parser.add_argument("--include-single-identity-frames", action="store_true")
    parser.add_argument(
        "--max-negatives-per-positive",
        type=int,
        default=None,
        help=(
            "Limit stored negatives per positive detection. "
            "When set, keeps positives and only the K hardest negatives "
            "per detection, ranked by cosine similarity to the detection feature. "
            "Default keeps all negatives."
        ),
    )
    parser.add_argument("--feature-cache", default=None, type=Path)
    parser.add_argument("--max-frames", type=int, default=None, help="Debug option: stop after N scanned frames.")
    parser.add_argument(
        "--store-feature-dtype",
        choices=["float32", "float16"],
        default="float32",
        help="Feature dtype stored in the output npz. float16 greatly reduces RAM/disk usage.",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Use np.savez instead of np.savez_compressed to reduce save-time CPU/RAM pressure.",
    )
    parser.add_argument(
        "--shard-size-frames",
        type=int,
        default=None,
        help=(
            "Write samples every N scanned frames into shard files and clear RAM. "
            "Example output for foo.npz: foo_part001.npz, foo_part002.npz. "
            "If unset, writes a single npz to --output-npz."
        ),
    )
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def collect_images(sequence_dir: Path) -> Dict[int, Path]:
    img_dir = sequence_dir / "img1"
    if not img_dir.exists():
        raise FileNotFoundError(f"img1 directory not found: {img_dir}")
    images = sorted(p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise FileNotFoundError(f"No images found in: {img_dir}")
    return {idx: path for idx, path in enumerate(images, start=1)}


def load_mot_rows(path: Path) -> Dict[int, List[dict]]:
    by_frame: Dict[int, List[dict]] = defaultdict(list)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            frame = int(float(parts[0]))
            obj_id = int(float(parts[1]))
            tlwh = np.asarray([float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])], dtype=np.float32)
            if obj_id <= 0 or tlwh[2] <= 0.0 or tlwh[3] <= 0.0:
                continue
            conf = float(parts[6]) if len(parts) > 6 else 1.0
            by_frame[frame].append({"id": obj_id, "tlwh": tlwh, "confidence": conf})
    return by_frame


def load_proposals(path: Path, min_confidence: float, min_detection_height: float) -> Dict[int, List[Proposal]]:
    by_frame: Dict[int, List[Proposal]] = defaultdict(list)
    rows = load_mot_rows(path)
    det_counter = 0
    for frame, frame_rows in rows.items():
        for row in frame_rows:
            tlwh = row["tlwh"]
            conf = float(row["confidence"])
            if conf < min_confidence or tlwh[3] < min_detection_height:
                continue
            by_frame[frame].append(Proposal(frame, det_counter, tlwh.copy(), conf))
            det_counter += 1
    return by_frame


def iou_tlwh(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return 0.0 if union <= 0.0 else inter / union


def infer_proposal_gt_ids(
    proposals: List[Proposal],
    gt_rows: List[dict],
    iou_threshold: float,
) -> Tuple[List[Optional[int]], List[float]]:
    gt_ids: List[Optional[int]] = []
    gt_ious: List[float] = []
    for proposal in proposals:
        best_iou = 0.0
        best_id: Optional[int] = None
        for gt in gt_rows:
            iou = iou_tlwh(proposal.tlwh, gt["tlwh"])
            if iou > best_iou:
                best_iou = iou
                best_id = int(gt["id"])
        if best_iou >= iou_threshold:
            gt_ids.append(best_id)
            gt_ious.append(best_iou)
        else:
            gt_ids.append(None)
            gt_ious.append(0.0)
    return gt_ids, gt_ious


def tlwh_to_xyxy(tlwh: Sequence[float]) -> np.ndarray:
    x, y, w, h = [float(v) for v in tlwh]
    return np.asarray([x, y, x + w, y + h], dtype=np.float32)


def extract_frame_features(
    image: np.ndarray,
    proposals: List[Proposal],
    extractor: FastReIDFeatureExtractor,
) -> None:
    if not proposals:
        return
    boxes = [tlwh_to_xyxy(p.tlwh) for p in proposals]
    feats = extractor(image, boxes)
    if feats.shape[0] != len(proposals):
        raise ValueError(f"Feature extractor returned {feats.shape[0]} rows for {len(proposals)} proposals")
    for proposal, feat in zip(proposals, feats):
        proposal.feature = np.asarray(feat, dtype=np.float32)


def build_short_history(track: Track, short_history_len: int):
    memory = getattr(track, "short_memory", [])
    if not memory:
        return None
    valid_len = min(len(memory), short_history_len)
    items = [np.asarray(feat, dtype=np.float32) for feat in memory[-short_history_len:]][::-1]
    while len(items) < short_history_len:
        items.append(items[-1])
    return np.stack(items, axis=0), valid_len


def build_long_history(track: Track, long_history_len: int):
    memory = getattr(track, "long_memory", [])
    if not memory:
        return None
    valid_len = min(len(memory), long_history_len)
    items = [np.asarray(feat, dtype=np.float32) for feat in memory[-long_history_len:]]
    while len(items) < long_history_len:
        items.append(items[-1])
    return np.stack(items, axis=0), valid_len


def normalize_feature(feat: np.ndarray) -> np.ndarray:
    feat = np.asarray(feat, dtype=np.float32)
    norm = np.linalg.norm(feat)
    return feat if norm < 1e-12 else feat / norm


def clear_sample_lists(
    det_feat_list: List[np.ndarray],
    short_hist_feat_list: List[np.ndarray],
    long_hist_feat_list: List[np.ndarray],
    short_hist_len_list: List[int],
    long_hist_len_list: List[int],
    label_list: List[float],
    frame_list: List[int],
    track_id_list: List[int],
    det_index_list: List[int],
) -> None:
    det_feat_list.clear()
    short_hist_feat_list.clear()
    long_hist_feat_list.clear()
    short_hist_len_list.clear()
    long_hist_len_list.clear()
    label_list.clear()
    frame_list.clear()
    track_id_list.clear()
    det_index_list.clear()


def save_pair_npz(
    output_npz: Path,
    det_feat_list: List[np.ndarray],
    short_hist_feat_list: List[np.ndarray],
    long_hist_feat_list: List[np.ndarray],
    short_hist_len_list: List[int],
    long_hist_len_list: List[int],
    label_list: List[float],
    frame_list: List[int],
    track_id_list: List[int],
    det_index_list: List[int],
    stride: int,
    feature_store_dtype,
    compress: bool,
) -> Tuple[int, int, int]:
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(label_list, dtype=np.float32)
    save_fn = np.savez_compressed if compress else np.savez
    save_fn(
        output_npz,
        det_feat=np.asarray(det_feat_list, dtype=feature_store_dtype),
        short_hist_feat=np.asarray(short_hist_feat_list, dtype=feature_store_dtype),
        long_hist_feat=np.asarray(long_hist_feat_list, dtype=feature_store_dtype),
        short_hist_len=np.asarray(short_hist_len_list, dtype=np.int32),
        long_hist_len=np.asarray(long_hist_len_list, dtype=np.int32),
        label=labels,
        frame=np.asarray(frame_list, dtype=np.int32),
        track_id=np.asarray(track_id_list, dtype=np.int32),
        det_index=np.asarray(det_index_list, dtype=np.int32),
        temporal_stride=np.asarray([stride], dtype=np.int32),
    )
    num_samples = int(labels.size)
    num_pos = int(labels.sum()) if labels.size else 0
    return num_samples, num_pos, num_samples - num_pos


def make_shard_path(output_npz: Path, shard_index: int) -> Path:
    return output_npz.with_name(f"{output_npz.stem}_part{shard_index:03d}{output_npz.suffix}")


def select_pair_memory_items(
    memory_items: List[Tuple[int, Track, np.ndarray, int, np.ndarray, int]],
    det_gt_id: int,
    det_feature: np.ndarray,
    max_negatives_per_positive: Optional[int],
) -> List[Tuple[int, Track, np.ndarray, int, np.ndarray, int]]:
    """Keep all positives and optionally only top-K hard negatives.

    Hard negatives are selected by current feature similarity to the newest
    short-memory feature. This avoids materializing every wrong identity in
    crowded scenes while keeping the most confusing competitors.
    """
    positives = [item for item in memory_items if item[0] == det_gt_id]
    negatives = [item for item in memory_items if item[0] != det_gt_id]
    if max_negatives_per_positive is None or max_negatives_per_positive < 0:
        return positives + negatives
    if max_negatives_per_positive == 0:
        return positives
    if not negatives:
        return positives

    det_feature = normalize_feature(det_feature)
    scored_negatives = []
    for item in negatives:
        gt_id, _memory, short_hist, _short_len, _long_hist, _long_len = item
        ref_feature = normalize_feature(short_hist[0])
        similarity = float(np.dot(det_feature, ref_feature))
        scored_negatives.append((similarity, gt_id, item))
    scored_negatives.sort(key=lambda row: (row[0], row[1]), reverse=True)
    selected_negatives = [item for _sim, _gt_id, item in scored_negatives[:max_negatives_per_positive]]
    return positives + selected_negatives


def main() -> None:
    args = parse_args()

    # The imported Track class reads tracker_pipeline.strongsort.opts.opt.
    # Set the memory flags used by the online tracker before creating tracks.
    from tracker_pipeline.strongsort.opts import opt

    opt.enable_stm_ltm = True
    opt.memory_init = True
    opt.memory_aware = True
    opt.topk = True

    sequence_dir = args.sequence_dir
    gt_txt = args.gt_txt or sequence_dir / "gt" / "gt.txt"
    proposal_txt = args.proposal_txt or sequence_dir / "result.txt"

    images = collect_images(sequence_dir)
    gt_by_frame = load_mot_rows(gt_txt)
    proposals_by_frame = load_proposals(proposal_txt, args.min_confidence, args.min_detection_height)

    print("Loading FastReID extractor...", flush=True)
    extractor = FastReIDFeatureExtractor(
        fastreid_root=str(resolve_project_path(args.fastreid_root)),
        config_file=str(resolve_project_path(args.fastreid_config)),
        weights=str(resolve_project_path(args.fastreid_weights)),
        device=args.fastreid_device,
        batch_size=args.fastreid_batch_size,
        normalize=True,
    )
    print("FastReID extractor ready.", flush=True)

    memories: Dict[int, Track] = {}
    det_feat_list: List[np.ndarray] = []
    short_hist_feat_list: List[np.ndarray] = []
    long_hist_feat_list: List[np.ndarray] = []
    short_hist_len_list: List[int] = []
    long_hist_len_list: List[int] = []
    label_list: List[float] = []
    frame_list: List[int] = []
    track_id_list: List[int] = []
    det_index_list: List[int] = []

    stride = max(1, int(args.temporal_stride))
    short_history_len = max(2, int(args.short_history_len))
    long_history_len = max(1, int(args.long_history_len))
    min_history_observations = 2 * stride + 1
    feature_store_dtype = np.float16 if args.store_feature_dtype == "float16" else np.float32
    frames = sorted(set(images) | set(gt_by_frame) | set(proposals_by_frame))

    total_valid_dets = 0
    total_frames_with_pairs = 0
    total_samples_written = 0
    total_pos_written = 0
    total_neg_written = 0
    shard_index = 0
    shard_size_frames = args.shard_size_frames
    shard_start_scan_idx = 1
    written_paths: List[Path] = []
    if args.max_frames is not None:
        frames = frames[: max(0, int(args.max_frames))]

    for scan_idx, frame in enumerate(frames, start=1):
        if args.log_every and (scan_idx == 1 or scan_idx % args.log_every == 0):
            print(f"Processing frame {frame} ({scan_idx}/{len(frames)})", flush=True)
        proposals = proposals_by_frame.get(frame, [])
        gt_rows = gt_by_frame.get(frame, [])
        if proposals:
            image_path = images.get(frame)
            if image_path is None:
                raise FileNotFoundError(f"No image for frame {frame}")
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Could not read image: {image_path}")
            extract_frame_features(image, proposals, extractor)

        proposal_gt_ids, proposal_gt_ious = infer_proposal_gt_ids(proposals, gt_rows, args.iou_threshold)
        valid_indices = [idx for idx, gt_id in enumerate(proposal_gt_ids) if gt_id is not None]
        if not args.include_single_identity_frames:
            valid_gt_ids = {proposal_gt_ids[idx] for idx in valid_indices}
            if len(valid_gt_ids) < 2:
                valid_indices_for_pairs: List[int] = []
            else:
                valid_indices_for_pairs = valid_indices
        else:
            valid_indices_for_pairs = valid_indices

        if valid_indices_for_pairs:
            total_frames_with_pairs += 1
        total_valid_dets += len(valid_indices)

        memory_items: List[Tuple[int, Track, np.ndarray, int, np.ndarray, int]] = []
        for gt_id, memory in sorted(memories.items()):
            if len(getattr(memory, "det_feat_history", [])) < min_history_observations:
                continue
            short_result = build_short_history(memory, short_history_len)
            long_result = build_long_history(memory, long_history_len)
            if short_result is None or long_result is None:
                continue
            short_hist, short_len = short_result
            long_hist, long_len = long_result
            memory_items.append((gt_id, memory, short_hist, short_len, long_hist, long_len))

        for det_idx in valid_indices_for_pairs:
            det_gt_id = proposal_gt_ids[det_idx]
            if det_gt_id is None:
                continue
            feat = normalize_feature(proposals[det_idx].feature)
            selected_memory_items = select_pair_memory_items(
                memory_items,
                int(det_gt_id),
                feat,
                args.max_negatives_per_positive,
            )
            if not any(gt_id == det_gt_id for gt_id, *_rest in selected_memory_items):
                continue
            if args.max_negatives_per_positive is not None and args.max_negatives_per_positive >= 0:
                if not any(gt_id != det_gt_id for gt_id, *_rest in selected_memory_items):
                    continue
            for gt_id, _memory, short_hist, short_len, long_hist, long_len in selected_memory_items:
                det_feat_list.append(feat.astype(feature_store_dtype, copy=False))
                short_hist_feat_list.append(short_hist.astype(feature_store_dtype, copy=False))
                long_hist_feat_list.append(long_hist.astype(feature_store_dtype, copy=False))
                short_hist_len_list.append(short_len)
                long_hist_len_list.append(long_len)
                label_list.append(1.0 if proposal_gt_ids[det_idx] == gt_id else 0.0)
                frame_list.append(frame)
                track_id_list.append(gt_id)
                det_index_list.append(det_idx)

        best_det_by_gt: Dict[int, Tuple[float, int]] = {}
        for det_idx in valid_indices:
            gt_id = proposal_gt_ids[det_idx]
            if gt_id is None:
                continue
            iou = proposal_gt_ious[det_idx]
            if gt_id not in best_det_by_gt or iou > best_det_by_gt[gt_id][0]:
                best_det_by_gt[int(gt_id)] = (iou, det_idx)

        for gt_id, (_iou, det_idx) in best_det_by_gt.items():
            proposal = proposals[det_idx]
            feature = normalize_feature(proposal.feature)
            detection = Detection(proposal.tlwh, proposal.confidence, feature)
            if gt_id not in memories:
                memories[gt_id] = Track(
                    detection.to_xyah(),
                    gt_id,
                    n_init=1,
                    max_age=10**9,
                    feature=feature.copy(),
                    score=detection.confidence,
                )
                memories[gt_id].state = TrackState.Confirmed
            else:
                memories[gt_id].update(detection)

        if shard_size_frames is not None and shard_size_frames > 0:
            should_flush = (
                scan_idx - shard_start_scan_idx + 1 >= shard_size_frames
                or scan_idx == len(frames)
            )
            if should_flush:
                if label_list:
                    shard_index += 1
                    shard_path = make_shard_path(args.output_npz, shard_index)
                    samples, pos, neg = save_pair_npz(
                        shard_path,
                        det_feat_list,
                        short_hist_feat_list,
                        long_hist_feat_list,
                        short_hist_len_list,
                        long_hist_len_list,
                        label_list,
                        frame_list,
                        track_id_list,
                        det_index_list,
                        stride,
                        feature_store_dtype,
                        compress=not args.no_compress,
                    )
                    total_samples_written += samples
                    total_pos_written += pos
                    total_neg_written += neg
                    written_paths.append(shard_path)
                    print(
                        f"Saved shard {shard_index:03d}: {shard_path} "
                        f"(samples={samples}, positives={pos}, negatives={neg})",
                        flush=True,
                    )
                    clear_sample_lists(
                        det_feat_list,
                        short_hist_feat_list,
                        long_hist_feat_list,
                        short_hist_len_list,
                        long_hist_len_list,
                        label_list,
                        frame_list,
                        track_id_list,
                        det_index_list,
                    )
                shard_start_scan_idx = scan_idx + 1

    if shard_size_frames is None or shard_size_frames <= 0:
        samples, pos, neg = save_pair_npz(
            args.output_npz,
            det_feat_list,
            short_hist_feat_list,
            long_hist_feat_list,
            short_hist_len_list,
            long_hist_len_list,
            label_list,
            frame_list,
            track_id_list,
            det_index_list,
            stride,
            feature_store_dtype,
            compress=not args.no_compress,
        )
        total_samples_written = samples
        total_pos_written = pos
        total_neg_written = neg
        written_paths = [args.output_npz]
        print(f"Saved pair dataset: {args.output_npz}")
    else:
        print(f"Saved shard datasets: {len(written_paths)} files")
        for path in written_paths:
            print(f"  {path}")
    print(f"Frames scanned: {len(frames)}")
    print(f"Valid GT-matched detections: {total_valid_dets}")
    print(f"Frames contributing pairs: {total_frames_with_pairs}")
    print(
        f"Samples: {total_samples_written}, positives: {total_pos_written}, "
        f"negatives: {total_neg_written}"
    )
    if total_samples_written == 0:
        print("WARNING: no samples were generated. Check GT/proposal overlap or use --include-single-identity-frames.")


if __name__ == "__main__":
    main()
