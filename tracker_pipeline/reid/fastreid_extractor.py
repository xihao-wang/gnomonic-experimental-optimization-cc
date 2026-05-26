"""Callable FastReID/BoT feature extractor for gnomonic tracker detections.

This wraps the old file-based FastReID extraction logic into a per-frame
callable compatible with ``tracker_pipeline.gnomonic_adapter``:

    extractor(composite_image, crop_boxes_xyxy) -> N x D feature matrix

The crop boxes are in composite/perspective image coordinates, not fisheye
coordinates.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence
import collections
import collections.abc

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def _load_fastreid(fastreid_root: str):
    # Older FastReID revisions import Mapping from collections, which breaks on
    # Python 3.10+. Patch the symbol before importing FastReID without editing
    # third_party sources.
    if not hasattr(collections, "Mapping"):
        collections.Mapping = collections.abc.Mapping

    root = str(Path(fastreid_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)

    from fastreid.config import get_cfg
    from fastreid.engine import DefaultTrainer
    from fastreid.utils.checkpoint import Checkpointer

    return get_cfg, DefaultTrainer, Checkpointer


def _build_transform():
    return transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255.0),
    ])


def _clip_xyxy(box: Sequence[float], width: int, height: int):
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


class FastReIDFeatureExtractor:
    """FastReID model wrapper with the adapter callable contract."""

    def __init__(
        self,
        fastreid_root: str,
        config_file: str,
        weights: str,
        device: str = "cpu",
        batch_size: int = 32,
        normalize: bool = True,
    ):
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.normalize = bool(normalize)
        self.transform = _build_transform()

        get_cfg, DefaultTrainer, Checkpointer = _load_fastreid(fastreid_root)
        cfg = get_cfg()
        cfg.merge_from_file(config_file)
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        cfg.MODEL.WEIGHTS = weights
        cfg.MODEL.DEVICE = device
        cfg.freeze()

        model = DefaultTrainer.build_model(cfg)
        model.eval()
        Checkpointer(model).load(cfg.MODEL.WEIGHTS)
        model.to(self.device)

        self.cfg = cfg
        self.model = model

    def _make_patches(self, composite_image: np.ndarray, crop_boxes_xyxy: Sequence[Sequence[float]]):
        if composite_image is None:
            raise ValueError("composite_image is required for FastReID feature extraction")

        rgb = cv2.cvtColor(composite_image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        width, height = image.size

        patches = []
        valid_indices = []
        for idx, box in enumerate(crop_boxes_xyxy):
            clipped = _clip_xyxy(box, width, height)
            if clipped is None:
                continue
            patch = image.crop(clipped)
            patches.append(self.transform(patch))
            valid_indices.append(idx)
        return patches, valid_indices

    def __call__(self, composite_image: np.ndarray, crop_boxes_xyxy: Sequence[Sequence[float]]) -> np.ndarray:
        patches, valid_indices = self._make_patches(composite_image, crop_boxes_xyxy)
        if not crop_boxes_xyxy:
            return np.zeros((0, 0), dtype=np.float32)
        if not patches:
            raise ValueError("No valid ReID crops were produced from the provided boxes")

        outputs = []
        with torch.no_grad():
            for start in range(0, len(patches), self.batch_size):
                batch = torch.stack(patches[start:start + self.batch_size], dim=0).to(self.device)
                feat = self.model(batch).detach().cpu().numpy().astype(np.float32)
                outputs.append(feat)

        valid_features = np.concatenate(outputs, axis=0)
        if self.normalize:
            norms = np.linalg.norm(valid_features, axis=1, keepdims=True)
            valid_features = valid_features / np.maximum(norms, 1e-12)

        feature_dim = valid_features.shape[1]
        features = np.zeros((len(crop_boxes_xyxy), feature_dim), dtype=np.float32)
        for row, idx in enumerate(valid_indices):
            features[idx] = valid_features[row]
        return features
