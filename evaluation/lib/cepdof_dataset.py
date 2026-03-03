"""
CEPDOF Dataset handler for rotated bounding box annotations.

Loads CEPDOF standard JSON annotations and provides image-annotation pairs
for evaluation. The dataset has a flat 2-level hierarchy:

    {root}/{Sequence}/{frame_id}.jpg
    {root}/standard-annotations/{Sequence}/{frame_id}.json

Sequences are taken from config (DATASETS.CEPDOF.SEQUENCES).

Standard JSON format (same as BOMNI and PIROPO):
{
    "center_x": float,
    "center_y": float,
    "width":    float,
    "height":   float,
    "angle":    float,
    "class_name": string
}

Author: Yassir Zardoua <y.zardoua@caplogy.com | yassirzardoua@gmail.com>
"""

import json
import cv2
from pathlib import Path
from typing import List, Dict, Optional


class CEPDOFDataset:
    """
    CEPDOF dataset handler for standard JSON rotated bbox annotations.

    Iterates over {Sequence}/{frame_id}.jpg with annotations at
    standard-annotations/{Sequence}/{frame_id}.json.
    """

    def __init__(self, cfg):
        """
        Initialize CEPDOF dataset.

        Args:
            cfg: YACS config object with DATASETS.CEPDOF settings
        """
        self.cfg = cfg
        self.root_dir = Path(cfg.DATASETS.CEPDOF.ROOT_DIR)
        self.annotations_dir = Path(cfg.DATASETS.CEPDOF.STANDARD_ANNOTATIONS_DIR)
        self.sequences = list(cfg.DATASETS.CEPDOF.SEQUENCES)
        self.image_ext = cfg.DATASETS.CEPDOF.IMAGE_EXT
        self.verbose = cfg.VERBOSE

        self._build_dataset()

    def _build_dataset(self):
        """
        Build flat list of (image_path, annotation_path) pairs.

        For each sequence, finds all annotation JSON files and the
        corresponding frames under {root}/{sequence}/.
        """
        self.data = []
        stats = {}

        for sequence in self.sequences:
            seq_ann_dir = self.annotations_dir / sequence
            if not seq_ann_dir.exists():
                if self.verbose:
                    print(f"WARNING: Annotations directory not found: {seq_ann_dir}")
                continue

            count = 0
            for ann_file in sorted(seq_ann_dir.glob("*.json")):
                frame_path = self.root_dir / sequence / (ann_file.stem + self.image_ext)

                if not frame_path.exists():
                    if self.verbose:
                        print(f"WARNING: Frame not found: {frame_path}")
                    continue

                self.data.append({
                    "sequence": sequence,
                    "image_path": str(frame_path),
                    "annotation_path": str(ann_file),
                    "image_name": ann_file.stem + self.image_ext
                })
                count += 1

            stats[sequence] = count

        if self.verbose:
            print(f"CEPDOF Dataset: Found {len(self.data)} image-annotation pairs")
            for seq, count in stats.items():
                print(f"  {seq}: {count} images")
        else:
            print(f"CEPDOF Dataset: {len(self.data)} images across {len(stats)} sequences")

    def __len__(self) -> int:
        """Return total number of image-annotation pairs."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get image and annotations by index.

        Args:
            idx: Index into the flat dataset list

        Returns:
            dict with keys:
                - image:            numpy array (H, W, 3)
                - annotations:      list of standard JSON annotation dicts
                - image_path:       str
                - annotation_path:  str
                - sequence:         str  (e.g., "Lunch1")
                - image_name:       str  (e.g., "Lunch1_000067.jpg")
        """
        item = self.data[idx]

        image = cv2.imread(item["image_path"])
        if image is None:
            raise ValueError(f"Could not load image: {item['image_path']}")

        with open(item["annotation_path"], "r") as f:
            annotations = json.load(f)

        return {
            "image": image,
            "annotations": annotations,
            "image_path": item["image_path"],
            "annotation_path": item["annotation_path"],
            "sequence": item["sequence"],
            "image_name": item["image_name"]
        }

    def get_sequence_data(self, sequence: str) -> List[Dict]:
        """Return all data items for a specific sequence."""
        return [d for d in self.data if d["sequence"] == sequence]
