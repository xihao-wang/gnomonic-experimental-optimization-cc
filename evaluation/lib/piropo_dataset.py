"""
PIROPO Dataset handler for rotated bounding box annotations.

Loads PIROPO standard JSON annotations and provides image-annotation pairs
for evaluation. The dataset has a 3-level hierarchy:

    {root}/{Room}/{Camera}/{Camera}_{Sequence}/{frame}.jpg
    {root}/standard-annotations/{Room}/{Camera}/{Camera}_{Sequence}/{frame}.json

Cameras and sequences are auto-discovered from the annotations directory.
Rooms can be filtered via config (e.g., only Room_A, only Room_B, or both).

Standard JSON format (same as BOMNI and CEPDOF):
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


class PIROPODataset:
    """
    PIROPO dataset handler for standard JSON rotated bbox annotations.

    Flattens the 3-level Room → Camera → Sequence folder hierarchy into a
    single list of (image_path, annotation_path) pairs for evaluation.
    """

    def __init__(self, cfg):
        """
        Initialize PIROPO dataset.

        Args:
            cfg: YACS config object with DATASETS.PIROPO settings
        """
        self.cfg = cfg
        self.root_dir = Path(cfg.DATASETS.PIROPO.ROOT_DIR)
        self.annotations_dir = Path(cfg.DATASETS.PIROPO.STANDARD_ANNOTATIONS_DIR)
        self.rooms = list(cfg.DATASETS.PIROPO.ROOMS)
        self.image_ext = cfg.DATASETS.PIROPO.IMAGE_EXT
        self.fisheye_center = (
            cfg.DATASETS.PIROPO.FISHEYE_CENTER_X,
            cfg.DATASETS.PIROPO.FISHEYE_CENTER_Y
        )
        self.verbose = cfg.VERBOSE

        self._build_dataset()

    def _build_dataset(self):
        """
        Build flat list of (image_path, annotation_path) pairs.

        Walks standard-annotations/{Room}/{Camera}/{Camera}_{Seq}/ and
        finds the corresponding frame at {root}/{Room}/{Camera}/{Camera}_{Seq}/.
        """
        self.data = []

        stats = {}  # room -> camera -> count

        for room in self.rooms:
            room_ann_dir = self.annotations_dir / room
            if not room_ann_dir.exists():
                if self.verbose:
                    print(f"WARNING: Annotations directory not found: {room_ann_dir}")
                continue

            stats[room] = {}

            for camera_dir in sorted(room_ann_dir.iterdir()):
                if not camera_dir.is_dir():
                    continue
                camera = camera_dir.name

                stats[room][camera] = 0

                for seq_dir in sorted(camera_dir.iterdir()):
                    if not seq_dir.is_dir():
                        continue

                    for ann_file in sorted(seq_dir.glob("*.json")):
                        # Relative path from annotations root:
                        # e.g., Room_A/omni_1A/omni_1A_training/frame.json
                        rel_path = ann_file.relative_to(self.annotations_dir)

                        # Corresponding frame path:
                        # root_dir / Room_A/omni_1A/omni_1A_training/frame.jpg
                        frame_path = self.root_dir / rel_path.with_suffix(self.image_ext)

                        if not frame_path.exists():
                            if self.verbose:
                                print(f"WARNING: Frame not found: {frame_path}")
                            continue

                        self.data.append({
                            "room": room,
                            "camera": camera,
                            "sequence": seq_dir.name,
                            "image_path": str(frame_path),
                            "annotation_path": str(ann_file),
                            "image_name": ann_file.stem + self.image_ext
                        })
                        stats[room][camera] += 1

        if self.verbose:
            print(f"PIROPO Dataset: Found {len(self.data)} image-annotation pairs")
            for room, cameras in stats.items():
                room_total = sum(cameras.values())
                print(f"  {room}: {room_total} images")
                for camera, count in cameras.items():
                    print(f"    {camera}: {count} images")
        else:
            print(f"PIROPO Dataset: {len(self.data)} images across {len(self.rooms)} room(s)")

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
                - room:             str  (e.g., "Room_A")
                - camera:           str  (e.g., "omni_1A")
                - sequence:         str  (e.g., "omni_1A_training")
                - image_name:       str  (e.g., "frame.jpg")
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
            "room": item["room"],
            "camera": item["camera"],
            "sequence": item["sequence"],
            "image_name": item["image_name"]
        }

    def get_room_data(self, room: str) -> List[Dict]:
        """Return all data items for a specific room."""
        return [d for d in self.data if d["room"] == room]

    def get_camera_data(self, room: str, camera: str) -> List[Dict]:
        """Return all data items for a specific room and camera."""
        return [d for d in self.data if d["room"] == room and d["camera"] == camera]
