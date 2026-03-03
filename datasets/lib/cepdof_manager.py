"""
CEPDOF Dataset Manager

Handles conversion of CEPDOF annotations (COCO-like JSON with rotated bboxes)
to our standard per-frame JSON format.

CEPDOF annotation format:
  bbox: [cx, cy, w, h, d]
    - cx, cy: center coordinates
    - w, h:   width and height (constrained: w <= h)
    - d:      clockwise rotation from vertical axis (up), degrees, range [-90, +90]

Standard format output:
  {center_x, center_y, width, height, angle, class_name}

The conversion is direct (no trigonometric derivation needed). The w <= h constraint
and angle range are normalized once here so the pipeline never needs to handle it.

Author: Yassir Zardoua <y.zardoua@caplogy.com | yassirzardoua@gmail.com>
"""

import json
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

from datasets.lib.base_manager import BaseDatasetManager


# Mapping from annotation file stem to sequence folder name
# (handles Edge_Cases.json -> Edge_cases/ naming mismatch)
_ANN_FILE_TO_FOLDER = {
    "All_off": "All_off",
    "Edge_Cases": "Edge_cases",
    "High_activity": "High_activity",
    "IRfilter": "IRfilter",
    "IRill": "IRill",
    "Lunch1": "Lunch1",
    "Lunch2": "Lunch2",
    "Lunch3": "Lunch3",
}


def _normalize_bbox(cx, cy, w, h, d):
    """
    Normalize CEPDOF bbox to enforce w <= h and d in [-90, +90].

    CEPDOF already enforces this constraint, but we normalize defensively
    during conversion so the standard format is always consistent.

    Returns:
        (center_x, center_y, width, height, angle)
    """
    if w > h:
        w, h = h, w
        # Rotating the box 90 degrees swaps width/height
        d = d - 90.0 if d >= 0 else d + 90.0
    # Clamp to [-90, +90]
    d = max(-90.0, min(90.0, d))
    return cx, cy, w, h, d


class CEPDOFManager(BaseDatasetManager):
    """
    Dataset manager for CEPDOF (Challenging Events for Person Detection
    from Overhead Fisheye images).

    Converts COCO-like JSON with [cx, cy, w, h, d] bboxes to our standard
    per-frame JSON format.
    """

    def get_dataset_name(self) -> str:
        return "cepdof"

    def extract_frames(self, **kwargs):
        """No-op: CEPDOF frames are pre-extracted (one folder per sequence)."""
        print("Skipping frame extraction (CEPDOF frames already available)")

    def cleanup_unannotated_frames(
        self,
        frames_dir: str,
        annotations_dir: str,
        source_dir: Optional[str] = None,
        sequences: Optional[List[str]] = None,
    ):
        """
        Copy only annotated frames from source to target directory.

        Reads the CEPDOF JSON annotation files to identify which frames have
        at least one annotation, then copies only those frames.

        Args:
            frames_dir:       Target directory (CEPDOF-step1/)
            annotations_dir:  CEPDOF annotation JSON files directory
            source_dir:       Source directory containing all frames
            sequences:        Sequences to process (default: cfg.CEPDOF.SEQUENCES)
        """
        import shutil

        dataset_cfg = self.cfg.CEPDOF
        source_dir = Path(source_dir) if source_dir else Path(dataset_cfg.FRAMES_SOURCE_DIR)
        frames_dir = Path(frames_dir)

        if sequences is None:
            sequences = dataset_cfg.SEQUENCES

        folder_to_ann = {v: k for k, v in _ANN_FILE_TO_FOLDER.items()}

        print("=" * 80)
        print("CEPDOF Annotated Frame Extraction")
        print("=" * 80)
        print(f"\nSource (all frames): {source_dir}")
        print(f"Target (annotated only): {frames_dir}")
        print(f"Annotations: {annotations_dir}")
        print("\n" + "-" * 80)

        total_copied = 0
        total_skipped = 0

        for sequence in sequences:
            ann_stem = folder_to_ann.get(sequence)
            if ann_stem is None:
                continue

            ann_file = Path(annotations_dir) / f"{ann_stem}.json"
            if not ann_file.exists():
                matches = [f for f in Path(annotations_dir).glob("*.json") if f.stem.lower() == ann_stem.lower()]
                if matches:
                    ann_file = matches[0]
                else:
                    print(f"  [WARNING] Annotation file not found for '{sequence}'")
                    continue

            with open(ann_file, "r") as f:
                data = json.load(f)

            # Collect frame filenames that have at least one annotation
            annotated_ids = set(a["image_id"] for a in data["annotations"])

            src_seq_dir = source_dir / sequence
            dst_seq_dir = frames_dir / sequence
            dst_seq_dir.mkdir(parents=True, exist_ok=True)

            copied = 0
            skipped = 0
            for image_id in annotated_ids:
                src_file = src_seq_dir / f"{image_id}.jpg"
                dst_file = dst_seq_dir / f"{image_id}.jpg"
                if src_file.exists():
                    shutil.copy2(str(src_file), str(dst_file))
                    copied += 1
                else:
                    skipped += 1

            print(f"  {sequence}: copied {copied}, skipped {skipped} (missing)")
            total_copied += copied
            total_skipped += skipped

        print("\n" + "-" * 80)
        print(f"\nTotal copied: {total_copied} frames")
        print(f"Total skipped: {total_skipped} frames (not found in source)")
        print(f"\n[DONE] Annotated frames extracted!")
        print(f"Target: {frames_dir}")
        print("=" * 80)

    def convert_to_standard_format(
        self,
        input_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        sequences: Optional[List[str]] = None,
    ):
        """
        Convert CEPDOF COCO-like JSON annotations to standard per-frame JSON.

        Reads one JSON file per sequence, writes one JSON file per annotated frame.
        Frames with no annotations (no persons visible) are skipped.

        Args:
            input_dir:  Directory containing CEPDOF annotation JSON files.
                        Default: cfg.CEPDOF.RAW_ANNOTATIONS_DIR
            output_dir: Root output directory for standard-annotations.
                        Default: datasets/all-datasets/{TARGET_NAME}/standard-annotations
            sequences:  List of sequence folder names to process.
                        Default: cfg.CEPDOF.SEQUENCES
        """
        dataset_cfg = self.cfg.CEPDOF

        if input_dir is None:
            input_dir = Path(dataset_cfg.RAW_ANNOTATIONS_DIR)
        else:
            input_dir = Path(input_dir)

        if output_dir is None:
            output_dir = Path("datasets/all-datasets") / dataset_cfg.TARGET_NAME / "standard-annotations"
        else:
            output_dir = Path(output_dir)

        if sequences is None:
            sequences = dataset_cfg.SEQUENCES

        # Build reverse mapping: folder name -> annotation file stem
        folder_to_ann = {v: k for k, v in _ANN_FILE_TO_FOLDER.items()}

        print("=" * 80)
        print("CEPDOF Annotation Format Conversion")
        print("=" * 80)
        print(f"\nInput directory:  {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Sequences: {sequences}")
        print("\n" + "-" * 80)

        total_files = 0
        total_annotations = 0

        for sequence in sequences:
            ann_stem = folder_to_ann.get(sequence)
            if ann_stem is None:
                print(f"  [WARNING] No annotation file mapping for sequence '{sequence}', skipping")
                continue

            # Try to find the annotation file (exact then case-insensitive fallback)
            ann_file = input_dir / f"{ann_stem}.json"
            if not ann_file.exists():
                # Try all json files with case-insensitive match
                matches = [f for f in input_dir.glob("*.json") if f.stem.lower() == ann_stem.lower()]
                if matches:
                    ann_file = matches[0]
                else:
                    print(f"  [WARNING] Annotation file not found for '{sequence}': {ann_file}")
                    continue

            with open(ann_file, "r") as f:
                data = json.load(f)

            # Build image metadata lookup: image_id -> {width, height}
            image_meta = {img["id"]: img for img in data["images"]}

            # Group annotations by image_id
            annotations_by_image = {}
            for ann in data["annotations"]:
                iid = ann["image_id"]
                if iid not in annotations_by_image:
                    annotations_by_image[iid] = []
                annotations_by_image[iid].append(ann)

            # Output folder for this sequence
            seq_output_dir = output_dir / sequence
            seq_output_dir.mkdir(parents=True, exist_ok=True)

            seq_files = 0
            seq_annotations = 0

            for image_id, ann_list in tqdm(annotations_by_image.items(), desc=f"  {sequence}"):
                img_meta = image_meta.get(image_id, {})
                img_w = img_meta.get("width", 2048)
                img_h = img_meta.get("height", 2048)

                standard_annotations = []
                for ann in ann_list:
                    cx, cy, w, h, d = ann["bbox"]
                    cx, cy, w, h, d = _normalize_bbox(cx, cy, w, h, d)
                    standard_annotations.append({
                        "center_x": round(cx, 4),
                        "center_y": round(cy, 4),
                        "width": round(w, 4),
                        "height": round(h, 4),
                        "angle": round(d, 4),
                        "class_name": "person"
                    })

                # image_id is the frame stem (e.g., "Lunch1_000067")
                output_file = seq_output_dir / f"{image_id}.json"
                with open(output_file, "w") as f:
                    json.dump(standard_annotations, f, indent=2)

                seq_files += 1
                seq_annotations += len(standard_annotations)

            print(f"  {sequence}: {seq_files} files ({seq_annotations} annotations)")
            total_files += seq_files
            total_annotations += seq_annotations

        print("\n" + "-" * 80)
        print(f"\n[DONE] Conversion complete!")
        print(f"  Total files: {total_files}")
        print(f"  Total annotations: {total_annotations}")
        print(f"\nOutput directory: {output_dir}")
        print("=" * 80)

    def visualize_annotations(
        self,
        annotations_dir: str,
        frames_dir: str,
        output_dir: Optional[str] = None,
        sequences: Optional[List[str]] = None,
        max_images: Optional[int] = None,
        spread: bool = True,
        fisheye_center: Optional[tuple] = None,
        image_ext: str = ".jpg"
    ):
        """
        Visualize CEPDOF annotations using the base class implementation.

        CEPDOF has a flat structure ({sequence}/{frame}.jpg) matching the base class
        expectations, so we delegate directly to the base class method.
        """
        if sequences is None:
            sequences = self.cfg.CEPDOF.SEQUENCES

        super().visualize_annotations(
            annotations_dir=annotations_dir,
            frames_dir=frames_dir,
            output_dir=output_dir,
            sequences=sequences,
            max_images=max_images,
            spread=spread,
            fisheye_center=fisheye_center,
            image_ext=image_ext
        )
