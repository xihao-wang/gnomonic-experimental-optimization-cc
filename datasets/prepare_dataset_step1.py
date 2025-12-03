"""
Dataset Preparation - STEP 1 (Automated)

This script performs Steps 1-4 of the dataset preparation workflow:
  1. Extract frames from videos
  2. Cleanup unannotated frames
  3. Convert annotations to standard JSON format
  4. Generate visualizations for manual quality review

After running this script, the user MUST manually review the visualizations
and DELETE any images with incorrect annotations. Then run prepare_dataset_step2.py.

IMPORTANT: User should ONLY delete visualization images, NEVER annotation files!

Configuration:
  - Edit evaluation/config.py to specify:
    - DATASETS.{DATASET}.PREPARATION.VIDEO_DIR (input videos)
    - DATASETS.{DATASET}.PREPARATION.RAW_ANNOTATIONS_DIR (input annotations)
    - DATASETS.{DATASET}.PREPARATION.RAW_ANNOTATION_FORMAT (e.g., "tamura")
    - DATASETS.{DATASET}.PREPARATION.TARGET_NAME (output folder name)
    - DATASETS.{DATASET}.PREPARATION.SEQUENCES (sequences to process)
    - DATASETS.{DATASET}.FISHEYE_CENTER_X/Y (for angle calculation)

  - Edit DATASET_NAME variable below to select which dataset to prepare

Outputs (under datasets/all-datasets/{TARGET_NAME}/):
  - frames/scenario1/{sequence}/       - Extracted frames (all)
  - frames/scenario1/{sequence}/       - After cleanup (annotated only)
  - Standard-annotations/scenario1/{sequence}/  - Converted JSON annotations
  - visualizations/{sequence}/         - Images for manual review

Usage:
  1. Edit evaluation/config.py to set PREPARATION.TARGET_NAME
  2. python datasets/prepare_dataset_step1.py
  3. Manually review visualizations and delete incorrect ones
  4. Run datasets/prepare_dataset_step2.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.config import get_cfg
from datasets.bomni_manager import BOMNIManager
from datasets.piropo_manager import PIROPOManager


# ============================================================================
# CONFIGURATION: Select dataset to prepare
# ============================================================================

DATASET_NAME = "bomni"  # Options: "bomni", "piropo"


# ============================================================================
# Main Script
# ============================================================================

def get_manager_for_dataset(dataset_name: str, cfg):
    """Get the appropriate dataset manager."""
    if dataset_name == "bomni":
        return BOMNIManager(cfg)
    elif dataset_name == "piropo":
        return PIROPOManager(cfg)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def main():
    """Run Steps 1-4 of dataset preparation."""
    print("=" * 80)
    print("DATASET PREPARATION - STEP 1 (Automated)")
    print("=" * 80)
    print(f"\nDataset: {DATASET_NAME.upper()}")
    print("\nThis script will:")
    print("  1. Extract frames from videos")
    print("  2. Cleanup unannotated frames")
    print("  3. Convert annotations to standard JSON format")
    print("  4. Generate visualizations for manual quality review")
    print("\n" + "=" * 80)

    # Load config
    cfg = get_cfg()
    cfg.VERBOSE = True

    # Get dataset-specific config
    if DATASET_NAME == "bomni":
        prep_cfg = cfg.DATASETS.BOMNI.PREPARATION
        fisheye_center = (cfg.DATASETS.BOMNI.FISHEYE_CENTER_X, cfg.DATASETS.BOMNI.FISHEYE_CENTER_Y)
    elif DATASET_NAME == "piropo":
        prep_cfg = cfg.DATASETS.PIROPO.PREPARATION
        fisheye_center = (cfg.DATASETS.PIROPO.FISHEYE_CENTER_X, cfg.DATASETS.PIROPO.FISHEYE_CENTER_Y)
    else:
        raise ValueError(f"Unknown dataset: {DATASET_NAME}")

    # Build output paths
    base_dir = Path("datasets/all-datasets") / prep_cfg.TARGET_NAME
    frames_dir = base_dir / "frames" / "scenario1"
    annotations_dir = base_dir / "Standard-annotations" / "scenario1"
    visualizations_dir = base_dir / "visualizations"

    print(f"\nConfiguration:")
    print(f"  Input videos: {prep_cfg.VIDEO_DIR}")
    print(f"  Input annotations: {prep_cfg.RAW_ANNOTATIONS_DIR}")
    print(f"  Input format: {prep_cfg.RAW_ANNOTATION_FORMAT}")
    print(f"  Target directory: {base_dir}")
    print(f"  Sequences: {prep_cfg.SEQUENCES}")
    print(f"  Fisheye center: {fisheye_center}")

    # Create manager
    manager = get_manager_for_dataset(DATASET_NAME, cfg)

    # Step 1: Extract frames
    print("\n" + "=" * 80)
    print("STEP 1: Extracting frames from videos")
    print("=" * 80)
    manager.extract_frames(
        video_dir=prep_cfg.VIDEO_DIR,
        output_dir=str(frames_dir),
        sequences=prep_cfg.SEQUENCES
    )

    # Step 2: Cleanup unannotated frames
    print("\n" + "=" * 80)
    print("STEP 2: Removing unannotated frames")
    print("=" * 80)
    manager.cleanup_unannotated_frames(
        frames_dir=str(frames_dir),
        annotations_dir=prep_cfg.RAW_ANNOTATIONS_DIR,
        sequences=prep_cfg.SEQUENCES
    )

    # Step 3: Convert to standard JSON format
    print("\n" + "=" * 80)
    print("STEP 3: Converting annotations to standard JSON format")
    print("=" * 80)
    manager.convert_to_standard_format(
        input_format=prep_cfg.RAW_ANNOTATION_FORMAT,
        input_dir=prep_cfg.RAW_ANNOTATIONS_DIR,
        output_dir=str(annotations_dir),
        sequences=prep_cfg.SEQUENCES
    )

    # Step 4: Generate visualizations for manual review
    print("\n" + "=" * 80)
    print("STEP 4: Generating visualizations for manual quality review")
    print("=" * 80)
    manager.visualize_annotations(
        annotations_dir=str(annotations_dir),
        frames_dir=str(frames_dir),
        sequences=prep_cfg.SEQUENCES,
        max_images="all",
        output_dir=str(visualizations_dir),
        fisheye_center=fisheye_center
    )

    # Final instructions
    print("\n" + "=" * 80)
    print("STEP 1 COMPLETE - MANUAL REVIEW REQUIRED")
    print("=" * 80)
    print(f"\nVisualization images saved to:")
    print(f"  {visualizations_dir}/{DATASET_NAME}/annotation_visualization/")
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("1. Open the visualizations directory above")
    print("2. Review each image and DELETE any with incorrect annotations:")
    print("   - Wrong bbox position")
    print("   - Missing detection")
    print("   - False positive")
    print("   - Inaccurate bbox size/orientation")
    print("3. Keep ONLY images with correct annotations")
    print("4. Run: python datasets/prepare_dataset_step2.py")
    print("\nIMPORTANT: Only delete VISUALIZATION images, never annotation files!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as e:
        print(f"\n[ERROR] Dataset not implemented: {e}")
        print("Please implement the convert_to_standard_format() method in the dataset manager.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Preparation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
