"""
Dataset Preparation - STEP 2 (Automated)

This script performs Steps 6-7 of the dataset preparation workflow:
  6. Cleanup incorrect annotations (based on deleted visualizations)
  7. Regenerate final clean visualizations

PREREQUISITE: You must have:
  1. Run prepare_dataset_step1.py
  2. Manually reviewed visualizations and DELETED incorrect ones

This script will:
  - Compare visualization images against annotation files
  - DELETE annotation files and frames that have no corresponding visualization
  - Keep ONLY verified correct data
  - Regenerate clean visualizations for final verification

IMPORTANT: This script determines what to keep based on which visualization
images exist. If you deleted a visualization image, the corresponding annotation
and frame will be removed.

Configuration:
  - Use the SAME dataset and target directory as Step 1
  - Edit DATASET_NAME variable below to select which dataset

Outputs (under datasets/all-datasets/{TARGET_NAME}/):
  - Standard-annotations/scenario1/{sequence}/  - Only verified annotations remain
  - frames/scenario1/{sequence}/                - Only verified frames remain
  - visualizations/{sequence}/                  - Final clean visualizations

Usage:
  1. Ensure you've manually reviewed and deleted incorrect visualizations
  2. Edit DATASET_NAME variable below (must match Step 1)
  3. python datasets/prepare_dataset_step2.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.lib.config import get_cfg
from datasets.lib.bomni_manager import BOMNIManager
from datasets.lib.piropo_manager import PIROPOManager


# ============================================================================
# CONFIGURATION: Select dataset (MUST MATCH STEP 1)
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
    """Run Steps 6-7 of dataset preparation."""
    print("=" * 80)
    print("DATASET PREPARATION - STEP 2 (Automated Cleanup)")
    print("=" * 80)
    print(f"\nDataset: {DATASET_NAME.upper()}")
    print("\nThis script will:")
    print("  6. Remove annotation files with no corresponding visualization")
    print("  7. Regenerate final clean visualizations")
    print("\n" + "=" * 80)

    # Load config
    cfg = get_cfg()
    cfg.VERBOSE = True

    # Get dataset-specific config
    if DATASET_NAME == "bomni":
        dataset_cfg = cfg.BOMNI
    elif DATASET_NAME == "piropo":
        dataset_cfg = cfg.PIROPO
    else:
        raise ValueError(f"Unknown dataset: {DATASET_NAME}")

    fisheye_center = (dataset_cfg.FISHEYE_CENTER_X, dataset_cfg.FISHEYE_CENTER_Y)

    # Build output paths
    base_dir = Path("datasets/all-datasets") / dataset_cfg.TARGET_NAME
    frames_dir = base_dir / "frames" / "scenario1"
    annotations_dir = base_dir / "Standard-annotations" / "scenario1"
    # Visualizations are nested: visualizations/bomni/annotation_visualization/{sequence}/
    visualizations_dir = base_dir / "visualizations" / DATASET_NAME / "annotation_visualization"

    print(f"\nConfiguration:")
    print(f"  Target directory: {base_dir}")
    print(f"  Visualization directory: {visualizations_dir}")
    print(f"  Annotations directory: {annotations_dir}")
    print(f"  Frames directory: {frames_dir}")
    print(f"  Sequences: {dataset_cfg.SEQUENCES}")

    # Verify paths exist
    if not visualizations_dir.exists():
        print(f"\n[ERROR] Visualization directory not found: {visualizations_dir}")
        print("Did you run prepare_dataset_step1.py first?")
        sys.exit(1)

    if not annotations_dir.exists():
        print(f"\n[ERROR] Annotations directory not found: {annotations_dir}")
        print("Did you run prepare_dataset_step1.py first?")
        sys.exit(1)

    # Create manager
    manager = get_manager_for_dataset(DATASET_NAME, cfg)

    # Step 6: Cleanup incorrect annotations based on deleted visualizations
    print("\n" + "=" * 80)
    print("STEP 6: Removing annotations without corresponding visualizations")
    print("=" * 80)
    print("\nThis will DELETE:")
    print("  - Annotation files with no matching visualization image")
    print("  - Frame files with no matching visualization image")
    print("\nThis will KEEP:")
    print("  - Only annotations/frames where visualization image still exists")

    input("\nPress Enter to continue or Ctrl+C to abort...")

    manager.cleanup_incorrect_annotations(
        visualization_dir=str(visualizations_dir),
        annotations_dir=str(annotations_dir),
        frames_dir=str(frames_dir),
        sequences=dataset_cfg.SEQUENCES
    )

    # Step 7: Regenerate final clean visualizations
    print("\n" + "=" * 80)
    print("STEP 7: Regenerating final verified visualizations")
    print("=" * 80)

    # Clear old visualizations first
    import shutil
    viz_base_dir = base_dir / "visualizations"
    if viz_base_dir.exists():
        shutil.rmtree(viz_base_dir)

    manager.visualize_annotations(
        annotations_dir=str(annotations_dir),
        frames_dir=str(frames_dir),
        sequences=dataset_cfg.SEQUENCES,
        max_images="all",
        output_dir=str(viz_base_dir),
        fisheye_center=fisheye_center
    )

    # Final summary
    print("\n" + "=" * 80)
    print("STEP 2 COMPLETE - Dataset preparation finished!")
    print("=" * 80)
    print(f"\nFinal dataset location:")
    print(f"  Annotations: {annotations_dir}")
    print(f"  Frames: {frames_dir}")
    print(f"  Visualizations: {visualizations_dir}")
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("1. Review final visualizations to verify quality")
    print("2. Update evaluation/config.py to use this dataset:")
    print(f"     DATASETS.{DATASET_NAME.upper()}.STANDARD_ANNOTATIONS_DIR = \"{annotations_dir}\"")
    print(f"     DATASETS.{DATASET_NAME.upper()}.FRAMES_DIR = \"{frames_dir}\"")
    print("3. Run evaluation with this dataset")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as e:
        print(f"\n[ERROR] Dataset not implemented: {e}")
        print("Please implement the cleanup_incorrect_annotations() method in the dataset manager.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n[ABORTED] Cleanup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Cleanup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
