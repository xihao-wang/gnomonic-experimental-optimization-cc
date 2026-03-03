"""
Visualize annotations for multiple datasets.

This script generates visualization images with rotated bounding boxes
for specified datasets. Configure which datasets to visualize by editing
the DATASETS_TO_VISUALIZE list below.

All configuration is set in evaluation/config.py:
- VISUALIZATION.MAX_IMAGES_PER_SEQUENCE (set to "all" for all images)
- Dataset-specific paths and settings

Usage:
    python -m evaluation.visualize_datasets
"""

from evaluation.lib.config import get_cfg
from datasets.lib.bomni_manager import BOMNIManager
from datasets.lib.piropo_manager import PIROPOManager


# ============================================================================
# CONFIGURATION: Which datasets to visualize
# ============================================================================

DATASETS_TO_VISUALIZE = [
    "bomni",
    # "piropo",  # Uncomment when PIROPO is implemented
]


# ============================================================================
# Main Script
# ============================================================================

def get_manager_for_dataset(dataset_name: str, cfg):
    """
    Get the appropriate dataset manager for a given dataset name.

    Args:
        dataset_name: Dataset identifier (e.g., "bomni", "piropo")
        cfg: YACS config object

    Returns:
        Dataset manager instance

    Raises:
        ValueError: If dataset name is not recognized
    """
    if dataset_name == "bomni":
        return BOMNIManager(cfg)
    elif dataset_name == "piropo":
        return PIROPOManager(cfg)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def visualize_dataset(dataset_name: str, cfg):
    """
    Visualize annotations for a specific dataset.

    Args:
        dataset_name: Dataset identifier (e.g., "bomni", "piropo")
        cfg: YACS config object
    """
    print("\n" + "="*70)
    print(f"Visualizing {dataset_name.upper()} Dataset")
    print("="*70)

    try:
        # Get appropriate manager
        manager = get_manager_for_dataset(dataset_name, cfg)

        # Get dataset-specific config
        if dataset_name == "bomni":
            dataset_cfg = cfg.DATASETS.BOMNI
            annotations_dir = dataset_cfg.STANDARD_ANNOTATIONS_DIR
            frames_dir = dataset_cfg.FRAMES_DIR
            sequences = dataset_cfg.SEQUENCES
            fisheye_center = (dataset_cfg.FISHEYE_CENTER_X, dataset_cfg.FISHEYE_CENTER_Y)
        elif dataset_name == "piropo":
            dataset_cfg = cfg.DATASETS.PIROPO
            annotations_dir = dataset_cfg.STANDARD_ANNOTATIONS_DIR
            frames_dir = dataset_cfg.FRAMES_DIR
            sequences = dataset_cfg.SEQUENCES
            fisheye_center = (dataset_cfg.FISHEYE_CENTER_X, dataset_cfg.FISHEYE_CENTER_Y)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        print(f"\nAnnotation format: {dataset_cfg.ANNOTATION_FORMAT}")
        print(f"Sequences: {sequences}")
        print(f"Max images per sequence: {cfg.VISUALIZATION.MAX_IMAGES_PER_SEQUENCE}")

        # Visualize annotations
        manager.visualize_annotations(
            annotations_dir=annotations_dir,
            frames_dir=frames_dir,
            sequences=sequences,
            max_images=cfg.VISUALIZATION.MAX_IMAGES_PER_SEQUENCE,
            fisheye_center=fisheye_center
        )

        print(f"\n[SUCCESS] {dataset_name.upper()} visualization complete!")

    except NotImplementedError as e:
        print(f"\n[SKIP] {dataset_name.upper()} not implemented yet: {e}")
    except Exception as e:
        print(f"\n[ERROR] Failed to visualize {dataset_name.upper()}: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Visualize all specified datasets."""
    print("="*70)
    print("Multi-Dataset Annotation Visualization")
    print("="*70)

    # Load config
    cfg = get_cfg()
    cfg.VERBOSE = True

    print(f"\nDatasets to visualize: {DATASETS_TO_VISUALIZE}")

    # Visualize each dataset
    for dataset_name in DATASETS_TO_VISUALIZE:
        visualize_dataset(dataset_name, cfg)

    print("\n" + "="*70)
    print("[DONE] All visualizations complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
