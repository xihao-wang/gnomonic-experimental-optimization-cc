"""
Test script to validate dataset manager implementation.

This script tests:
1. BOMNI manager can load and validate standard annotations
2. BOMNI dataset loader works with standard format
3. Visualization with standard annotations
4. Statistics and validation

Author: Generated for gnomonic projection pedestrian detection project
Date: 2025-12-03
"""

from evaluation.config import get_cfg
from datasets.bomni_manager import BOMNIManager
from evaluation.bomni_dataset import BOMNIDataset


def test_bomni_manager():
    """Test BOMNI dataset manager operations."""
    print("="*70)
    print("Testing BOMNI Dataset Manager")
    print("="*70)

    # Load config
    cfg = get_cfg()
    cfg.VERBOSE = True

    # Create manager
    manager = BOMNIManager(cfg)

    print(f"\nDataset name: {manager.get_dataset_name()}")

    # Test get_statistics
    print("\n" + "-"*70)
    print("Getting dataset statistics...")
    print("-"*70)
    stats = manager.get_statistics(
        annotations_dir=cfg.DATASETS.BOMNI.STANDARD_ANNOTATIONS_DIR,
        sequences=cfg.DATASETS.BOMNI.SEQUENCES
    )

    print(f"\nDataset: {stats['dataset']}")
    print(f"Total images: {stats['total_images']}")
    print(f"Total annotations: {stats['total_annotations']}")
    print("\nPer-sequence breakdown:")
    for seq, seq_stats in stats['sequences'].items():
        print(f"  {seq}: {seq_stats['images']} images, {seq_stats['annotations']} annotations")

    # Test validation
    print("\n" + "-"*70)
    print("Validating dataset integrity...")
    print("-"*70)
    validation_results = manager.validate(
        annotations_dir=cfg.DATASETS.BOMNI.STANDARD_ANNOTATIONS_DIR,
        frames_dir=cfg.DATASETS.BOMNI.FRAMES_DIR,
        sequences=cfg.DATASETS.BOMNI.SEQUENCES
    )

    if len(validation_results['errors']) == 0:
        print("\n[SUCCESS] Dataset validation passed!")
    else:
        print(f"\n[WARNING] Found {len(validation_results['errors'])} errors")

    print("\n" + "="*70)


def test_bomni_dataset_loader():
    """Test BOMNI dataset loader (runtime loading)."""
    print("\n" + "="*70)
    print("Testing BOMNI Dataset Loader")
    print("="*70)

    # Load config
    cfg = get_cfg()
    cfg.VERBOSE = True

    print(f"\nAnnotation format: {cfg.DATASETS.BOMNI.ANNOTATION_FORMAT}")

    # Create dataset
    dataset = BOMNIDataset(cfg)

    print(f"\nDataset size: {len(dataset)}")

    # Test loading one item
    if len(dataset) > 0:
        item = dataset[0]
        print(f"\nSample item:")
        print(f"  Image shape: {item['image'].shape}")
        print(f"  Number of annotations: {len(item['annotations'])}")
        print(f"  Sequence: {item['sequence']}")
        if item['annotations']:
            ann = item['annotations'][0]
            print(f"  First annotation:")
            print(f"    - center: ({ann['center_x']:.1f}, {ann['center_y']:.1f})")
            print(f"    - size: {ann['width']:.1f} x {ann['height']:.1f}")
            print(f"    - angle: {ann['angle']:.1f} deg")
            print(f"    - class: {ann['class_name']}")

    print("\n[SUCCESS] Dataset loader working correctly!")
    print("="*70)


def test_visualization():
    """Test visualization with manager."""
    print("\n" + "="*70)
    print("Testing Visualization (2 images per sequence)")
    print("="*70)

    # Load config
    cfg = get_cfg()
    cfg.VERBOSE = False
    cfg.VISUALIZATION.MAX_IMAGES_PER_SEQUENCE = 2

    # Create manager
    manager = BOMNIManager(cfg)

    # Visualize annotations
    manager.visualize_annotations(
        annotations_dir=cfg.DATASETS.BOMNI.STANDARD_ANNOTATIONS_DIR,
        frames_dir=cfg.DATASETS.BOMNI.FRAMES_DIR,
        sequences=cfg.DATASETS.BOMNI.SEQUENCES,
        max_images=2,
        fisheye_center=(cfg.DATASETS.BOMNI.FISHEYE_CENTER_X, cfg.DATASETS.BOMNI.FISHEYE_CENTER_Y)
    )

    print("\n[SUCCESS] Visualization completed!")
    print("="*70)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("DATASET MANAGER VALIDATION TEST")
    print("="*70)

    try:
        # Test manager
        test_bomni_manager()

        # Test dataset loader
        test_bomni_dataset_loader()

        # Test visualization
        test_visualization()

        print("\n" + "="*70)
        print("[SUCCESS] All tests passed!")
        print("="*70)
        print("\nNext steps:")
        print("1. Review visualization outputs in evaluation/results/bomni/annotation_visualization/")
        print("2. If everything looks good, delete standalone scripts:")
        print("   - datasets/extract_bomni_frames.py")
        print("   - datasets/preprocess_bomni_annotations.py")
        print("   - datasets/cleanup_bomni_frames.py")
        print("   - datasets/cleanup_incorrect_annotations.py")
        print("3. Use BOMNIManager for all future dataset preparation")
        print("="*70 + "\n")

    except Exception as e:
        print("\n" + "="*70)
        print("[FAILED] Test failed with error:")
        print("="*70)
        print(f"\n{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "="*70 + "\n")
