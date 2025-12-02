"""
Visualize BOMNI dataset rotated bounding box annotations.

This script loads the BOMNI dataset and visualizes all rotated bbox annotations
on the fisheye images. Results are saved to evaluation/results/bomni/annotation_visualization/.

Usage:
    python -m evaluation.visualize_bomni_annotations [--max-images N]

Options:
    --max-images N    Limit visualization to N images per sequence (default: all)
"""

import argparse
from evaluation.config import get_cfg
from evaluation.bomni_dataset import BOMNIDataset


def main():
    """Visualize BOMNI annotations."""
    parser = argparse.ArgumentParser(description="Visualize BOMNI rotated bbox annotations")
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to visualize per sequence (default: all)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for visualizations (default: from config)"
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="Sequences to visualize (default: all enabled sequences)"
    )
    parser.add_argument(
        "--no-rotation",
        action="store_true",
        help="Draw axis-aligned boxes instead of rotated boxes"
    )
    parser.add_argument(
        "--show-center",
        action="store_true",
        help="Draw image center and radial lines"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information"
    )

    args = parser.parse_args()

    # Load configuration
    cfg = get_cfg()

    # Apply command-line overrides
    if args.sequences is not None:
        cfg.DATASETS.BOMNI.SEQUENCES = args.sequences

    if args.no_rotation:
        cfg.VISUALIZATION.DRAW_ROTATED = False

    if args.show_center:
        cfg.VISUALIZATION.DRAW_CENTER_AND_RADIAL = True

    if args.verbose:
        cfg.VERBOSE = True

    if args.max_images is not None:
        cfg.VISUALIZATION.MAX_IMAGES_PER_SEQUENCE = args.max_images

    # Freeze config
    cfg.freeze()

    # Create dataset
    print("=" * 60)
    print("BOMNI Dataset Annotation Visualization")
    print("=" * 60)

    dataset = BOMNIDataset(cfg)

    print(f"\nDataset loaded: {len(dataset)} image-annotation pairs")

    # Run visualization
    dataset.visualize_annotations(
        output_dir=args.output_dir,
        max_images=args.max_images
    )

    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
