"""
Cleanup incorrect BOMNI annotations based on manual review.

This script:
1. Scans remaining visualization images (only correct annotations kept)
2. Extracts frame IDs from visualization filenames
3. Deletes XML annotations that don't match remaining visualizations
4. Deletes frame images that don't match remaining visualizations
"""
import os
from pathlib import Path

def cleanup_annotations(viz_dir, annotations_dir, frames_dir, sequences):
    """
    Delete XML annotations and frames that don't have corresponding visualizations.

    Args:
        viz_dir: Directory with visualization images (user deleted bad ones)
        annotations_dir: Directory with XML annotation files
        frames_dir: Directory with extracted frame images
        sequences: List of sequence names
    """
    total_xml_deleted = 0
    total_frames_deleted = 0
    total_kept = 0

    for sequence in sequences:
        print(f"\n{sequence}:")

        # Get list of remaining visualization files
        viz_sequence_dir = viz_dir / sequence
        if not viz_sequence_dir.exists():
            print(f"  WARNING: Visualization directory not found: {viz_sequence_dir}")
            continue

        # Extract frame IDs from remaining visualizations
        viz_files = list(viz_sequence_dir.glob("annotated_*.jpg"))
        kept_ids = set()
        for viz_file in viz_files:
            # Extract frame ID: "annotated_0001.jpg" -> "0001"
            frame_id = viz_file.stem.replace("annotated_", "")
            kept_ids.add(frame_id)

        print(f"  Kept annotations: {len(kept_ids)}")

        # Delete XML annotations not in kept list
        xml_sequence_dir = annotations_dir / sequence
        if not xml_sequence_dir.exists():
            print(f"  WARNING: XML directory not found: {xml_sequence_dir}")
            continue

        xml_files = list(xml_sequence_dir.glob("*.xml"))
        xml_deleted = 0
        for xml_file in xml_files:
            frame_id = xml_file.stem
            if frame_id not in kept_ids:
                xml_file.unlink()
                xml_deleted += 1

        print(f"  XML files deleted: {xml_deleted}")
        total_xml_deleted += xml_deleted

        # Delete frame images not in kept list
        frames_sequence_dir = frames_dir / sequence
        if not frames_sequence_dir.exists():
            print(f"  WARNING: Frames directory not found: {frames_sequence_dir}")
            continue

        frame_files = list(frames_sequence_dir.glob("*.jpg"))
        frames_deleted = 0
        for frame_file in frame_files:
            frame_id = frame_file.stem
            if frame_id not in kept_ids:
                frame_file.unlink()
                frames_deleted += 1

        print(f"  Frame images deleted: {frames_deleted}")
        total_frames_deleted += frames_deleted
        total_kept += len(kept_ids)

    print(f"\n{'='*60}")
    print(f"TOTAL:")
    print(f"  Annotations kept: {total_kept}")
    print(f"  XML files deleted: {total_xml_deleted}")
    print(f"  Frame images deleted: {total_frames_deleted}")
    print(f"{'='*60}")


def main():
    """Main cleanup script."""
    base_dir = Path(__file__).parent.parent

    # Paths
    viz_dir = base_dir / "evaluation" / "results" / "bomni" / "annotation_visualization"
    annotations_dir = base_dir / "datasets" / "all-datasets" / "BOMNI-corrected" / "Rotated-annotations" / "scenario1"
    frames_dir = base_dir / "datasets" / "all-datasets" / "bomni-5841" / "frames" / "scenario1"
    sequences = ["top-0", "top-1", "top-2", "top-3"]

    print("="*60)
    print("BOMNI Incorrect Annotation Cleanup")
    print("="*60)
    print(f"Visualization directory: {viz_dir}")
    print(f"XML annotations directory: {annotations_dir}")
    print(f"Frames directory: {frames_dir}")
    print(f"Sequences: {sequences}")
    print("="*60)

    # Confirm before deletion
    response = input("\nThis will DELETE XML and frame files. Continue? (yes/no): ")
    if response.lower() != 'yes':
        print("Cleanup cancelled.")
        return

    cleanup_annotations(viz_dir, annotations_dir, frames_dir, sequences)
    print("\nCleanup complete!")


if __name__ == "__main__":
    main()
