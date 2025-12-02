"""
Cleanup BOMNI extracted frames - keep only frames with corresponding annotations.
"""
import os
from pathlib import Path

def cleanup_frames(frames_dir, annotations_dir, sequences):
    """
    Delete frames that don't have corresponding annotation files.

    Args:
        frames_dir: Directory containing extracted frames
        annotations_dir: Directory containing annotation XML files
        sequences: List of sequence names (e.g., ['top-0', 'top-1', ...])
    """
    total_deleted = 0
    total_kept = 0

    for sequence in sequences:
        sequence_frames_dir = frames_dir / sequence
        sequence_annotations_dir = annotations_dir / sequence

        if not sequence_frames_dir.exists():
            print(f"WARNING: Frames directory not found: {sequence_frames_dir}")
            continue

        if not sequence_annotations_dir.exists():
            print(f"WARNING: Annotations directory not found: {sequence_annotations_dir}")
            continue

        # Get all annotation files (without extension)
        annotation_files = list(sequence_annotations_dir.glob("*.xml"))
        annotation_stems = {ann_file.stem for ann_file in annotation_files}

        print(f"\n{sequence}:")
        print(f"  Annotations found: {len(annotation_stems)}")

        # Get all frame files
        frame_files = list(sequence_frames_dir.glob("*.jpg"))
        print(f"  Frames found: {len(frame_files)}")

        deleted_count = 0
        kept_count = 0

        # Delete frames without corresponding annotations
        for frame_file in frame_files:
            frame_stem = frame_file.stem
            if frame_stem not in annotation_stems:
                frame_file.unlink()  # Delete the file
                deleted_count += 1
            else:
                kept_count += 1

        print(f"  Frames deleted: {deleted_count}")
        print(f"  Frames kept: {kept_count}")

        total_deleted += deleted_count
        total_kept += kept_count

    print(f"\n{'='*60}")
    print(f"TOTAL:")
    print(f"  Frames deleted: {total_deleted}")
    print(f"  Frames kept: {total_kept}")
    print(f"{'='*60}")

def main():
    # Setup paths
    base_dir = Path(__file__).parent.parent
    frames_dir = base_dir / "datasets" / "all-datasets" / "bomni-5841" / "frames" / "scenario1"
    annotations_dir = base_dir / "datasets" / "all-datasets" / "omnidet-rotinv-master" / "omnidet-rotinv-master" / "rotate" / "bomni" / "rotate" / "scenario1"
    sequences = ["top-0", "top-1", "top-2", "top-3"]

    print("BOMNI Frames Cleanup Script")
    print("="*60)
    print(f"Frames directory: {frames_dir}")
    print(f"Annotations directory: {annotations_dir}")
    print(f"Sequences: {sequences}")
    print("="*60)

    # Confirm before deletion
    response = input("\nThis will DELETE frames without annotations. Continue? (yes/no): ")
    if response.lower() != 'yes':
        print("Cleanup cancelled.")
        return

    cleanup_frames(frames_dir, annotations_dir, sequences)
    print("\nCleanup complete!")

if __name__ == "__main__":
    main()
