"""
Extract frames from BOMNI dataset videos.

This script extracts frames from the top camera videos (top-0, top-1, top-2, top-3)
and saves them with 4-digit serial numbers (0001.jpg, 0002.jpg, etc.) as expected
by the rotated bbox annotations.
"""

import cv2
import os
from pathlib import Path


def extract_frames_from_video(video_path, output_dir, start_index=1):
    """
    Extract all frames from a video file.

    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        start_index: Starting index for frame numbering (default: 1)

    Returns:
        Number of frames extracted
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frame_count = 0
    frame_index = start_index

    print(f"Extracting frames from {video_path.name}...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame with 4-digit number (0001.jpg, 0002.jpg, etc.)
        frame_filename = f"{frame_index:04d}.jpg"
        frame_path = output_dir / frame_filename
        cv2.imwrite(str(frame_path), frame)

        frame_count += 1
        frame_index += 1

        if frame_count % 100 == 0:
            print(f"  Extracted {frame_count} frames...")

    cap.release()
    print(f"  Total: {frame_count} frames extracted to {output_dir}")

    return frame_count


def main():
    """Extract frames from all BOMNI top camera videos."""

    # Paths
    base_dir = Path(__file__).parent.parent
    video_dir = base_dir / "datasets" / "all-datasets" / "bomni-5841" / "scenario1"
    output_base = base_dir / "datasets" / "all-datasets" / "bomni-5841" / "frames" / "scenario1"

    # Videos to process (only those with rotated annotations)
    videos_to_process = ["top-0", "top-1", "top-2", "top-3"]

    print("=" * 60)
    print("BOMNI Frame Extraction")
    print("=" * 60)

    total_frames = 0

    for video_name in videos_to_process:
        video_path = video_dir / f"{video_name}.mp4"
        output_dir = output_base / video_name

        if not video_path.exists():
            print(f"WARNING: Video not found: {video_path}")
            continue

        try:
            frame_count = extract_frames_from_video(video_path, output_dir)
            total_frames += frame_count
        except Exception as e:
            print(f"ERROR processing {video_name}: {e}")

    print("=" * 60)
    print(f"Extraction complete! Total frames: {total_frames}")
    print("=" * 60)


if __name__ == "__main__":
    main()
