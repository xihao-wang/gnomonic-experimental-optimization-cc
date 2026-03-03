"""
Dataset structure registry for evaluation framework.

This module defines the folder structure and metadata for each supported dataset.
The registry allows the evaluation runner to understand how annotations and frames
are organized for different datasets without hardcoding dataset-specific logic.

Author: Generated for gnomonic projection pedestrian detection project
Date: 2025-12-04
"""

DATASET_STRUCTURES = {
    "bomni": {
        "annotation_root": "Standard-annotations",
        "frames_root": "frames",
        "scenarios": ["scenario1"],
        "sequences": {
            "scenario1": ["top-0", "top-1", "top-2", "top-3"]
        },
        "fisheye_center": (320.0, 240.0),
        "image_size": (640, 480)
    },
    "piropo": {
        # 3-level hierarchy: Room / Camera / CameraSeq
        # Frames:      {root}/{Room}/{Camera}/{Camera}_{Seq}/{frame}.jpg
        # Annotations: {root}/standard-annotations/{Room}/{Camera}/{Camera}_{Seq}/{frame}.json
        # Cameras and sequences are auto-discovered by PIROPODataset
        "annotation_root": "standard-annotations",
        "frames_root": "",          # Frames live directly under root (no subfolder)
        "rooms": ["Room_A", "Room_B"],
        "cameras": {
            "Room_A": ["omni_1A", "omni_2A", "omni_3A"],
            "Room_B": ["omni_1B"]
        },
        "fisheye_center": (400.0, 300.0),
        "image_size": (800, 600)
    },
    "cepdof": {
        # Flat 2-level hierarchy: {Sequence}/{frame_id}.jpg
        # Annotations: standard-annotations/{Sequence}/{frame_id}.json
        "annotation_root": "standard-annotations",
        "frames_root": "",  # Sequences sit directly under root
        "sequences": [
            "Lunch1", "Lunch2", "Lunch3", "Edge_cases",
            "High_activity", "All_off", "IRfilter", "IRill"
        ],
        # Mixed resolutions (2048x2048 and 1080x1080); center auto-computed per image
        "fisheye_center": None,
        "image_size": None
    }
}


def get_dataset_structure(dataset_name):
    """
    Get structure information for a dataset.

    Args:
        dataset_name: Name of the dataset (e.g., "bomni")

    Returns:
        Dictionary containing dataset structure information

    Raises:
        ValueError: If dataset_name is not found in registry
    """
    if dataset_name not in DATASET_STRUCTURES:
        available = ', '.join(DATASET_STRUCTURES.keys())
        raise ValueError(
            f"Dataset '{dataset_name}' not found in registry. "
            f"Available datasets: {available}"
        )
    return DATASET_STRUCTURES[dataset_name]


def list_available_datasets():
    """
    List all datasets registered in the system.

    Returns:
        List of dataset names
    """
    return list(DATASET_STRUCTURES.keys())
