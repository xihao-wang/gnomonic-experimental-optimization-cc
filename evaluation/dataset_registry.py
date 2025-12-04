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
    }
    # Add other datasets here when ready (e.g., PIROPO, custom datasets)
    # Each dataset must define: annotation_root, frames_root, scenarios, sequences, fisheye_center, image_size
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
