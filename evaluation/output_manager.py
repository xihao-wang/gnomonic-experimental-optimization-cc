"""
Output path management for evaluation framework.

This module generates consistent output paths for bbox predictions and visualizations.
Paths follow the structure: configs-comparison/{json-name}/{config-id}/{dataset}/...

Key design principle: DO NOT include dataset-specific folder structure (scenario, sequence)
as method arguments. The registry knows the structure for each dataset.

Author: Generated for gnomonic projection pedestrian detection project
Date: 2025-12-04
"""

from pathlib import Path


class OutputPathManager:
    """Manages output paths for evaluation results."""

    def __init__(self, base_dir="evaluation/configs-comparison"):
        """
        Initialize output path manager.

        Args:
            base_dir: Base directory for all evaluation outputs
        """
        self.base_dir = Path(base_dir)

    def get_config_base_path(self, json_name, config_id, dataset_name):
        """
        Get base path for a specific configuration-dataset combination.

        Args:
            json_name: Name of JSON file (without .json extension)
            config_id: Configuration ID from JSON
            dataset_name: Dataset name (e.g., "bomni")

        Returns:
            Path object
        """
        return self.base_dir / json_name / config_id / dataset_name

    def get_bboxes_numeric_path(self, json_name, config_id, dataset_name, relative_path):
        """
        Get path for saving numeric bbox coordinates.

        Args:
            json_name: Name of JSON file (without .json extension)
            config_id: Configuration ID from JSON
            dataset_name: Dataset name
            relative_path: Relative path from dataset root (e.g., "scenario1/top-0/0001.json")

        Returns:
            Path object for numeric bbox JSON file
        """
        base = self.get_config_base_path(json_name, config_id, dataset_name)
        return base / "bboxes-numeric" / relative_path

    def get_bboxes_visuals_composite_path(self, json_name, config_id, dataset_name, image_name):
        """
        Get path for composite image visualization.

        Args:
            json_name: Name of JSON file (without .json extension)
            config_id: Configuration ID from JSON
            dataset_name: Dataset name
            image_name: Image filename (e.g., "0001.jpg")

        Returns:
            Path object for composite visualization
        """
        base = self.get_config_base_path(json_name, config_id, dataset_name)
        return base / "bboxes-visuals" / "composite" / image_name

    def get_bboxes_visuals_fisheye_path(self, json_name, config_id, dataset_name, relative_path):
        """
        Get path for fisheye image visualization (GT + predictions).

        Args:
            json_name: Name of JSON file (without .json extension)
            config_id: Configuration ID from JSON
            dataset_name: Dataset name
            relative_path: Relative path from dataset root (e.g., "scenario1/top-0/0001.jpg")

        Returns:
            Path object for fisheye visualization
        """
        base = self.get_config_base_path(json_name, config_id, dataset_name)
        return base / "bboxes-visuals" / "fisheye" / relative_path

    def ensure_directories_exist(self, file_path):
        """
        Ensure all parent directories exist for a file path.

        Args:
            file_path: Path object or string
        """
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    def get_all_output_dirs(self, json_name, config_id, dataset_name):
        """
        Get all output directories for a configuration.

        Args:
            json_name: Name of JSON file
            config_id: Configuration ID
            dataset_name: Dataset name

        Returns:
            Dictionary with keys: 'base', 'numeric', 'composite', 'fisheye'
        """
        base = self.get_config_base_path(json_name, config_id, dataset_name)
        return {
            'base': base,
            'numeric': base / "bboxes-numeric",
            'composite': base / "bboxes-visuals" / "composite",
            'fisheye': base / "bboxes-visuals" / "fisheye"
        }
