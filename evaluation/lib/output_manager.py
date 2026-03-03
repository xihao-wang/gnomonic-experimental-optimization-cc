"""
Output path management for evaluation framework.

This module generates consistent output paths for bbox predictions and visualizations.
Paths follow the structure: configs-comparison/{session-folder}/{config-id}/{dataset}/...

Key design principle: DO NOT include dataset-specific folder structure (scenario, sequence)
as method arguments. The registry knows the structure for each dataset.

Author: Generated for gnomonic projection pedestrian detection project
Date: 2025-12-04
"""

from pathlib import Path
import shutil


class OutputPathManager:
    """Manages output paths for evaluation results."""

    def __init__(self, config_json_path=None, base_dir="evaluation/configs-comparison"):
        """
        Initialize output path manager with versioned session folder.

        Args:
            config_json_path: Path to JSON configuration file (if provided, creates versioned session)
            base_dir: Base directory for all evaluation outputs
        """
        self.base_dir = Path(base_dir)

        if config_json_path is not None:
            # Create versioned session folder and copy JSON config
            self.session_name = self._create_session_folder(Path(config_json_path))
        else:
            # Fallback for backward compatibility (if no config path provided)
            self.session_name = None

    def _create_session_folder(self, config_json_path):
        """
        Create versioned session folder and copy JSON config.

        Args:
            config_json_path: Path to JSON configuration file

        Returns:
            Session folder name (e.g., "backproj_visual_test_session_1")
        """
        # Find next available session number
        session_number = 1
        while True:
            session_name = f"backproj_visual_test_session_{session_number}"
            session_path = self.base_dir / session_name
            if not session_path.exists():
                break
            session_number += 1

        # Create session folder
        session_path.mkdir(parents=True, exist_ok=True)

        # Copy JSON config to session folder with session folder name
        dest_json_path = session_path / f"{session_name}.json"
        shutil.copy(config_json_path, dest_json_path)

        return session_name

    def get_config_base_path(self, config_id, dataset_name):
        """
        Get base path for a specific configuration-dataset combination.

        Args:
            config_id: Configuration ID from JSON
            dataset_name: Dataset name (e.g., "bomni")

        Returns:
            Path object
        """
        return self.base_dir / self.session_name / config_id / dataset_name

    def get_bboxes_numeric_path(self, config_id, dataset_name, relative_path):
        """
        Get path for saving numeric bbox coordinates.

        Args:
            config_id: Configuration ID from JSON
            dataset_name: Dataset name
            relative_path: Relative path from dataset root (e.g., "scenario1/top-0/0001.json")

        Returns:
            Path object for numeric bbox JSON file
        """
        base = self.get_config_base_path(config_id, dataset_name)
        return base / "bboxes-numeric" / relative_path

    def get_bboxes_visuals_composite_path(self, config_id, dataset_name, relative_path):
        """
        Get path for composite image visualization.

        Args:
            config_id: Configuration ID from JSON
            dataset_name: Dataset name
            relative_path: Relative path from dataset root (e.g., "scenario1/top-0/0001.jpg")

        Returns:
            Path object for composite visualization
        """
        base = self.get_config_base_path(config_id, dataset_name)
        return base / "bboxes-visuals" / "composite" / relative_path

    def get_bboxes_visuals_fisheye_path(self, config_id, dataset_name, relative_path):
        """
        Get path for fisheye image visualization (GT + predictions).

        Args:
            config_id: Configuration ID from JSON
            dataset_name: Dataset name
            relative_path: Relative path from dataset root (e.g., "scenario1/top-0/0001.jpg")

        Returns:
            Path object for fisheye visualization
        """
        base = self.get_config_base_path(config_id, dataset_name)
        return base / "bboxes-visuals" / "fisheye" / relative_path

    def ensure_directories_exist(self, file_path):
        """
        Ensure all parent directories exist for a file path.

        Args:
            file_path: Path object or string
        """
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    def get_all_output_dirs(self, config_id, dataset_name):
        """
        Get all output directories for a configuration.

        Args:
            config_id: Configuration ID
            dataset_name: Dataset name

        Returns:
            Dictionary with keys: 'base', 'numeric', 'composite', 'fisheye'
        """
        base = self.get_config_base_path(config_id, dataset_name)
        return {
            'base': base,
            'numeric': base / "bboxes-numeric",
            'composite': base / "bboxes-visuals" / "composite",
            'fisheye': base / "bboxes-visuals" / "fisheye"
        }

    def _flatten_relative_path(self, relative_path):
        """
        Convert hierarchical relative path to flattened filename.

        Args:
            relative_path: Relative path like "scenario1/top-0/0001.json"

        Returns:
            Flattened filename like "scenario1-top-0-0001.json"
        """
        path_obj = Path(relative_path)
        parts = path_obj.parts
        stem = path_obj.stem
        suffix = path_obj.suffix

        # Join all directory parts with stem, separated by dashes
        flattened_name = "-".join(parts[:-1] + (stem,)) + suffix
        return flattened_name

    def get_all_annotations_path(self, config_id, dataset_name, relative_path):
        """
        Get path for flattened annotations folder.

        Args:
            config_id: Configuration ID
            dataset_name: Dataset name
            relative_path: Relative path from dataset root (e.g., "scenario1/top-0/0001.json")

        Returns:
            Path object for flattened annotation file
        """
        base = self.get_config_base_path(config_id, dataset_name)
        flattened_name = self._flatten_relative_path(relative_path)
        return base / "all-annotations" / flattened_name

    def get_all_images_composite_path(self, config_id, dataset_name, relative_path):
        """
        Get path for flattened composite images folder.

        Args:
            config_id: Configuration ID
            dataset_name: Dataset name
            relative_path: Relative path from dataset root (e.g., "scenario1/top-0/0001.jpg")

        Returns:
            Path object for flattened composite image
        """
        base = self.get_config_base_path(config_id, dataset_name)
        flattened_name = self._flatten_relative_path(relative_path)
        return base / "all-images" / "composite" / flattened_name

    def get_all_images_fisheye_path(self, config_id, dataset_name, relative_path):
        """
        Get path for flattened fisheye images folder.

        Args:
            config_id: Configuration ID
            dataset_name: Dataset name
            relative_path: Relative path from dataset root (e.g., "scenario1/top-0/0001.jpg")

        Returns:
            Path object for flattened fisheye image
        """
        base = self.get_config_base_path(config_id, dataset_name)
        flattened_name = self._flatten_relative_path(relative_path)
        return base / "all-images" / "fisheye" / flattened_name

    def get_all_images_side_by_side_path(self, config_id, dataset_name, relative_path):
        """
        Get path for flattened side-by-side images folder.

        Side-by-side images contain fisheye (left) and composite (right) with bboxes
        for easy visual comparison of backprojection accuracy.

        Args:
            config_id: Configuration ID
            dataset_name: Dataset name
            relative_path: Relative path from dataset root (e.g., "scenario1/top-0/0001.jpg")

        Returns:
            Path object for flattened side-by-side image
        """
        base = self.get_config_base_path(config_id, dataset_name)
        flattened_name = self._flatten_relative_path(relative_path)
        return base / "all-images" / "side-by-side" / flattened_name
