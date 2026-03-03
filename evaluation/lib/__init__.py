"""
Evaluation library - internal utilities and infrastructure.

This module contains the core infrastructure for visual validation testing:
- Configuration system (config.py)
- Dataset registry and loaders (dataset_registry.py, bomni_dataset.py)
- Visual validation orchestrator (backprojection_visual_validator.py)
- Visualization functions (visualization.py)
- Output management (output_manager.py)

Entry points that use this library:
- run_backprojection_visual_test.py
- visualize_datasets.py
"""

# Core exports (if needed for convenience imports)
from evaluation.lib.config import get_cfg, get_pred_bbox_color
from evaluation.lib.backprojection_visual_validator import BackprojectionVisualValidator
from evaluation.lib.bomni_dataset import BOMNIDataset

__all__ = [
    'get_cfg',
    'get_pred_bbox_color',
    'BackprojectionVisualValidator',
    'BOMNIDataset',
]
