"""
Image Composer Module - Multi-perspective fisheye projection API

This module provides functionality to convert fisheye images into composite images
by creating multiple gnomonic projections arranged in a grid.

Main Functions:
    - generate_composite_from_config: Convert fisheye to composite using config dict
    - get_preset: Get a pre-defined configuration
    - list_presets: List all available presets
"""

import sys
from pathlib import Path

# Add image-composer directory to path so it can import its local modules
_img_composer_dir = Path(__file__).parent
if str(_img_composer_dir) not in sys.path:
    sys.path.insert(0, str(_img_composer_dir))

from presets import get_preset, list_presets, PRESETS
from multi_persp import generate_composite_from_config

__all__ = ['generate_composite_from_config', 'get_preset', 'list_presets', 'PRESETS']
