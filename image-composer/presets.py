"""
Preset configurations for multi-persp.py

This file contains predefined configurations for different projection scenarios.
Add your own presets or modify existing ones as needed.
"""

# Default image path used by all presets unless overridden
DEFAULT_IMG_PATH = "imgs/fisheye.png"

# Output directory used by all presets
OUTPUT_DIR = "multi-persp-out"

# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

# Dictionary of named presets
# Each preset is a dictionary of configuration parameters
PRESETS = {
    # --------------------------------------------------------------------
    # Default configuration - 6 projections with 60° steps
    # --------------------------------------------------------------------
    "default": {
        "name": "Default 6-view Configuration",
        "description": "Standard 6 projections with 60° longitude steps at 45° latitude",
        "img_path": DEFAULT_IMG_PATH,
        "proj_nbr": 6,
        "fov_h": 90.0,
        "fov_v": 60.0,
        "latitude": 45.0,
        "lon_0": 0.0,
        "lon_step": 60.0,
        "grid": None,  # Automatically determined
        "comp_sz": (640, 640),
        "target_mp": 0.48
    },
    
    # --------------------------------------------------------------------
    # High coverage configuration - 8 projections with 45° steps
    # --------------------------------------------------------------------
    "chiang_conf": {
        "name": "Configuration from Chiang et al. 2021 IAVC",
        "description": "8 projections with 45° longitude steps for better coverage",
        "img_path": DEFAULT_IMG_PATH,
        "proj_nbr": 8,
        "fov_h": 96.0,
        "fov_v": 48.0,
        "latitude": 36.0,
        "lon_0": 0.0,
        "lon_step": 45.0,
        "grid": None,
        "comp_sz": (640, 640),
        "target_mp": 0.48
    },
    
    # --------------------------------------------------------------------
    # Horizontal panorama configuration - 4 projections at horizon level
    # --------------------------------------------------------------------
    "horizon": {
        "name": "Horizon View Configuration",
        "description": "4 projections at horizon level (90° latitude) for panoramic view",
        "img_path": DEFAULT_IMG_PATH,
        "proj_nbr": 4,
        "fov_h": 90.0,
        "fov_v": 60.0,
        "latitude": 80.0,  # Near horizon
        "lon_0": 0.0,
        "lon_step": 90.0,
        "grid": (1, 4),  # Single row, 4 columns
        "comp_sz": (640, 640),
        "target_mp": 0.48
    },
    
    # --------------------------------------------------------------------
    # YOLOv8 optimized configuration - 9 projections in 3x3 grid
    # --------------------------------------------------------------------
    "yolo_grid": {
        "name": "YOLOv8 Optimized Grid",
        "description": "9 projections in a 3x3 grid optimized for YOLOv8 detection",
        "img_path": DEFAULT_IMG_PATH,
        "proj_nbr": 9,
        "fov_h": 60.0,
        "fov_v": 60.0,
        "latitude": 45.0,
        "lon_0": 0.0,
        "lon_step": 40.0,
        "grid": (3, 3),  # 3x3 grid
        "comp_sz": (640, 640),
        "target_mp": 'auto'  # Auto: each projection will be comp_sz / grid_dims (no resizing needed)
    },
    
    # --------------------------------------------------------------------
    # Wide angle configuration - 6 projections with wider FOV
    # --------------------------------------------------------------------
    "wide_angle": {
        "name": "Wide Angle Configuration",
        "description": "6 projections with wide 120° horizontal FOV",
        "img_path": DEFAULT_IMG_PATH,
        "proj_nbr": 6,
        "fov_h": 120.0,
        "fov_v": 90.0,
        "latitude": 45.0,
        "lon_0": 0.0,
        "lon_step": 60.0,
        "grid": None,
        "comp_sz": (640, 640),
        "target_mp": 0.48
    },
    
    # --------------------------------------------------------------------
    # Higher resolution configuration - same coverage but higher resolution
    # --------------------------------------------------------------------
    "high_res": {
        "name": "High Resolution Configuration",
        "description": "6 projections with higher resolution output (1MP per projection)",
        "img_path": DEFAULT_IMG_PATH,
        "proj_nbr": 6,
        "fov_h": 90.0,
        "fov_v": 60.0,
        "latitude": 45.0,
        "lon_0": 0.0,
        "lon_step": 60.0,
        "grid": None,
        "comp_sz": (1024, 1024),
        "target_mp": 1.0  # Higher resolution per projection
    }
}

# List of available preset names
PRESET_NAMES = list(PRESETS.keys())

# Function to get a preset configuration
def get_preset(preset_name):
    """
    Get a preset configuration by name.
    
    :param preset_name: Name of the preset
    :return: Dictionary of configuration parameters or None if not found
    """
    if preset_name in PRESETS:
        # Make a copy to avoid modifying the original
        preset = PRESETS[preset_name].copy()
        # Add output_dir which is common to all presets
        preset["output_dir"] = OUTPUT_DIR
        return preset
    return None

# Function to list all available presets
def list_presets():
    """
    Return a formatted string listing all available presets.
    
    :return: String with preset information
    """
    result = "Available Presets:\n"
    result += "=" * 70 + "\n"
    
    for name, preset in PRESETS.items():
        result += f"- {name}: {preset['name']}\n"
        result += f"  {preset['description']}\n"
        result += f"  {preset['proj_nbr']} projections, FOV: {preset['fov_h']}°×{preset['fov_v']}°, Latitude: {preset['latitude']}°\n"
        result += "-" * 70 + "\n"
    
    return result
