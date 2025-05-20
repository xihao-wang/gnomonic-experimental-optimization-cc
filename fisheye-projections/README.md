# Multi-Perspective Fisheye Projection Tool

This repository contains two related tools for working with fisheye images:

1. `fisheye-to-persp.py`: The original interactive single-view projection tool
2. `multi-persp.py`: An extended version that generates multiple projections with presets

---

## 1. fisheye-to-persp.py - Interactive Single Projection

The original `fisheye-to-persp.py` script provides an interactive interface for creating perspective projections from fisheye images:

- **Interactive Controls**: Adjust longitude (0-360°), latitude (0-90°), and FOV in real-time
- **FOV Visualization**: See the projection's field of view overlaid on the fisheye image
- **Performance Metrics**: Track remapping latency
- **Single View**: Creates one perspective-corrected view at a time

To use this tool, simply run:
```
python fisheye-to-persp.py
```

---

## 2. multi-persp.py - Multiple Projections Tool

The `multi-persp.py` script extends the original tool to generate multiple projections and combines them into one composite image for use with object detection models pre-trained on perspective images like YOLOv8.

### Features

- Generate multiple perspective projections at different viewing angles
- Create composite image by arranging projections in a grid layout
- Visualize all projections' FOVs simultaneously on the original fisheye
- Calculate coverage statistics (overlap and uncovered areas) !! statistics may be inaccurate !!
- Use preset configurations for common scenarios
- Generate detailed configuration and performance logs

### Configuration Options

#### Using Preset Configurations

Edit `config.py` and set:

```python
# To use a preset, specify its name here:
SELECTED_PRESET = "high_res"

# Optionally override the preset's default image:
CUSTOM_IMAGE = "path/to/your/image.jpg"  # or None to use default
```

Available presets:
- **default**: Standard 6 projections with 60° longitude steps
- **high_coverage**: 8 projections with 45° steps for better coverage
- **horizon**: 4 projections at horizon level for panoramic view
- **yolo_grid**: 9 projections in a 3x3 grid (assumed suitable for YOLOv8)
- **wide_angle**: 6 projections with wide 120° horizontal FOV
- **high_res**: Higher resolution output (1MP per projection)

#### Creating Custom Presets

You can easily add your own presets or delete existing ones by editing the `presets.py` file:

1. Open `presets.py` in your editor
2. Find the `PRESETS` dictionary
3. Add a new entry using this template:

```python
"your_preset_name": {
    "name": "Human-Readable Preset Name",
    "description": "Description of what this preset does",
    "img_path": "imgs/fisheye.png",
    "proj_nbr": 8,  # Number of projections
    "fov_h": 90.0,  # Horizontal FOV in degrees
    "fov_v": 60.0,  # Vertical FOV in degrees
    "latitude": 45.0,  # In degrees (0=nadir, 90=horizon)
    "lon_0": 0.0,  # Starting longitude
    "lon_step": 45.0,  # Step between projections
    "grid": (2, 4),  # Grid layout or None for automatic
    "comp_sz": (640, 640),  # Output composite size
    "target_mp": 0.48,  # Target megapixels per projection
    "output_dir": "multi-persp-out"
},
```

#### Using Custom Configuration

Optionally, you can manually define the configuration parameters (e.g., PROJ_NBR) in the config.py file:

```python
# Set to None to use custom configuration:
SELECTED_PRESET = None

# Then adjust all parameters below:
IMG_PATH = "imgs/fisheye.png"
PROJ_NBR = 6
FOV_H = 90.0
# etc...
```

### Running the Multi-Projection Tool

Run without arguments:

```
python multi-persp.py
```

### Output Files

The tool creates a new folder named `pconf-X` (where X is an incremental number) in the `multi-persp-out` directory, containing:

- Individual perspective projections (`persp-X-Y.png`), where`Y`is an incremental number indexing the projected image.
- Original-size composite image (`or-composit-X.png`)
- Resized composite image for YOLO input (`rz-composit-X.png`)
- Original fisheye image (`fish-X.png`)
- Fisheye with FOV visualizations (`fovs-fish-X.png`) 
- Configuration and performance log (`log-pconf-X.txt`)