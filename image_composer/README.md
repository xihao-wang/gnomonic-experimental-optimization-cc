# Multi-Perspective Fisheye Projection Tool

This tool transforms fisheye images into multiple perspective projections arranged in a composite image for object detection.

## Features

- Create multiple perspective projections from a single fisheye image
- Configure projection parameters (FOV, latitude, longitude spacing)
- Automatically size projections to fit target composite dimensions
- Generate a grid-based composite image suitable for object detection models
- Export mapping matrices for converting detections back to fisheye coordinates
- Visualize FOV coverage on the original fisheye image

## Configuration

Edit `config.py` to set your parameters:

```python
# Input image
IMG_PATH = "imgs/fisheye.png"

# Projection parameters
PROJ_NBR = 6         # Number of projections
FOV_H = 90.0         # Horizontal field of view in degrees
FOV_V = 60.0         # Vertical field of view in degrees
LATITUDE = 45.0      # Latitude angle in degrees
LON_0 = 0.0          # Starting longitude in degrees
LON_STEP = 60.0      # Longitude step between projections

# Output parameters
GRID = None          # Grid layout as (rows, cols) or None for automatic
COMP_SZ = (640, 640) # Composite size for YOLOv8 input
TARGET_MP = 'auto'   # Target megapixels per projection or 'auto'
OUTPUT_DIR = "multi-persp-out"
```

When `TARGET_MP` is set to `'auto'`, projection dimensions will be automatically calculated to fit the composite size based on the grid layout.

## Usage

```bash
python multi-persp.py
```

## Output Files

Each run creates a new folder `pconf-X` with:

- `fish-X.png`: Original fisheye image
- `fovs-fish-X.png`: Fisheye with FOV visualization
- `persp-X-Y.png`: Individual projections (Y = projection index)
- `or-composit-X.png`: Original composite grid
- `rz-composit-X.png`: Resized composite for detection
- `mappings-X.npy`: Combined mapping matrices (shape [num_projections, 2, height, width])
- `log-pconf-X.txt`: Detailed configuration log

## Using Mapping Matrices

The mapping matrices allow converting detection coordinates from perspective views back to the fisheye image:

```python
import numpy as np

# Load mapping matrices
mappings = np.load('mappings-X.npy')
# Shape: [num_projections, 2, height, width]

# Get map_x and map_y for projection 0
map_x = mappings[0, 0]  # First 0: projection index, Second 0: map_x
map_y = mappings[0, 1]  # First 0: projection index, Second 1: map_y

# Convert a point (100,150) in perspective view to fisheye coordinates
fisheye_x = map_x[150, 100]  # Note: y,x indexing
fisheye_y = map_y[150, 100]
```

## Presets

You can use predefined configurations by setting a preset in `config.py`:

```python
SELECTED_PRESET = "high_coverage"  # Name of preset to use
```

Or list available presets:

```bash
python multi-persp.py list
```