# Multi-Perspective Fisheye Projection Tool

Transforms a fisheye image into multiple gnomonic perspective projections arranged in a composite grid, suitable for object detection models.

## Features

- Multiple perspective projections from a single fisheye image
- Configurable FOV, latitude, longitude spacing, and grid layout
- Auto-sizing of projections to fit target composite dimensions
- Optional padding to reduce aspect-ratio distortion in non-square grids
- Exports mapping matrices for backprojecting detections to fisheye coordinates
- FOV coverage visualization on the original fisheye image

---

## Configuration

Edit `config.py`:

```python
# Use a named preset, or set to None for custom configuration
SELECTED_PRESET = None  # "default", "high_coverage", "horizon", "yolo_grid", "wide_angle", "high_res"

# Input image (used when SELECTED_PRESET is None, or to override a preset's image)
IMG_PATH = "imgs/fisheye.png"
CUSTOM_IMAGE = None  # Overrides preset image path if set

# Projection parameters
PROJ_NBR = 8        # Number of projections
FOV_H = 48.0        # Horizontal FOV in degrees
FOV_V = 96.0        # Vertical FOV in degrees
LATITUDE = 36.0     # 0 = nadir (straight down), 90 = horizon
LON_0 = 0.0         # Starting longitude
LON_STEP = 45.0     # Longitude step between projections

# Output
GRID = None         # (rows, cols) or None for automatic
COMP_SZ = (640, 640)
TARGET_MP = 'auto'  # 'auto' = fit each projection to its grid cell
OUTPUT_DIR = "multi-persp-out"
```

### Padding (optional)

For non-square grids (e.g. 2×4), projections are tall and narrow, which compresses pedestrians vertically. Padding adds black borders inside each cell to preserve aspect ratio.

```python
PADDING_DIRECTION = "none"   # "none" | "vertical" | "horizontal" | "both"
PADDING_PCT = 0.10           # Fraction of cell dimension used as total black border
                             # Split evenly on both sides. Use [v_pct, h_pct] for "both".
                             # Example: 0.10 → 5% black bar on each side vertically
```

Requires `TARGET_MP = 'auto'`.

---

## Usage

```bash
python multi_persp.py
```

List available presets:

```bash
python multi_persp.py list
```

---

## Output Files

Each run creates a new folder `pconf-X` inside `OUTPUT_DIR`:

| File | Description |
|------|-------------|
| `fish-X.png` | Original fisheye image |
| `fovs-fish-X.png` | Fisheye with FOV coverage visualization |
| `persp-X-Y.png` | Individual projection Y |
| `or-composit-X.png` | Original composite grid (before resize) |
| `rz-composit-X.png` | Resized composite (COMP_SZ, for detection input) |
| `mappings-X.npy` | Mapping matrices `[num_proj, 2, H, W]` |
| `log-pconf-X.txt` | Detailed configuration log |

---

## Using Mapping Matrices

```python
import numpy as np

mappings = np.load('mappings-X.npy')
# Shape: [num_projections, 2, height, width]

# Fisheye coordinates for a point (px=100, py=150) in projection 0
map_x = mappings[0, 0]
map_y = mappings[0, 1]
fisheye_x = map_x[150, 100]   # note: [row, col] = [y, x]
fisheye_y = map_y[150, 100]
```
