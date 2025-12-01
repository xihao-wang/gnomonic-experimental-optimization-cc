# Detection Pipeline

Pedestrian detection pipeline for fisheye images using projection-based approach.

## Overview

The detection pipeline converts fisheye images into composite images (multiple gnomonic projections stitched together) and runs YOLO detection on the composite to identify pedestrians.

**Workflow**:
```
Fisheye Image
    ↓
[Build Projection Config]
    ↓
[Generate Composite (9 projections in 3×3 grid)]
    ↓
[Run YOLO Detection]
    ↓
[Filter to Person Class Only]
    ↓
Pedestrian Detections (in composite coordinates)
```

## Quick Start

### 1. Configuration

Edit `config.py` to customize the pipeline. Key parameters:

```python
# Input image path
INPUT.IMAGE_PATH = "path/to/fisheye.png"

# Projection preset: "yolo_grid", "default", "high_coverage", "horizon", "wide_angle", "high_res"
PROJECTION.PRESET = "yolo_grid"  # 9 projections (3×3 grid, 640×640 composite)

# YOLO model from detection_pipeline/models/
YOLO.MODEL = "models/yolov8n.pt"

# Detection confidence threshold
YOLO.CONFIDENCE_THRESHOLD = 0.5

# NMS IoU threshold
YOLO.IOU_THRESHOLD = 0.45

# Device: None (auto-detect GPU if available, else CPU), "cuda", or "cpu"
YOLO.DEVICE = None  # Auto-detects: GPU if available, else CPU
```

### 2. Run Detection

**Option A: Run test script**
```bash
cd detection_pipeline
python test_pipeline.py
```

This will:
- Load `fisheye-sample.png`
- Generate composite using "yolo_grid" preset
- Run YOLOv8n detection
- Save visualization to `detection-composite-fisheye-sample.png`

**Option B: Use as module**
```python
from detection_pipeline.config import get_cfg
from detection_pipeline.pipeline import DetectionPipeline

# Get default config
cfg = get_cfg()

# Customize
cfg.INPUT.IMAGE_PATH = "my_fisheye.png"
cfg.YOLO.DEVICE = "cuda"

# Run pipeline
pipeline = DetectionPipeline(cfg)
detections, composite, metadata = pipeline.run()

# detections: list of detected pedestrians
# composite: the 640×640 composite image
# metadata: projection info (includes mapping matrices for future backprojection)
```

## Components

### `config.py` (YACS Configuration)

Defines all pipeline parameters using YACS. Sections:

- **INPUT**: Fisheye image path
- **PROJECTION**: Composite generation settings
  - Preset selection or manual configuration
  - FOV, latitude, grid layout
  - Composite size and per-projection megapixels
- **YOLO**: Detection model and inference parameters
  - Model path (must be in `models/` directory)
  - Device (CPU/GPU)
  - Confidence threshold
  - IoU threshold for NMS
- **OUTPUT**: Visualization options
- **VERBOSE**: Debug logging flag

### `yolo_detector.py`

`YOLODetector` class wrapping YOLO inference.

**Key Method**:
```python
def detect(image, class_filter=None):
    """
    Run YOLO detection on image.

    Args:
        image: numpy array (H, W, 3) in BGR
        class_filter: Filter detections to specific class(es)
                     e.g., "person" or ["person", "car"]

    Returns:
        List of detections with keys:
        {
            'x': center x (0-1 normalized),
            'y': center y (0-1 normalized),
            'w': width (0-1 normalized),
            'h': height (0-1 normalized),
            'confidence': confidence score,
            'class_id': YOLO class ID,
            'class_name': class name string
        }
    """
```

**Features**:
- Version-agnostic (works with YOLOv8, v11, v12, etc.)
- Supports class filtering for person-only detection
- Returns normalized coordinates (0-1)

### `pipeline.py`

`DetectionPipeline` class orchestrating the full workflow.

**Key Method**:
```python
def run():
    """
    Execute full detection pipeline.

    Returns:
        (detections, composite_image, metadata) tuple

        detections: List of person detections
        composite_image: Stitched composite image (640×640)
        metadata: Projection info with mapping matrices
    """
```

**Features**:
- Integrates image_composer API for composite generation
- Supports preset and custom projection configurations
- Filters detections to "person" class by default
- Optional visualization saving
- Verbose logging

### `test_pipeline.py`

End-to-end test script demonstrating pipeline usage.

**What it does**:
1. Loads `fisheye-sample.png`
2. Generates 3×3 composite (640×640) using "yolo_grid" preset
3. Runs YOLOv8n detection
4. Saves visualizations and results

**Usage**:
```bash
python test_pipeline.py
```

## Available YOLO Models

Models available in `models/` directory:

**YOLOv8** (recommended for most use cases):
- `yolov8n.pt` - Nano (fastest, lowest accuracy)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra Large (slowest, highest accuracy)

**YOLOv11** (latest):
- `yolo11n.pt`
- `yolo11s.pt`

**YOLOv12** (experimental):
- `yolo12n.pt`
- `yolo12s.pt`
- `yolo12x.pt`

To use a different model, update `config.py`:
```python
YOLO.MODEL = "models/yolov8m.pt"  # Medium model (slower but more accurate)
```

## Available Projection Presets

Configure in `config.py` via `PROJECTION.PRESET`:

- **"yolo_grid"** - 9 projections (3×3 grid), optimized for YOLOv8
  - FOV: 60°×60°, Composite: 640×640, Latitude: 45°

- **"default"** - 6 projections with 60° steps
  - FOV: 90°×60°, Composite: 640×640

- **"high_coverage"** - 8 projections with 45° steps
  - FOV: 75°×60°, Composite: 640×640

- **"horizon"** - 4 projections at horizon level
  - FOV: 90°×60°, Composite: 640×640, Latitude: 80°

- **"wide_angle"** - 6 projections with wide FOV
  - FOV: 120°×90°, Composite: 640×640

- **"high_res"** - 6 projections at higher resolution
  - FOV: 90°×60°, Composite: 1024×1024

Or configure manually by setting `PROJECTION.PRESET = None` and specifying:
```python
PROJECTION.PROJ_NBR = 9
PROJECTION.FOV_H = 60.0
PROJECTION.FOV_V = 60.0
PROJECTION.LATITUDE = 45.0
PROJECTION.GRID = (3, 3)
PROJECTION.COMP_SIZE = (640, 640)
PROJECTION.TARGET_MP = 'auto'  # Efficient: each projection = comp_size / grid_dims
```

## Backprojection

Backprojection maps detections from composite coordinates back to fisheye with radial alignment.

**Method**:

1. **Map center**: Compute the center of each bbox on the composite image, then map it directly to the fisheye image.

2. **Transform dimensions**: Given a width and height (w, h) of a bbox on the composite image, compute (width_fish, height_fish) - the width and height in pixels on the fisheye that correspond to (w, h) on composite. This is a distance transformation operation:
   - Backproject left and right edges at center height, measure distance between them → width_fish
   - Backproject top and bottom edges at center width, measure distance between them → height_fish

3. **Radial alignment**: Establish the line that passes through the bbox center and the fisheye image center, then draw a bbox using (width_fish, height_fish) rotated to align with this radial direction. Height points radially (toward/away from fisheye center), width points tangentially (perpendicular).

**Implementation details**:
- Width: (x - w/2, y) → (x_left, y_left), (x + w/2, y) → (x_right, y_right), width_fish = distance(left, right)
- Height: (x, y - h/2) → (x_top, y_top), (x, y + h/2) → (x_bottom, y_bottom), height_fish = distance(top, bottom)
- Corners: Built from center ± height_fish/2 (radial), ± width_fish/2 (tangential), rotated by angle from fisheye center to bbox center

**Lattice visualization**: Aspect-ratio aware grid (width_samples = height_samples × aspect_ratio) backprojected to show distortion. One image per bbox: `fisheye_bbox_lattice_N.png`

## Output

Results are automatically saved in organized directory structure:

```
detection_pipeline/results/{image_name}/{timestamp}/
├── fisheye-sample.png                  # Original fisheye image
├── composite.png                       # Stitched composite image
├── detections.png                      # Composite with bounding boxes
└── metadata.txt                        # Configuration and results metadata
```

Each run creates a timestamped subfolder, so multiple runs on the same image are preserved without overwriting.

The metadata.txt file contains:
- Projection configuration (grid, FOV, latitude, etc.)
- YOLO configuration (model, device, thresholds)
- All detected pedestrians with coordinates and confidence scores

## Example: Custom Workflow

```python
from detection_pipeline.config import get_cfg
from detection_pipeline.pipeline import DetectionPipeline
import cv2

# Configure
cfg = get_cfg()
cfg.INPUT.IMAGE_PATH = "fisheye.png"
cfg.YOLO.MODEL = "models/yolov8m.pt"  # Use medium model (slower but more accurate)
cfg.YOLO.DEVICE = None                 # Auto-detect (GPU if available, else CPU)
cfg.PROJECTION.PRESET = "high_coverage" # 8 projections
cfg.VERBOSE = True

# Run (results automatically saved to organized directory)
pipeline = DetectionPipeline(cfg)
detections, composite, metadata, results_dir = pipeline.run()

# Process results
print(f"Found {len(detections)} pedestrians")
for det in detections:
    print(f"  Person at ({det['x']:.3f}, {det['y']:.3f}), "
          f"confidence={det['confidence']:.3f}")

print(f"\nResults saved to: {results_dir}")
print(f"Files created:")
print(f"  - fisheye-sample.png (original)")
print(f"  - composite.png (projected composite)")
print(f"  - detections.png (with bounding boxes)")
print(f"  - metadata.txt (configuration & results)")

# Access composite image
cv2.imshow("Composite", composite)
cv2.waitKey(0)

# Metadata contains mapping matrices for future backprojection
print(f"Projection grid: {metadata['grid']}")
print(f"Mapping matrices shape: {metadata['mapping_matrices'].shape}")
```

## Troubleshooting

**Issue**: YOLO model not found
- **Solution**: Ensure model exists in `detection_pipeline/models/` directory
- Check config: `YOLO.MODEL = "models/yolov8n.pt"`

**Issue**: Detection too slow
- **Solution**: Device auto-detects GPU if available. Ensure CUDA is installed for GPU acceleration. Use smaller model if needed: `YOLO.MODEL = "models/yolov8n.pt"`

**Issue**: Running on GPU instead of CPU
- **Solution**: Set `YOLO.DEVICE = "cpu"` in config.py

**Issue**: Running on CPU instead of GPU
- **Solution**: Set `YOLO.DEVICE = "cuda"` or `None` (auto-detect). Ensure CUDA is installed and compatible.

**Issue**: Detecting non-pedestrian objects
- **Solution**: Person filtering is automatic in `pipeline.run()`, only pedestrians should be detected

**Issue**: GPU out of memory
- **Solution**: Use smaller model (`yolov8n.pt`) or use CPU (`YOLO.DEVICE = "cpu"`)

## Next Steps

- **Phase 2B** (coming soon): Backprojection of detections from composite to fisheye coordinates
- **Evaluation**: Compare detection results across different projection configurations
- **Configuration Search**: Find optimal projection settings for your datasets

## Files

- `config.py` - YACS configuration
- `yolo_detector.py` - YOLO wrapper
- `pipeline.py` - Main orchestrator
- `test_pipeline.py` - Test script
- `models/` - YOLO model files
- `fisheye-sample.png` - Sample fisheye image

## References

- Image Composer API: `../image_composer/`
- Project Architecture: `../docs/ARCHITECTURE.md`
- Full API Reference: `../docs/API.md`
