# Detection Pipeline

Pedestrian detection pipeline for fisheye images using a projection-based approach.

## Overview

Fisheye images are decomposed into gnomonic perspective projections, stitched into a composite, fed to YOLO, then detections are backprojected to fisheye coordinates with radial alignment.

**Workflow**:
```
Fisheye Image
    ↓
[Generate Composite (gnomonic projections → grid)]
    ↓
[Run YOLO Detection]
    ↓
[Stage 1 NMS — Standard NMS on composite]
    ↓
[Backprojection — composite coords → fisheye coords]
    ↓
[Stage 2 NMS — Soft-NMS (Gaussian) on fisheye]
    ↓
Pedestrian Detections (fisheye coordinates, radially aligned)
```

---

## Quick Start

### 1. Configuration

Edit `config.py`. Key parameters:

```python
# Input image
INPUT.IMAGE_PATH = "path/to/fisheye.png"

# Projection preset: "yolo_grid", "default", "high_coverage", "horizon", "wide_angle", "high_res"
# Set to None to configure manually via PROJECTION.PROJ_NBR / FOV_H / FOV_V / etc.
PROJECTION.PRESET = "yolo_grid"

# YOLO model (path relative to project root)
YOLO.MODEL = "models/yolo12x.pt"

# Confidence threshold (pre-NMS filter)
YOLO.CONFIDENCE_THRESHOLD = 0.25

# Device: None = auto (GPU if available, else CPU), "cuda", or "cpu"
YOLO.DEVICE = None

# Two-stage NMS
NMS.STAGE1.IOU_THRESHOLD = 0.8       # Standard NMS on composite
NMS.STAGE2.SIGMA = 0.2               # Soft-NMS Gaussian sigma on fisheye
NMS.STAGE2.SCORE_THRESHOLD = 0.3     # Discard below this after decay

# Backprojection
BACKPROJECTION.ENABLED = True
BACKPROJECTION.LATTICE_HEIGHT_SAMPLES = 10  # Distortion grid density
```

### 2. Run Detection

**Option A: Test script**
```bash
python detection_pipeline/test_pipeline.py
```

Loads `fisheye-sample.png`, runs full pipeline with "yolo_grid" preset, saves results.

**Option B: Module API**
```python
from detection_pipeline.config import get_cfg
from detection_pipeline.pipeline import DetectionPipeline

cfg = get_cfg()
cfg.INPUT.IMAGE_PATH = "my_fisheye.png"
cfg.YOLO.DEVICE = "cuda"

pipeline = DetectionPipeline(cfg)
detections, composite, metadata, results_dir = pipeline.run()

print(f"Found {len(detections)} pedestrians")
print(f"Results saved to: {results_dir}")
```

---

## Configuration Reference

### INPUT
| Key | Default | Description |
|-----|---------|-------------|
| `IMAGE_PATH` | `"fisheye-sample.png"` | Path to fisheye image |

### PROJECTION
| Key | Default | Description |
|-----|---------|-------------|
| `PRESET` | `"yolo_grid"` | Named preset (set to `None` for manual config) |
| `PROJ_NBR` | `9` | Number of projections (manual mode) |
| `FOV_H` | `60.0` | Horizontal FOV in degrees |
| `FOV_V` | `60.0` | Vertical FOV in degrees |
| `LATITUDE` | `45.0` | Camera latitude (0=nadir, 90=horizon) |
| `LON_0` | `0.0` | Starting longitude |
| `LON_STEP` | `40.0` | Longitude step between projections |
| `GRID` | `None` | Grid layout `(rows, cols)` or `None` for auto |
| `COMP_SIZE` | `(640, 640)` | Composite image size |
| `TARGET_MP` | `'auto'` | Pixels per projection (`'auto'` = fit grid cell) |

### YOLO
| Key | Default | Description |
|-----|---------|-------------|
| `MODEL` | `"models/yolo12x.pt"` | Model path (relative to project root) |
| `DEVICE` | `None` | `None`=auto, `"cuda"`, or `"cpu"` |
| `CONFIDENCE_THRESHOLD` | `0.25` | Pre-NMS confidence filter |
| `MAX_DETECTIONS` | `300` | Maximum detections per image |

### NMS
| Key | Default | Description |
|-----|---------|-------------|
| `STAGE1.ENABLED` | `True` | Standard NMS on composite image |
| `STAGE1.IOU_THRESHOLD` | `0.8` | IoU threshold for Stage 1 (high = keep more for Stage 2) |
| `STAGE2.ENABLED` | `True` | Soft-NMS on fisheye image after backprojection |
| `STAGE2.SIGMA` | `0.2` | Gaussian decay sigma (lower = more aggressive suppression) |
| `STAGE2.SCORE_THRESHOLD` | `0.3` | Discard detections below this score after decay |

### BACKPROJECTION
| Key | Default | Description |
|-----|---------|-------------|
| `ENABLED` | `True` | Map detections from composite → fisheye coordinates |
| `LATTICE_HEIGHT_SAMPLES` | `10` | Grid points along bbox height for distortion visualization |

### OUTPUT
| Key | Default | Description |
|-----|---------|-------------|
| `SAVE_COMPOSITE` | `True` | Save composite image |
| `SAVE_COMPOSITE_VIZ` | `True` | Save composite with YOLO detections |
| `SAVE_FISHEYE_VIZ` | `True` | Save fisheye with backprojected detections |
| `SAVE_LATTICE_VIZ` | `True` | Save lattice distortion visualization per bbox |
| `SAVE_DIR` | `"results"` | Output directory |

---

## Backprojection

Detections are mapped from composite coordinates back to fisheye with radial alignment:

1. **Center**: bbox center on composite → fisheye pixel via mapping matrices
2. **Dimensions**: left/right edges backprojected → `width_fish`; top/bottom edges → `height_fish`
3. **Radial alignment**: rotated bbox drawn with height pointing radially (toward fisheye center) and width tangentially

Lattice visualization shows the distortion grid for each backprojected bbox (`fisheye_bbox_lattice_N.png`).

---

## Available Projection Presets

| Preset | Projections | FOV | Grid | Composite |
|--------|-------------|-----|------|-----------|
| `yolo_grid` | 9 | 60°×60° | 3×3 | 640×640 |
| `default` | 6 | 90°×60° | auto | 640×640 |
| `high_coverage` | 8 | 75°×60° | auto | 640×640 |
| `horizon` | 4 | 90°×60° | auto | 640×640 (lat=80°) |
| `wide_angle` | 6 | 120°×90° | auto | 640×640 |
| `high_res` | 6 | 90°×60° | auto | 1024×1024 |

---

## Available YOLO Models

Models are stored in `models/` at the project root.

| Family | Variants | Notes |
|--------|----------|-------|
| YOLO12 | n / s / x | Current default (`yolo12x.pt`) |
| YOLO11 | n / s | — |
| YOLOv10 | n / s / m / l / x | — |
| YOLOv9 | c / e | — |
| YOLOv8 | n / s / m / l / x | — |

---

## Output Structure

```
detection_pipeline/results/{image_name}/{timestamp}/
├── fisheye.png                  # Original fisheye
├── composite.png                # Stitched composite
├── composite_detections.png     # Composite with YOLO boxes
├── fisheye_detections.png       # Fisheye with backprojected boxes
├── fisheye_bbox_lattice_N.png   # Lattice distortion per bbox
└── metadata.txt                 # Config + detection results
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Model not found | Check `YOLO.MODEL` path; models live in `models/` at project root |
| Slow inference | Use a smaller model or ensure CUDA is available |
| Force CPU | Set `YOLO.DEVICE = "cpu"` |
| Force GPU | Set `YOLO.DEVICE = "cuda"` |
| GPU out of memory | Use `yolo12n.pt` or switch to CPU |
| Non-person detections | Person filtering is applied automatically in `pipeline.run()` |

---

## Files

| File | Role |
|------|------|
| `config.py` | YACS configuration |
| `pipeline.py` | Main orchestrator |
| `yolo_detector.py` | YOLO inference wrapper |
| `backprojection.py` | Composite → fisheye coordinate mapping |
| `nms.py` | Two-stage NMS (standard + Soft-NMS) |
| `process_video.py` | Per-frame video processing entry point |
| `test_pipeline.py` | End-to-end test on `fisheye-sample.png` |

---

## References

- Image Composer API: `../image_composer/`
- Project Architecture: `../docs/ARCHITECTURE.md`
- Full API Reference: `../docs/API.md`
