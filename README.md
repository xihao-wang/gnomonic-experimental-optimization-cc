# Pedestrian Detection in Fisheye Images via Gnomonic Projections

This project detects pedestrians in omnidirectional (fisheye) images using a projection-based approach: fisheye frames are decomposed into gnomonic perspective projections, fed to a YOLO detector, and detections are backprojected to fisheye coordinates. A systematic configuration search identifies the optimal projection layout.

---

## Approach

1. **Projection**: Generate multiple gnomonic projections from a fisheye image and stitch them into a composite
2. **Detection**: Run a YOLO model on the composite image
3. **Backprojection**: Map detections back to fisheye coordinates using a 5-point geometric transformation
4. **Evaluation**: Compute mAP, AP50, AP75, AR, F1 against ground-truth annotations
5. **Configuration Search**: Identify the best projection layout (count, FOV, grid arrangement)

The proposed best configuration is **TEST#32-central-9×9**.

---

## Project Structure

```
project-root/
├── image_composer/          # Gnomonic composite generation from fisheye
│   ├── config.py
│   ├── multi_persp.py
│   └── presets.py
├── detection_pipeline/      # YOLO detection + backprojection (single image / video)
│   ├── config.py
│   ├── pipeline.py
│   ├── yolo_detector.py
│   ├── backprojection.py
│   └── nms.py
├── tracker_pipeline/        # Multi-frame tracking (ByteTrack + Kalman filter) — placeholder, to be replaced with a fisheye-specific tracker from literature
│   ├── config.py
│   ├── process_video.py
│   └── lib/                 # byte_tracker.py, kalman_filter.py, matching.py
├── evaluation/              # Metrics evaluation and cross-model comparison
│   ├── lib/
│   │   └── config.py        # YACS config (datasets, metrics, output)
│   ├── run_single_config.py
│   ├── run_metrics_evaluation.py
│   ├── run_cross_model_comparator.py
│   ├── run_comparator.py
│   └── projection_configs_for_metrics.py
├── datasets/                # Dataset managers and preparation utilities
│   ├── adapters/
│   ├── lib/
│   └── utils/
├── models/                  # YOLO model weights
├── config_search/           # Configuration search (Phase 5)
│   └── config.py
└── docs/                    # API, architecture, and setup documentation
```

---

## Quick Start

### 1. Setup

```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install ultralytics opencv-python numpy scikit-learn matplotlib yacs pillow
```

### 2. Configure

All parameters are controlled via `config.py` files — no command-line arguments:

| File | Controls |
|------|----------|
| `detection_pipeline/config.py` | YOLO model, projection preset, NMS, backprojection |
| `evaluation/lib/config.py` | Dataset paths, metrics, IoU thresholds, YOLO imgsz |
| `tracker_pipeline/config.py` | ByteTrack parameters, video input/output |
| `config_search/config.py` | Search space for configuration search |

### 3. Prepare Datasets

```bash
python datasets/prepare_dataset_step1.py   # Automated preparation
# Manually review and delete incorrect visualizations
python datasets/prepare_dataset_step2.py   # Semi-automated cleanup
```

### 4. Run Detection

```bash
python detection_pipeline/pipeline.py         # Single image
python detection_pipeline/process_video.py    # Video
python tracker_pipeline/process_video.py      # Video with tracking
```

### 5. Evaluate

```bash
python evaluation/run_single_config.py          # One projection config
python evaluation/run_metrics_evaluation.py     # All configs (batch)
python evaluation/run_cross_model_comparator.py # Cross-YOLO-model comparison
```

---

## Datasets

### BOMNI
- Omnidirectional pedestrian tracking dataset
- 245 verified images with rotated bounding box annotations across 4 sequences
- Annotations sourced from Tamura et al. (omnidet-rotinv, WACV 2019)

### PIROPO
- Panoramic indoor pedestrian dataset (~3,004 images)
- 2 rooms (Room_A, Room_B), multiple cameras

### CEPDOF
- 25,358 frames across 8 sequences (Lunch1–3, Edge_cases, High_activity, All_off, IRfilter, IRill)
- Mixed resolutions: 2048×2048 and 1080×1080

---

## Evaluation Metrics

- **Primary**: mAP (AP@[0.50:0.95])
- **Secondary**: AP50, AP75, AR, F1
- IoU thresholds: 0.50–0.95 (step 0.05)
- Confidence intervals: 95%
- Baseline for comparison: Chiang et al. 2021 (8 projections, 2×4 grid, 48×96 FOV)

---

## Configuration Philosophy

All runtime parameters live in `config.py` files using the YACS framework. No argparse. Configurations are committed to git for full experiment reproducibility.

---

## Dependencies

```
ultralytics      # YOLO inference
opencv-python    # Image processing
numpy            # Numerical computing
scikit-learn     # Metrics
matplotlib       # Visualization
yacs             # Configuration framework
pillow           # Image I/O
```

---

## Documentation

- `docs/API.md` — Module API reference
- `docs/ARCHITECTURE.md` — System architecture
- `docs/SETUP.md` — Detailed setup guide
- `datasets/PREPARATION_README.md` — Dataset preparation workflow
