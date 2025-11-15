# Setup & Installation

## Prerequisites

- Python 3.8+
- YOLO: `pip install ultralytics` (YOLOv8)
- Computer vision: `pip install opencv-python numpy`
- Dataset handling: `pip install pillow`
- Metrics: `pip install scikit-learn`  (for evaluation metrics)
- Optional: `pip install matplotlib` (for visualization)
- Optional: GPU support with CUDA

## Installation Steps

1. **Navigate to project root**:
   ```bash
   cd projection-configuration-search-pedestrian-detection
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install ultralytics opencv-python numpy pillow scikit-learn matplotlib
   ```

4. **Download YOLO model** (optional, will auto-download on first use):
   ```bash
   # Models will be auto-downloaded to detection-pipeline/models/
   # Or manually download from: https://github.com/ultralytics/assets/releases
   ```

## Configuration

### 1. Project-level Configuration

Edit `project_config.py`:
- Set `DEFAULT_DATASET` to your preferred dataset
- Verify `DATASETS_DIR` path
- Verify `RESULTS_DIR` path

### 2. Dataset Paths

Edit `datasets/config.py`:
- Set `BOMNI_ROOT` to your BOMNI dataset location
- Set `PIROPO_ROOT` to your PIROPO dataset location
- Check annotation format matches your datasets

Example:
```python
BOMNI_ROOT = Path("/mnt/data/bomni")
PIROPO_ROOT = Path("/mnt/data/piropo")
```

### 3. Detection Pipeline

Edit `detection-pipeline/config.py`:
- `YOLO_MODEL`: Which YOLOv8 model to use (nano, small, medium, large)
- `YOLO_CONFIDENCE_THRESHOLD`: Detection confidence cutoff
- `COMPOSITE_SIZE`: Input image size for YOLO (typically 640×640)
- `DEFAULT_PROJECTIONS`: Default field of view and camera angles

Example:
```python
DEFAULT_PROJECTIONS = {
    "count": 4,      # 2x2 grid
    "fov_h": 48.0,   # 48 degrees horizontal
    "fov_v": 96.0,   # 96 degrees vertical
    "latitude": 36.0, # 36 degrees from nadir
}
```

### 4. Evaluation Settings

Edit `evaluation/config.py`:
- `COMPUTE_METRICS`: Which metrics to calculate
- `EVALUATION_STAGE`: Evaluate on "composite", "fisheye", or "both"
- `PRIMARY_METRIC`: Metric to optimize in configuration search

### 5. Configuration Search

Edit `config_search/config.py`:
- `SEARCH_STRATEGY`: "grid", "random", "bayesian", or "genetic"
- `PROJECTION_COUNTS`: Which grid sizes to try
- `FOV_H_OPTIONS`, `FOV_V_OPTIONS`: Field of view ranges
- `LATITUDE_OPTIONS`: Camera angle options
- `SEARCH_DATASETS`: Which datasets to search on
- `USE_SUBSET`: Use subset of data for faster search

Example grid search:
```python
PROJECTION_COUNTS = [4, 6, 8, 9, 12, 16]  # 2x2, 2x3, 2x4, 3x3, 3x4, 4x4
FOV_H_OPTIONS = [40.0, 48.0, 60.0, 75.0, 90.0]
FOV_V_OPTIONS = [60.0, 96.0, 120.0, 150.0, 180.0]
LATITUDE_OPTIONS = [0.0, 36.0, 90.0]
# This generates 5×5×3 = 75 configurations to try
```

## Running the Project

### Quick Test

```bash
python main.py
```

This will print the project structure and next steps.

### (Later) Run Full Pipeline

Once you implement the modules:

```bash
python -m detection_pipeline.pipeline --config detection-pipeline/config.py
```

### (Later) Run Configuration Search

```bash
python -m config_search.searcher --config config_search/config.py
```

## Project Structure Checklist

After setup, verify these files exist:

```
✓ project_config.py
✓ main.py
✓ detection-pipeline/config.py
✓ detection-pipeline/__init__.py
✓ datasets/config.py
✓ datasets/__init__.py
✓ evaluation/config.py
✓ evaluation/__init__.py
✓ config_search/config.py
✓ config_search/__init__.py
✓ results/ (directory)
✓ docs/ARCHITECTURE.md
```

## Dataset Preparation

Before running the pipeline, ensure your datasets are set up correctly:

### BOMNI Dataset Structure
```
datasets/data/bomni/
├── images/
│   ├── seq_001/
│   │   ├── img_0001.jpg
│   │   ├── img_0002.jpg
│   │   └── ...
│   └── seq_002/
│       └── ...
└── annotations/
    ├── seq_001/
    │   ├── img_0001.xml
    │   ├── img_0002.xml
    │   └── ...
    └── seq_002/
        └── ...
```

### PIROPO Dataset Structure
```
datasets/data/piropo/
├── images/
│   ├── scene_001/
│   │   ├── frame_00001.jpg
│   │   ├── frame_00002.jpg
│   │   └── ...
│   └── scene_002/
│       └── ...
└── annotations/
    ├── scene_001/
    │   ├── frame_00001.txt
    │   ├── frame_00002.txt
    │   └── ...
    └── scene_002/
        └── ...
```

If your datasets don't match this structure, you'll need to:
1. Create custom loader in `datasets/`
2. Create annotation adapter in `datasets/adapters/`
3. Update `datasets/config.py` with new paths

## Troubleshooting

### YOLO Model Download Issues
- Models auto-download to `~/.yolov8/`
- If slow, download manually from: https://github.com/ultralytics/assets/releases

### GPU Not Detected
- Check CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
- Edit `detection-pipeline/config.py`: `YOLO_DEVICE = "cpu"` to use CPU instead

### Dataset Not Found
- Verify paths in `datasets/config.py`
- Check file permissions
- Use absolute paths, not relative paths

### Memory Issues
- Reduce `COMPOSITE_SIZE` in `detection-pipeline/config.py`
- Use smaller YOLO model: `YOLO_MODEL = "yolov8n.pt"`
- Reduce `SUBSET_SIZE` in `config_search/config.py`

## Next: Implementation

Once setup is complete, implement modules in this order:

1. **Dataset loaders** - `datasets/dataset_loader.py`, `bomni_loader.py`, `piropo_loader.py`
2. **Detection pipeline** - `detection-pipeline/pipeline.py`, `yolo_detector.py`, `backprojection.py`
3. **Evaluation** - `evaluation/evaluator.py`, `metrics.py`, `results_aggregator.py`
4. **Configuration search** - `config_search/searcher.py`, `comparator.py`, `best_config_selector.py`
5. **Utilities** - `utils/logging_utils.py`, `visualization_utils.py`, `io_utils.py`
6. **Tests** - `tests/` - Unit and integration tests

See `docs/ARCHITECTURE.md` for module interfaces and responsibilities.
