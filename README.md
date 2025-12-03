# Pedestrian Detection in Fisheye Images

⚠️ **PROJECT STATUS: Early Stage Development**

**Current Progress:**
- ✅ Phase 1: Infrastructure complete
- ✅ Phase 2.1-2.3: Detection pipeline functional
- ✅ Phase 2.4: Results organization & GPU auto-detection
- ✅ Phase 2B.1: Backprojection first pass (5-point geometric transformation)
- ⏳ Phase 2B.2: Post-processing (NMS for duplicate detections - deferred)
- ✅ Phase 3: Dataset infrastructure complete (extensible architecture for other datasets, BOMNI dataset fully implemented)
- ⏳ Phase 4: Evaluation metrics (next)
- ⏳ Phase 5: Configuration search (planned)

---

## Project Overview

This research project aims to detect pedestrians in omnidirectional (fisheye) images using a **projection-based approach**:

1. **Composition**: Generate multiple gnomonic projections from a fisheye image and stitch them into a single composite image
2. **Detection**: Apply YOLOv8 detector on the composite image
3. **Backprojection**: Project detections back to fisheye coordinates
4. **Optimization**: Find the best projection configuration (2×2, 2×4, 4×4, etc.) through systematic search

## Key Features

- **No Model Training**: Leverages existing YOLOv8 models
- **Configuration-Driven**: All parameters controlled via `config.py` files, not argument parsers
- **Multi-Dataset Support**: BOMNI, PIROPO, and extensible for custom datasets
- **Systematic Search**: Grid search for optimal projection parameters
- **Comprehensive Evaluation**: Standard metrics (AP, mAP, AR, F1) computed at multiple IoU thresholds

## Project Structure

```
project-root/
├── image_composer/              # ✓ Existing API - creates composite images
├── detection_pipeline/          # Detection pipeline implementation
├── datasets/                    # Dataset loaders & adapters
├── evaluation/                  # Metrics computation & analysis
├── config_search/               # Configuration search & comparison
├── results/                     # Experiment outputs (metrics, visualizations, reports)
├── utils/                       # Shared utilities
├── tests/                       # Unit & integration tests
├── docs/                        # Documentation
│   ├── ARCHITECTURE.md         # Detailed architecture & data flow
│   ├── SETUP.md                # Installation & configuration guide
│   └── API.md                  # Module interfaces & classes
├── project_config.py            # Global project configuration
└── main.py                      # Project entry point
```

## Quick Start

### 1. Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install ultralytics opencv-python numpy scikit-learn matplotlib
```

### 2. Configure
Edit configuration files in order:
1. `project_config.py` - Project-wide settings
2. `datasets/config.py` - Dataset paths
3. `detection_pipeline/config.py` - Pipeline parameters
4. `evaluation/config.py` - Evaluation metrics
5. `config_search/config.py` - Search parameters

### 3. Run
```bash
python main.py
```

For detailed setup instructions, see `docs/SETUP.md`

## Documentation

- **`docs/ARCHITECTURE.md`**: System design, module responsibilities, data flow
- **`docs/SETUP.md`**: Installation, configuration, dataset preparation, troubleshooting
- **`docs/API.md`**: Module interfaces, class signatures, usage examples

## Project Implementation Plan

### Phase 1: Infrastructure (✓ Done)
- [x] Create folder structure
- [x] Define configuration system
- [x] Document architecture & APIs
- [x] Create entry point

### Phase 2: Core Modules (Next)
- [ ] Implement dataset loaders (BOMNI, PIROPO adapters)
- [ ] Implement detection pipeline (YOLO integration, backprojection)
- [ ] Implement evaluation metrics
- [ ] Implement configuration search

### Phase 3: Testing & Optimization
- [ ] Unit tests for each module
- [ ] Integration tests for full pipeline
- [ ] Performance profiling
- [ ] Documentation of results

## Module Overview

### `detection_pipeline/`
Executes detection on individual images. Uses the image_composer API to generate composites, runs YOLO, and backprojects results to fisheye coordinates.

### `datasets/`
Abstracts different dataset formats (BOMNI, PIROPO). Provides unified `DatasetLoader` interface that handles format differences transparently.

### `evaluation/`
Computes standardized metrics (AP, mAP, AR, F1) comparing detections to ground truth. Aggregates results across images and datasets.

### `config_search/`
Systematically searches for optimal projection configurations. Supports grid search (and extensible to random, Bayesian, genetic algorithms).

## Configuration Philosophy

**Principle**: Use config files, not command-line arguments.

- **Global**: `project_config.py`
- **Per-Module**: `<module>/config.py`
- **Minimal Runtime**: Can pass config objects, but prefer modifying files

This keeps experiments reproducible and configurations explicit.

## Supported Datasets

### BOMNI
- Omnidirectional pedestrian detection dataset
- XML-format annotations
- Configurable at: `datasets/config.py` (`BOMNI_ROOT`)

### PIROPO
- Panoramic indoor pedestrian dataset
- Text-format annotations
- Configurable at: `datasets/config.py` (`PIROPO_ROOT`)

### Adding New Datasets
1. Create `datasets/<dataset_name>_loader.py` extending `DatasetLoader`
2. Add entry to `DATASETS_META` in `datasets/config.py`
3. Create annotation parser in `datasets/adapters/` if needed

## Key Design Decisions

1. **Config-Driven Design**: All runtime parameters in config files
2. **Modular Pipeline**: Each stage (composite → detect → backproject → evaluate) is independent
3. **Dataset Abstraction**: Single loader interface supports multiple formats
4. **Results Isolation**: Experiment outputs in separate `results/` directory
5. **Incremental Search**: Configuration search can use dataset subsets for speed

## Next Steps

1. **Review** this structure and suggest changes/improvements
2. **Implement** modules following the interfaces in `docs/API.md`
3. **Create** unit tests in `tests/`
4. **Configure** dataset paths in `datasets/config.py`
5. **Run** `python main.py` to verify setup
6. **Execute** experiments once implementations are complete

## Dependencies

```
ultralytics    # YOLOv8
opencv-python  # Image processing
numpy          # Numerical computing
scikit-learn   # Metrics (AP, etc.)
matplotlib     # Visualization (optional)
pillow         # Image I/O
```

## Troubleshooting

**YOLO model not found**: Models auto-download to `~/.yolov8/`. Check internet connection or download manually.

**Dataset not found**: Verify paths in `datasets/config.py` are correct and use absolute paths.

**GPU not detected**: Set `YOLO_DEVICE = "cpu"` in `detection_pipeline/config.py`.

**Memory issues**: Reduce `COMPOSITE_SIZE` or use smaller YOLO model (`yolov8n.pt`).

See `docs/SETUP.md` for more troubleshooting.

## Research Objective

Find the optimal projection configuration that maximizes detection performance (mAP) while considering:
- Number of projections (computational cost)
- Field of view coverage
- Camera angles
- Computational efficiency

Results will be compared across datasets to identify configurations that generalize well.
