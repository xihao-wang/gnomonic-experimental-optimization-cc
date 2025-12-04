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
- **Multi-Dataset Support**: Extensible for custom datasets
- **More features to come**: ...

## Project Structure

```
project-root/
├── image_composer/              # ✅ Composite image generation from fisheye
│   ├── config.py
│   ├── multi_persp.py
│   ├── presets.py
│   └── README.md
├── detection_pipeline/          # ✅ YOLO detection + backprojection
│   ├── config.py                # YACS configuration
│   ├── pipeline.py              # Main orchestrator
│   ├── yolo_detector.py         # YOLO wrapper
│   ├── backprojection.py        # Geometric transformation
│   ├── models/                  # YOLO models
│   └── results/                 # Detection outputs
├── datasets/                    # ✅ Dataset managers & preparation
│   ├── base_manager.py          # Abstract base class
│   ├── bomni_manager.py         # BOMNI-specific operations
│   ├── piropo_manager.py        # PIROPO placeholder
│   ├── prepare_dataset_step1.py # Automated preparation
│   ├── prepare_dataset_step2.py # Semi-automated cleanup
│   ├── PREPARATION_README.md    # Two-step workflow guide
│   ├── utils/
│   │   └── visualization.py     # Shared visualization
│   └── all-datasets/            # Raw and processed datasets
├── evaluation/                  # 🔄 Evaluation infrastructure
│   ├── config.py                # YACS configuration
│   ├── bomni_dataset.py         # BOMNI runtime loader
│   ├── visualize_datasets.py    # Multi-dataset visualization
│   ├── README.md                # Dataset documentation
│   └── results/                 # Evaluation outputs
├── config_search/               # ⏸️ Configuration search (Phase 5)
│   └── config.py
└── results/                     # Experiment outputs
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
Edit YACS configuration files:
1. `detection_pipeline/config.py` - Pipeline parameters (YOLO, backprojection)
2. `evaluation/config.py` - Dataset paths, evaluation settings
3. `config_search/config.py` - Search parameters (Phase 5)

### 3. Prepare Datasets
See `datasets/PREPARATION_README.md` for complete workflow:
```bash
# Step 1: Automated preparation
python datasets/prepare_dataset_step1.py

# Manual review: Delete incorrect visualizations

# Step 2: Semi-automated cleanup
python datasets/prepare_dataset_step2.py
```

### 4. Run Detection Pipeline
```bash
python detection_pipeline/test_pipeline.py
```

## Documentation

- **`detection_pipeline/README.md`**: Detection pipeline usage and architecture
- **`datasets/PREPARATION_README.md`**: Two-step dataset preparation workflow
- **`evaluation/README.md`**: Dataset infrastructure and annotation formats
- **`image_composer/README.md`**: Composite image generation API

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

### `image_composer/`
Generates gnomonic projection composites from fisheye images. Configurable projection layouts (2×2, 3×3, 2×4, etc.) and FOV parameters.

### `detection_pipeline/`
Executes detection on individual images. Uses image_composer API to generate composites, runs YOLO, backprojects detections to fisheye coordinates using 5-point geometric transformation.

### `datasets/`
Class-based dataset managers for preparation and validation. Two-step workflow: automated preparation → manual review → semi-automated cleanup. Converts all datasets to unified standard JSON format.

### `evaluation/`
Runtime dataset loaders (BOMNI implemented) and visualization tools. Will compute metrics (AP, mAP, AR, F1) comparing detections to ground truth (Phase 4).

### `config_search/`
Will systematically search for optimal projection configurations (Phase 5). Grid search across projection counts, FOV values, and layouts.

## Configuration Philosophy

**Principle**: Use YACS config files, not command-line arguments.

- **Per-Module**: Each module has its own `config.py` using YACS framework
- **No argparse**: All parameters configured by editing `.py` files directly
- **Reproducible**: Configuration committed to git for experiment tracking

This keeps experiments reproducible and configurations explicit.

## Supported Datasets

### BOMNI
- Omnidirectional pedestrian detection dataset (251 verified annotations)
- Standard JSON format (unified across all datasets)
- Configurable at: `evaluation/config.py` (DATASETS.BOMNI node)
- Two-step preparation workflow implemented

### PIROPO
- Panoramic indoor pedestrian dataset
- Placeholder implementation (not yet integrated)

### Adding New Datasets
1. Create `datasets/<dataset_name>_manager.py` extending `BaseDatasetManager`
2. Implement `convert_to_standard_format()` method
3. Add dataset configuration in `evaluation/config.py`
4. Use two-step preparation workflow

## Key Design Decisions

1. **Config-Driven Design**: All runtime parameters in config files
2. **Modular Pipeline**: Each stage (composite → detect → backproject → evaluate) is independent
3. **Dataset Abstraction**: Single loader interface supports multiple formats
4. **Results Isolation**: Experiment outputs in separate `results/` directory
5. **Incremental Search**: Configuration search can use dataset subsets for speed

## Next Steps

1. **Phase 4**: Implement evaluation metrics (IoU for rotated boxes, precision, recall, mAP)
2. **Phase 5**: Configuration search to find optimal projection setup
3. **Testing**: Run detection pipeline on different projection configurations
4. **Analysis**: Compare performance across configurations on BOMNI dataset
5. **Documentation**: Document optimal configuration findings

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

**YOLO model not found**: Models auto-download to `detection_pipeline/models/`. Check internet connection or download manually.

**Dataset not found**: Verify paths in `evaluation/config.py` (DATASETS node) are correct and use absolute paths.

**GPU not detected**: Detection pipeline auto-detects GPU. Check `detection_pipeline/config.py` for device settings.

**Visualization fails**: Ensure you run scripts from project root, not from subdirectories.

## Research Objective

Find the optimal projection configuration that maximizes detection performance (mAP) while considering:
- Number of projections (computational cost)
- Field of view coverage
- Camera angles
- Computational efficiency

Results will be compared across datasets to identify configurations that generalize well.
