# Project Architecture

## Overview

This project implements a pedestrian detection pipeline for fisheye images using a novel approach:

1. **Image Composition**: Generate multiple gnomonic projections from a single fisheye image
2. **Detection**: Apply YOLOv8 detector on the stitched composite image
3. **Backprojection**: Project detections back to fisheye coordinates
4. **Configuration Search**: Find optimal projection configurations (2×2, 2×4, 4×4, etc.)

## Directory Structure

```
project-root/
├── image-composer/              # [EXISTING] Creates composite images
│   ├── multi-persp.py          # Main composite generation
│   ├── fisheye-to-persp.py     # Gnomonic projection implementation
│   ├── config.py               # Projection parameters
│   └── presets.py              # Pre-defined configurations
│
├── detection-pipeline/          # Detection pipeline
│   ├── pipeline.py             # Main orchestrator
│   ├── yolo_detector.py        # YOLO integration
│   ├── backprojection.py       # Fisheye reprojection
│   ├── config.py               # Pipeline parameters
│   └── models/                 # YOLO model files
│
├── datasets/                    # Dataset management
│   ├── dataset_loader.py       # Abstract base class
│   ├── bomni_loader.py         # BOMNI adapter
│   ├── piropo_loader.py        # PIROPO adapter
│   ├── config.py               # Dataset paths & metadata
│   ├── adapters/               # Annotation format adapters
│   │   ├── xml_adapter.py      # PASCAL VOC XML parsing
│   │   └── txt_adapter.py      # Text format parsing
│   └── data/                   # [NOT TRACKED] Dataset files
│
├── evaluation/                  # Evaluation & metrics
│   ├── evaluator.py            # Main evaluator
│   ├── metrics.py              # Metric computations (AP, mAP, etc.)
│   ├── results_aggregator.py   # Aggregate across images/datasets
│   └── config.py               # Evaluation settings
│
├── config_search/              # Configuration search
│   ├── searcher.py             # Grid/random/Bayesian search
│   ├── comparator.py           # Configuration comparison
│   ├── best_config_selector.py # Select best config(s)
│   └── config.py               # Search parameters
│
├── results/                     # Output directory (NOT TRACKED)
│   ├── metrics/                # JSON metric files
│   ├── visualizations/         # Plots and images
│   └── reports/                # Comparison reports
│
├── utils/                       # Shared utilities
│   ├── logging_utils.py        # Logging configuration
│   ├── visualization_utils.py  # Plotting helpers
│   └── io_utils.py             # File I/O utilities
│
├── tests/                       # Unit & integration tests
│   ├── test_pipeline.py
│   ├── test_datasets.py
│   └── test_evaluation.py
│
├── docs/                        # Documentation
│   ├── ARCHITECTURE.md         # This file
│   ├── SETUP.md                # Installation & setup
│   └── API.md                  # Module APIs
│
├── project_config.py           # Global project config
├── main.py                     # Entry point
└── .gitignore
```

## Data Flow

### Full Pipeline Execution
```
Dataset → Image Loader → Projection Config
    ↓
Create Composite (image-composer)
    ↓
Run YOLO Detection
    ↓
Backproject to Fisheye
    ↓
Evaluate Metrics (vs ground truth)
    ↓
Store Results
```

### Configuration Search
```
Generate Config Combinations (grid search)
    ↓
For each config: Run full pipeline on dataset subset
    ↓
Collect metrics for each configuration
    ↓
Rank & compare
    ↓
Select best configuration
    ↓
Generate comparison report
```

## Module Responsibilities

### `detection-pipeline/`
**Responsibility**: Execute detection on individual images

- **pipeline.py**: Orchestrates the full pipeline for one image
- **yolo_detector.py**: Wraps YOLOv8 model inference
- **backprojection.py**: Handles geometric transformation from composite to fisheye
- **config.py**: Controls YOLO parameters, projection settings, visualization options

**Key Interface**:
```python
DetectionPipeline.run(
    fisheye_image: np.ndarray,
    projection_config: dict,
) -> Detections  # Detections in fisheye coordinates
```

### `datasets/`
**Responsibility**: Abstract dataset heterogeneity

- **dataset_loader.py**: Abstract base class defining the interface
- **bomni_loader.py**: Handles BOMNI-specific folder structure & XML annotations
- **piropo_loader.py**: Handles PIROPO-specific structure & text annotations
- **adapters/**: Pluggable format parsers for different annotation types
- **config.py**: Dataset paths and metadata

**Key Interface**:
```python
loader = DatasetLoader.create("piropo")  # Factory method
for image, annotations in loader.iterate():
    # annotations is standardized format regardless of source
```

### `evaluation/`
**Responsibility**: Compute standardized metrics

- **evaluator.py**: Main class that runs evaluation
- **metrics.py**: Individual metric implementations (AP, mAP, AR, F1, etc.)
- **results_aggregator.py**: Aggregate per-image results into dataset-level metrics
- **config.py**: Which metrics to compute, thresholds, comparison strategy

**Key Interface**:
```python
evaluator = Evaluator(config)
metrics = evaluator.evaluate(
    detections: List[Detection],
    ground_truth: List[Annotation]
)
# metrics = {"map": 0.75, "ap50": 0.82, "ar": 0.68, ...}
```

### `config_search/`
**Responsibility**: Find optimal configurations

- **searcher.py**: Implements search algorithm (grid, random, Bayesian, genetic)
- **comparator.py**: Compares two configurations, returns winner
- **best_config_selector.py**: Extracts top-N configurations from search results
- **config.py**: Search parameters (grid points, datasets, optimization metric)

**Key Interface**:
```python
searcher = Searcher.create(strategy="grid")
results = searcher.search(
    dataset_name="piropo",
    evaluation_fn=lambda cfg: evaluate_config(cfg)
)
# results = sorted list of (config, metrics) tuples
```

### `utils/`
**Responsibility**: Shared utilities

- **logging_utils.py**: Centralized logging configuration
- **visualization_utils.py**: Plotting (detection visualizations, metric curves, comparison plots)
- **io_utils.py**: JSON/pickle serialization, path utilities

## Configuration Philosophy

**Principle**: No command-line arguments; use config files for everything.

- **Global**: `project_config.py` - Dataset paths, project directories, debug flags
- **Per-Module**: `<module>/config.py` - Component-specific settings
- **Runtime Override**: Minimal - can pass config objects, but prefer modifying config.py files

This keeps experiments reproducible and configurations clear.

## Dataset Structure Requirements

Your datasets must have this structure to work with the loaders:

**BOMNI**:
```
datasets/data/bomni/
├── images/
│   └── <sequence>/<image_id>.jpg
└── annotations/
    └── <sequence>/<image_id>.<format>
```

**PIROPO**:
```
datasets/data/piropo/
├── images/
│   └── <scene>/<frame_id>.jpg
└── annotations/
    └── <scene>/<frame_id>.<format>
```

To add new datasets, create:
1. `datasets/<dataset_name>_loader.py` extending `DatasetLoader`
2. Update `datasets/config.py` with paths and metadata
3. Add annotation parser to `datasets/adapters/` if needed

## Extending the Project

### Add a new dataset
1. Create `datasets/mynewdataset_loader.py`
2. Add entry to `DATASETS_META` in `datasets/config.py`
3. Implement `load_image()` and `load_annotations()` methods

### Add a new evaluation metric
1. Add function to `evaluation/metrics.py`
2. Add to `COMPUTE_METRICS` in `evaluation/config.py`
3. Update `results_aggregator.py` to include it

### Change the search strategy
1. Modify `SEARCH_STRATEGY` in `config_search/config.py`
2. Implement strategy class in `config_search/searcher.py`

## Next Steps

1. Review this structure and suggest changes
2. Implement individual modules following the interfaces above
3. Create unit tests in `tests/`
4. Set dataset paths in `datasets/config.py`
5. Run `python main.py` to start experiments
