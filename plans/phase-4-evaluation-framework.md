# Phase 4: Evaluation Framework - Projection Configuration Testing

## Overview
Create extensible evaluation infrastructure to test multiple projection configurations on a configurable list of datasets. Currently BOMNI is the only dataset in the list, but architecture is designed to support multiple datasets.

## Requirements Summary
1. JSON file defines multiple projection configurations with common YOLO model
2. Run detection on BOMNI dataset for each configuration
3. Save predicted bboxes in standard JSON format (same structure as GT)
4. Generate visualizations (GT + predictions overlaid)
5. Organize outputs by: JSON-filename → config-ID → dataset-name → results
6. Program knows dataset structure based on dataset name

## Implementation Plan

### Step 1: Create Projection Configuration JSON Schema

**File**: `evaluation/projection_configs/test_configs_v1.json`

**Structure**:
```json
{
  "yolo_model": "models/yolov8n.pt",
  "configurations": [
    {
      "id": "grid-2x2-fov60",
      "name": "2×2 Grid, 60° FOV",
      "proj_nbr": 4,
      "fov_h": 60.0,
      "fov_v": 60.0,
      "latitude": 45.0,
      "lon_0": 0.0,
      "lon_step": 90.0,
      "grid": [2, 2],
      "comp_sz": [640, 640],
      "target_mp": "auto"
    },
    {
      "id": "grid-3x3-fov60",
      "name": "3×3 Grid, 60° FOV (baseline)",
      "proj_nbr": 9,
      "fov_h": 60.0,
      "fov_v": 60.0,
      "latitude": 45.0,
      "lon_0": 0.0,
      "lon_step": 40.0,
      "grid": [3, 3],
      "comp_sz": [640, 640],
      "target_mp": "auto"
    },
    {
      "id": "grid-2x4-fov96-chiang",
      "name": "2×4 Grid, 96° FOV (Chiang baseline)",
      "proj_nbr": 8,
      "fov_h": 96.0,
      "fov_v": 48.0,
      "latitude": 36.0,
      "lon_0": 0.0,
      "lon_step": 45.0,
      "grid": [2, 4],
      "comp_sz": [640, 640],
      "target_mp": 0.48
    }
  ]
}
```

### Step 2: Create Dataset Structure Registry

**File**: `evaluation/dataset_registry.py`

**Purpose**: Define folder structure for each supported dataset.

**Structure**:
```python
DATASET_STRUCTURES = {
    "bomni": {
        "annotation_root": "Standard-annotations-ours",
        "frames_root": "frames",
        "scenarios": ["scenario1"],
        "sequences": {
            "scenario1": ["top-0", "top-1", "top-2", "top-3"]
        },
        "fisheye_center": (320.0, 240.0),
        "image_size": (640, 480)
    }
    # Add other datasets later when needed
}
```

**Note**: Only BOMNI defined for now. Other datasets will be added when they're ready.

### Step 3: Create Evaluation Runner

**File**: `evaluation/run_projection_evaluation.py`

**Class**: `ProjectionEvaluator`

**Key Methods**:

1. **`__init__(config_json_path, dataset_names)`**
   - Load JSON configuration
   - Parse projection configs and YOLO model
   - Validate dataset names exist in registry

2. **`run_evaluation()`**
   - Main orchestrator
   - For each configuration ID:
     - For each dataset:
       - Run detection on all images
       - Save predicted bboxes
       - Generate visualizations

3. **`_run_detection_on_dataset(config, dataset_name)`**
   - Load dataset images using registry structure
   - For each image:
     - Configure detection pipeline with projection params
     - Run detection
     - Backproject to fisheye
     - Convert to standard JSON
     - Save predictions

4. **`_save_predictions(predictions, output_path)`**
   - Save to: `configs-comparison/{json-name}/{config-id}/{dataset}/bboxes-numeric/{dataset_structure}/*`
   - Format: Same as GT standard JSON (raw coordinate data, no metrics)

5. **`_generate_visualizations(dataset_name, config_id)`**
   - For each image:
     - **Composite visualization**: Save composite image with YOLO detections
       - Save to: `bboxes-visuals/composite/{image_name}.jpg`
     - **Fisheye visualization**: Load GT and predicted annotations
       - Draw GT bboxes FIRST (green, thickness=2)
       - Draw predicted bboxes on top (red, thickness=1, thinner)
       - Save to: `bboxes-visuals/fisheye/{dataset_structure}/*` (follows dataset folder structure)

### Step 4: Visualization Module

**File**: `evaluation/visualization.py`

**Function**: `visualize_gt_and_predictions(image, gt_annotations, pred_annotations)`

**Implementation**:
- Draw GT bboxes FIRST: Green (#00FF00), thickness=2
- Draw predicted bboxes on top: Red (#FF0000), thickness=1 (thinner so both visible when aligned)
- Add legend indicating colors
- Return annotated image

**Two visualization outputs**:
1. **Composite visualizations**: Detections on composite images (before backprojection)
   - Saved to: `bboxes-visuals/composite/{image_name}.jpg`
   - Shows YOLO detections on stitched composite

2. **Fisheye visualizations**: Backprojected bboxes on fisheye images (after backprojection)
   - Saved to: `bboxes-visuals/fisheye/{dataset_structure}/*`
   - Shows GT (green, thick) + predictions (red, thin) on original fisheye
   - Allows verification of backprojection correctness

### Step 5: Output Structure Manager

**File**: `evaluation/output_manager.py`

**Class**: `OutputPathManager`

**Purpose**: Generate consistent output paths following user's structure.

**Key Methods**:
- `get_bboxes_numeric_path(json_name, config_id, dataset_name, relative_path)`
- `get_bboxes_visuals_composite_path(json_name, config_id, dataset_name, image_name)`
- `get_bboxes_visuals_fisheye_path(json_name, config_id, dataset_name, relative_path)`
- `ensure_directories_exist(base_path)`

**Design Philosophy**:
- DO NOT include dataset-specific folder structure (scenario, sequence, etc.) as arguments
- Registry knows the structure for each dataset
- Methods only need: json_name, config_id, dataset_name, and relative_path
- Relative path is computed based on dataset structure from registry

**Naming Convention**:
- `bboxes-numeric/` - Raw bbox coordinates in JSON format
- `bboxes-visuals/composite/` - Detections on composite images (before backprojection)
- `bboxes-visuals/fisheye/` - Backprojected bboxes on fisheye images (GT + predictions)

### Step 6: Integration with Existing Pipeline

**Use existing modules**:
- `detection_pipeline.pipeline.DetectionPipeline` - For running detection
- `detection_pipeline.backprojection.backproject_detections` - For backprojection
- `image_composer.generate_composite_from_config` - For composite generation
- `datasets.utils.visualization.draw_rotated_bbox_on_image` - For drawing

**Modifications needed**:
- None to existing modules (use as-is)
- Create wrapper to convert detection output to standard JSON format

### Step 7: Configuration Script

**File**: `evaluation/run_backprojection_visual_test.py`

**Purpose**: User-editable configuration for running evaluation.

**Settings**:
```python
# Configuration JSON file to use
CONFIG_JSON = "evaluation/projection_configs/test_configs_v1.json"

# Datasets to evaluate on (extensible list - designed to support multiple datasets)
DATASETS = ["bomni"]  # Only BOMNI for now, will extend later

# Dataset root paths (must specify root for each dataset in DATASETS list)
DATASET_ROOTS = {
    "bomni": "datasets/all-datasets/BOMNI-corrected"
    # Add other dataset roots when ready
}

# Verbosity
VERBOSE = True
```

**Design**: Code is EXTENSIBLE - iterates through all datasets in DATASETS list. Currently only BOMNI is in the list. When other datasets are ready, user adds them to the list and specifies their root paths.

**Execution**:
```bash
python evaluation/run_backprojection_visual_test.py
```

## File Changes Summary

### New Files:
1. `evaluation/projection_configs/test_configs_v1.json` - Projection configurations
2. `evaluation/dataset_registry.py` - Dataset structure definitions
3. `evaluation/run_projection_evaluation.py` - Main evaluation runner
4. `evaluation/visualization.py` - GT + prediction visualization
5. `evaluation/output_manager.py` - Output path management
6. `evaluation/run_backprojection_visual_test.py` - User configuration script

### Modified Files:
- None (all existing modules used as-is)

## Output Structure

```
evaluation/
├── configs-comparison/
│   └── test_configs_v1/
│       ├── grid-2x2-fov60/
│       │   └── bomni/                        # Only BOMNI for now
│       │       ├── bboxes-numeric/           # Raw bbox coordinates (JSON)
│       │       │   └── scenario1/            # BOMNI-specific structure
│       │       │       ├── top-0/
│       │       │       │   ├── 0001.json
│       │       │       │   ├── 0011.json
│       │       │       │   └── ...
│       │       │       ├── top-1/
│       │       │       ├── top-2/
│       │       │       └── top-3/
│       │       └── bboxes-visuals/
│       │           ├── composite/            # Detections on composite images
│       │           │   ├── 0001.jpg
│       │           │   ├── 0011.jpg
│       │           │   └── ...
│       │           └── fisheye/              # Backprojected on fisheye (GT + pred)
│       │               └── scenario1/        # BOMNI-specific structure
│       │                   ├── top-0/
│       │                   │   ├── 0001.jpg
│       │                   │   ├── 0011.jpg
│       │                   │   └── ...
│       │                   ├── top-1/
│       │                   ├── top-2/
│       │                   └── top-3/
│       ├── grid-3x3-fov60/
│       │   └── bomni/
│       │       └── ...
│       └── grid-2x4-fov96-chiang/
│           └── bomni/
│               └── ...
└── projection_configs/
    └── test_configs_v1.json
```

**Key Points**:
- `bboxes-numeric/`: Raw coordinate data (JSON), follows dataset structure
- `bboxes-visuals/composite/`: Detections on composite images (verification of YOLO)
- `bboxes-visuals/fisheye/`: Backprojected on fisheye with GT + predictions (verification of backprojection)
- GT drawn FIRST (green, thick), predictions on top (red, thin) - both visible when aligned
- Currently only BOMNI in output
- Dataset structure (scenario1, top-0, etc.) comes from registry, not hardcoded in path methods

## Execution Flow

1. User edits `evaluation/run_backprojection_visual_test.py` to specify:
   - CONFIG_JSON path
   - DATASETS list
   - DATASET_ROOTS paths

2. User runs: `python evaluation/run_backprojection_visual_test.py`

3. Program:
   - Loads JSON configuration
   - Validates dataset names (all in DATASETS must have root in DATASET_ROOTS)
   - For each configuration in JSON:
     - For each dataset in DATASETS list:
       - Loads all images from dataset (using registry structure)
       - Runs detection pipeline with projection config
       - Saves predicted bboxes to bboxes-numeric/ (standard JSON format)
       - Generates visualizations to bboxes-visuals/ (GT + predictions)

4. Output:
   - Raw bbox coordinates: `configs-comparison/{json-name}/{config-id}/{dataset}/bboxes-numeric/{dataset_structure}/*`
   - Composite visualizations: `configs-comparison/{json-name}/{config-id}/{dataset}/bboxes-visuals/composite/*`
   - Fisheye visualizations: `configs-comparison/{json-name}/{config-id}/{dataset}/bboxes-visuals/fisheye/{dataset_structure}/*`

5. Extensible design:
   - Currently DATASETS = ["bomni"] (only one dataset)
   - When other datasets are ready, add to DATASETS list and they'll be evaluated automatically
   - Each dataset gets independent folder structure within each config folder

## Phase 4A (Later): Metrics Computation

**File**: `evaluation/compute_metrics.py`

**Purpose**: Compute mAP, precision, recall for rotated bboxes.

**Input**: Predicted bboxes + GT annotations from `configs-comparison/` structure

**Output**: CSV/JSON with metrics per configuration per dataset

**Will implement after predictions are generated and validated.**

## Critical Design Decisions

1. **Standard JSON format for predictions**: Ensures consistency with GT, enables easy metrics computation
2. **Config ID naming**: User controls unique IDs in JSON, used as folder names
3. **Dataset registry**: Program knows structure, user only provides dataset root + name
4. **Multi-dataset support**: Code designed to be extensible - user specifies list of datasets in config
5. **Clear output naming**:
   - `bboxes-numeric/` = raw coordinate data (JSON)
   - `bboxes-visuals/composite/` = detections on composite (pre-backprojection)
   - `bboxes-visuals/fisheye/` = backprojected on fisheye with GT
6. **Visualization layers**: GT drawn first (green, thick=2), predictions on top (red, thin=1) - both visible
7. **No dataset-specific arguments**: Path methods use relative paths, registry handles dataset structure
8. **No argument parsers**: All config in `.py` files (following project rules)
9. **Modular design**: Separation of concerns (runner, output manager, visualization, registry)

## Testing Strategy

1. Create `test_configs_v1.json` with 3 configurations
2. Run on BOMNI dataset (251 images) - it's the only dataset in DATASETS list for now
3. Verify output structure matches specification
4. Manually inspect 5-10 visualizations to validate GT + prediction overlay
5. Verify predicted JSON files match GT structure
6. Once validated, can add more projection configurations
7. Later when other datasets are ready, add them to DATASETS list and registry
