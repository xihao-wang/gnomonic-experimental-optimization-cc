# Evaluation Module

This module handles dataset loading, annotation visualization, and evaluation metrics for pedestrian detection in fisheye images.

## BOMNI Dataset Implementation

### Overview

The BOMNI (Boğaziçi University Multi-Omnidirectional Video Tracking Database) dataset has been integrated with support for rotated bounding box annotations. These annotations were provided by a third-party research project ([omnidet-rotinv](https://github.com/hitachi-rd-cv/omnidet-rotinv)) specifically for omnidirectional pedestrian detection.

## Dataset Preparation Workflow (From Raw Data)

If you have just downloaded the raw BOMNI dataset with Tamura annotations from this repository ([omnidet-rotinv](https://github.com/hitachi-rd-cv/omnidet-rotinv)), follow these steps to prepare the standard annotation format and generate visualizations.

### Prerequisites

```
datasets/all-datasets/
├── bomni-5841/
│   └── scenario1/
│       ├── top-0.mp4, top-1.mp4, top-2.mp4, top-3.mp4
│       └── ...
└── omnidet-rotinv-master/omnidet-rotinv-master/
    └── rotate/bomni/rotate/scenario1/
        ├── top-0/ (*.xml files)
        ├── top-1/ (*.xml files)
        ├── top-2/ (*.xml files)
        └── top-3/ (*.xml files)
```

### Step 1: Extract Frames from Videos

```python
from evaluation.lib.config import get_cfg
from datasets.bomni_manager import BOMNIManager

cfg = get_cfg()
manager = BOMNIManager(cfg)

# Extract all frames from the 4 video sequences
manager.extract_frames(
    video_dir="datasets/all-datasets/bomni-5841/scenario1",
    output_dir="datasets/all-datasets/bomni-5841/frames/scenario1",
    sequences=["top-0", "top-1", "top-2", "top-3"]
)
```

**Output**: 3345 frames extracted across 4 sequences
- top-0: 1001 frames
- top-1: 794 frames
- top-2: 643 frames
- top-3: 907 frames

### Step 2: Cleanup Unannotated Frames

Tamura annotations are sparse (~10% of frames). Remove frames without annotations to save disk space:

```python
manager.cleanup_unannotated_frames(
    frames_dir="datasets/all-datasets/bomni-5841/frames/scenario1",
    annotations_dir="datasets/all-datasets/omnidet-rotinv-master/omnidet-rotinv-master/rotate/bomni/rotate/scenario1",
    sequences=["top-0", "top-1", "top-2", "top-3"]
)
```

**Output**: 337 frames kept (only annotated frames)

### Step 3: Convert to Standard JSON Format

Convert Tamura's Pascal VOC XML format to our unified standard JSON format:

```python
manager.convert_to_standard_format(
    input_format="tamura",
    input_dir="datasets/all-datasets/omnidet-rotinv-master/omnidet-rotinv-master/rotate/bomni/rotate/scenario1",
    output_dir="datasets/all-datasets/BOMNI-corrected/Standard-annotations-ours/scenario1",
    sequences=["top-0", "top-1", "top-2", "top-3"]
)
```

**Output**: 337 JSON files created with precomputed values:
```json
{
  "center_x": float,
  "center_y": float,
  "width": float,
  "height": float,
  "angle": float,
  "class_name": "person"
}
```

### Step 4: Visualize for Quality Review

Generate visualizations to manually review annotation quality:

```python
manager.visualize_annotations(
    annotations_dir="datasets/all-datasets/BOMNI-corrected/Standard-annotations-ours/scenario1",
    frames_dir="datasets/all-datasets/bomni-5841/frames/scenario1",
    sequences=["top-0", "top-1", "top-2", "top-3"],
    max_images="all",
    fisheye_center=(320.0, 240.0)
)
```

**Output**: 337 visualization images in `evaluation/results/bomni/annotation_visualization/`

### Step 5: Manual Quality Review

1. Open `evaluation/results/bomni/annotation_visualization/{sequence}/`
2. Review each visualization image
3. **DELETE images with incorrect annotations** (wrong bbox position, missing detection, false positive, etc.)
4. Keep only images with correct annotations

### Step 6: Cleanup Incorrect Annotations

After manual review, remove annotation files and frames that correspond to deleted visualizations:

```python
manager.cleanup_incorrect_annotations(
    visualization_dir="evaluation/results/bomni/annotation_visualization",
    annotations_dir="datasets/all-datasets/BOMNI-corrected/Standard-annotations-ours/scenario1",
    frames_dir="datasets/all-datasets/bomni-5841/frames/scenario1",
    sequences=["top-0", "top-1", "top-2", "top-3"]
)
```

**Output**: Only verified correct annotations remain (e.g., 251 out of 337)

### Step 7: Regenerate Final Visualizations

Generate clean visualizations with only correct annotations:

```python
# Update config to use corrected annotations
cfg.DATASETS.BOMNI.STANDARD_ANNOTATIONS_DIR = "datasets/all-datasets/BOMNI-corrected/Standard-annotations-ours/scenario1"

# Visualize all corrected annotations
from evaluation.visualize_datasets import main
main()
```

**Output**: Final verified visualizations in `evaluation/results/bomni/annotation_visualization/`

### Complete Python Script

Create a file `datasets/prepare_bomni.py`:

```python
"""
Complete BOMNI dataset preparation workflow.
"""
from evaluation.lib.config import get_cfg
from datasets.bomni_manager import BOMNIManager

def main():
    cfg = get_cfg()
    manager = BOMNIManager(cfg)

    print("Step 1: Extract frames from videos")
    manager.extract_frames()

    print("\nStep 2: Cleanup unannotated frames")
    manager.cleanup_unannotated_frames()

    print("\nStep 3: Convert to standard JSON format")
    manager.convert_to_standard_format(input_format="tamura")

    print("\nStep 4: Visualize for quality review")
    manager.visualize_annotations(
        annotations_dir=cfg.DATASETS.BOMNI.STANDARD_ANNOTATIONS_DIR,
        frames_dir=cfg.DATASETS.BOMNI.FRAMES_DIR,
        sequences=cfg.DATASETS.BOMNI.SEQUENCES,
        max_images="all"
    )

    print("\n[DONE] Review visualizations and delete incorrect ones")
    print("Then run Step 6 to cleanup incorrect annotations")

if __name__ == "__main__":
    main()
```

Run: `python datasets/prepare_bomni.py`

### Dataset Structure

```
datasets/all-datasets/
├── bomni-5841/                                    # Original BOMNI dataset
│   ├── scenario1/                                 # Scenario 1 (single person)
│   │   ├── top-0.mp4, top-1.mp4, top-2.mp4,      # Top camera videos
│   │   ├── top-3.mp4, top-4.mp4                   # (640x480, variable frame rate)
│   │   ├── side-0.mp4, ... side-4.mp4            # Side camera videos
│   │   └── *.info files                           # Timing and sync information
│   ├── scenario2/                                 # Scenario 2 (three people)
│   ├── annotations/                               # Original VATIC annotations
│   ├── frames/scenario1/                          # Extracted frames (generated)
│   │   ├── top-0/                                 # 0001.jpg, 0002.jpg, ... 1001.jpg
│   │   ├── top-1/                                 # 0001.jpg, 0002.jpg, ... 0794.jpg
│   │   ├── top-2/                                 # 0001.jpg, 0002.jpg, ... 0643.jpg
│   │   └── top-3/                                 # 0001.jpg, 0002.jpg, ... 0907.jpg
│   └── calibration-*.txt                          # Camera calibration files
│
└── omnidet-rotinv-master/omnidet-rotinv-master/
    └── rotate/bomni/rotate/scenario1/             # Rotated bbox annotations (Pascal VOC XML)
        ├── top-0/                                 # 0001.xml, 0011.xml, ... (101 files)
        ├── top-1/                                 # 0001.xml, 0011.xml, ... (80 files)
        ├── top-2/                                 # 0001.xml, 0011.xml, ... (65 files)
        └── top-3/                                 # 0001.xml, 0011.xml, ... (91 files)
```

### Videos Included vs. Excluded

**INCLUDED** (with rotated bbox annotations):
- `scenario1/top-0` - 1001 frames, 101 annotated (single person, top camera)
- `scenario1/top-1` - 794 frames, 80 annotated (single person, top camera)
- `scenario1/top-2` - 643 frames, 65 annotated (single person, top camera)
- `scenario1/top-3` - 907 frames, 91 annotated (single person, top camera)

**EXCLUDED** (reasons):
- `scenario1/top-4` - **No rotated annotations available**
- `scenario1/side-*` (all 5 side camera videos) - **No rotated annotations available**
- `scenario2/*` (all scenario 2 videos) - **No rotated annotations available**

**Initial Total:** 337 annotated images across 4 sequences.

### Annotation Quality Review and Correction

After frame extraction, all annotations were visualized and manually reviewed. Due to frame alignment issues (likely caused by variable frame rate and missing frames in the original videos), **86 annotations (25.5%)** were found to have incorrect bbox positions, inaccurate bboxes, missing bboxes, or false positives.

**Manual Correction Process:**
1. Generated visualizations for all 337 annotations using `evaluation/visualize_bomni_annotations.py`
2. Manually reviewed each visualization
3. Deleted visualization images with incorrect annotations
4. Used `datasets/cleanup_incorrect_annotations.py` to remove corresponding XML and frame files

**Corrected Annotation Statistics:**

| Sequence | Original | Removed | Final | Accuracy |
|----------|----------|---------|-------|----------|
| top-0    | 101      | 25      | 76    | 75.2%    |
| top-1    | 80       | 28      | 52    | 65.0%    |
| top-2    | 65       | 32      | 33    | 50.8%    |
| top-3    | 91       | 1       | 90    | 98.9%    |
| **TOTAL** | **337** | **86** | **251** | **74.5%** |

**Final Dataset:** **251 manually verified correct annotations** across 4 sequences.

**Note:** The corrected annotations are stored in `datasets/all-datasets/BOMNI-corrected/Rotated-annotations/scenario1/` and are used by default in the evaluation configuration.

### Annotation Format

Annotations are provided in **Pascal VOC XML format**, but the fields are **repurposed as storage containers** for rotated bbox parameters rather than traditional axis-aligned boxes.

**What's STORED in the XML files:**
- `xmin, ymin, xmax, ymax` - Repurposed to encode rotated bbox center and dimensions (NOT axis-aligned coordinates)
- `class_name` - Always "person"

**What we DERIVE when loading annotations:**
- `center_x = (xmin + xmax) / 2` - Bbox center X coordinate
- `center_y = (ymin + ymax) / 2` - Bbox center Y coordinate
- `width = xmax - xmin` - Width of the tight-fit **rotated** bbox
- `height = ymax - ymin` - Height of the tight-fit **rotated** bbox
- `angle` - Rotation angle in degrees, calculated geometrically as: *"the angle between a vertical line and a line connecting the image center and bounding box center"* using `θ = arctan2(Δx, -Δy)`

The width and height represent the dimensions of the **rotated rectangle** that tightly fits the pedestrian, not an axis-aligned box.

### Frame Extraction

Frames were extracted from the 4 top camera videos using `datasets/extract_bomni_frames.py`. The script:
- Extracts all frames from each video
- Saves them with 4-digit serial numbers (0001.jpg, 0002.jpg, etc.) as expected by the annotation files
- Total extracted: **3345 frames** across all 4 sequences

Note: Only a subset of frames have annotations (~10% of frames are annotated).

### Configuration

The evaluation system uses **YACS** for configuration management. See `evaluation/lib/config.py` for all configurable parameters.

Key configuration nodes:
- `DATASETS.BOMNI` - Dataset paths, sequences, image dimensions, fisheye center
- `METRICS` - Evaluation metrics to compute (AP, mAP, F1, etc.)
- `VISUALIZATION` - Annotation visualization settings
- `OUTPUT` - Result storage options

### Usage

#### Loading the Dataset

```python
from evaluation.lib.config import get_cfg
from evaluation.bomni_dataset import BOMNIDataset

# Load configuration
cfg = get_cfg()
cfg.DATASETS.BOMNI.ENABLED = True
cfg.freeze()

# Create dataset
dataset = BOMNIDataset(cfg)

# Get dataset info
print(f"Total images: {len(dataset)}")

# Load a single image with annotations
item = dataset[0]
image = item["image"]           # numpy array (H, W, 3)
annotations = item["annotations"]  # List of rotated bbox dicts
sequence = item["sequence"]     # "top-0", "top-1", etc.
```

#### Visualizing Annotations

```python
# Visualize all annotations and save to disk
dataset.visualize_annotations()

# Or limit to first N images per sequence
dataset.visualize_annotations(max_images=10)

# Customize output directory
dataset.visualize_annotations(output_dir="evaluation/results/test_viz")
```

Visualizations are saved to: `evaluation/results/bomni/annotation_visualization/{sequence}/`

#### Testing the Implementation

Run the dataset module directly to test:
```bash
cd gnomonic-experimental-optimization-v2
python -m evaluation.bomni_dataset
```

This will:
1. Load the dataset
2. Print statistics
3. Visualize 2 images per sequence (8 total)
4. Save results to `evaluation/results/bomni/annotation_visualization/`

### Dataset Statistics

| Sequence | Total Frames | Annotated Frames | Annotations (person instances) |
|----------|--------------|------------------|-------------------------------|
| top-0    | 1001         | 101              | 288                           |
| top-1    | 794          | 80               | ~310 (estimated)              |
| top-2    | 643          | 65               | ~245 (estimated)              |
| top-3    | 907          | 91               | ~351 (estimated)              |
| **Total** | **3345**    | **337**          | **~1194**                     |

Note: Statistics from the omnidet-rotinv README match our extracted data.

### Rotation Angle Calculation

The rotation angle is computed using:
```python
def _calculate_rotation_angle(bbox_center_x, bbox_center_y):
    """Calculate angle from vertical (up) to vector from image center to bbox center."""
    img_center_x, img_center_y = fisheye_center  # (320, 240)
    dx = bbox_center_x - img_center_x
    dy = bbox_center_y - img_center_y
    angle_rad = np.arctan2(dx, -dy)  # -dy because y-axis points down
    angle_deg = np.degrees(angle_rad)
    return angle_deg
```

## Visualizing Dataset Annotations

### Quick Start

To visualize annotations for all configured datasets:

```bash
python -m evaluation.visualize_datasets
```

This will generate visualization images with rotated bounding boxes drawn on the fisheye images.

### Configuration

Edit `evaluation/visualize_datasets.py` to specify which datasets to visualize:

```python
DATASETS_TO_VISUALIZE = [
    "bomni",
    # "piropo",  # Uncomment when PIROPO is implemented
]
```

Control visualization settings in `evaluation/lib/config.py`:

```python
# Visualize ALL images
_C.VISUALIZATION.MAX_IMAGES_PER_SEQUENCE = "all"

# Or limit to N images per sequence for quick testing
_C.VISUALIZATION.MAX_IMAGES_PER_SEQUENCE = 10
```

### Output Location

Visualizations are saved to:
```
evaluation/results/{dataset_name}/annotation_visualization/{sequence}/
```

For example, BOMNI visualizations are saved to:
```
evaluation/results/bomni/annotation_visualization/
├── top-0/  (76 images)
├── top-1/  (52 images)
├── top-2/  (33 images)
└── top-3/  (90 images)
```

Each dataset gets its own folder to avoid conflicts.

## Metrics Evaluation System

### Overview

The metrics evaluation system compares different projection configurations by running the full detection pipeline and computing comprehensive metrics against ground truth annotations.

There are two independent workflows:

| Workflow | When to use |
|----------|-------------|
| **Incremental** (`run_single_config.py` + `run_comparator.py`) | Standard use — evaluate configs one at a time, compare whenever you like, never re-run a finished config |
| **Batch** (`run_metrics_evaluation.py`) | Legacy — evaluates all configs in one session; adding a new config reruns everything |

---

### Incremental Workflow (recommended)

Each configuration is evaluated once and its results are stored permanently. Adding a new configuration or re-running the comparator never touches previously evaluated configs.

**Step 1 — Define your projection configurations**

Add or edit entries in `evaluation/projection_configs_for_metrics.py`:

```python
class ProjectionConfigs:
    CONFIGS = [
        {
            "id": "grid-2x2-fov60",        # unique identifier (used as folder name)
            "name": "2x2 Grid, 60deg FOV",
            "proj_nbr": 4,
            "fov_h": 60.0, "fov_v": 60.0,
            "latitude": 45.0,
            "lon_0": 0.0, "lon_step": 90.0,
            "grid": [2, 2],
            "comp_sz": [640, 640],
            "target_mp": "auto"
        },
        # add more configs here...
    ]
```

**Step 2 — Pick one configuration to evaluate**

In `evaluation/lib/config.py`, set:

```python
_C.SINGLE_CONFIG_RUN.CONFIG_ID = "grid-2x2-fov60"   # must match an 'id' in step 1
```

Adjust any other run parameters in the same `SINGLE_CONFIG_RUN` section if needed
(datasets, max images, YOLO model, visuals, etc.).

**Step 3 — Run the evaluation for that configuration**

```bash
python evaluation/run_single_config.py
```

Results are saved to `evaluation/proj-conf-comparison/configs/grid-2x2-fov60/`.
Repeat steps 2–3 for every configuration you want to evaluate. Each run is fully
independent; completed configs are never re-evaluated (unless `OVERWRITE_EXISTING=True`).

**Step 4 — Compare all evaluated configurations**

```bash
python evaluation/run_comparator.py
```

The comparator auto-discovers every folder inside
`evaluation/proj-conf-comparison/configs/` and compares them. You can restrict which
configs to compare by listing them explicitly in `config.py`:

```python
_C.COMPARATOR.CONFIG_IDS = []                          # empty = compare all found
# or:
_C.COMPARATOR.CONFIG_IDS = ["grid-2x2-fov60", "chiang-2021-baseline"]
```

Output is written to a timestamped folder under
`evaluation/proj-conf-comparison/comparisons/comparison_YYYYMMDD_HHMMSS/`.

**Incremental output structure:**

```
evaluation/proj-conf-comparison/
├── configs/                              # one subfolder per evaluated config
│   ├── grid-2x2-fov60/
│   │   ├── config.json                  # projection params + sample indices
│   │   ├── bomni/
│   │   │   ├── metrics.json
│   │   │   ├── timing.json
│   │   │   ├── pr_curves/
│   │   │   └── visuals/
│   │   ├── piropo/
│   │   └── cepdof/
│   └── chiang-2021-baseline/
│       └── ...
└── comparisons/                         # comparator outputs (no re-execution)
    └── comparison_YYYYMMDD_HHMMSS/
        ├── comparison_config.json
        ├── consistency_check.txt
        ├── bomni_comparison_table.txt
        ├── piropo_comparison_table.txt
        ├── cepdof_comparison_table.txt
        ├── overall_winner.txt
        └── figures/
            ├── bomni_ap_bar.png
            ├── bomni_pr_curves_overlay.png
            └── ...
```

---

### Batch Workflow (legacy)

Evaluates all configurations defined in `projection_configs_for_metrics.py` in a single
session. Adding a new config requires re-running everything from scratch.

**1. Configure which projection configs to evaluate:**

Edit `evaluation/projection_configs_for_metrics.py` (same format as above).

**2. Configure evaluation settings (optional):**

Edit `evaluation/lib/config.py` → `METRICS_EVALUATION` section:
```python
_C.METRICS_EVALUATION.MAX_IMAGES = None  # None = all, 5 = quick test
_C.METRICS_EVALUATION.YOLO_MODEL = "models/yolov8m.pt"
_C.METRICS_EVALUATION.ENABLE_PR_CURVES = True
```

**3. Run evaluation:**

```bash
python evaluation/run_metrics_evaluation.py
```

### Metrics Computed

For each projection configuration, the system computes:

**Per-IoU Threshold (0.30 to 0.95 with 0.05 step):**
- True Positives (TP), False Positives (FP), False Negatives (FN)
- Precision, Recall, F1 Score
- Average Precision (AP) using 11-point interpolation

**Summary Metrics:**
- **AP@[0.30:0.95]**: Mean AP across all IoU thresholds (comprehensive metric)
- **AP@0.50**: Standard PASCAL VOC baseline (50% overlap required)
- **AP@0.75**: Strict COCO-style metric (75% overlap required)
- **Precision@0.5**, **Recall@0.5**, **F1@0.5**: Per-class metrics at IoU=0.5

**Timing Measurements:**
- Composite generation time (varies with projection config)
- YOLO detection time (should be constant across configs)
- Backprojection time (scales with detections)
- Soft-NMS time (scales with overlapping boxes)
- Total end-to-end time per image

### Understanding the Metrics

**Which metric to trust?**

When comparing configurations, focus on:
1. **AP@0.50** - Industry standard (PASCAL VOC, COCO)
2. **F1@0.5** - Best balance between precision and recall
3. **Precision@0.5** - Fewer false positives (higher is better)
4. **Recall@0.5** - More true detections (higher is better)

**AP@[0.30:0.95]** gives a comprehensive view across all strictness levels, but can be dominated by very loose matching (IoU 0.30-0.40) which may not be practically meaningful.

**Example Interpretation:**
- Config A: AP@[0.30:0.95]=0.22, AP@0.50=0.29, Precision@0.5=0.48
- Config B: AP@[0.30:0.95]=0.19, AP@0.50=0.31, Precision@0.5=0.69

→ **Config B is better** because it has higher AP@0.50 and much better precision at the standard IoU=0.5 threshold, even though Config A has slightly higher mean AP.

### Output Structure

Each evaluation run creates a versioned session folder:

```
evaluation/proj-conf-comparison/
└── metrics_eval_session_N/
    ├── metadata.txt                     # Session info (model, configs, date)
    ├── projection_configs_snapshot.py   # Exact config used (reproducibility)
    └── bomni/
        ├── metrics.json                 # Full metrics per config
        ├── timing.json                  # Timing stats per config
        ├── summary.txt                  # Human-readable summary
        ├── comparison_table.txt         # Side-by-side comparison
        └── pr_curves/                   # Precision-Recall curve plots
            ├── config1_pr_curve_iou0.50.png
            ├── config1_pr_curve_iou0.75.png
            └── ...
```

**Key Files:**
- **comparison_table.txt** - Shows best config for each metric
- **metrics.json** - Full numerical results for analysis
- **pr_curves/** - Visualize precision-recall trade-offs

### How It Works

**Pipeline Flow:**
1. Load projection configurations from `projection_configs_for_metrics.py`
2. Load ground truth dataset (BOMNI)
3. For each configuration:
   - Generate composite image
   - Run YOLO detection
   - Backproject to fisheye coordinates
   - Apply two-stage NMS
   - Measure timing at each stage
4. Evaluate predictions against ground truth at 14 IoU thresholds
5. Compute AP using 11-point interpolation (PASCAL VOC style)
6. Generate comparison tables and PR curves
7. Identify best configuration for each metric

**Two-Stage NMS:**
- **Stage 1 (Composite)**: Standard NMS with IoU=0.8 (keeps overlapping boxes)
- **Stage 2 (Fisheye)**: Soft-NMS with Gaussian decay for rotated boxes

**IoU Threshold Range:**
- Evaluation uses 0.30-0.95 (lower than COCO's 0.50-0.95)
- Starting from 0.30 captures easier detections
- Helps diagnose if model detects objects but with poor localization

### Interpreting Detection Coverage

Check the timing.json file to see detection coverage:
- If backprojection count < total images → some images had NO detections
- Example: 228/245 images = 93% detection coverage (7% complete misses)

This is crucial for understanding low AP scores - configurations may be missing detections entirely on some images rather than just having poor localization.

### Multi-Dataset Support

The evaluation pipeline supports three datasets. Configure which to run in `evaluation/lib/config.py`:

```python
_C.METRICS_EVALUATION.DATASETS = ["bomni", "piropo", "cepdof"]
```

Each dataset is evaluated independently. Results are stored in separate per-dataset folders under the session directory:

```
evaluation/proj-conf-comparison/
└── metrics_eval_session_N/
    ├── metadata.txt
    ├── projection_configs_snapshot.py
    ├── bomni/           # BOMNI results (245 images)
    │   ├── metrics.json
    │   ├── timing.json
    │   ├── summary.txt
    │   ├── comparison_table.txt
    │   ├── pr_curves/
    │   └── visuals/
    │       └── {config_id}/
    │           ├── 00001_composite.jpg
    │           └── 00001_fisheye.jpg
    ├── piropo/          # PIROPO results (3,004 images)
    └── cepdof/          # CEPDOF results (up to MAX_IMAGES, spread across 25,358)
```

### Sampling Configuration

For large datasets (especially CEPDOF with 25K frames), configure:

```python
# In evaluation/lib/config.py

# Limit images per dataset (None = all)
_C.METRICS_EVALUATION.MAX_IMAGES = 3000

# Spread samples evenly across full dataset instead of taking first N
_C.METRICS_EVALUATION.SPREAD_SAMPLES = True
```

**How spread sampling works:**
- `step = total / max_images`
- `indices = [int(i * step) for i in range(max_images)]`
- Frames are evenly distributed across the entire dataset (e.g., every 8th frame for 3000 of 25358)
- If `max_images >= total`, all frames are used (no truncation)
- Applied per dataset independently: BOMNI (245 frames) uses all even when MAX_IMAGES=3000

### Visual Output

Enable visual results for qualitative inspection:

```python
# In evaluation/lib/config.py
_C.METRICS_EVALUATION.ENABLE_VISUALS = True
```

For each evaluated frame, two images are saved under `{session}/{dataset}/visuals/{config_id}/`:
- `{idx:05d}_composite.jpg` — composite image with YOLO axis-aligned detections (green)
- `{idx:05d}_fisheye.jpg` — fisheye image with GT rotated boxes (green) and backprojected predictions after Soft-NMS (yellow)

Images are saved flat (no dataset hierarchy), one pair per frame.

### Dataset Loaders

| Dataset | Loader | Frames | Structure |
|---------|--------|--------|-----------|
| BOMNI | `evaluation/lib/bomni_dataset.py` | 245 | `scenario1/{sequence}/` |
| PIROPO | `evaluation/lib/piropo_dataset.py` | 3,004 | `{Room}/{Camera}/{Camera}_{Seq}/` |
| CEPDOF | `evaluation/lib/cepdof_dataset.py` | 25,358 | `{Sequence}/` |

All loaders read standard JSON annotations (center_x, center_y, width, height, angle, class_name).

### Next Steps

1. ~~**Implement evaluation metrics**~~ ✅ COMPLETE
2. ~~**Integrate with detection pipeline**~~ ✅ COMPLETE
3. ~~**Add PIROPO and CEPDOF datasets**~~ ✅ COMPLETE
4. ~~**Incremental per-config evaluation**~~ ✅ COMPLETE
5. **Configuration search** - Systematically test parameter ranges (Phase 6)

### Files in this Module

Entry points (run directly):
- `run_single_config.py` - Evaluate one projection config (incremental workflow)
- `run_comparator.py` - Compare pre-computed config results (incremental workflow)
- `run_metrics_evaluation.py` - Evaluate all configs in one session (batch/legacy)
- `run_backprojection_visual_test.py` - Visual backprojection test
- `visualize_datasets.py` - Visualize dataset annotations

Library (imported by other modules, in `lib/`):
- `lib/config.py` - YACS configuration for evaluation
- `lib/single_config_runner.py` - SingleConfigRunner (incremental workflow)
- `lib/comparator.py` - ConfigComparator (incremental workflow)
- `lib/bomni_dataset.py` - BOMNI runtime loader
- `lib/piropo_dataset.py` - PIROPO runtime loader
- `lib/cepdof_dataset.py` - CEPDOF runtime loader
- `lib/dataset_registry.py` - Dataset metadata registry
- `lib/metrics_evaluator_runner.py` - Batch evaluation orchestrator (legacy)
- `lib/evaluator.py` - Per-image evaluation logic
- `lib/aggregator.py` - Results aggregation + AP + PR curves
- `lib/timing.py` - Pipeline timing utilities
- `README.md` - This file

### References

1. Original BOMNI Dataset: [Boğaziçi University PI Lab](https://www.cmpe.boun.edu.tr/pilab/pilabfiles/databases/bomni/)
2. Rotated Annotations: [omnidet-rotinv GitHub](https://github.com/hitachi-rd-cv/omnidet-rotinv) (Tamura et al., WACV 2019)
3. Paper: "Omnidirectional Pedestrian Detection by Rotation Invariant Training" (Tamura et al., 2019)
