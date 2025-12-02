# Evaluation Module

This module handles dataset loading, annotation visualization, and evaluation metrics for pedestrian detection in fisheye images.

## BOMNI Dataset Implementation

### Overview

The BOMNI (Boğaziçi University Multi-Omnidirectional Video Tracking Database) dataset has been integrated with support for rotated bounding box annotations. These annotations were provided by a third-party research project ([omnidet-rotinv](https://github.com/your-path-here)) specifically for omnidirectional pedestrian detection.

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

Annotations are provided in **Pascal VOC XML format** with axis-aligned bounding boxes. The rotation angle for each bbox is calculated as:

> "Rotation angle for each bounding box is the angle between a vertical line and a line connecting a image center and bounding box center."

Each annotation contains:
- `xmin, ymin, xmax, ymax` - Axis-aligned bounding box coordinates
- `center_x, center_y` - Calculated bbox center
- `width, height` - Bbox dimensions
- `angle` - Rotation angle in degrees (0° = vertical up, clockwise positive)
- `class_name` - Always "person"

### Frame Extraction

Frames were extracted from the 4 top camera videos using `datasets/extract_bomni_frames.py`. The script:
- Extracts all frames from each video
- Saves them with 4-digit serial numbers (0001.jpg, 0002.jpg, etc.) as expected by the annotation files
- Total extracted: **3345 frames** across all 4 sequences

Note: Only a subset of frames have annotations (~10% of frames are annotated).

### Configuration

The evaluation system uses **YACS** for configuration management. See `evaluation/config.py` for all configurable parameters.

Key configuration nodes:
- `DATASETS.BOMNI` - Dataset paths, sequences, image dimensions, fisheye center
- `METRICS` - Evaluation metrics to compute (AP, mAP, F1, etc.)
- `VISUALIZATION` - Annotation visualization settings
- `OUTPUT` - Result storage options

### Usage

#### Loading the Dataset

```python
from evaluation.config import get_cfg
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

### Next Steps

1. **Implement evaluation metrics** - IoU for rotated boxes, precision, recall, mAP
2. **Integrate with detection pipeline** - Run YOLO on BOMNI images, backproject, and evaluate
3. **Configuration search** - Test different projection configurations on BOMNI dataset
4. **Add more datasets** - PIROPO, Mirror Worlds, CVRG (if needed)

### Files in this Module

- `config.py` - YACS configuration for evaluation
- `bomni_dataset.py` - BOMNI dataset handler with rotated bbox support
- `README.md` - This file
- `__init__.py` - Module initialization

### References

1. Original BOMNI Dataset: [Boğaziçi University PI Lab](https://www.cmpe.boun.edu.tr/pilab/pilabfiles/databases/bomni/)
2. Rotated Annotations: [omnidet-rotinv GitHub](https://github.com/your-path-here) (Tamura et al., WACV 2019)
3. Paper: "Omnidirectional Pedestrian Detection by Rotation Invariant Training" (Tamura et al., 2019)
