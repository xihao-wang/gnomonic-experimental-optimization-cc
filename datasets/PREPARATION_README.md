# Dataset Preparation - Quick Guide

## Two-Script Workflow

The preparation pipeline splits into two scripts with manual review in between:

1. **`prepare_dataset_step1.py`** - Extract frames, convert annotations, generate visualizations
2. **Manual Review** - User deletes incorrect visualization images
3. **`prepare_dataset_step2.py`** - Remove annotations based on deleted visualizations

## How to Run

**IMPORTANT**: Always run from the project root directory:

```bash
cd gnomonic-experimental-optimization-v2

# Step 1
python datasets/prepare_dataset_step1.py

# [Manual review]

# Step 2
python datasets/prepare_dataset_step2.py
```

**If using PyCharm**: Set working directory to project root, not the datasets folder.

---

## Configuration

Before running, edit `evaluation/config.py`:

```python
# Choose output folder name (change for different test runs)
_C.DATASETS.BOMNI.PREPARATION.TARGET_NAME = "BOMNI-test-run-1"

# Verify input paths (should already be correct)
_C.DATASETS.BOMNI.PREPARATION.VIDEO_DIR = "datasets/all-datasets/bomni-5841/scenario1"
_C.DATASETS.BOMNI.PREPARATION.RAW_ANNOTATIONS_DIR = "datasets/all-datasets/omnidet-rotinv-master/omnidet-rotinv-master/rotate/bomni/rotate/scenario1"
```

In each script, set the dataset name:

```python
# In prepare_dataset_step1.py and prepare_dataset_step2.py
DATASET_NAME = "bomni"  # or "piropo"
```

---

## Step 1: Generate Visualizations

```bash
python datasets/prepare_dataset_step1.py
```

**Output**: `datasets/all-datasets/{TARGET_NAME}/visualizations/bomni/annotation_visualization/`

**What it does**:
1. Extracts frames from videos
2. Removes unannotated frames
3. Converts annotations to standard JSON format
4. Generates visualization images

---

## Manual Review

**Path to check**: `datasets/all-datasets/{TARGET_NAME}/visualizations/{dataset}/annotation_visualization/`

For BOMNI: `datasets/all-datasets/BOMNI-test-run-1/visualizations/bomni/annotation_visualization/`

**What to do**:
1. Open each sequence folder (top-0, top-1, top-2, top-3)
2. Review each visualization image
3. **DELETE images with incorrect annotations** (wrong bbox, missing detection, false positive)
4. **KEEP images with correct annotations**

**CRITICAL**: Only delete visualization images, NEVER delete annotation or frame files!

---

## Step 2: Cleanup Annotations

```bash
python datasets/prepare_dataset_step2.py
```

**What it does**:
1. Finds annotations with no corresponding visualization (because you deleted it)
2. Deletes those annotation and frame files
3. Regenerates clean visualizations

**Result**: Only verified correct data remains.

---

## Final Step: Update Config

After preparation completes, update `evaluation/config.py`:

```python
_C.DATASETS.BOMNI.ANNOTATION_FORMAT = "standard"
_C.DATASETS.BOMNI.STANDARD_ANNOTATIONS_DIR = "datasets/all-datasets/BOMNI-test-run-1/Standard-annotations/scenario1"
_C.DATASETS.BOMNI.FRAMES_DIR = "datasets/all-datasets/BOMNI-test-run-1/frames/scenario1"
```

---

## Output Structure

```
datasets/all-datasets/{TARGET_NAME}/
├── frames/scenario1/{sequence}/              - Extracted frames (verified only)
├── Standard-annotations/scenario1/{sequence}/ - JSON annotations (verified only)
└── visualizations/bomni/annotation_visualization/{sequence}/  - Visualization images
```

**Standard JSON format** (all annotations):
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

---

## Multiple Test Runs

Change `TARGET_NAME` to create separate output folders:

```python
_C.DATASETS.BOMNI.PREPARATION.TARGET_NAME = "BOMNI-test-run-1"
_C.DATASETS.BOMNI.PREPARATION.TARGET_NAME = "BOMNI-test-run-2"
_C.DATASETS.BOMNI.PREPARATION.TARGET_NAME = "BOMNI-corrected"  # production
```

---

## Extending to New Datasets

For PIROPO or any new dataset:

1. **Add config** in `evaluation/config.py` (see PIROPO section)
2. **Create manager** in `datasets/{dataset}_manager.py` (inherit from BaseDatasetManager)
3. **Implement** `convert_to_standard_format()` method
4. **Update both scripts**: Add manager import and case in `get_manager_for_dataset()`
5. **Set** `DATASET_NAME = "piropo"` in both scripts
6. **Run** same two-step workflow

The workflow is identical for all datasets.
