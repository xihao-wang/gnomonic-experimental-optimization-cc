"""
Configuration for dataset loaders.

Defines paths and metadata for different datasets (BOMNI, PIROPO, etc.).
Each dataset may have different folder structure and annotation formats.

Dataset adapters in this module should handle these differences transparently.
"""

from pathlib import Path
import project_config

# ============================================================================
# DATASET PATHS
# ============================================================================

# BOMNI dataset
# Folder structure example:
#   bomni/
#   ├── images/
#   │   ├── sequence_01/
#   │   │   ├── image_0001.jpg
#   │   │   ├── image_0002.jpg
#   │   └── ...
#   └── annotations/
#       ├── sequence_01/
#       │   ├── image_0001.xml
#       │   ├── image_0002.xml

BOMNI_ROOT = project_config.DATASETS_DIR / "bomni"
BOMNI_IMAGES_DIR = BOMNI_ROOT / "images"
BOMNI_ANNOTATIONS_DIR = BOMNI_ROOT / "annotations"

# PIROPO dataset
# Folder structure example:
#   piropo/
#   ├── images/
#   │   ├── scene_001/
#   │   │   ├── frame_001.jpg
#   │   │   ├── frame_002.jpg
#   │   └── ...
#   └── annotations/
#       ├── scene_001/
#       │   ├── frame_001.txt
#       │   ├── frame_002.txt

PIROPO_ROOT = project_config.DATASETS_DIR / "piropo"
PIROPO_IMAGES_DIR = PIROPO_ROOT / "images"
PIROPO_ANNOTATIONS_DIR = PIROPO_ROOT / "annotations"

# ============================================================================
# ANNOTATION FORMATS
# ============================================================================

# BOMNI: XML format (possibly PASCAL VOC style)
BOMNI_ANNOTATION_FORMAT = "xml"

# PIROPO: likely text format (possibly YOLO txt or custom)
PIROPO_ANNOTATION_FORMAT = "txt"

# ============================================================================
# DATASET METADATA
# ============================================================================

DATASETS_META = {
    "bomni": {
        "name": "BOMNI Dataset",
        "description": "Omnidirectional pedestrian detection dataset",
        "root": BOMNI_ROOT,
        "images_dir": BOMNI_IMAGES_DIR,
        "annotations_dir": BOMNI_ANNOTATIONS_DIR,
        "annotation_format": BOMNI_ANNOTATION_FORMAT,
        "image_extension": ".jpg",
    },
    "piropo": {
        "name": "PIROPO Dataset",
        "description": "Panoramic indoor pedestrian dataset",
        "root": PIROPO_ROOT,
        "images_dir": PIROPO_IMAGES_DIR,
        "annotations_dir": PIROPO_ANNOTATIONS_DIR,
        "annotation_format": PIROPO_ANNOTATION_FORMAT,
        "image_extension": ".jpg",
    },
}

# ============================================================================
# DATASET SPLITTING
# ============================================================================

# For reproducibility, define train/val/test split ratios
# If these are None, will use whatever splits are in the dataset folders
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# ============================================================================
# LOADING OPTIONS
# ============================================================================

# Maximum images to load per dataset (None = load all)
MAX_IMAGES_PER_DATASET = None

# Filter images by minimum number of annotations
MIN_ANNOTATIONS_PER_IMAGE = 0

# Downsample images during loading (e.g., 0.5 = half resolution)
# Set to 1.0 for no downsampling
DOWNSAMPLE_FACTOR = 1.0
