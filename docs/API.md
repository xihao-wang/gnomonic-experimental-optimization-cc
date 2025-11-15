# API Reference

This document describes the key interfaces and classes you'll need to implement.

## Detection Pipeline Module

### `DetectionPipeline` class
**Location**: `detection-pipeline/pipeline.py`

```python
class DetectionPipeline:
    def __init__(self, config: dict):
        """Initialize the pipeline with configuration."""
        pass

    def run(self,
            fisheye_image: np.ndarray,
            projection_config: dict) -> List[Detection]:
        """
        Execute full detection pipeline on one image.

        Args:
            fisheye_image: Input fisheye image (H, W, 3) RGB
            projection_config: Configuration with keys:
                - count: Number of projections
                - fov_h: Horizontal field of view
                - fov_v: Vertical field of view
                - latitude: Camera angle (0=nadir, 90=horizon)

        Returns:
            List of Detection objects with coordinates in fisheye space
        """
        pass
```

### `Detection` dataclass
**Location**: `detection-pipeline/pipeline.py`

```python
@dataclass
class Detection:
    x: float  # Bounding box center x (in fisheye coords, 0-1 normalized)
    y: float  # Bounding box center y (in fisheye coords, 0-1 normalized)
    w: float  # Width (normalized)
    h: float  # Height (normalized)
    confidence: float  # Detection confidence (0-1)
    class_id: int  # Object class ID
    source_projection: int  # Which projection this came from (0 to count-1)
```

### `YOLODetector` class
**Location**: `detection-pipeline/yolo_detector.py`

```python
class YOLODetector:
    def __init__(self, model_name: str, device: str = "cpu"):
        """Initialize YOLO model."""
        pass

    def detect(self, image: np.ndarray) -> List[dict]:
        """
        Run YOLO detection on image.

        Args:
            image: Input image (H, W, 3) RGB

        Returns:
            List of detections from YOLO with keys:
                - x, y, w, h: Bounding box (in image coordinates)
                - confidence: Detection confidence
                - class_id: Class ID
        """
        pass
```

### Backprojection functions
**Location**: `detection-pipeline/backprojection.py`

```python
def backproject_detections(
    detections_composite: List[dict],
    composite_metadata: dict,
    fisheye_shape: tuple) -> List[Detection]:
    """
    Project detections from composite space back to fisheye coordinates.

    Args:
        detections_composite: Detections in composite image coords
        composite_metadata: Info about how composite was created:
            - projection_configs: List of projection configs used
            - grid: (rows, cols) layout of projections
            - proj_bbox: Bounding boxes of each projection in composite
        fisheye_shape: (H, W) shape of original fisheye image

    Returns:
        List of Detection objects in fisheye coordinates
    """
    pass
```

## Dataset Module

### `DatasetLoader` abstract class
**Location**: `datasets/dataset_loader.py`

```python
class DatasetLoader:
    @staticmethod
    def create(dataset_name: str) -> 'DatasetLoader':
        """Factory method to create appropriate loader."""
        pass

    def __init__(self, config: dict):
        """Initialize with dataset config."""
        pass

    def iterate(self, split: str = "all") -> Iterator[Tuple[np.ndarray, List[Annotation]]]:
        """
        Iterate over dataset images and annotations.

        Args:
            split: "train", "val", "test", or "all"

        Yields:
            (image, annotations) tuples
                - image: (H, W, 3) RGB numpy array
                - annotations: List of Annotation objects
        """
        pass

    def get_image(self, image_id: str) -> Tuple[np.ndarray, List[Annotation]]:
        """Get a specific image by ID."""
        pass
```

### `Annotation` dataclass
**Location**: `datasets/dataset_loader.py`

```python
@dataclass
class Annotation:
    class_name: str  # "person", "pedestrian", etc.
    x: float  # Bounding box center x (0-1 normalized)
    y: float  # Bounding box center y (0-1 normalized)
    w: float  # Bounding box width (0-1 normalized)
    h: float  # Bounding box height (0-1 normalized)
    difficult: bool = False  # Mark as hard/ambiguous
    occluded: bool = False   # Partially occluded
    truncated: bool = False  # Extends beyond image
```

### Example Loaders

**BOMNI Loader**: `datasets/bomni_loader.py`
```python
class BOMMILoader(DatasetLoader):
    """Loads BOMNI omnidirectional dataset."""
    pass
```

**PIROPO Loader**: `datasets/piropo_loader.py`
```python
class PIROPOLoader(DatasetLoader):
    """Loads PIROPO panoramic indoor dataset."""
    pass
```

## Evaluation Module

### `Evaluator` class
**Location**: `evaluation/evaluator.py`

```python
class Evaluator:
    def __init__(self, config: dict):
        """Initialize evaluator with metric config."""
        pass

    def evaluate(self,
                 detections: List[Detection],
                 ground_truth: List[Annotation],
                 iou_thresholds: List[float] = None) -> dict:
        """
        Compute metrics comparing detections to ground truth.

        Args:
            detections: Predicted detections
            ground_truth: Ground truth annotations
            iou_thresholds: IoU thresholds to evaluate at

        Returns:
            Dictionary with metrics:
                {
                    "ap": 0.75,    # Average Precision @ 0.50:0.95
                    "ap50": 0.82,  # @ 0.50
                    "ap75": 0.70,  # @ 0.75
                    "ar": 0.68,    # Average Recall
                    "f1": 0.74,    # F1 score
                    ...
                }
        """
        pass
```

### Metric Functions
**Location**: `evaluation/metrics.py`

```python
def compute_ap(precision: np.ndarray, recall: np.ndarray) -> float:
    """Compute Average Precision from P-R curve."""
    pass

def compute_iou(box1: tuple, box2: tuple) -> float:
    """Compute IoU between two boxes (x, y, w, h)."""
    pass

def match_detections(detections: List[Detection],
                    ground_truth: List[Annotation],
                    iou_threshold: float) -> Tuple[List[bool], List[bool]]:
    """
    Match detections to ground truth annotations.

    Returns:
        (true_positives, false_positives) boolean arrays
    """
    pass
```

### Results Aggregator
**Location**: `evaluation/results_aggregator.py`

```python
class ResultsAggregator:
    def __init__(self):
        """Initialize aggregator."""
        pass

    def add_image_results(self,
                         image_id: str,
                         metrics: dict,
                         detections: List[Detection]):
        """Add results for one image."""
        pass

    def aggregate(self) -> dict:
        """
        Aggregate results across all images.

        Returns:
            {
                "overall": { metrics averaged across all images },
                "per_image": { image_id: metrics },
                "statistics": { mean, std, min, max for each metric }
            }
        """
        pass
```

## Configuration Search Module

### `Searcher` abstract class
**Location**: `config_search/searcher.py`

```python
class Searcher:
    @staticmethod
    def create(strategy: str) -> 'Searcher':
        """Factory method. Strategies: "grid", "random", "bayesian", "genetic"."""
        pass

    def search(self,
               config_space: dict,
               evaluation_fn: Callable,
               dataset_names: List[str]) -> List[Tuple[dict, dict]]:
        """
        Search for optimal configuration.

        Args:
            config_space: Configuration space to search
                {
                    "projection_counts": [4, 6, 8, 9, 12, 16],
                    "fov_h": [40, 48, 60, 75, 90],
                    "fov_v": [60, 96, 120, 150, 180],
                    "latitude": [0, 36, 90]
                }
            evaluation_fn: Function that takes (config, dataset) and returns metrics dict
            dataset_names: ["bomni"], ["piropo"], or ["bomni", "piropo"]

        Returns:
            Sorted list of (config, metrics) tuples, best first
                config: {"projection_count": 4, "fov_h": 48.0, ...}
                metrics: {"map": 0.75, "ap50": 0.82, ...}
        """
        pass
```

### `Comparator` class
**Location**: `config_search/comparator.py`

```python
class Comparator:
    @staticmethod
    def compare(config1_metrics: dict,
               config2_metrics: dict,
               primary_metric: str = "map") -> int:
        """
        Compare two configurations.

        Returns:
            1 if config1 is better
            -1 if config2 is better
            0 if tied
        """
        pass

    @staticmethod
    def rank_configurations(results: List[Tuple[dict, dict]],
                           primary_metric: str = "map") -> List[Tuple[dict, dict]]:
        """Sort configurations by primary metric."""
        pass
```

### Best Config Selector
**Location**: `config_search/best_config_selector.py`

```python
class BestConfigSelector:
    @staticmethod
    def select_top_n(results: List[Tuple[dict, dict]],
                    n: int = 10,
                    primary_metric: str = "map") -> List[Tuple[dict, dict]]:
        """Select top N configurations."""
        pass

    @staticmethod
    def select_pareto_optimal(results: List[Tuple[dict, dict]],
                             metrics_to_optimize: List[str]) -> List[Tuple[dict, dict]]:
        """Select Pareto-optimal configurations (trade-off curve)."""
        pass
```

## Utilities

### Logging
**Location**: `utils/logging_utils.py`

```python
def setup_logging(log_file: Path = None, level: str = "INFO"):
    """Configure logging for the project."""
    pass

def get_logger(name: str) -> logging.Logger:
    """Get logger for a module."""
    pass
```

### Visualization
**Location**: `utils/visualization_utils.py`

```python
def draw_detections(image: np.ndarray,
                   detections: List[Detection],
                   ground_truth: List[Annotation] = None,
                   title: str = "") -> np.ndarray:
    """Draw detections and ground truth on image."""
    pass

def plot_metrics(results: List[Tuple[dict, dict]],
                metric: str = "map",
                title: str = ""):
    """Plot metric across configurations."""
    pass
```

### I/O Utilities
**Location**: `utils/io_utils.py`

```python
def save_json(data: dict, path: Path):
    """Save dict to JSON file."""
    pass

def load_json(path: Path) -> dict:
    """Load dict from JSON file."""
    pass

def save_detections(detections: List[Detection], path: Path):
    """Serialize detections to file."""
    pass
```

## Data Flow Example

```python
# Load dataset
loader = DatasetLoader.create("piropo")
for image, ground_truth in loader.iterate(split="test"):

    # Run pipeline with specific configuration
    pipeline = DetectionPipeline(config)
    projection_config = {
        "count": 4,
        "fov_h": 48.0,
        "fov_v": 96.0,
        "latitude": 36.0
    }
    detections = pipeline.run(image, projection_config)

    # Evaluate
    evaluator = Evaluator(eval_config)
    metrics = evaluator.evaluate(detections, ground_truth)

    # Store results
    aggregator.add_image_results(image_id, metrics, detections)

# Get aggregate metrics
final_metrics = aggregator.aggregate()
print(f"mAP: {final_metrics['overall']['map']}")
```

## Configuration Objects

All config dictionaries should use these standard keys where applicable:

```python
# Projection configuration
projection_config = {
    "count": 4,           # Number of projections
    "fov_h": 48.0,        # Horizontal FOV (degrees)
    "fov_v": 96.0,        # Vertical FOV (degrees)
    "latitude": 36.0,     # Camera latitude (0-90)
}

# Pipeline configuration
pipeline_config = {
    "yolo_model": "yolov8n.pt",
    "yolo_device": "cpu",
    "yolo_confidence": 0.5,
    "composite_size": (640, 640),
}

# Search configuration
search_config = {
    "strategy": "grid",
    "projection_counts": [4, 6, 8, 9, 12, 16],
    "fov_h_options": [40.0, 48.0, 60.0, 75.0, 90.0],
    "optimization_metric": "map",
}
```

---

**Note**: This is a design document. Implementation details and method signatures may evolve based on actual requirements. Follow the spirit of these interfaces while adapting to your specific needs.
