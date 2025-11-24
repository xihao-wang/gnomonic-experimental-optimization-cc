"""
Comprehensive backprojection test with multiple projection configurations.

This script tests the backprojection system across different projection configurations
to verify that bounding boxes correctly map from composite to fisheye coordinates.

Tests multiple configurations:
1. 3x3 grid (yolo_grid preset)
2. 2x4 grid (wide configuration)
3. 2x2 grid (simple configuration)
4. 4x2 grid (tall configuration)

For each configuration, generates visualizations showing:
- Composite image with detections
- Fisheye image with backprojected detections (radially aligned)

All results saved to results/backprojection_test/ for visual verification.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime

# Standard Python imports
from config import get_cfg
from pipeline import DetectionPipeline


def create_side_by_side_viz(composite_viz, fisheye_viz, config_name):
    """
    Create side-by-side visualization of composite and fisheye detections.

    Args:
        composite_viz: Composite image with detections
        fisheye_viz: Fisheye image with backprojected detections
        config_name: Name of the configuration for labeling

    Returns:
        Combined side-by-side image
    """
    # Resize images to same height for side-by-side display
    target_height = 800

    comp_h, comp_w = composite_viz.shape[:2]
    comp_scale = target_height / comp_h
    comp_resized = cv2.resize(composite_viz, (int(comp_w * comp_scale), target_height))

    fish_h, fish_w = fisheye_viz.shape[:2]
    fish_scale = target_height / fish_h
    fish_resized = cv2.resize(fisheye_viz, (int(fish_w * fish_scale), target_height))

    # Create side-by-side image
    combined = np.hstack([comp_resized, fish_resized])

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, f"{config_name} - Composite", (10, 30),
               font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(combined, "Fisheye (Backprojected)", (comp_resized.shape[1] + 10, 30),
               font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return combined


def test_projection_config(config_name, proj_nbr, grid, fov_h, fov_v, latitude, lon_step,
                          test_results_dir):
    """
    Test backprojection for a specific projection configuration.

    Args:
        config_name: Name of the configuration
        proj_nbr: Number of projections
        grid: (rows, cols) grid layout
        fov_h: Horizontal FOV in degrees
        fov_v: Vertical FOV in degrees
        latitude: Latitude angle in degrees
        lon_step: Longitude step in degrees
        test_results_dir: Directory to save test results

    Returns:
        (success, num_detections, num_backprojected)
    """
    print("\n" + "=" * 80)
    print(f"TESTING CONFIGURATION: {config_name}")
    print("=" * 80)
    print(f"  Grid: {grid[0]}x{grid[1]} ({proj_nbr} projections)")
    print(f"  FOV: {fov_h}°x{fov_v}°")
    print(f"  Latitude: {latitude}°, Lon step: {lon_step}°")

    # Create configuration
    cfg = get_cfg()
    cfg.INPUT.IMAGE_PATH = str(Path(__file__).parent / "fisheye-sample.png")

    # Set projection configuration (manual, not preset)
    cfg.PROJECTION.PRESET = None
    cfg.PROJECTION.PROJ_NBR = proj_nbr
    cfg.PROJECTION.FOV_H = fov_h
    cfg.PROJECTION.FOV_V = fov_v
    cfg.PROJECTION.LATITUDE = latitude
    cfg.PROJECTION.LON_0 = 0.0
    cfg.PROJECTION.LON_STEP = lon_step
    cfg.PROJECTION.GRID = grid
    cfg.PROJECTION.COMP_SIZE = (640, 640)
    cfg.PROJECTION.TARGET_MP = 'auto'

    # YOLO configuration
    cfg.YOLO.MODEL = str(Path(__file__).parent / "models" / "yolo12x.pt")
    cfg.YOLO.DEVICE = None  # Auto-detect
    cfg.YOLO.CONFIDENCE_THRESHOLD = 0.5

    # Backprojection enabled
    cfg.BACKPROJECTION.ENABLED = True

    # Output configuration
    cfg.OUTPUT.SAVE_DIR = str(test_results_dir / config_name)
    cfg.VERBOSE = False  # Reduce output verbosity for batch testing

    try:
        # Run pipeline
        pipeline = DetectionPipeline(cfg)
        detections, composite, metadata, results_dir, fisheye_bboxes = pipeline.run()

        print(f"\n✓ Detection complete:")
        print(f"  Detections in composite: {len(detections)}")
        print(f"  Backprojected to fisheye: {len(fisheye_bboxes) if fisheye_bboxes else 0}")

        # Load visualization images
        composite_viz_path = results_dir / "detections.png"
        fisheye_viz_path = results_dir / "fisheye_detections.png"

        if composite_viz_path.exists() and fisheye_viz_path.exists():
            composite_viz = cv2.imread(str(composite_viz_path))
            fisheye_viz = cv2.imread(str(fisheye_viz_path))

            # Create side-by-side visualization
            side_by_side = create_side_by_side_viz(composite_viz, fisheye_viz, config_name)

            # Save side-by-side visualization
            side_by_side_path = test_results_dir / f"{config_name}_comparison.png"
            cv2.imwrite(str(side_by_side_path), side_by_side)
            print(f"  Side-by-side visualization: {side_by_side_path}")

        return (True, len(detections), len(fisheye_bboxes) if fisheye_bboxes else 0)

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return (False, 0, 0)


def main():
    """Run comprehensive backprojection tests."""
    print("=" * 80)
    print("BACKPROJECTION COMPREHENSIVE TEST")
    print("=" * 80)
    print("\nThis test will run detection pipeline with multiple projection configurations")
    print("and generate visualizations for manual verification.")
    print("\nVerify that bounding boxes on fisheye images match those on composite images.")

    # Create test results directory
    test_results_dir = Path(__file__).parent / "results" / "backprojection_test"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_results_dir = test_results_dir / timestamp
    test_results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nResults will be saved to: {test_results_dir}")

    # Define test configurations
    test_configs = [
        {
            "name": "3x3_grid",
            "proj_nbr": 9,
            "grid": (3, 3),
            "fov_h": 60.0,
            "fov_v": 60.0,
            "latitude": 45.0,
            "lon_step": 40.0,
        },
        {
            "name": "2x4_grid",
            "proj_nbr": 8,
            "grid": (2, 4),
            "fov_h": 90.0,
            "fov_v": 60.0,
            "latitude": 45.0,
            "lon_step": 45.0,
        },
        {
            "name": "2x2_grid",
            "proj_nbr": 4,
            "grid": (2, 2),
            "fov_h": 90.0,
            "fov_v": 90.0,
            "latitude": 50.0,
            "lon_step": 90.0,
        },
        {
            "name": "4x2_grid",
            "proj_nbr": 8,
            "grid": (4, 2),
            "fov_h": 60.0,
            "fov_v": 45.0,
            "latitude": 40.0,
            "lon_step": 45.0,
        },
        {
            "name": "2x3_grid",
            "proj_nbr": 6,
            "grid": (2, 3),
            "fov_h": 90.0,
            "fov_v": 60.0,
            "latitude": 45.0,
            "lon_step": 60.0,
        },
    ]

    # Run tests
    results = []
    for config in test_configs:
        success, num_detections, num_backprojected = test_projection_config(
            config_name=config["name"],
            proj_nbr=config["proj_nbr"],
            grid=config["grid"],
            fov_h=config["fov_h"],
            fov_v=config["fov_v"],
            latitude=config["latitude"],
            lon_step=config["lon_step"],
            test_results_dir=test_results_dir
        )
        results.append({
            "name": config["name"],
            "success": success,
            "detections": num_detections,
            "backprojected": num_backprojected,
        })

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"\nTotal configurations tested: {len(test_configs)}")
    print(f"Successful: {sum(1 for r in results if r['success'])}")
    print(f"Failed: {sum(1 for r in results if not r['success'])}")

    print("\nDetailed Results:")
    print(f"{'Configuration':<20} {'Status':<10} {'Detections':<12} {'Backprojected':<15}")
    print("-" * 80)
    for r in results:
        status = "✓ PASS" if r['success'] else "✗ FAIL"
        print(f"{r['name']:<20} {status:<10} {r['detections']:<12} {r['backprojected']:<15}")

    print(f"\n{'='*80}")
    print("VERIFICATION INSTRUCTIONS")
    print("=" * 80)
    print(f"\nOpen the comparison images in: {test_results_dir}")
    print("\nFor each *_comparison.png file:")
    print("  - Left side: Composite image with detections (green boxes)")
    print("  - Right side: Fisheye image with backprojected detections (green rotated boxes)")
    print("\nVerify that:")
    print("  1. Each detection in composite has a corresponding detection in fisheye")
    print("  2. Bounding boxes are radially aligned (pointing toward/away from fisheye center)")
    print("  3. Box positions match the pedestrian locations")
    print("  4. Box sizes are reasonable (not too large/small)")

    return 0 if all(r['success'] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
