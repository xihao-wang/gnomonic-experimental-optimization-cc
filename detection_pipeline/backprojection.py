"""
Backprojection module for mapping bounding boxes from composite coordinates to fisheye coordinates.

This module handles the inverse transformation from the composite image (multiple gnomonic projections
stitched together) back to the original fisheye image, with radial alignment to ensure bounding boxes
are oriented toward the fisheye center.

Key Features:
- Maps composite bbox coordinates to fisheye coordinates using projection metadata
- Handles bboxes spanning multiple projection cells
- Ensures radially-aligned rectangles (sides parallel/perpendicular to fisheye center)
- Returns rotated bounding boxes (center, size, angle) for fisheye rendering
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional


def identify_projection_cell(bbox_center_x: float, bbox_center_y: float,
                             comp_width: int, comp_height: int,
                             grid: Tuple[int, int]) -> Tuple[int, int]:
    """
    Identify which projection cell a point belongs to in the composite grid.

    Args:
        bbox_center_x: X coordinate in composite image (pixels)
        bbox_center_y: Y coordinate in composite image (pixels)
        comp_width: Composite image width
        comp_height: Composite image height
        grid: Grid layout (rows, cols)

    Returns:
        (cell_row, cell_col) indices
    """
    rows, cols = grid
    cell_width = comp_width / cols
    cell_height = comp_height / rows

    cell_col = min(int(bbox_center_x / cell_width), cols - 1)
    cell_row = min(int(bbox_center_y / cell_height), rows - 1)

    return (cell_row, cell_col)


def composite_to_projection_coords(x_comp: float, y_comp: float,
                                   cell_row: int, cell_col: int,
                                   comp_width: int, comp_height: int,
                                   proj_width: int, proj_height: int,
                                   grid: Tuple[int, int]) -> Tuple[float, float]:
    """
    Convert composite image coordinates to projection-local coordinates.

    Args:
        x_comp, y_comp: Coordinates in composite image (pixels)
        cell_row, cell_col: Which projection cell
        comp_width, comp_height: Composite image dimensions
        proj_width, proj_height: Individual projection dimensions
        grid: Grid layout (rows, cols)

    Returns:
        (x_proj, y_proj) coordinates in projection-local space
    """
    rows, cols = grid
    cell_width = comp_width / cols
    cell_height = comp_height / rows

    # Coordinates within the cell (0 to cell_width/height)
    x_in_cell = x_comp - (cell_col * cell_width)
    y_in_cell = y_comp - (cell_row * cell_height)

    # Scale to projection dimensions (accounting for potential resizing)
    x_proj = (x_in_cell / cell_width) * proj_width
    y_proj = (y_in_cell / cell_height) * proj_height

    return (x_proj, y_proj)


def projection_to_fisheye_coords(x_proj: float, y_proj: float,
                                 cell_row: int, cell_col: int,
                                 mapping_matrices: np.ndarray,
                                 grid: Tuple[int, int]) -> Tuple[float, float]:
    """
    Map projection coordinates to fisheye coordinates using the forward mapping matrices.

    Args:
        x_proj, y_proj: Coordinates in projection-local space
        cell_row, cell_col: Which projection cell
        mapping_matrices: Shape [proj_nbr, 2, height, width] - forward mapping (proj -> fisheye)
        grid: Grid layout (rows, cols)

    Returns:
        (x_fisheye, y_fisheye) coordinates in fisheye image
    """
    rows, cols = grid
    proj_idx = cell_row * cols + cell_col

    # Get mapping matrices for this projection
    map_x = mapping_matrices[proj_idx, 0]  # Shape: [height, width]
    map_y = mapping_matrices[proj_idx, 1]

    # Clamp coordinates to valid range
    x_idx = int(np.clip(x_proj, 0, map_x.shape[1] - 1))
    y_idx = int(np.clip(y_proj, 0, map_x.shape[0] - 1))

    # Look up fisheye coordinates
    x_fisheye = map_x[y_idx, x_idx]
    y_fisheye = map_y[y_idx, x_idx]

    return (x_fisheye, y_fisheye)


def calculate_radial_angle(x: float, y: float, cx: float, cy: float) -> float:
    """
    Calculate the radial angle from fisheye center to a point.

    Args:
        x, y: Point coordinates
        cx, cy: Fisheye center coordinates

    Returns:
        Angle in radians (0 to 2π)
    """
    dx = x - cx
    dy = y - cy
    angle = np.arctan2(dy, dx)
    return angle


def fit_radial_bbox(points_fisheye: List[Tuple[float, float]],
                   fisheye_center: Tuple[float, float]) -> Dict:
    """
    Fit a radially-aligned rotated rectangle to a set of fisheye points.

    The rectangle is oriented such that one pair of sides points toward/away from
    the fisheye center (radial direction), and the other pair is perpendicular
    (tangential direction).

    Args:
        points_fisheye: List of (x, y) coordinates in fisheye image
        fisheye_center: (cx, cy) center of fisheye image

    Returns:
        Dict with keys:
            - center: (x, y) center of rotated bbox
            - size: (width, height) of rotated bbox
            - angle: rotation angle in degrees (radial alignment)
            - corners: List of 4 (x, y) corner points
    """
    cx, cy = fisheye_center
    points = np.array(points_fisheye)

    # Calculate center of backprojected points
    center_x = np.mean(points[:, 0])
    center_y = np.mean(points[:, 1])

    # Calculate radial angle at this center point
    radial_angle = calculate_radial_angle(center_x, center_y, cx, cy)

    # Create rotation matrix for radial alignment
    # Radial direction is along the angle, tangential is perpendicular
    cos_a = np.cos(radial_angle)
    sin_a = np.sin(radial_angle)

    # Transform points to radial coordinate system (centered at bbox center)
    points_centered = points - np.array([center_x, center_y])

    # Rotate to align with radial direction (radial = x-axis, tangential = y-axis)
    rotation_matrix = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    points_rotated = points_centered @ rotation_matrix.T

    # Find extent in rotated frame
    min_x = np.min(points_rotated[:, 0])
    max_x = np.max(points_rotated[:, 0])
    min_y = np.min(points_rotated[:, 1])
    max_y = np.max(points_rotated[:, 1])

    # Size of the aligned bbox
    width = max_x - min_x
    height = max_y - min_y

    # Calculate the 4 corners in rotated frame
    corners_rotated = np.array([
        [min_x, min_y],  # Bottom-left in rotated frame
        [max_x, min_y],  # Bottom-right
        [max_x, max_y],  # Top-right
        [min_x, max_y],  # Top-left
    ])

    # Rotate corners back to image frame
    inv_rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    corners_image = corners_rotated @ inv_rotation.T + np.array([center_x, center_y])

    # Convert angle to degrees for OpenCV
    angle_deg = np.degrees(radial_angle)

    return {
        'center': (center_x, center_y),
        'size': (width, height),
        'angle': angle_deg,
        'corners': corners_image.tolist()
    }


def backproject_bbox(bbox: Dict, metadata: Dict,
                    fisheye_shape: Tuple[int, int],
                    lattice_height_samples: int = 10) -> Optional[Dict]:
    """
    Backproject a bounding box from composite coordinates to fisheye coordinates.

    Samples a lattice of points within the bbox (respecting aspect ratio), backprojects them,
    and fits a radially-aligned rotated rectangle.

    Args:
        bbox: Detection dict with keys 'x', 'y', 'w', 'h' (normalized 0-1)
              and optionally 'confidence', 'class_name'
        metadata: Projection metadata dict with keys:
                  - 'comp_sz': (width, height) of composite
                  - 'grid': (rows, cols) grid layout
                  - 'mapping_matrices': [proj_nbr, 2, height, width] array
        fisheye_shape: (height, width) of fisheye image
        lattice_height_samples: Number of samples along bbox height (width samples calculated from aspect ratio)

    Returns:
        Dict with fisheye bbox info:
            - center: (x, y) center in fisheye coordinates
            - size: (width, height) of rotated bbox
            - angle: rotation angle in degrees
            - corners: List of 4 corner points
            - lattice_points: List of all backprojected lattice points
            - confidence: detection confidence (if available)
            - class_name: class name (if available)
        Returns None if backprojection fails
    """
    # Extract metadata
    comp_width, comp_height = metadata['comp_sz']
    grid = metadata['grid']
    mapping_matrices = metadata['mapping_matrices']

    # Fisheye center
    fisheye_height, fisheye_width = fisheye_shape
    fisheye_cx = fisheye_width / 2
    fisheye_cy = fisheye_height / 2

    # Get projection dimensions from mapping matrices
    proj_height, proj_width = mapping_matrices.shape[2], mapping_matrices.shape[3]

    # Convert normalized bbox to pixel coordinates in composite
    x_norm, y_norm, w_norm, h_norm = bbox['x'], bbox['y'], bbox['w'], bbox['h']
    x_comp = x_norm * comp_width
    y_comp = y_norm * comp_height
    w_comp = w_norm * comp_width
    h_comp = h_norm * comp_height

    # Bbox bounds in composite
    x1_comp = x_comp - w_comp / 2
    y1_comp = y_comp - h_comp / 2
    x2_comp = x_comp + w_comp / 2
    y2_comp = y_comp + h_comp / 2

    # Identify which projection cell the bbox center belongs to
    cell_row, cell_col = identify_projection_cell(x_comp, y_comp, comp_width, comp_height, grid)

    # Calculate lattice sampling based on aspect ratio
    # If bbox is w=50, h=100 (ratio 0.5), and height_samples=10, then width_samples=5
    aspect_ratio = w_comp / h_comp if h_comp > 0 else 1.0
    lattice_width_samples = max(2, int(lattice_height_samples * aspect_ratio))

    # Sample lattice points within bbox
    x_samples = np.linspace(x1_comp, x2_comp, lattice_width_samples)
    y_samples = np.linspace(y1_comp, y2_comp, lattice_height_samples)

    # Backproject all lattice points
    fisheye_lattice = []
    for y_s in y_samples:
        for x_s in x_samples:
            # Convert to projection coordinates
            x_proj, y_proj = composite_to_projection_coords(
                x_s, y_s, cell_row, cell_col,
                comp_width, comp_height, proj_width, proj_height, grid
            )

            # Map to fisheye coordinates
            x_fish, y_fish = projection_to_fisheye_coords(
                x_proj, y_proj, cell_row, cell_col,
                mapping_matrices, grid
            )

            fisheye_lattice.append((x_fish, y_fish))

    # Check if backprojection was successful (valid fisheye coords)
    fisheye_lattice = np.array(fisheye_lattice)
    if np.any(np.isnan(fisheye_lattice)) or np.any(np.isinf(fisheye_lattice)):
        return None

    # Fit radially-aligned bbox to all lattice points
    radial_bbox = fit_radial_bbox(fisheye_lattice.tolist(), (fisheye_cx, fisheye_cy))

    # Add lattice points to bbox dict
    radial_bbox['lattice_points'] = fisheye_lattice.tolist()

    # Add detection metadata if available
    if 'confidence' in bbox:
        radial_bbox['confidence'] = bbox['confidence']
    if 'class_name' in bbox:
        radial_bbox['class_name'] = bbox['class_name']

    return radial_bbox


def backproject_detections(detections: List[Dict], metadata: Dict,
                          fisheye_shape: Tuple[int, int],
                          lattice_height_samples: int = 10) -> List[Dict]:
    """
    Backproject all detections from composite to fisheye coordinates.

    Args:
        detections: List of detection dicts with normalized coords (0-1)
        metadata: Projection metadata from image_composer
        fisheye_shape: (height, width) of original fisheye image
        lattice_height_samples: Number of samples along bbox height for lattice

    Returns:
        List of fisheye bboxes (rotated rectangles with radial alignment and lattice points)
    """
    fisheye_bboxes = []

    for det in detections:
        fisheye_bbox = backproject_bbox(det, metadata, fisheye_shape, lattice_height_samples)
        if fisheye_bbox is not None:
            fisheye_bboxes.append(fisheye_bbox)

    return fisheye_bboxes


def draw_rotated_bbox(image: np.ndarray, bbox: Dict,
                     color: Tuple[int, int, int] = (0, 255, 0),
                     thickness: int = 2,
                     draw_label: bool = True) -> np.ndarray:
    """
    Draw a rotated bounding box on an image.

    Args:
        image: Image to draw on (will be modified in-place)
        bbox: Dict with 'corners' list of 4 (x, y) points, optionally 'confidence' and 'class_name'
        color: BGR color tuple
        thickness: Line thickness
        draw_label: Whether to draw label with class name and confidence

    Returns:
        Modified image
    """
    corners = np.array(bbox['corners'], dtype=np.int32)

    # Draw the 4 edges
    for i in range(4):
        pt1 = tuple(corners[i])
        pt2 = tuple(corners[(i + 1) % 4])
        cv2.line(image, pt1, pt2, color, thickness)

    # Draw label if requested
    if draw_label and 'class_name' in bbox:
        label = bbox['class_name']
        if 'confidence' in bbox:
            label += f" {bbox['confidence']:.2f}"

        # Draw label at the first corner
        label_pos = tuple(corners[0])
        cv2.putText(image, label, label_pos,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    return image


def visualize_backprojection(fisheye_image: np.ndarray,
                            fisheye_bboxes: List[Dict],
                            output_path: Optional[str] = None) -> np.ndarray:
    """
    Create visualization of backprojected bounding boxes on fisheye image.

    Args:
        fisheye_image: Original fisheye image
        fisheye_bboxes: List of backprojected bbox dicts
        output_path: Optional path to save visualization

    Returns:
        Image with drawn bboxes
    """
    viz_image = fisheye_image.copy()

    for bbox in fisheye_bboxes:
        draw_rotated_bbox(viz_image, bbox, color=(0, 255, 0), thickness=2)

    if output_path:
        cv2.imwrite(output_path, viz_image)

    return viz_image


def visualize_bbox_lattice(fisheye_image: np.ndarray,
                          fisheye_bboxes: List[Dict],
                          output_path: Optional[str] = None) -> np.ndarray:
    """
    Create visualization showing the backprojected lattice points for each bbox.

    Each lattice point is drawn as a small red circle, showing how the bbox grid
    deforms during backprojection from composite to fisheye coordinates.

    Args:
        fisheye_image: Original fisheye image
        fisheye_bboxes: List of backprojected bbox dicts with 'lattice_points' key
        output_path: Optional path to save visualization

    Returns:
        Image with lattice points drawn as red filled circles
    """
    viz_image = fisheye_image.copy()

    for bbox in fisheye_bboxes:
        if 'lattice_points' in bbox:
            lattice_points = bbox['lattice_points']
            # Draw each lattice point as a red filled circle
            for point in lattice_points:
                x, y = int(point[0]), int(point[1])
                cv2.circle(viz_image, (x, y), 3, (0, 0, 255), -1)  # Red in BGR, filled

    if output_path:
        cv2.imwrite(output_path, viz_image)

    return viz_image
