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


def build_radial_bbox(bbox_center: Tuple[float, float],
                     width: float,
                     height: float,
                     fisheye_center: Tuple[float, float]) -> Dict:
    """
    Build a radially-aligned rotated rectangle from center and dimensions.

    The rectangle is oriented such that one pair of sides points toward/away from
    the fisheye center (radial direction), and the other pair is perpendicular
    (tangential direction).

    Args:
        bbox_center: (x, y) center of bbox in fisheye coordinates
        width: Width of bbox in fisheye (distance in pixels)
        height: Height of bbox in fisheye (distance in pixels)
        fisheye_center: (cx, cy) center of fisheye image

    Returns:
        Dict with keys:
            - center: (x, y) center of rotated bbox
            - size: (width, height) of rotated bbox
            - angle: rotation angle in degrees (radial alignment)
            - corners: List of 4 (x, y) corner points
    """
    cx, cy = fisheye_center
    center_x, center_y = bbox_center

    # Calculate radial angle at bbox center
    radial_angle = calculate_radial_angle(center_x, center_y, cx, cy)

    # Create rotation matrix for radial alignment
    cos_a = np.cos(radial_angle)
    sin_a = np.sin(radial_angle)

    # Build 4 corners in unrotated frame (centered at origin)
    # Height along x-axis (radial direction after rotation)
    # Width along y-axis (tangential direction after rotation)
    half_width = width / 2
    half_height = height / 2

    corners_unrotated = np.array([
        [-half_height, -half_width],  # Bottom-left
        [ half_height, -half_width],  # Bottom-right
        [ half_height,  half_width],  # Top-right
        [-half_height,  half_width],  # Top-left
    ])

    # Rotate corners by radial angle
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    corners_rotated = corners_unrotated @ rotation_matrix.T

    # Shift corners to bbox center
    corners_image = corners_rotated + np.array([center_x, center_y])

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
                    compute_lattice: bool = False,
                    lattice_height_samples: int = 10) -> Optional[Dict]:
    """
    Backproject a bounding box from composite coordinates to fisheye coordinates.

    Args:
        bbox: Detection dict with keys 'x', 'y', 'w', 'h' (normalized 0-1)
              and optionally 'confidence', 'class_name'
        metadata: Projection metadata dict with keys:
                  - 'comp_sz': (width, height) of composite
                  - 'grid': (rows, cols) grid layout
                  - 'mapping_matrices': [proj_nbr, 2, height, width] array
        fisheye_shape: (height, width) of fisheye image
        compute_lattice: If True, compute lattice points for visualization (default: False)
        lattice_height_samples: Number of samples along bbox height (only used if compute_lattice=True)

    Returns:
        Dict with fisheye bbox info:
            - center: (x, y) center in fisheye coordinates
            - size: (width, height) of rotated bbox
            - angle: rotation angle in degrees
            - corners: List of 4 corner points
            - lattice_points: List of backprojected lattice points (only if compute_lattice=True)
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

    # Backproject the bbox center directly
    x_proj_center, y_proj_center = composite_to_projection_coords(
        x_comp, y_comp, cell_row, cell_col,
        comp_width, comp_height, proj_width, proj_height, grid
    )
    x_fish_center, y_fish_center = projection_to_fisheye_coords(
        x_proj_center, y_proj_center, cell_row, cell_col,
        mapping_matrices, grid
    )

    # Check if center backprojection was successful
    if np.isnan(x_fish_center) or np.isnan(y_fish_center) or np.isinf(x_fish_center) or np.isinf(y_fish_center):
        return None

    # Backproject width: left and right edges at center height
    x_proj_left, y_proj_left = composite_to_projection_coords(
        x1_comp, y_comp, cell_row, cell_col,
        comp_width, comp_height, proj_width, proj_height, grid
    )
    x_fish_left, y_fish_left = projection_to_fisheye_coords(
        x_proj_left, y_proj_left, cell_row, cell_col,
        mapping_matrices, grid
    )

    x_proj_right, y_proj_right = composite_to_projection_coords(
        x2_comp, y_comp, cell_row, cell_col,
        comp_width, comp_height, proj_width, proj_height, grid
    )
    x_fish_right, y_fish_right = projection_to_fisheye_coords(
        x_proj_right, y_proj_right, cell_row, cell_col,
        mapping_matrices, grid
    )

    # Calculate width in fisheye as distance between left and right edges
    width_fish = np.sqrt((x_fish_right - x_fish_left)**2 + (y_fish_right - y_fish_left)**2)

    # Backproject height: top and bottom edges at center width
    x_proj_top, y_proj_top = composite_to_projection_coords(
        x_comp, y1_comp, cell_row, cell_col,
        comp_width, comp_height, proj_width, proj_height, grid
    )
    x_fish_top, y_fish_top = projection_to_fisheye_coords(
        x_proj_top, y_proj_top, cell_row, cell_col,
        mapping_matrices, grid
    )

    x_proj_bottom, y_proj_bottom = composite_to_projection_coords(
        x_comp, y2_comp, cell_row, cell_col,
        comp_width, comp_height, proj_width, proj_height, grid
    )
    x_fish_bottom, y_fish_bottom = projection_to_fisheye_coords(
        x_proj_bottom, y_proj_bottom, cell_row, cell_col,
        mapping_matrices, grid
    )

    # Calculate height in fisheye as distance between top and bottom edges
    height_fish = np.sqrt((x_fish_bottom - x_fish_top)**2 + (y_fish_bottom - y_fish_top)**2)

    # Check for invalid dimensions
    if np.isnan(width_fish) or np.isnan(height_fish) or width_fish <= 0 or height_fish <= 0:
        return None

    # Build radially-aligned bbox from center and dimensions
    fisheye_bbox_center = (x_fish_center, y_fish_center)
    radial_bbox = build_radial_bbox(fisheye_bbox_center, width_fish, height_fish, (fisheye_cx, fisheye_cy))

    # Optionally sample lattice points for visualization (not for fitting)
    if compute_lattice:
        aspect_ratio = w_comp / h_comp if h_comp > 0 else 1.0
        lattice_width_samples = max(2, int(lattice_height_samples * aspect_ratio))

        x_samples = np.linspace(x1_comp, x2_comp, lattice_width_samples)
        y_samples = np.linspace(y1_comp, y2_comp, lattice_height_samples)

        # Backproject all lattice points for visualization
        fisheye_lattice = []
        for y_s in y_samples:
            for x_s in x_samples:
                x_proj, y_proj = composite_to_projection_coords(
                    x_s, y_s, cell_row, cell_col,
                    comp_width, comp_height, proj_width, proj_height, grid
                )
                x_fish, y_fish = projection_to_fisheye_coords(
                    x_proj, y_proj, cell_row, cell_col,
                    mapping_matrices, grid
                )
                fisheye_lattice.append((x_fish, y_fish))

        # Add lattice points to bbox dict (for visualization only)
        radial_bbox['lattice_points'] = fisheye_lattice

    # Add detection metadata if available
    if 'confidence' in bbox:
        radial_bbox['confidence'] = bbox['confidence']
    if 'class_name' in bbox:
        radial_bbox['class_name'] = bbox['class_name']

    return radial_bbox


def backproject_detections(detections: List[Dict], metadata: Dict,
                          fisheye_shape: Tuple[int, int],
                          compute_lattice: bool = False,
                          lattice_height_samples: int = 10) -> List[Dict]:
    """
    Backproject all detections from composite to fisheye coordinates.

    Args:
        detections: List of detection dicts with normalized coords (0-1)
        metadata: Projection metadata from image_composer
        fisheye_shape: (height, width) of original fisheye image
        compute_lattice: If True, compute lattice points for visualization (default: False)
        lattice_height_samples: Number of samples along bbox height (only used if compute_lattice=True)

    Returns:
        List of fisheye bboxes (rotated rectangles with radial alignment)
    """
    fisheye_bboxes = []

    for det in detections:
        fisheye_bbox = backproject_bbox(det, metadata, fisheye_shape, compute_lattice, lattice_height_samples)
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
