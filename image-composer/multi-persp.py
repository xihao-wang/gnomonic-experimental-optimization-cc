import cv2
import numpy as np
from time import time
import math
import os
import sys
from datetime import datetime

# Import presets
import presets

# Import configuration parameters
try:
    import config
except ImportError:
    # Create a default config if it doesn't exist
    config = type('', (), {})
    config.IMG_PATH = "imgs/fisheye.png"
    config.PROJ_NBR = 6
    config.FOV_H = 90.0
    config.FOV_V = 60.0
    config.LATITUDE = 45.0
    config.LON_0 = 0.0
    config.LON_STEP = 60.0
    config.GRID = None
    config.COMP_SZ = (640, 640)
    config.TARGET_MP = 0.48
    config.OUTPUT_DIR = "multi-persp-out"


def compute_viewing_basis(longitude, latitude):
    """
    Computes viewing basis vector (xyz) for fisheye projection:
    - view_vector: Direction from viewer to target point on sphere
    - right_vector: Horizontal basis of tangent plane (negated to prevent mirroring)
    - up_vector: Vertical basis of tangent plane
    :param longitude: Horizontal angle in degrees (0-360°)
    :param latitude: Vertical angle in degrees (0° = nadir (the point directly below the observer), 90° = horizon)
    :return: 3x3 matrix with columns [right_vector, up_vector, view_vector]
    """
    lon = np.radians(longitude)
    lat = np.radians(latitude)

    # Compute viewing vector xyz (points downward for latitude=0, horizon for latitude=90)
    view_vector_x = np.sin(lat) * np.cos(lon)
    view_vector_y = np.sin(lat) * np.sin(lon)
    view_vector_z = -np.cos(lat)  # Negative Z for downward direction
    view_vector = np.array([view_vector_x, view_vector_y, view_vector_z])
    view_vector /= np.linalg.norm(view_vector)  # Normalize

    # Right vector (cross product with world-up axis Y)
    world_up = np.array([0.0, 0.0, 1.0])
    R = np.cross(view_vector, world_up)

    # Normalize right vector to ensure unit length
    # The small constant (1e-6) prevents division by zero in edge cases
    R = R / (np.linalg.norm(R) + 1e-6)

    # Up vector (corrected perpendicular to forward and right)
    U = np.cross(R, view_vector)
    U /= np.linalg.norm(U) + 1e-6

    return np.column_stack((-R, U, view_vector))  # -R avoids projection mirroring


def calculate_dimensions(fov_h_deg, fov_v_deg, target_mp=0.48, grid=None, comp_sz=None):
    """
    Calculate output dimensions based on FOV and target megapixels.

    :param fov_h_deg: Horizontal field of view in degrees
    :param fov_v_deg: Vertical field of view in degrees
    :param target_mp: Target megapixels (default 0.48) or 'auto' for automatic sizing based on grid and comp_sz
    :param grid: Grid layout (rows, cols) for auto sizing
    :param comp_sz: Composite size (width, height) for auto sizing
    :return: Tuple of (width, height) in pixels
    """
    # Calculate the aspect ratio based on FOV
    aspect_ratio = fov_h_deg / fov_v_deg

    # Auto mode: calculate based on final composite size and grid layout
    if target_mp == 'auto' and grid is not None and comp_sz is not None:
        rows, cols = grid
        comp_width, comp_height = comp_sz

        # Calculate the maximum width and height of each projection
        # This is the exact division of the composite size by the grid dimensions
        width = int(comp_width / cols)
        height = int(comp_height / rows)

        # No need to adjust dimensions - we want to exactly fill the composite
        # This ensures that when projections are arranged in the grid and resized,
        # they will perfectly match the desired composite size
        return (width, height)

    # Standard mode: calculate based on target megapixels
    # Solving: width * height = target_mp * 1,000,000 and width = aspect_ratio * height
    height = int(math.sqrt((target_mp * 1_000_000) / aspect_ratio))
    width = int(height * aspect_ratio)

    # Ensure we're as close as possible to the target MP
    actual_mp = (width * height) / 1_000_000
    if abs(actual_mp - target_mp) > 0.01:  # More than 0.01 MP off
        # Adjust to get closer to target MP
        scale_factor = math.sqrt(target_mp / actual_mp)
        width = int(width * scale_factor)
        height = int(height * scale_factor)

    return (width, height)


def fisheye_to_perspective(fisheye_img, cx, cy, r, longitude, latitude, fov_h_deg, fov_v_deg, target_mp=0.48, grid=None,
                           comp_sz=None):
    """
    Project a perspective view from a top-view fisheye image.

    :param fisheye_img: Input fisheye image (numpy array)
    :param cx: X-coordinate of the fisheye center in the input image
    :param cy: Y-coordinate of the fisheye center in the input image
    :param r: Radius of the fisheye circle in pixels
    :param longitude: Horizontal viewing angle in degrees (0-360°)
    :param latitude: Vertical viewing angle in degrees (0-90°, where 0=nadir, 90=horizon)
    :param fov_h_deg: Horizontal field of view in degrees
    :param fov_v_deg: Vertical field of view in degrees
    :param target_mp: Target megapixels for output image or 'auto'
    :param grid: Grid layout for auto sizing
    :param comp_sz: Composite size for auto sizing
    :return: Perspective projection image, latency, and mapping matrices
    """
    # Calculate output dimensions based on FOV ratio
    output_size = calculate_dimensions(fov_h_deg, fov_v_deg, target_mp, grid, comp_sz)
    W_out, H_out = output_size

    rot_matrix = compute_viewing_basis(longitude, latitude)

    # Generate output pixel grid
    u, v = np.meshgrid(np.arange(W_out), np.arange(H_out))

    # Focal lengths from FOV
    fov_h_rad = np.radians(fov_h_deg)
    fov_v_rad = np.radians(fov_v_deg)
    fx = (W_out / 2) / np.tan(fov_h_rad / 2)
    fy = (H_out / 2) / np.tan(fov_v_rad / 2)

    # Camera local coordinates
    cx_out, cy_out = W_out / 2, H_out / 2
    x_local = (u - cx_out) / fx
    y_local = (cy_out - v) / fy  # OpenCV's Y increases downward
    z_local = np.ones_like(x_local)

    # Normalize direction vectors
    norm = np.sqrt(x_local ** 2 + y_local ** 2 + z_local ** 2)
    dirs = np.stack([x_local / norm, y_local / norm, z_local / norm], axis=2)

    # Rotate to align with fisheye's orientation
    world_dirs = np.dot(dirs, rot_matrix.T)

    # Convert to spherical coordinates (theta: angle from NADIR)
    theta = np.arccos(-world_dirs[..., 2])  # Key fix for top-view
    phi = np.arctan2(world_dirs[..., 1], world_dirs[..., 0])

    # Map to fisheye image coordinates
    valid_mask = theta <= (np.pi / 2)  # Fisheye only captures <= 90° from nadir
    r_pixel = np.where(valid_mask, (theta / (np.pi / 2)) * r, 0)
    x_fisheye = cx + r_pixel * np.cos(phi)
    y_fisheye = cy + r_pixel * np.sin(phi)

    # Remap and interpolate
    map_x = x_fisheye.astype(np.float32)
    map_y = y_fisheye.astype(np.float32)
    st = time()
    viewport = cv2.remap(fisheye_img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    latency = round((time() - st) * 1000, 3)

    # Return the projection, latency, and mapping matrices
    return viewport, latency, (map_x, map_y)


def draw_fov_on_fisheye(fisheye_img, cx, cy, r, longitude, latitude, fov_h_deg, fov_v_deg, color=(0, 255, 0)):
    """
    Draw the field of view projection area on the original fisheye image.
    Handles cases where FOV extends beyond fisheye boundary.

    :param fisheye_img: Original fisheye image
    :param cx: X-coordinate of fisheye center
    :param cy: Y-coordinate of fisheye center
    :param r: Radius of fisheye
    :param longitude: Horizontal viewing angle in degrees
    :param latitude: Vertical viewing angle in degrees
    :param fov_h_deg: Horizontal field of view in degrees
    :param fov_v_deg: Vertical field of view in degrees
    :param color: Color to use for drawing FOV (default green)
    :return: Fisheye image with FOV visualization overlay
    """
    # Create a copy of the fisheye image to avoid modifying the original
    viz_img = fisheye_img.copy()

    # Get the rotation matrix
    rot_matrix = compute_viewing_basis(longitude, latitude)

    # Create boundary points for the FOV rectangle
    fov_h_rad = np.radians(fov_h_deg)
    fov_v_rad = np.radians(fov_v_deg)

    # Define the four corners of the FOV in camera space
    corners = [
        [-np.tan(fov_h_rad / 2), -np.tan(fov_v_rad / 2), 1.0],  # Top-left
        [np.tan(fov_h_rad / 2), -np.tan(fov_v_rad / 2), 1.0],  # Top-right
        [np.tan(fov_h_rad / 2), np.tan(fov_v_rad / 2), 1.0],  # Bottom-right
        [-np.tan(fov_h_rad / 2), np.tan(fov_v_rad / 2), 1.0]  # Bottom-left
    ]
    corners = np.array(corners)

    # Normalize the direction vectors
    norms = np.sqrt(np.sum(corners ** 2, axis=1))
    normalized_corners = corners / norms[:, np.newaxis]

    # Rotate vectors to world space
    world_corners = np.dot(normalized_corners, rot_matrix.T)

    # Convert to spherical coordinates (theta from NADIR)
    theta = np.arccos(-world_corners[:, 2])
    phi = np.arctan2(world_corners[:, 1], world_corners[:, 0])

    # Check which corners are within fisheye view (less than 90° from nadir)
    valid_mask = theta <= (np.pi / 2)

    # Function to find intersection of a ray with the fisheye boundary
    def find_intersection(p1, p2):
        """Find intersection of line p1->p2 with fisheye boundary."""
        # Get spherical coordinates
        theta1 = np.arccos(-p1[2])
        phi1 = np.arctan2(p1[1], p1[0])
        theta2 = np.arccos(-p2[2])
        phi2 = np.arctan2(p2[1], p2[0])

        # If p1 is valid and p2 is invalid, we need to find where the ray crosses pi/2
        if theta1 <= np.pi / 2 and theta2 > np.pi / 2:
            # Parametrize the line from p1 to p2
            # p(t) = p1 + t*(p2-p1) where t in [0,1]
            # We need to find t where theta(p(t)) = pi/2

            # This is somewhat complex in 3D, so we'll use a binary search approximation
            t_min, t_max = 0.0, 1.0
            t_mid = 0.5
            max_iterations = 10

            for _ in range(max_iterations):
                t_mid = (t_min + t_max) / 2
                p_mid = p1 + t_mid * (p2 - p1)
                p_mid = p_mid / np.linalg.norm(p_mid)  # Normalize
                theta_mid = np.arccos(-p_mid[2])

                if abs(theta_mid - np.pi / 2) < 1e-6:
                    break
                elif theta_mid < np.pi / 2:
                    t_min = t_mid
                else:
                    t_max = t_mid

            # Now t_mid gives us the intersection point
            p_intersect = p1 + t_mid * (p2 - p1)
            p_intersect = p_intersect / np.linalg.norm(p_intersect)

            # Convert to fisheye coordinates
            theta_intersect = np.pi / 2  # By definition
            phi_intersect = np.arctan2(p_intersect[1], p_intersect[0])

            x_intersect = int(cx + r * np.cos(phi_intersect))
            y_intersect = int(cy + r * np.sin(phi_intersect))

            return (x_intersect, y_intersect)

        return None

    # Draw edges with intersection handling
    segments = 20
    for edge in range(4):
        start_idx = edge
        end_idx = (edge + 1) % 4

        start_corner = normalized_corners[start_idx]
        end_corner = normalized_corners[end_idx]

        start_world = world_corners[start_idx]
        end_world = world_corners[end_idx]

        start_valid = valid_mask[start_idx]
        end_valid = valid_mask[end_idx]

        if start_valid and end_valid:
            # Both endpoints are valid - use the original approach with segments
            edge_points = []
            for i in range(segments + 1):
                t = i / segments
                pt = (1 - t) * start_corner + t * end_corner
                pt = pt / np.linalg.norm(pt)
                edge_points.append(pt)

            edge_points = np.array(edge_points)
            world_edge = np.dot(edge_points, rot_matrix.T)

            edge_theta = np.arccos(-world_edge[:, 2])
            edge_phi = np.arctan2(world_edge[:, 1], world_edge[:, 0])

            # All edge points should be valid since endpoints are valid
            edge_r = (edge_theta / (np.pi / 2)) * r
            edge_x = (cx + edge_r * np.cos(edge_phi)).astype(int)
            edge_y = (cy + edge_r * np.sin(edge_phi)).astype(int)

            # Draw line segments
            for i in range(len(edge_x) - 1):
                cv2.line(viz_img, (edge_x[i], edge_y[i]),
                         (edge_x[i + 1], edge_y[i + 1]), color, 2)

        elif start_valid or end_valid:
            # Only one endpoint is valid - find intersection with fisheye boundary
            if start_valid:
                # Convert valid point to image coordinates
                theta_start = np.arccos(-start_world[2])
                phi_start = np.arctan2(start_world[1], start_world[0])
                x_start = int(cx + r * (theta_start / (np.pi / 2)) * np.cos(phi_start))
                y_start = int(cy + r * (theta_start / (np.pi / 2)) * np.sin(phi_start))

                # Find intersection
                intersection = find_intersection(start_world, end_world)
                if intersection:
                    cv2.line(viz_img, (x_start, y_start), intersection, color, 2)
            else:
                # Convert valid point to image coordinates
                theta_end = np.arccos(-end_world[2])
                phi_end = np.arctan2(end_world[1], end_world[0])
                x_end = int(cx + r * (theta_end / (np.pi / 2)) * np.cos(phi_end))
                y_end = int(cy + r * (theta_end / (np.pi / 2)) * np.sin(phi_end))

                # Find intersection
                intersection = find_intersection(end_world, start_world)
                if intersection:
                    cv2.line(viz_img, (x_end, y_end), intersection, color, 2)

        # If both endpoints are invalid, we don't draw anything
        # (the edge is completely outside the fisheye view)

    # Mark the valid corners with circles
    for i in range(4):
        if valid_mask[i]:
            theta_corner = np.arccos(-world_corners[i, 2])
            phi_corner = np.arctan2(world_corners[i, 1], world_corners[i, 0])
            x_corner = int(cx + r * (theta_corner / (np.pi / 2)) * np.cos(phi_corner))
            y_corner = int(cy + r * (theta_corner / (np.pi / 2)) * np.sin(phi_corner))
            cv2.circle(viz_img, (x_corner, y_corner), 4, color, -1)

    return viz_img


def calculate_coverage(lon_0, lon_step, proj_nbr, fov_h_deg, fov_v_deg, latitude):
    """
    Calculate FOV coverage and overlap between projections with improved accuracy.

    This function computes both horizontal and vertical coverage, accounting for:
    - Spacing between projections (gaps)
    - Projection distortion at different latitudes
    - Vertical coverage limitations

    :param lon_0: Starting longitude in degrees
    :param lon_step: Longitude step in degrees
    :param proj_nbr: Number of projections
    :param fov_h_deg: Horizontal field of view in degrees
    :param fov_v_deg: Vertical field of view in degrees
    :param latitude: Latitude in degrees
    :return: tuple of (overlap_percentage, uncovered_percentage)
    """
    # Create a discretized representation of the hemisphere (fisheye view)
    resolution = 2  # degrees per sample (higher = more accurate but slower)
    lat_samples = int(90 / resolution)
    lon_samples = int(360 / resolution)

    # Create a hemisphere coverage map (1 = covered, 0 = not covered)
    # Note: Only need to track the viewable hemisphere (0-90° latitude, 0-360° longitude)
    coverage_map = np.zeros((lat_samples, lon_samples), dtype=np.int8)

    # For each projection, mark the covered areas
    for i in range(proj_nbr):
        # Center of this projection
        center_lon = (lon_0 + i * lon_step) % 360
        center_lat = latitude

        # Calculate FOV boundaries in spherical coordinates
        # These are approximations since the actual FOV boundaries are not perfect spherical rectangles
        half_fov_h = fov_h_deg / 2
        half_fov_v = fov_v_deg / 2

        # Handle FOV that extends beyond the horizon or below nadir
        min_lat = max(0, center_lat - half_fov_v)  # Can't go below nadir (0°)
        max_lat = min(90, center_lat + half_fov_v)  # Can't go above horizon (90°)

        # Longitude range needs to account for the reduced effective longitude span at higher latitudes
        # At the equator, lon_span = fov_h_deg
        # At the poles, lon_span approaches 360°

        # For simplicity and to avoid complex math, we'll use a reasonable approximation
        # As we approach the horizon, the longitude span increases
        # This factor is an approximation that works reasonably well
        lat_factor = 1.0
        if center_lat > 45:  # Adjust factor for higher latitudes
            lat_factor = 1.0 + (center_lat - 45) / 45 * 0.5  # Scale from 1.0 to 1.5

        effective_half_fov_h = half_fov_h * lat_factor

        min_lon = (center_lon - effective_half_fov_h) % 360
        max_lon = (center_lon + effective_half_fov_h) % 360

        # Convert boundaries to indices in our coverage map
        min_lat_idx = int(min_lat / resolution)
        max_lat_idx = min(lat_samples - 1, int(max_lat / resolution))

        # Mark the coverage area
        for lat_idx in range(min_lat_idx, max_lat_idx + 1):
            # Actual latitude value
            lat = lat_idx * resolution

            # Adjust longitude span based on latitude
            # This handles the fact that longitude lines converge at the poles
            # At higher latitudes, the same FOV covers more longitude
            if lat > 0:  # Avoid division by zero at nadir
                # This formula approximates the widening of longitude span at higher latitudes
                lat_scale = min(3.0, 1.0 / np.cos(np.radians(min(85, lat))))  # Cap at 3x to avoid extremes
                current_half_fov_h = effective_half_fov_h * lat_scale
            else:
                current_half_fov_h = effective_half_fov_h

            # Adjusted longitude bounds for this latitude
            current_min_lon = (center_lon - current_half_fov_h) % 360
            current_max_lon = (center_lon + current_half_fov_h) % 360

            min_lon_idx = int(current_min_lon / resolution)
            max_lon_idx = int(current_max_lon / resolution)

            # Handle wrap-around
            if current_min_lon > current_max_lon:
                # FOV wraps around the 0/360 boundary
                for lon_idx in range(min_lon_idx, lon_samples):
                    coverage_map[lat_idx, lon_idx] = 1
                for lon_idx in range(0, max_lon_idx + 1):
                    coverage_map[lat_idx, lon_idx] = 1
            else:
                # Normal case
                for lon_idx in range(min_lon_idx, max_lon_idx + 1):
                    coverage_map[lat_idx, lon_idx % lon_samples] = 1

    # Calculate coverage statistics
    total_cells = coverage_map.shape[0] * coverage_map.shape[1]
    covered_cells = np.sum(coverage_map)

    # Calculate the uncovered percentage
    uncovered_percentage = 100.0 - (covered_cells / total_cells * 100.0)

    # Calculate overlap
    # For each projection, count how many cells it covers that are already covered
    overlap_cells = 0
    temp_map = np.zeros_like(coverage_map)

    for i in range(proj_nbr):
        # Reset temporary map
        temp_map.fill(0)

        # Center of this projection
        center_lon = (lon_0 + i * lon_step) % 360
        center_lat = latitude

        # Calculate FOV boundaries
        half_fov_h = fov_h_deg / 2
        half_fov_v = fov_v_deg / 2

        min_lat = max(0, center_lat - half_fov_v)
        max_lat = min(90, center_lat + half_fov_v)

        lat_factor = 1.0
        if center_lat > 45:
            lat_factor = 1.0 + (center_lat - 45) / 45 * 0.5

        effective_half_fov_h = half_fov_h * lat_factor

        min_lat_idx = int(min_lat / resolution)
        max_lat_idx = min(lat_samples - 1, int(max_lat / resolution))

        for lat_idx in range(min_lat_idx, max_lat_idx + 1):
            lat = lat_idx * resolution

            if lat > 0:
                lat_scale = min(3.0, 1.0 / np.cos(np.radians(min(85, lat))))
                current_half_fov_h = effective_half_fov_h * lat_scale
            else:
                current_half_fov_h = effective_half_fov_h

            current_min_lon = (center_lon - current_half_fov_h) % 360
            current_max_lon = (center_lon + current_half_fov_h) % 360

            min_lon_idx = int(current_min_lon / resolution)
            max_lon_idx = int(current_max_lon / resolution)

            if current_min_lon > current_max_lon:
                for lon_idx in range(min_lon_idx, lon_samples):
                    temp_map[lat_idx, lon_idx] = 1
                for lon_idx in range(0, max_lon_idx + 1):
                    temp_map[lat_idx, lon_idx] = 1
            else:
                for lon_idx in range(min_lon_idx, max_lon_idx + 1):
                    temp_map[lat_idx, lon_idx % lon_samples] = 1

        # Count cells that were already covered before this projection
        # Exclude this iteration by using the original coverage_map - temp_map
        overlap_mask = temp_map & (coverage_map - temp_map)
        overlap_cells += np.sum(overlap_mask)

    # Calculate overlap percentage (relative to total covered area)
    if covered_cells > 0:
        overlap_percentage = (overlap_cells / covered_cells) * 100.0
    else:
        overlap_percentage = 0.0

    return overlap_percentage, uncovered_percentage


def is_valid_proj_nbr(proj_nbr):
    """
    Check if proj_nbr is valid (either even or has integer square root).

    :param proj_nbr: Number of projections
    :return: True if valid, False otherwise
    """
    # Check if even
    if proj_nbr % 2 == 0:
        return True

    # Check if square root is an integer
    sqrt_proj = math.sqrt(proj_nbr)
    return sqrt_proj.is_integer()


def determine_grid(proj_nbr, grid=None):
    """
    Determine the grid layout (rows, cols) for the projections.

    :param proj_nbr: Number of projections
    :param grid: User-specified grid as (rows, cols) or None
    :return: (rows, cols) tuple
    """
    if grid is not None:
        rows, cols = grid
        if rows * cols < proj_nbr:
            print(f"Warning: Specified grid {grid} is too small for {proj_nbr} projections.")
            print("Adjusting grid automatically.")
            grid = None

    if grid is None:
        # Determine grid automatically
        sqrt_proj = math.sqrt(proj_nbr)
        if sqrt_proj.is_integer():
            # Perfect square
            rows = cols = int(sqrt_proj)
        else:
            # For even numbers, try to get close to square
            factors = []
            for i in range(1, int(math.sqrt(proj_nbr)) + 1):
                if proj_nbr % i == 0:
                    factors.append((i, proj_nbr // i))

            # Choose the most square-like factor pair
            min_diff = float('inf')
            rows, cols = 1, proj_nbr
            for r, c in factors:
                if abs(r - c) < min_diff:
                    min_diff = abs(r - c)
                    rows, cols = r, c

    return (rows, cols)


def create_composite_image(projections, grid, comp_sz):
    """
    Create a composite image by stacking all projections in a grid.

    :param projections: List of projection images
    :param grid: (rows, cols) tuple
    :param comp_sz: Desired composite size (width, height)
    :return: Original composite and resized composite
    """
    rows, cols = grid

    # Get the dimensions of the first projection
    proj_h, proj_w = projections[0].shape[:2]

    # Create the composite image
    composite_h = rows * proj_h
    composite_w = cols * proj_w
    composite = np.zeros((composite_h, composite_w, 3), dtype=np.uint8)

    # Place projections in the grid (first projection at top-left)
    for i, projection in enumerate(projections):
        if i >= rows * cols:
            break

        row = i // cols
        col = i % cols

        y_start = row * proj_h
        y_end = y_start + proj_h
        x_start = col * proj_w
        x_end = x_start + proj_w

        composite[y_start:y_end, x_start:x_end] = projection

    # Resize the composite image
    resized_composite = cv2.resize(composite, comp_sz)

    return composite, resized_composite


def generate_rainbow_colors(n):
    """
    Generate n distinct colors in rainbow-like sequence.

    :param n: Number of colors to generate
    :return: List of (B,G,R) color tuples
    """
    colors = []
    for i in range(n):
        hue = i * 179 // n  # Hue ranges from 0-179 in OpenCV
        hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        colors.append((int(bgr[0][0][0]), int(bgr[0][0][1]), int(bgr[0][0][2])))
    return colors


def load_configuration():
    """
    Load configuration from config.py, with preset support.

    :return: Dictionary with configuration parameters
    """
    # Check if a preset is selected in config.py
    if hasattr(config, 'SELECTED_PRESET') and config.SELECTED_PRESET in presets.PRESET_NAMES:
        preset = presets.PRESETS[config.SELECTED_PRESET].copy()

        # Override image if custom image is specified in config.py
        if hasattr(config, 'CUSTOM_IMAGE') and config.CUSTOM_IMAGE:
            preset["img_path"] = config.CUSTOM_IMAGE
            print(f"\n=== Configuration Source ===")
            print(f"Using preset '{config.SELECTED_PRESET}' ({preset['name']}) with custom image: {preset['img_path']}")
        else:
            print(f"\n=== Configuration Source ===")
            print(f"Using preset '{config.SELECTED_PRESET}' ({preset['name']})")
            print(f"Image path: {preset['img_path']}")

        # Ensure output directory is set
        if "output_dir" not in preset:
            preset["output_dir"] = presets.OUTPUT_DIR

        # Print key preset parameters
        print(f"Description: {preset['description']}")
        print(f"Projections: {preset['proj_nbr']} with FOV: {preset['fov_h']}°×{preset['fov_v']}°")
        print(f"Latitude: {preset['latitude']}°, Starting longitude: {preset['lon_0']}°, Step: {preset['lon_step']}°")
        print(f"Target MP: {preset['target_mp']}, Composite size: {preset['comp_sz'][0]}×{preset['comp_sz'][1]}")
        print(f"Output directory: {preset['output_dir']}")
        print("=" * 30)
        return preset

    # If no preset specified or preset not found, use config.py custom configuration
    custom_config = {
        "name": "Custom Configuration",
        "description": "Configuration from config.py",
        "img_path": config.IMG_PATH,
        "proj_nbr": config.PROJ_NBR,
        "fov_h": config.FOV_H,
        "fov_v": config.FOV_V,
        "latitude": config.LATITUDE,
        "lon_0": config.LON_0,
        "lon_step": config.LON_STEP,
        "grid": config.GRID,
        "comp_sz": config.COMP_SZ,
        "target_mp": config.TARGET_MP,
        "output_dir": config.OUTPUT_DIR
    }

    # Print custom configuration details
    print(f"\n=== Configuration Source ===")
    print(f"Using custom configuration from config.py")
    print(
        f"SELECTED_PRESET is {'None or not defined' if not hasattr(config, 'SELECTED_PRESET') else 'set to an invalid preset name' if config.SELECTED_PRESET else 'None'}")
    print(f"Image path: {custom_config['img_path']}")
    print(f"Projections: {custom_config['proj_nbr']} with FOV: {custom_config['fov_h']}°×{custom_config['fov_v']}°")
    print(
        f"Latitude: {custom_config['latitude']}°, Starting longitude: {custom_config['lon_0']}°, Step: {custom_config['lon_step']}°")
    print(f"Grid: {custom_config['grid']}")
    print(
        f"Target MP: {custom_config['target_mp']}, Composite size: {custom_config['comp_sz'][0]}×{custom_config['comp_sz'][1]}")
    print(f"Output directory: {custom_config['output_dir']}")
    print("=" * 30)

    return custom_config


def generate_composite_from_config(cfg_dict):
    """
    Generate composite image from configuration dictionary (programmatic API).

    Args:
        cfg_dict: Dict with keys img_path, proj_nbr, fov_h, fov_v, latitude, lon_0, lon_step, grid, comp_sz, target_mp

    Returns:
        (resized_composite, metadata) where metadata includes mapping_matrices for backprojection
    """
    img_path = cfg_dict["img_path"]
    proj_nbr = cfg_dict["proj_nbr"]
    fov_h = cfg_dict["fov_h"]
    fov_v = cfg_dict["fov_v"]
    latitude = cfg_dict["latitude"]
    lon_0 = cfg_dict["lon_0"]
    lon_step = cfg_dict["lon_step"]
    grid = cfg_dict["grid"]
    comp_sz = cfg_dict["comp_sz"]
    target_mp = cfg_dict["target_mp"]

    # Validate proj_nbr
    if not is_valid_proj_nbr(proj_nbr):
        raise ValueError(f"Invalid number of projections ({proj_nbr}). Must be even or have integer square root.")

    # Determine grid layout
    grid = determine_grid(proj_nbr, grid)

    # Load fisheye image
    fisheye_img = cv2.imread(img_path)
    if fisheye_img is None:
        raise FileNotFoundError(f"Could not load image at {img_path}")

    # Fisheye parameters
    cx, cy = fisheye_img.shape[1] // 2, fisheye_img.shape[0] // 2
    r = min(cx, cy)

    # Generate projections and mapping matrices
    projections = []
    mapping_matrices = []

    for i in range(proj_nbr):
        longitude = (lon_0 + i * lon_step) % 360
        projection, latency, mapping = fisheye_to_perspective(
            fisheye_img, cx, cy, r, longitude, latitude,
            fov_h, fov_v, target_mp, grid, comp_sz
        )
        projections.append(projection)
        mapping_matrices.append(mapping)

    # Create composite image
    composite, resized_composite = create_composite_image(projections, grid, comp_sz)

    # Create combined mapping matrices array [proj_nbr, 2, height, width]
    combined_maps = np.zeros((proj_nbr, 2,
                              mapping_matrices[0][0].shape[0],
                              mapping_matrices[0][0].shape[1]), dtype=np.float32)

    for i, (map_x, map_y) in enumerate(mapping_matrices):
        combined_maps[i, 0] = map_x
        combined_maps[i, 1] = map_y

    # Create metadata
    metadata = {
        'proj_nbr': proj_nbr,
        'grid': grid,
        'fov_h': fov_h,
        'fov_v': fov_v,
        'latitude': latitude,
        'lon_0': lon_0,
        'lon_step': lon_step,
        'comp_sz': comp_sz,
        'mapping_matrices': combined_maps,
    }

    return resized_composite, metadata


def main():
    # Load configuration
    cfg = load_configuration()

    # Extract configuration parameters
    img_path = cfg["img_path"]
    proj_nbr = cfg["proj_nbr"]
    fov_h = cfg["fov_h"]
    fov_v = cfg["fov_v"]
    latitude = cfg["latitude"]
    lon_0 = cfg["lon_0"]
    lon_step = cfg["lon_step"]
    grid = cfg["grid"]
    comp_sz = cfg["comp_sz"]
    target_mp = cfg["target_mp"]
    output_dir = cfg["output_dir"]

    # Validate proj_nbr
    if not is_valid_proj_nbr(proj_nbr):
        print(f"Error: Invalid number of projections ({proj_nbr}).")
        print("The number must be even or have an integer square root.")
        return 1

    # Determine grid layout
    grid = determine_grid(proj_nbr, grid)

    # Load fisheye image
    fisheye_img = cv2.imread(img_path)
    if fisheye_img is None:
        print(f"Error: Could not load image at {img_path}")
        return 1

    # Fisheye parameters
    cx, cy = fisheye_img.shape[1] // 2, fisheye_img.shape[0] // 2
    r = min(cx, cy)

    # Create output directory structure
    base_dir = output_dir
    os.makedirs(base_dir, exist_ok=True)

    # Count existing folders to determine the next index
    existing_folders = [d for d in os.listdir(base_dir) if
                        d.startswith("pconf-") and os.path.isdir(os.path.join(base_dir, d))]
    next_index = len(existing_folders) + 1

    out_dir = os.path.join(base_dir, f"pconf-{next_index}")
    os.makedirs(out_dir, exist_ok=True)

    # Initialize log file
    log_file = os.path.join(out_dir, f"log-pconf-{next_index}.txt")
    with open(log_file, 'w') as f:
        f.write(f"Multi-Perspective Projection Configuration #{next_index}\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # If using a preset, include that information
        if "name" in cfg and "description" in cfg and cfg["name"] != "Custom Configuration":
            f.write(f"Preset: {cfg['name']}\n")
            f.write(f"Description: {cfg['description']}\n\n")

        f.write("Input Parameters:\n")
        f.write(f"- Input Image: {img_path}\n")
        f.write(f"- Number of Projections: {proj_nbr}\n")
        f.write(f"- Horizontal FOV: {fov_h} degrees\n")
        f.write(f"- Vertical FOV: {fov_v} degrees\n")
        f.write(f"- Latitude: {latitude} degrees\n")
        f.write(f"- Starting Longitude: {lon_0} degrees\n")
        f.write(f"- Longitude Step: {lon_step} degrees\n")
        f.write(f"- Grid Layout: {grid[0]} rows x {grid[1]} columns\n")
        f.write(f"- Composite Size: {comp_sz[0]}x{comp_sz[1]} pixels\n")
        f.write(f"- Target MP per projection: {target_mp}\n\n")

        # Calculate and log coverage information
        overlap, uncovered = calculate_coverage(lon_0, lon_step, proj_nbr,
                                                fov_h, fov_v, latitude)
        f.write("Coverage Analysis:\n")
        f.write(f"- Projection Overlap: {overlap:.2f}%\n")
        f.write(f"- Uncovered FOV: {uncovered:.2f}%\n\n")

    # Generate projections
    projections = []
    mapping_matrices = []
    total_latency = 0
    fov_colors = generate_rainbow_colors(proj_nbr)

    # Create a copy of the fisheye for FOV visualization
    fisheye_with_fovs = fisheye_img.copy()

    print(f"Generating {proj_nbr} projections...")
    for i in range(proj_nbr):
        # Calculate longitude for this projection
        longitude = (lon_0 + i * lon_step) % 360

        # Generate projection
        projection, latency, mapping = fisheye_to_perspective(
            fisheye_img, cx, cy, r, longitude, latitude,
            fov_h, fov_v, target_mp, grid, comp_sz
        )
        projections.append(projection)
        mapping_matrices.append(mapping)
        total_latency += latency

        # Save projection
        proj_file = os.path.join(out_dir, f"persp-{next_index}-{i}.png")
        cv2.imwrite(proj_file, projection)

        # Store mapping matrices in the list (will save all together later)

        # Draw FOV on fisheye image
        fisheye_with_fovs = draw_fov_on_fisheye(
            fisheye_with_fovs, cx, cy, r, longitude, latitude,
            fov_h, fov_v, fov_colors[i]
        )

        print(f"  Projection {i + 1}/{proj_nbr} at longitude {longitude:.1f}° - Latency: {latency:.1f}ms")

    # Create and save composite image
    composite, resized_composite = create_composite_image(projections, grid, comp_sz)

    composite_file = os.path.join(out_dir, f"or-composit-{next_index}.png")
    cv2.imwrite(composite_file, composite)

    resized_file = os.path.join(out_dir, f"rz-composit-{next_index}.png")
    cv2.imwrite(resized_file, resized_composite)

    # Save original fisheye and FOV visualizations
    fisheye_file = os.path.join(out_dir, f"fish-{next_index}.png")
    cv2.imwrite(fisheye_file, fisheye_img)

    fovs_file = os.path.join(out_dir, f"fovs-fish-{next_index}.png")
    cv2.imwrite(fovs_file, fisheye_with_fovs)

    # Save all mapping matrices in one combined file
    print("Saving combined mapping matrices...")
    # Create array with shape [proj_nbr, 2, height, width]
    combined_maps = np.zeros((proj_nbr, 2,
                              mapping_matrices[0][0].shape[0],
                              mapping_matrices[0][0].shape[1]), dtype=np.float32)

    # Fill the array with mapping data
    for i, (map_x, map_y) in enumerate(mapping_matrices):
        combined_maps[i, 0] = map_x
        combined_maps[i, 1] = map_y

    # Save combined mappings
    map_file = os.path.join(out_dir, f"mappings-{next_index}.npy")
    np.save(map_file, combined_maps)
    with open(log_file, 'a') as f:
        f.write("Performance Information:\n")
        f.write(f"- Total Remapping Latency: {total_latency:.2f} ms\n")
        f.write(f"- Average Latency per Projection: {total_latency / proj_nbr:.2f} ms\n")

        # Output dimensions
        proj_dims = projections[0].shape[:2]
        f.write(f"- Projection Dimensions: {proj_dims[1]}x{proj_dims[0]} pixels\n")
        f.write(f"- Original Composite Dimensions: {composite.shape[1]}x{composite.shape[0]} pixels\n")
        f.write(f"- Resized Composite Dimensions: {comp_sz[0]}x{comp_sz[1]} pixels\n")

        # Mapping matrices information
        f.write("\nMapping Matrices Information:\n")
        f.write(f"- All mapping matrices saved to mappings-{next_index}.npy file\n")
        f.write(
            f"- Combined array shape: [{proj_nbr}, 2, {mapping_matrices[0][0].shape[0]}, {mapping_matrices[0][0].shape[1]}]\n")
        f.write(f"- First dimension: Projection index (0-{proj_nbr - 1})\n")
        f.write(f"- Second dimension: Map type (0=map_x, 1=map_y)\n")

    print("\nProcessing complete!")
    print(f"All outputs saved to: {out_dir}")
    print(f"Configuration log saved to: {log_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())