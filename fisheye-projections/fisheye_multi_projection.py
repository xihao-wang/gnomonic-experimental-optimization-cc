import cv2
import numpy as np
import math
from time import time

def compute_viewing_basis(longitude, latitude):
    """
    Computes viewing basis vector (xyz) for fisheye projection:
    - view_vector: Direction from viewer to target point on sphere
    - right_vector: Horizontal basis of tangent plane (negated to prevent mirroring)
    - up_vector: Vertical basis of tangent plane
    :param longitude: Horizontal angle in degrees (0-360°)
    :param latitude: Vertical angle in degrees (0° = nadir (point directly below), 90° = horizon)
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

    # Right vector (cross product with world-up axis Z)
    world_up = np.array([0.0, 0.0, 1.0])
    R = np.cross(view_vector, world_up)

    # Normalize right vector to ensure unit length
    # The small constant (1e-6) prevents division by zero in edge cases
    R = R / (np.linalg.norm(R) + 1e-6)

    # Up vector (corrected perpendicular to forward and right)
    U = np.cross(R, view_vector)
    U /= np.linalg.norm(U) + 1e-6

    return np.column_stack((-R, U, view_vector))  # -R avoids projection mirroring


def fisheye_to_perspective(fisheye_img, cx, cy, r, longitude, latitude, h_fov_deg, v_fov_deg, output_size):
    """
    Project perspective view from a top-view fisheye.
    :param fisheye_img: Input fisheye image
    :param cx, cy: Center of the fisheye image
    :param r: Radius of the fisheye image
    :param longitude, latitude: Viewing direction in degrees
    :param h_fov_deg, v_fov_deg: Horizontal and vertical field of view in degrees
    :param output_size: (width, height) of the output perspective image
    :return: Perspective view and mapping data
    """
    H_out, W_out = output_size[1], output_size[0]
    rot_matrix = compute_viewing_basis(longitude, latitude)

    # Generate output pixel grid
    u, v = np.meshgrid(np.arange(W_out), np.arange(H_out))

    # Focal lengths from FOV
    h_fov_rad = np.radians(h_fov_deg)
    v_fov_rad = np.radians(v_fov_deg)
    fx = (W_out / 2) / np.tan(h_fov_rad / 2)
    fy = (H_out / 2) / np.tan(v_fov_rad / 2)

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
    viewport = cv2.remap(fisheye_img, map_x, map_y, cv2.INTER_LINEAR, 
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    return viewport, (map_x, map_y, valid_mask)


def calculate_longitudes(num_projections, h_fov, desired_intersection):
    """
    Calculate longitudes for all projections based on number of projections and desired overlap.
    :param num_projections: Total number of projections
    :param h_fov: Horizontal field of view in degrees
    :param desired_intersection: Desired overlap between adjacent projections (%)
    :return: List of longitudes for each projection
    """
    # Calculate grid dimensions (make it a square grid)
    n_dim = int(math.sqrt(num_projections))
    
    # Calculate effective angle step considering overlap
    overlap_fraction = desired_intersection / 100
    effective_fov = h_fov * (1 - overlap_fraction)
    lon_step = 360 / n_dim
    
    longitudes = []
    for i in range(n_dim):
        for j in range(n_dim):
            # Calculate longitude to distribute projections evenly
            lon = (j * lon_step) % 360
            longitudes.append(lon)
    
    # Trim to exact number required
    return longitudes[:num_projections]


def draw_projection_boundaries(fisheye_img, cx, cy, r, longitude, latitude, h_fov, v_fov, output_size, color):
    """
    Draw the boundary of a projection on the fisheye image.
    :return: Center point of the boundary (for adding a label)
    """
    H_out, W_out = output_size[1], output_size[0]
    
    # Generate boundary points in the perspective view
    boundary_points = []
    
    # Top edge
    for x in range(0, W_out+1, W_out//10):
        boundary_points.append([x, 0])
    
    # Right edge
    for y in range(0, H_out+1, H_out//10):
        boundary_points.append([W_out-1, y])
    
    # Bottom edge
    for x in range(W_out-1, -1, -W_out//10):
        boundary_points.append([x, H_out-1])
    
    # Left edge
    for y in range(H_out-1, -1, -H_out//10):
        boundary_points.append([0, y])
    
    # Convert boundary points to fisheye coordinates
    rot_matrix = compute_viewing_basis(longitude, latitude)
    
    # Focal lengths from FOV
    h_fov_rad = np.radians(h_fov)
    v_fov_rad = np.radians(v_fov)
    fx = (W_out / 2) / np.tan(h_fov_rad / 2)
    fy = (H_out / 2) / np.tan(v_fov_rad / 2)
    
    # Camera center
    cx_out, cy_out = W_out / 2, H_out / 2
    
    # Convert points to fisheye coordinates
    fisheye_points = []
    
    for point in boundary_points:
        x, y = point
        
        # Camera local coordinates
        x_local = (x - cx_out) / fx
        y_local = (cy_out - y) / fy  # OpenCV's Y increases downward
        z_local = 1.0
        
        # Normalize direction vector
        norm = math.sqrt(x_local**2 + y_local**2 + z_local**2)
        dir_vector = np.array([x_local / norm, y_local / norm, z_local / norm])
        
        # Rotate to align with fisheye's orientation
        world_dir = np.dot(rot_matrix, dir_vector)
        
        # Convert to spherical coordinates
        theta = np.arccos(-world_dir[2])  # Angle from nadir
        phi = np.arctan2(world_dir[1], world_dir[0])
        
        # Check if point is within fisheye FOV
        if theta <= (np.pi / 2):  # Fisheye only captures <= 90° from nadir
            # Map to fisheye image coordinates
            r_pixel = (theta / (np.pi / 2)) * r
            x_fisheye = int(cx + r_pixel * np.cos(phi))
            y_fisheye = int(cy + r_pixel * np.sin(phi))
            
            fisheye_points.append((x_fisheye, y_fisheye))
    
    # Draw boundary lines
    if len(fisheye_points) > 1:
        for i in range(len(fisheye_points)):
            cv2.line(fisheye_img, fisheye_points[i], fisheye_points[(i+1) % len(fisheye_points)], color, 2)
    
    # Calculate center point for label
    if fisheye_points:
        center_x = sum(p[0] for p in fisheye_points) // len(fisheye_points)
        center_y = sum(p[1] for p in fisheye_points) // len(fisheye_points)
        
        return (center_x, center_y)
    
    return None


def update_intersection_from_fov(num_projections, h_fov):
    """
    Calculate intersection percentage based on h_fov and number of projections.
    """
    if num_projections <= 1:
        return 0
    
    # Number of projections in each dimension
    n_dim = int(math.sqrt(num_projections))
    if n_dim <= 1:
        return 0
    
    # Total FOV for panorama (full 360° horizontally)
    full_h_fov = 360
    
    # Calculate the necessary overlap
    non_overlap_portion = full_h_fov / (n_dim * h_fov)
    if non_overlap_portion >= 1:
        # FOV is too small to have any overlap
        return 0
    else:
        # Calculate overlap percentage
        overlap_percentage = (1 - non_overlap_portion) * 100
        # Clamp to valid range
        overlap_percentage = max(0, min(80, overlap_percentage))
        return int(overlap_percentage)


def update_fov_from_intersection(num_projections, desired_intersection):
    """
    Calculate horizontal FOV based on intersection and number of projections.
    """
    if num_projections <= 1:
        return 90
    
    # Number of projections in each dimension
    n_dim = int(math.sqrt(num_projections))
    if n_dim <= 1:
        return 90
    
    # Total FOV for panorama (full 360° horizontally)
    full_h_fov = 360
    
    # Calculate FOV with desired overlap
    overlap_fraction = desired_intersection / 100
    if overlap_fraction >= 1:  # Prevent division by zero
        return 180
    
    fov_required = full_h_fov / (n_dim * (1 - overlap_fraction))
    
    # Clamp to valid range
    h_fov = max(10, min(180, fov_required))
    
    return int(h_fov)


def nothing(x):
    pass


def main():
    # Load image
    imgs_path = "imgs/fisheye.png"
    fisheye_img = cv2.imread(imgs_path)
    if fisheye_img is None:
        print(f"Error: Could not load image from {imgs_path}")
        return
    
    # Calculate image center and radius
    cy, cx = fisheye_img.shape[0] // 2, fisheye_img.shape[1] // 2
    r = min(cx, cy)
    
    # Default parameters
    output_width = 400
    output_height = 300
    num_projections = 4
    desired_intersection = 20
    latitude = 45
    h_fov = 90
    v_fov = 90
    
    # Create parameter window
    cv2.namedWindow('Parameters', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Parameters', 600, 300)
    cv2.createTrackbar('Num Projections', 'Parameters', num_projections, 16, nothing)
    cv2.createTrackbar('Intersection %', 'Parameters', desired_intersection, 80, nothing)
    cv2.createTrackbar('Latitude', 'Parameters', latitude, 90, nothing)
    cv2.createTrackbar('H-FOV', 'Parameters', h_fov, 180, nothing)
    cv2.createTrackbar('V-FOV', 'Parameters', v_fov, 180, nothing)
    cv2.createTrackbar('Output Width', 'Parameters', output_width, 800, nothing)
    cv2.createTrackbar('Output Height', 'Parameters', output_height, 600, nothing)
    
    # Create output windows
    cv2.namedWindow('Fisheye with Boundaries', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Composite Image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Info', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Info', 400, 400)
    
    # Initial flags for parameter linking
    last_update_from = None
    
    while True:
        # Get current parameters
        current_num_projections = max(1, cv2.getTrackbarPos('Num Projections', 'Parameters'))
        current_intersection = cv2.getTrackbarPos('Intersection %', 'Parameters')
        current_latitude = cv2.getTrackbarPos('Latitude', 'Parameters')
        current_h_fov = max(10, cv2.getTrackbarPos('H-FOV', 'Parameters'))
        current_v_fov = max(10, cv2.getTrackbarPos('V-FOV', 'Parameters'))
        current_output_width = max(100, cv2.getTrackbarPos('Output Width', 'Parameters'))
        current_output_height = max(100, cv2.getTrackbarPos('Output Height', 'Parameters'))
        
        # Check for parameter changes and update linked parameters
        if num_projections != current_num_projections and last_update_from != "intersection":
            num_projections = current_num_projections
            # When num_projections changes, update intersection
            desired_intersection = update_intersection_from_fov(num_projections, h_fov)
            cv2.setTrackbarPos('Intersection %', 'Parameters', desired_intersection)
            last_update_from = "num_projections"
        
        if h_fov != current_h_fov and last_update_from != "intersection":
            h_fov = current_h_fov
            # When h_fov changes, update intersection
            desired_intersection = update_intersection_from_fov(num_projections, h_fov)
            cv2.setTrackbarPos('Intersection %', 'Parameters', desired_intersection)
            last_update_from = "h_fov"
        
        if desired_intersection != current_intersection and last_update_from != "num_projections" and last_update_from != "h_fov":
            desired_intersection = current_intersection
            # When intersection changes, update h_fov
            h_fov = update_fov_from_intersection(num_projections, desired_intersection)
            cv2.setTrackbarPos('H-FOV', 'Parameters', h_fov)
            last_update_from = "intersection"
        
        # Update other parameters
        latitude = current_latitude
        v_fov = current_v_fov
        output_width = current_output_width
        output_height = current_output_height
        
        # Calculate grid dimensions
        n_dim = int(math.sqrt(num_projections))
        
        # Calculate longitudes for all projections
        longitudes = calculate_longitudes(num_projections, h_fov, desired_intersection)
        
        # Create composite image
        composite_width = n_dim * output_width
        composite_height = n_dim * output_height
        composite = np.zeros((composite_height, composite_width, 3), dtype=np.uint8)
        
        # Create copy of fisheye image for drawing boundaries
        fisheye_with_boundaries = fisheye_img.copy()
        
        # Draw central circle representing the fisheye boundary
        cv2.circle(fisheye_with_boundaries, (cx, cy), r, (255, 255, 255), 2)
        
        # Generate all projections
        for i, lon in enumerate(longitudes):
            # Generate projection
            st = time()
            proj, _ = fisheye_to_perspective(
                fisheye_img, cx, cy, r, lon, latitude, h_fov, v_fov, 
                (output_width, output_height)
            )
            proj_time = (time() - st) * 1000
            
            # Add to composite image
            row = i // n_dim
            col = i % n_dim
            y_start = row * output_height
            y_end = y_start + output_height
            x_start = col * output_width
            x_end = x_start + output_width
            
            # Only add the projection if we have enough space
            if y_end <= composite_height and x_end <= composite_width:
                composite[y_start:y_end, x_start:x_end] = proj
            
            # Colors for each projection (up to 16 distinct colors)
            colors = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
                (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
                (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 128, 255)
            ]
            color = colors[i % len(colors)]
            
            # Draw projection boundary on fisheye image
            center = draw_projection_boundaries(
                fisheye_with_boundaries, cx, cy, r, lon, latitude, h_fov, v_fov, 
                (output_width, output_height), color
            )
            
            # Add projection number if center is valid
            if center:
                cv2.putText(fisheye_with_boundaries, str(i+1), center, 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        
        # Create info image
        info_img = np.ones((400, 400, 3), dtype=np.uint8) * 255
        
        # Add info text
        info_text = [
            f"Number of Projections: {num_projections}",
            f"Grid: {n_dim}x{n_dim}",
            f"Output Size: {output_width}x{output_height}",
            f"Intersection: {desired_intersection}%",
            f"Latitude: {latitude}°",
            f"H-FOV: {h_fov}°",
            f"V-FOV: {v_fov}°",
            f"Projection time: {proj_time:.1f} ms",
            "",
            "Longitudes:"
        ]
        
        for i, line in enumerate(info_text):
            cv2.putText(info_img, line, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (0, 0, 0), 1, cv2.LINE_AA)
        
        # Add longitudes
        max_display = min(12, len(longitudes))  # Show up to 12 longitudes
        for i, lon in enumerate(longitudes[:max_display]):
            row = i // 2
            col = i % 2
            x_pos = 10 + col * 200
            y_pos = 280 + row * 25
            text = f"Proj {i+1}: {lon:.1f}°"
            cv2.putText(info_img, text, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (0, 0, 0), 1, cv2.LINE_AA)
        
        # Display images
        cv2.imshow('Fisheye with Boundaries', fisheye_with_boundaries)
        cv2.imshow('Composite Image', composite)
        cv2.imshow('Info', info_img)
        
        # Reset update flag
        if last_update_from:
            last_update_from = None
        
        # Check for exit
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save images
            cv2.imwrite("composite.png", composite)
            cv2.imwrite("fisheye_boundaries.png", fisheye_with_boundaries)
            print("Saved images as 'composite.png' and 'fisheye_boundaries.png'")
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
