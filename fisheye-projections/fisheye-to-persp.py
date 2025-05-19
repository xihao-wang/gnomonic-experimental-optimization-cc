"""
Fisheye to Perspective Projection

This script converts a fisheye image to a perspective projection with configurable parameters.
It maintains a 'target_mp' megapixel (defaults to 0.48 MP)  output resolution while ensuring the width/height ratio
matches the horizontal/vertical field of view ratio. Controls are separated into a dedicated window.

Features:
- Adjustable longitude (0-360°) and latitude (0-90°)
- Independent horizontal and vertical FOV controls
- Automatic dimension calculation of the view port based on FOV ratio and target_mp
- Constant target_mp MP output resolution
- Performance tracking with interpolation remapping latency measurement
"""

import cv2
import numpy as np
from time import time
import math


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
    # world_up = np.array([0.0, 1.0, 0.0])
    world_up = np.array([0.0, 0.0, 1.0])
    R = np.cross(view_vector, world_up)

    # Normalize right vector to ensure unit length
    # The small constant (1e-6) prevents division by zero in edge cases
    R = R / (np.linalg.norm(R) + 1e-6)

    # Up vector (corrected perpendicular to forward and right)
    U = np.cross(R, view_vector)
    U /= np.linalg.norm(U) + 1e-6

    return np.column_stack((-R, U, view_vector))  # -R avoids projection mirroring


def calculate_dimensions(fov_h_deg, fov_v_deg, target_mp=0.48):
    """
    Calculate output dimensions based on FOV and target megapixels.

    :param fov_h_deg: Horizontal field of view in degrees
    :param fov_v_deg: Vertical field of view in degrees
    :param target_mp: Target megapixels (default 0.48)
    :return: Tuple of (width, height) in pixels
    """
    # Calculate the aspect ratio based on FOV
    aspect_ratio = fov_h_deg / fov_v_deg

    # Calculate dimensions that maintain the aspect ratio and hit the target megapixels
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

def fisheye_to_perspective(fisheye_img, cx, cy, r, longitude, latitude, fov_h_deg, fov_v_deg):
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
    :return: Perspective projection image with dimensions calculated from FOV ratio and 0.48MP target
    """
    # Calculate output dimensions based on FOV ratio
    output_size = calculate_dimensions(fov_h_deg, fov_v_deg, target_mp=0.48)
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
    print(map_x.shape)
    print(map_y.shape)
    viewport = cv2.remap(fisheye_img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    latency = round((time() - st) * 1000, 3)
    print(
        f"Remapping latency: {latency} ms | Output size: {output_size[0]}x{output_size[1]} "
        f"({output_size[0] * output_size[1] / 1_000_000:.2f} MP)")
    return viewport


def main():
    # Load fisheye image
    imgs_path = "imgs/fisheye.png"
    fisheye_img = cv2.imread(imgs_path)
    if fisheye_img is None:
        print(f"Error: Could not load image at {imgs_path}")
        return

    cx, cy = fisheye_img.shape[1] // 2, fisheye_img.shape[0] // 2
    r = min(cx, cy)

    # Create separate windows for controls and output
    cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Controls', 600, 200)
    cv2.namedWindow('Perspective Output', cv2.WINDOW_NORMAL)

    # Trackbar callback
    def nothing(x):
        pass

    # Create trackbars in the control window
    trackbar_Fh = 'FOV Horiz'
    trackbar_Fv = 'FOV Verti'
    cv2.createTrackbar('Longitude', 'Controls', 0, 360, nothing)  # 0-360°
    cv2.createTrackbar('Latitude', 'Controls', 45, 90, nothing)  # 0-90° (nadir to horizon)
    cv2.createTrackbar(trackbar_Fh, 'Controls', 90, 180, nothing)  # 1-180°
    cv2.createTrackbar(trackbar_Fv, 'Controls', 60, 180, nothing)  # 1-180°

    prev_lon = None
    prev_lat = None
    prev_fov_h = None
    prev_fov_v = None

    # Initial render

    lon = cv2.getTrackbarPos('Longitude', 'Controls')
    lat = cv2.getTrackbarPos('Latitude', 'Controls')
    fov_h = max(1, cv2.getTrackbarPos(trackbar_Fh, 'Controls'))
    fov_v = max(1, cv2.getTrackbarPos(trackbar_Fv, 'Controls'))

    output = fisheye_to_perspective(fisheye_img, cx, cy, r, lon, lat, fov_h, fov_v)
    cv2.imshow('Perspective Output', output)

    # Main loop here
    while True:
        current_lon = cv2.getTrackbarPos('Longitude', 'Controls')
        current_lat = cv2.getTrackbarPos('Latitude', 'Controls')
        current_fov_h = max(1, cv2.getTrackbarPos(trackbar_Fh, 'Controls'))
        current_fov_v = max(1, cv2.getTrackbarPos(trackbar_Fv, 'Controls'))

        # Only update if parameters changed
        if (current_lon != prev_lon or current_lat != prev_lat or
                current_fov_h != prev_fov_h or current_fov_v != prev_fov_v):
            output = fisheye_to_perspective(
                fisheye_img, cx, cy, r, current_lon, current_lat, current_fov_h, current_fov_v
            )
            cv2.imshow('Perspective Output', output)

            # Update previous values
            prev_lon = current_lon
            prev_lat = current_lat
            prev_fov_h = current_fov_h
            prev_fov_v = current_fov_v
            output_h, output_w = output.shape[:2]
            cv2.resizeWindow('Perspective Output', output_w, output_h)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()