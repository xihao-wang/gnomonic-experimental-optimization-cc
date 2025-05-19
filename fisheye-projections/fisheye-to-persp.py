import cv2
import numpy as np
from time import time

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


def fisheye_to_perspective(fisheye_img, cx, cy, r, longitude, latitude, fov_deg, output_size):
    """
    Project perspective view from a top-view fisheye.
    """
    H_out, W_out = output_size[1], output_size[0]
    rot_matrix = compute_viewing_basis(longitude, latitude)

    # Generate output pixel grid
    u, v = np.meshgrid(np.arange(W_out), np.arange(H_out))

    # Focal length from horizontal FOV
    fov_rad = np.radians(fov_deg)
    f = (W_out / 2) / np.tan(fov_rad / 2)

    # Camera local coordinates
    cx_out, cy_out = W_out / 2, H_out / 2
    x_local = (u - cx_out) / f
    y_local = (cy_out - v) / f  # OpenCV's Y increases downward
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
    print(f"remapping latency : {latency} ms")
    return viewport


# Trackbar setup
def nothing(x): pass

imgs_path = "imgs/fisheye.png"
fisheye_img = cv2.imread(imgs_path)
cx, cy = fisheye_img.shape[1] // 2, fisheye_img.shape[0] // 2
r = min(cx, cy)
output_size = (800, 600)

cv2.namedWindow('Perspective Output')
cv2.createTrackbar('Longitude', 'Perspective Output', 0, 360, nothing)  # 0-360°
cv2.createTrackbar('Latitude', 'Perspective Output', 45, 90, nothing)  # 0-90° (nadir to horizon)
cv2.createTrackbar('FOV', 'Perspective Output', 90, 180, nothing)  # 1-180°

prev_lon = None
prev_lat = None
prev_fov = None

# Initial render
lon = cv2.getTrackbarPos('Longitude', 'Perspective Output')
lat = cv2.getTrackbarPos('Latitude', 'Perspective Output')
fov = cv2.getTrackbarPos('FOV', 'Perspective Output')
output = fisheye_to_perspective(fisheye_img, cx, cy, r, lon, lat, fov, output_size)
cv2.imshow('Perspective Output', output)

while True:
    current_lon = cv2.getTrackbarPos('Longitude', 'Perspective Output')
    current_lat = cv2.getTrackbarPos('Latitude', 'Perspective Output')
    current_fov = cv2.getTrackbarPos('FOV', 'Perspective Output')

    # Only update if parameters changed
    if current_lon != prev_lon or current_lat != prev_lat or current_fov != prev_fov:
        output = fisheye_to_perspective(fisheye_img, cx, cy, r, current_lon, current_lat, current_fov, output_size)
        cv2.imshow('Perspective Output', output)

        # Update previous values
        prev_lon = current_lon
        prev_lat = current_lat
        prev_fov = current_fov

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()