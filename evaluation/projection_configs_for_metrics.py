"""
Projection configurations for metrics evaluation.

Edit this file to define which projection configurations to compare.
Each configuration must specify: proj_nbr, fov_h, fov_v, latitude, lon_0,
lon_step, grid, comp_sz, target_mp.
"""


class ProjectionConfigs:
    """Define projection configurations to evaluate."""

    CONFIGS = [
        {
            "id": "stagiaire-grid-2x2-fov60",
            "name": "2x2 Grid, 60deg FOV",
            "proj_nbr": 4,
            "fov_h": 106.0,
            "fov_v": 106.0,
            "latitude": 37.0,
            "lon_0": 0.0,
            "lon_step": 90.0,
            "grid": [2, 2],
            "comp_sz": [640, 640],
            "target_mp": "auto"
        },
        {
            "id": "chiang-2021-baseline",
            "name": "Chiang 2021 Baseline (8 proj, 2x4 grid)",
            "proj_nbr": 8,
            "fov_h": 48.0,
            "fov_v": 96.0,
            "latitude": 36.0,
            "lon_0": 0.0,
            "lon_step": 45.0,
            "grid": [2, 4],
            "comp_sz": [640, 640],
            "target_mp": "auto"
        },
        {
            "id": "TEST#01-h90-v90-g(2,3)",
            "name": "Experimental test (6 proj, 2x3 grid)",
            "proj_nbr": 6,
            "fov_h": 90.0,
            "fov_v": 90.0,
            "latitude": 45.0,
            "lon_0": 0.0,
            "lon_step": 360/6,
            "grid": [2, 3],
            "comp_sz": [640, 640],
            "target_mp": "auto"
        },
        {
            "id": "TEST#02_h90-v90-g(3,2)",
            "name": "Experimental test (6 proj, 3x2 grid)",
            "proj_nbr": 6,
            "fov_h": 90.0,
            "fov_v": 90.0,
            "latitude": 45.0,
            "lon_0": 0.0,
            "lon_step": 360/6,
            "grid": [3, 2],
            "comp_sz": [640, 640],
            "target_mp": "auto"
        }
    ]
