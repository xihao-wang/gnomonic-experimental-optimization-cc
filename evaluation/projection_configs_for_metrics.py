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
        },
        {
            "id": "TEST#04-h60-v90-g(2,3)",
            "name": "Experimental test (6 proj, 2x3 grid)",
            "proj_nbr": 6,
            "fov_h": 60.0,
            "fov_v": 90.0,
            "latitude": 45.0,
            "lon_0": 0.0,
            "lon_step": 360 / 6,
            "grid": [2, 3],
            "comp_sz": [640, 640],
            "target_mp": "auto"
        },

        # ── 4-projection variants (lon_step=90°, fov_h ≥ 90° for full coverage) ──
        # For 4 projections only a [2,2] grid is possible — no mirror pairs here.

        {
            # Minimal-overlap square FOV.  Removes the 16° horizontal redundancy of
            # stagiaire; tests whether tight (zero-overlap) coverage is sufficient.
            "id": "TEST#05-h90-v90-g(2,2)",
            "name": "Experimental test (4 proj, 2x2 grid, 90x90 FOV)",
            "proj_nbr": 4,
            "fov_h": 90.0,
            "fov_v": 90.0,
            "latitude": 45.0,
            "lon_0": 0.0,
            "lon_step": 360 / 4,
            "grid": [2, 2],
            "comp_sz": [640, 640],
            "target_mp": "auto"
        },
        {
            # 10° horizontal overlap, shorter vertical, lower latitude.
            # Lower lat (40°) sweeps more mid-range distance — suited for larger rooms.
            "id": "TEST#06-h100-v80-g(2,2)",
            "name": "Experimental test (4 proj, 2x2 grid, 100x80 FOV)",
            "proj_nbr": 4,
            "fov_h": 100.0,
            "fov_v": 80.0,
            "latitude": 40.0,
            "lon_0": 0.0,
            "lon_step": 360 / 4,
            "grid": [2, 2],
            "comp_sz": [640, 640],
            "target_mp": "auto"
        },
        {
            # Same horizontal as stagiaire (16° overlap) but shorter vertical and higher
            # latitude.  Steeper look angle — favours pedestrians near the camera nadir.
            "id": "TEST#07-h106-v80-g(2,2)",
            "name": "Experimental test (4 proj, 2x2 grid, 106x80 FOV)",
            "proj_nbr": 4,
            "fov_h": 106.0,
            "fov_v": 80.0,
            "latitude": 50.0,
            "lon_0": 0.0,
            "lon_step": 360 / 4,
            "grid": [2, 2],
            "comp_sz": [640, 640],
            "target_mp": "auto"
        },

        # ── 6-projection variants (lon_step=60°, fov_h ≥ 60° for full coverage) ──
        # Each FOV variant is tested with both grid orientations to isolate the grid effect.

        {
            # Grid mirror of TEST#04.  Wider sub-view cells (landscape) give more
            # horizontal pixels per person vs the portrait layout of TEST#04.
            "id": "TEST#08-h60-v90-g(3,2)",
            "name": "Experimental test (6 proj, 3x2 grid, 60x90 FOV)",
            "proj_nbr": 6,
            "fov_h": 60.0,
            "fov_v": 90.0,
            "latitude": 45.0,
            "lon_0": 0.0,
            "lon_step": 360 / 6,
            "grid": [3, 2],
            "comp_sz": [640, 640],
            "target_mp": "auto"
        },
        {
            # 10° horizontal overlap (70° > 60° step).  Safer seam coverage than TEST#04/08
            # without going to the 30° overlap of TEST#01/02.  Portrait grid.
            "id": "TEST#09-h70-v90-g(2,3)",
            "name": "Experimental test (6 proj, 2x3 grid, 70x90 FOV)",
            "proj_nbr": 6,
            "fov_h": 70.0,
            "fov_v": 90.0,
            "latitude": 45.0,
            "lon_0": 0.0,
            "lon_step": 360 / 6,
            "grid": [2, 3],
            "comp_sz": [640, 640],
            "target_mp": "auto"
        },
        {
            # Grid mirror of TEST#09.  All other parameters identical.
            "id": "TEST#10-h70-v90-g(3,2)",
            "name": "Experimental test (6 proj, 3x2 grid, 70x90 FOV)",
            "proj_nbr": 6,
            "fov_h": 70.0,
            "fov_v": 90.0,
            "latitude": 45.0,
            "lon_0": 0.0,
            "lon_step": 360 / 6,
            "grid": [3, 2],
            "comp_sz": [640, 640],
            "target_mp": "auto"
        },
        {
            # Higher latitude (50°) with shorter vertical FOV.  Steeper downward look —
            # tighter vertical extent reduces distortion in the upper region of each sub-view.
            "id": "TEST#11-h60-v80-g(2,3)",
            "name": "Experimental test (6 proj, 2x3 grid, 60x80 FOV, lat50)",
            "proj_nbr": 6,
            "fov_h": 60.0,
            "fov_v": 80.0,
            "latitude": 50.0,
            "lon_0": 0.0,
            "lon_step": 360 / 6,
            "grid": [2, 3],
            "comp_sz": [640, 640],
            "target_mp": "auto"
        },
        {
            # Grid mirror of TEST#11.  All other parameters identical.
            "id": "TEST#12-h60-v80-g(3,2)",
            "name": "Experimental test (6 proj, 3x2 grid, 60x80 FOV, lat50)",
            "proj_nbr": 6,
            "fov_h": 60.0,
            "fov_v": 80.0,
            "latitude": 50.0,
            "lon_0": 0.0,
            "lon_step": 360 / 6,
            "grid": [3, 2],
            "comp_sz": [640, 640],
            "target_mp": "auto"
        },
        {
            # 20° horizontal overlap — intermediate between TEST#04/08 (0°) and
            # TEST#01/02 (30°).  Portrait grid.
            "id": "TEST#13-h80-v90-g(2,3)",
            "name": "Experimental test (6 proj, 2x3 grid, 80x90 FOV)",
            "proj_nbr": 6,
            "fov_h": 80.0,
            "fov_v": 90.0,
            "latitude": 45.0,
            "lon_0": 0.0,
            "lon_step": 360 / 6,
            "grid": [2, 3],
            "comp_sz": [640, 640],
            "target_mp": "auto"
        },
        {
            # Grid mirror of TEST#13.  All other parameters identical.
            "id": "TEST#14-h80-v90-g(3,2)",
            "name": "Experimental test (6 proj, 3x2 grid, 80x90 FOV)",
            "proj_nbr": 6,
            "fov_h": 80.0,
            "fov_v": 90.0,
            "latitude": 45.0,
            "lon_0": 0.0,
            "lon_step": 360 / 6,
            "grid": [3, 2],
            "comp_sz": [640, 640],
            "target_mp": "auto"
        },

        # ── 9-projection variants (lon_step=40°, only [3,3] grid is sensible) ──
        # fov_h starts at 40° (zero overlap = minimum full coverage).
        # Differentiation via horizontal overlap level, vertical FOV, and latitude.

        {
            # Zero horizontal overlap — minimum coverage, least distortion per sub-view.
            # Baseline for the 9-projection family.
            "id": "TEST#15-h40-v90-g(3,3)",
            "name": "Experimental test (9 proj, 3x3 grid, 40x90 FOV)",
            "proj_nbr": 9,
            "fov_h": 40.0,
            "fov_v": 90.0,
            "latitude": 45.0,
            "lon_0": 0.0,
            "lon_step": 360 / 9,
            "grid": [3, 3],
            "comp_sz": [640, 640],
            "target_mp": "auto"
        },
        {
            # 10° horizontal overlap — objects near sub-view seams appear in both
            # adjacent views, improving recall at boundaries.
            "id": "TEST#16-h50-v90-g(3,3)",
            "name": "Experimental test (9 proj, 3x3 grid, 50x90 FOV)",
            "proj_nbr": 9,
            "fov_h": 50.0,
            "fov_v": 90.0,
            "latitude": 45.0,
            "lon_0": 0.0,
            "lon_step": 360 / 9,
            "grid": [3, 3],
            "comp_sz": [640, 640],
            "target_mp": "auto"
        },
        {
            # 20° horizontal overlap — same overlap ratio as Chiang (8 proj, fov_h=48°
            # over 45° step) but with 9 views and a wider per-view angle.
            "id": "TEST#17-h60-v90-g(3,3)",
            "name": "Experimental test (9 proj, 3x3 grid, 60x90 FOV)",
            "proj_nbr": 9,
            "fov_h": 60.0,
            "fov_v": 90.0,
            "latitude": 45.0,
            "lon_0": 0.0,
            "lon_step": 360 / 9,
            "grid": [3, 3],
            "comp_sz": [640, 640],
            "target_mp": "auto"
        },
        {
            # 30° horizontal overlap — generous redundancy at seams.
            "id": "TEST#18-h70-v90-g(3,3)",
            "name": "Experimental test (9 proj, 3x3 grid, 70x90 FOV)",
            "proj_nbr": 9,
            "fov_h": 70.0,
            "fov_v": 90.0,
            "latitude": 45.0,
            "lon_0": 0.0,
            "lon_step": 360 / 9,
            "grid": [3, 3],
            "comp_sz": [640, 640],
            "target_mp": "auto"
        },
        {
            # Chiang-inspired aspect ratio (48x96) scaled to 9 projections.
            # Lower latitude (36°) matches Chiang's geometry.  Direct test of
            # whether finer angular sampling at the same per-view shape improves results.
            "id": "TEST#19-h48-v96-g(3,3)",
            "name": "Experimental test (9 proj, 3x3 grid, 48x96 FOV, lat36)",
            "proj_nbr": 9,
            "fov_h": 48.0,
            "fov_v": 96.0,
            "latitude": 36.0,
            "lon_0": 0.0,
            "lon_step": 360 / 9,
            "grid": [3, 3],
            "comp_sz": [640, 640],
            "target_mp": "auto"
        },
        {
            # Lower latitude (36°) at a round FOV — isolates the latitude effect
            # vs TEST#17 (same FOV, lat=45°).
            "id": "TEST#20-h60-v90-g(3,3)",
            "name": "Experimental test (9 proj, 3x3 grid, 60x90 FOV, lat36)",
            "proj_nbr": 9,
            "fov_h": 60.0,
            "fov_v": 90.0,
            "latitude": 36.0,
            "lon_0": 0.0,
            "lon_step": 360 / 9,
            "grid": [3, 3],
            "comp_sz": [640, 640],
            "target_mp": "auto"
        },
        {
            # Higher latitude (50°) with shorter vertical FOV — steeper downward look,
            # same philosophy as TEST#11/12 but with 9 sub-views.
            "id": "TEST#21-h50-v80-g(3,3)",
            "name": "Experimental test (9 proj, 3x3 grid, 50x80 FOV, lat50)",
            "proj_nbr": 9,
            "fov_h": 50.0,
            "fov_v": 80.0,
            "latitude": 50.0,
            "lon_0": 0.0,
            "lon_step": 360 / 9,
            "grid": [3, 3],
            "comp_sz": [640, 640],
            "target_mp": "auto"
        },
        {
            # Same steeper look (lat=50°) applied to the zero-overlap baseline —
            # isolates the latitude effect at the minimum FOV.
            "id": "TEST#22-h40-v80-g(3,3)",
            "name": "Experimental test (9 proj, 3x3 grid, 40x80 FOV, lat50)",
            "proj_nbr": 9,
            "fov_h": 40.0,
            "fov_v": 80.0,
            "latitude": 50.0,
            "lon_0": 0.0,
            "lon_step": 360 / 9,
            "grid": [3, 3],
            "comp_sz": [640, 640],
            "target_mp": "auto"
        },
        {
            # Same steeper look (lat=50°) applied to the zero-overlap baseline —
            # isolates the latitude effect at the minimum FOV.
            "id": "TEST#23-h60-v90-g(2,3)",
            "name": "Experimental test (6 proj, 2x3 grid, 60x90 FOV, lat50)",
            "proj_nbr": 6,
            "fov_h": 60.0,
            "fov_v": 90.0,
            "latitude": 50.0,
            "lon_0": 0.0,
            "lon_step": 360 / 6,
            "grid": [2, 3],
            "comp_sz": [640, 640],
            "target_mp": "auto"
        },
        {
        "id": "TEST#24-h60-v80-g(2,3)",
        "name": "Experimental test (6 proj, 2x3 grid, 60x80 FOV, lat50)",
        "proj_nbr": 6,
        "fov_h": 60.0,
        "fov_v": 80.0,
        "latitude": 40.0,
        "lon_0": 0.0,
        "lon_step": 360 / 6,
        "grid": [2, 3],
        "comp_sz": [640, 640],
        "target_mp": "auto"
    },
        {
        "id": "TEST#25-h60-v90-g(2,3)",
        "name": "Experimental test (6 proj, 2x3 grid, 60x80 FOV, lat50)",
        "proj_nbr": 6,
        "fov_h": 60.0,
        "fov_v": 90.0,
        "latitude": 40.0,
        "lon_0": 0.0,
        "lon_step": 360 / 6,
        "grid": [2, 3],
        "comp_sz": [640, 640],
        "target_mp": "auto"
    },
        {
        "id": "TEST#26-h60-v90-g(2,3)",
        "name": "Experimental test (6 proj, 2x3 grid, 60x80 FOV, lat50)",
        "proj_nbr": 6,
        "fov_h": 60.0,
        "fov_v": 85.0,
        "latitude": 45.0,
        "lon_0": 0.0,
        "lon_step": 360 / 6,
        "grid": [2, 3],
        "comp_sz": [640, 640],
        "target_mp": "auto"
    },
    {
        "id": "TEST#27-h60-v85-g(2,3)-pad14v",
        "name": "TEST#26 + 14% vertical padding (6 proj, 2x3 grid, 60x85 FOV, lat45)",
        "proj_nbr": 6,
        "fov_h": 60.0,
        "fov_v": 90.0,
        "latitude": 45.0,
        "lon_0": 0.0,
        "lon_step": 360 / 6,
        "grid": [2, 3],
        "comp_sz": [640, 640],
        "target_mp": "auto",
        "pad_direction": "vertical",
        "pad_pct": 0.14
    },
    {
        # 1 central projection (lat=0°, directly nadir — fisheye centre) with square 90x90 FOV,
        # placed at grid slot (0,0).  Covers theta=0° to 45° from nadir (inner half of fisheye).
        # + 5 peripheral projections at lat=65°, 80x70 FOV, uniformly spaced at 72° steps.
        # Peripheral coverage: 65-35=30° to 65+35=100°→clipped 90° — ~15° overlap with central,
        # full horizon-to-edge coverage.  Horizontal: 80° FOV on 72° step = 8° seam overlap.
        # Hypothesis: dedicating one sub-view to the nadir region (highest pedestrian density
        # directly under camera) improves detection vs uniform-latitude configurations.
        # NOTE — coordinate convention: latitude=0 = nadir (fisheye centre),
        #                               latitude=90 = horizon (fisheye edge).
        "id": "TEST#28-central-plus-5periph-g(2,3)",
        "name": "1 central (lat0, 90x90) + 5 peripheral (lat65, 80x70) — 2x3 grid",
        "proj_nbr": 5,
        "fov_h": 75.0,
        "fov_v": 80.0,
        "latitude": 50.0,
        "lon_0": 0.0,
        "lon_step": 360 / 5,
        "grid": [2, 3],
        "comp_sz": [640, 640],
        "target_mp": "auto",
        "extra_projections": [
            {"longitude": 0.0, "latitude": 1.0, "fov_h": 40.0, "fov_v": 40.0, "pos": [0, 0]}
        ]
    },
    {
        "id": "TEST#29-central-plus-5periph-g(2,3)",
        "name": "1 central (lat0, 90x90) + 5 peripheral (lat65, 80x70) — 2x3 grid",
        "proj_nbr": 5,
        "fov_h": 75.0,
        "fov_v": 70.0,
        "latitude": 55.0,
        "lon_0": 0.0,
        "lon_step": 360 / 5,
        "grid": [2, 3],
        "comp_sz": [640, 640],
        "target_mp": "auto",
        "extra_projections": [
            {"longitude": 0.0, "latitude": 1.0, "fov_h": 50.0, "fov_v": 50.0, "pos": [0, 0]}
        ]
    }
    ]
