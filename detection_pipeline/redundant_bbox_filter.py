"""
Redundant-bbox filter (border-based method).

Two-stage:
  1. Composite-side (geometric): TAG a YOLO detection whose bbox has a side
     flush (within a pixel tolerance) against an *active* border of the tile
     it belongs to. Flagged detections are NOT dropped — they continue
     through backprojection and Stage-2 Soft-NMS.
  2. Fisheye-side (containment confirmation): after Stage-2 Soft-NMS, drop
     a still-flagged bbox only if there exists another fisheye bbox that is
     sufficiently larger (area ratio threshold) AND partially contains it
     (IoS threshold).

All tunables live in detection_pipeline/config.py under
REDUNDANT_BBOX_FILTER.BORDER_BASED.

Author: Yassir Zardoua — y.zardoua@caplogy.com | yassirzardoua@gmail.com
"""

from typing import Dict, List, Tuple

import cv2
import numpy as np


# A rectangle has exactly four sides — structural invariant, not user-editable.
BBOX_SIDES = ("top", "right", "bottom", "left")


# ---------------------------------------------------------------------------
# Active-border construction (preset + overrides)
# ---------------------------------------------------------------------------

def build_active_borders(
    grid: Tuple[int, int],
    preset: str,
    overrides: List[List],
    has_extra_projections: bool = False,
) -> Dict[Tuple[int, int], Dict[str, bool]]:
    """
    Build the active-border map: for every (row, col) tile in the grid, which
    of its 4 bbox-aligned tile edges participate in flagging.
    """
    rows, cols = grid
    active: Dict[Tuple[int, int], Dict[str, bool]] = {}

    if preset == "internal_only":
        if has_extra_projections:
            print(
                "\n" + "!" * 78 + "\n"
                "WARNING: REDUNDANT_BBOX_FILTER.BORDER_BASED.FLAGGING.PRESET ==\n"
                "'internal_only' is being used together with image_composer\n"
                "extra_projections. The preset assumes grid-adjacent tiles are also\n"
                "fisheye-adjacent (true for uniform longitude stepping). With\n"
                "extra_projections this may NOT hold — a person truncated at a shared\n"
                "grid edge may not continue into the neighbour tile in fisheye space.\n"
                "Use REDUNDANT_BBOX_FILTER.BORDER_BASED.FLAGGING.OVERRIDES to disable\n"
                "suppression on the affected sides, or switch to 'all_but_tile_top' /\n"
                "'all' with explicit overrides.\n"
                + "!" * 78 + "\n"
            )
        for r in range(rows):
            for c in range(cols):
                active[(r, c)] = {
                    "top":    r > 0,
                    "bottom": r < rows - 1,
                    "left":   c > 0,
                    "right":  c < cols - 1,
                }
    elif preset == "all_but_tile_top":
        for r in range(rows):
            for c in range(cols):
                active[(r, c)] = {"top": False, "right": True, "bottom": True, "left": True}
    elif preset == "all_but_tile_top_and_bottom":
        # Only the vertical sides (left/right) of every tile participate.
        # Use when the top edge of each tile maps to the fisheye outer ring
        # AND the bottom edge maps to the fisheye centre (nadir/zenith) —
        # in both cases a flush bbox side cannot be a cross-tile fragment.
        for r in range(rows):
            for c in range(cols):
                active[(r, c)] = {"top": False, "right": True, "bottom": False, "left": True}
    elif preset == "all":
        for r in range(rows):
            for c in range(cols):
                active[(r, c)] = {s: True for s in BBOX_SIDES}
    else:
        raise ValueError(
            f"Unknown REDUNDANT_BBOX_FILTER.BORDER_BASED.FLAGGING.PRESET '{preset}'. "
            f"Expected one of: 'internal_only', 'all_but_tile_top', "
            f"'all_but_tile_top_and_bottom', 'all'."
        )

    for entry in overrides:
        if len(entry) != 4:
            raise ValueError(
                "REDUNDANT_BBOX_FILTER.BORDER_BASED.FLAGGING.OVERRIDES entry must be "
                f"[row, col, side, enabled], got {entry}"
            )
        r, c, side, enabled = entry
        if not (0 <= r < rows and 0 <= c < cols):
            raise ValueError(f"Override (row={r}, col={c}) out of bounds for grid {grid}")
        if side not in BBOX_SIDES:
            raise ValueError(f"Override side '{side}' must be one of {BBOX_SIDES}")
        active[(r, c)][side] = bool(enabled)

    return active


def _tile_of_box(cx_norm: float, cy_norm: float, grid: Tuple[int, int]) -> Tuple[int, int]:
    """Return (row, col) of the tile containing the box centre."""
    rows, cols = grid
    col = min(cols - 1, max(0, int(cx_norm * cols)))
    row = min(rows - 1, max(0, int(cy_norm * rows)))
    return row, col


# ---------------------------------------------------------------------------
# Stage 1: flag border-aligned detections (composite, normalized coords)
# ---------------------------------------------------------------------------

def flag_border_aligned_candidates(
    detections: List[Dict],
    composite_shape: Tuple[int, int],
    grid: Tuple[int, int],
    active_borders: Dict[Tuple[int, int], Dict[str, bool]],
    tolerance_px: float,
) -> List[Dict]:
    """
    Tag detections whose bbox has any side flush against an active tile border.

    Returns the same list of detections with two new keys added on flagged ones:
        '_flagged_border_side': side that triggered the flag
        '_flagged_tile':        (row, col) of the source tile

    Unflagged detections are returned unchanged. Nothing is dropped here.
    """
    if not detections:
        return []

    h, w = composite_shape[:2]
    rows, cols = grid
    cell_w = w / cols
    cell_h = h / rows

    out: List[Dict] = []
    for det in detections:
        cx_px = det["x"] * w
        cy_px = det["y"] * h
        bw_px = det["w"] * w
        bh_px = det["h"] * h

        left_px   = cx_px - bw_px / 2
        right_px  = cx_px + bw_px / 2
        top_px    = cy_px - bh_px / 2
        bottom_px = cy_px + bh_px / 2

        row, col = _tile_of_box(det["x"], det["y"], grid)
        tile_left   = col * cell_w
        tile_right  = (col + 1) * cell_w
        tile_top    = row * cell_h
        tile_bottom = (row + 1) * cell_h

        sides_active = active_borders[(row, col)]
        flagged_side = None
        if sides_active["top"]    and abs(top_px    - tile_top)    <= tolerance_px:
            flagged_side = "top"
        elif sides_active["bottom"] and abs(bottom_px - tile_bottom) <= tolerance_px:
            flagged_side = "bottom"
        elif sides_active["left"]   and abs(left_px   - tile_left)   <= tolerance_px:
            flagged_side = "left"
        elif sides_active["right"]  and abs(right_px  - tile_right)  <= tolerance_px:
            flagged_side = "right"

        if flagged_side is None:
            out.append(det)
        else:
            tagged = dict(det)
            tagged["_flagged_border_side"] = flagged_side
            tagged["_flagged_tile"] = (row, col)
            out.append(tagged)

    return out


# ---------------------------------------------------------------------------
# Stage 2: containment confirmation (fisheye, rotated boxes)
# ---------------------------------------------------------------------------

def _rotated_ios(bbox_a: Dict, bbox_b: Dict) -> float:
    """
    Intersection over Smaller for two rotated bboxes:
        IoS = |A ∩ B| / min(|A|, |B|).
    Returns 0.0 when the two boxes don't intersect.
    """
    rect_a = (tuple(bbox_a["center"]), tuple(bbox_a["size"]), float(bbox_a["angle"]))
    rect_b = (tuple(bbox_b["center"]), tuple(bbox_b["size"]), float(bbox_b["angle"]))

    inter_type, inter_pts = cv2.rotatedRectangleIntersection(rect_a, rect_b)
    area_a = bbox_a["size"][0] * bbox_a["size"][1]
    area_b = bbox_b["size"][0] * bbox_b["size"][1]
    smaller = min(area_a, area_b)

    if inter_type == cv2.INTERSECT_NONE or smaller <= 0:
        return 0.0
    if inter_type == cv2.INTERSECT_FULL:
        return 1.0
    if inter_pts is None or len(inter_pts) < 3:
        return 0.0

    inter_area = cv2.contourArea(inter_pts)
    return max(0.0, min(1.0, inter_area / smaller))


def drop_confirmed_redundant_bboxes(
    fisheye_bboxes: List[Dict],
    min_area_ratio_to_larger: float,
    min_overlap_ios: float,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Apply the fisheye-side containment confirmation.

    A bbox tagged '_flagged_border_side' is dropped iff there exists another
    bbox B in `fisheye_bboxes` such that:
        area(B) / area(flagged) >= min_area_ratio_to_larger
        IoS(B, flagged)         >= min_overlap_ios

    Unflagged bboxes are always kept.

    Returns (kept, dropped). Dropped bboxes are tagged with extra keys:
        '_drop_reason': "border_based_confirmed"
        '_dropped_by_index': index of the larger box that triggered the drop
    """
    if not fisheye_bboxes:
        return [], []

    n = len(fisheye_bboxes)
    areas = [bb["size"][0] * bb["size"][1] for bb in fisheye_bboxes]

    kept: List[Dict] = []
    dropped: List[Dict] = []

    for i, bb in enumerate(fisheye_bboxes):
        if "_flagged_border_side" not in bb:
            kept.append(bb)
            continue

        area_i = areas[i]
        triggered_by = None
        for j in range(n):
            if j == i or areas[j] <= 0 or area_i <= 0:
                continue
            if areas[j] / area_i < min_area_ratio_to_larger:
                continue
            if _rotated_ios(fisheye_bboxes[j], bb) < min_overlap_ios:
                continue
            triggered_by = j
            break

        if triggered_by is None:
            kept.append(bb)
        else:
            tagged = dict(bb)
            tagged["_drop_reason"] = "border_based_confirmed"
            tagged["_dropped_by_index"] = triggered_by
            dropped.append(tagged)

    return kept, dropped


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def draw_grid_overlay(image: np.ndarray, grid: Tuple[int, int]) -> np.ndarray:
    """Overlay tile-grid lines on a copy of the composite (cyan, thin)."""
    out = image.copy()
    h, w = out.shape[:2]
    rows, cols = grid
    cell_w = w / cols
    cell_h = h / rows
    color = (255, 255, 0)  # cyan in BGR
    for r in range(1, rows):
        y = int(round(r * cell_h))
        cv2.line(out, (0, y), (w - 1, y), color, 1)
    for c in range(1, cols):
        x = int(round(c * cell_w))
        cv2.line(out, (x, 0), (x, h - 1), color, 1)
    return out


def _draw_box_with_corners(
    image: np.ndarray,
    det: Dict,
    color: Tuple[int, int, int],
    thickness: int = 2,
    label: str = None,
) -> None:
    """Draw a normalized-coord bbox on `image` and mark its two diagonal corners."""
    h, w = image.shape[:2]
    cx, cy = det["x"] * w, det["y"] * h
    bw, bh = det["w"] * w, det["h"] * h
    x1, y1 = int(round(cx - bw / 2)), int(round(cy - bh / 2))
    x2, y2 = int(round(cx + bw / 2)), int(round(cy + bh / 2))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    cv2.circle(image, (x1, y1), 4, color, -1)
    cv2.circle(image, (x2, y2), 4, color, -1)
    if label:
        cv2.putText(image, label, (x1, max(0, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def visualize_composite_flags(
    composite_image: np.ndarray,
    detections: List[Dict],
    grid: Tuple[int, int],
) -> np.ndarray:
    """
    Composite-side inspection view (post-flagging, pre-backprojection).
      - Tile grid in cyan.
      - Unflagged boxes: green.
      - Flagged boxes:   orange, with the triggering side over-drawn in magenta.
    """
    out = draw_grid_overlay(composite_image, grid)
    h, w = out.shape[:2]
    orange  = (0, 165, 255)  # BGR
    magenta = (255, 0, 255)
    green   = (0, 255, 0)

    for det in detections:
        if "_flagged_border_side" in det:
            label = f"FLAG-{det['_flagged_border_side']} {det.get('confidence', 0):.2f}"
            _draw_box_with_corners(out, det, color=orange, thickness=2, label=label)
            cx, cy = det["x"] * w, det["y"] * h
            bw, bh = det["w"] * w, det["h"] * h
            x1, y1 = int(round(cx - bw / 2)), int(round(cy - bh / 2))
            x2, y2 = int(round(cx + bw / 2)), int(round(cy + bh / 2))
            side = det["_flagged_border_side"]
            if side == "top":
                cv2.line(out, (x1, y1), (x2, y1), magenta, 3)
            elif side == "bottom":
                cv2.line(out, (x1, y2), (x2, y2), magenta, 3)
            elif side == "left":
                cv2.line(out, (x1, y1), (x1, y2), magenta, 3)
            elif side == "right":
                cv2.line(out, (x2, y1), (x2, y2), magenta, 3)
        else:
            _draw_box_with_corners(
                out, det, color=green, thickness=2,
                label=f"{det.get('confidence', 0):.2f}"
            )

    return out


def visualize_detections_on_grid(
    composite_image: np.ndarray,
    detections: List[Dict],
    grid: Tuple[int, int],
    color: Tuple[int, int, int] = (0, 255, 0),
    label_conf: bool = True,
) -> np.ndarray:
    """Render boxes + corner dots over the composite with tile-grid lines."""
    out = draw_grid_overlay(composite_image, grid)
    for det in detections:
        label = f"{det.get('confidence', 0):.2f}" if label_conf else None
        _draw_box_with_corners(out, det, color=color, thickness=2, label=label)
    return out
