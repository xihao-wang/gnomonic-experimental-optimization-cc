#!/usr/bin/env python3
"""Minimal rotated-box GT editor for gnomonic fisheye tracking outputs.

This viewer is intentionally independent from the old ``debug_match_viewer``.
It edits rotated radial boxes, while still exporting a MOT-compatible
axis-aligned ``gt.txt`` for TrackEval.

Inputs expected from ``export_gt_viewer_sequence.py``:

    viewer_sequence/
      img1/000001.jpg
      result.txt
      rotated_boxes.jsonl

Outputs:

    gt/gt_work_rotated.json   editable working file
    gt/gt_rotated.jsonl       rotated GT sidecar
    gt/gt.txt                 MOT axis-aligned bounding boxes
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from tkinter import messagebox, simpledialog
import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class RotatedGTBox:
    gt_id: int
    cx: float
    cy: float
    w: float
    h: float
    angle: float

    def corners(self) -> np.ndarray:
        rect = ((float(self.cx), float(self.cy)), (float(self.w), float(self.h)), float(self.angle))
        return cv2.boxPoints(rect).astype(np.float32)

    def axis_tlwh(self) -> Tuple[float, float, float, float]:
        pts = self.corners()
        x1, y1 = pts[:, 0].min(), pts[:, 1].min()
        x2, y2 = pts[:, 0].max(), pts[:, 1].max()
        return float(x1), float(y1), float(x2 - x1), float(y2 - y1)


def _frame_from_name(path: Path) -> int:
    return int(path.stem)


def _load_images(sequence_dir: Path) -> Dict[int, Path]:
    img_dir = sequence_dir / "img1"
    if not img_dir.exists():
        raise FileNotFoundError(f"img1 directory not found: {img_dir}")
    images = {
        _frame_from_name(path): path
        for path in img_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    }
    if not images:
        raise FileNotFoundError(f"No image files found in: {img_dir}")
    return dict(sorted(images.items()))


def _load_rotated_sidecar(path: Path) -> Dict[int, List[RotatedGTBox]]:
    boxes_by_frame: Dict[int, List[RotatedGTBox]] = {}
    if not path.exists():
        return boxes_by_frame
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            frame = int(row["frame"])
            track_id = int(row["track_id"])
            rotated = row.get("rotated_box") or {}
            if rotated.get("center_x") is None:
                x, y, w, h = row["axis_aligned_tlwh"]
                cx, cy, angle = x + w / 2.0, y + h / 2.0, 0.0
            else:
                cx = float(rotated["center_x"])
                cy = float(rotated["center_y"])
                w = float(rotated["width"])
                h = float(rotated["height"])
                angle = float(rotated["angle"])
            boxes_by_frame.setdefault(frame, []).append(
                RotatedGTBox(track_id, cx, cy, float(w), float(h), angle)
            )
    return boxes_by_frame


def _load_mot_gt(path: Path) -> Dict[int, List[RotatedGTBox]]:
    boxes_by_frame: Dict[int, List[RotatedGTBox]] = {}
    if not path.exists():
        return boxes_by_frame
    with path.open(newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 6:
                continue
            frame = int(float(row[0]))
            gt_id = int(float(row[1]))
            x, y, w, h = map(float, row[2:6])
            if gt_id <= 0 or w <= 0 or h <= 0:
                continue
            boxes_by_frame.setdefault(frame, []).append(
                RotatedGTBox(
                    gt_id=gt_id,
                    cx=x + w / 2.0,
                    cy=y + h / 2.0,
                    w=w,
                    h=h,
                    angle=0.0,
                )
            )
    return boxes_by_frame


def _load_work(path: Path) -> Dict[int, List[RotatedGTBox]]:
    if not path.exists():
        return {}
    with path.open() as f:
        data = json.load(f)
    frames = data.get("frames", {})
    boxes_by_frame: Dict[int, List[RotatedGTBox]] = {}
    for frame, boxes in frames.items():
        boxes_by_frame[int(frame)] = [
            RotatedGTBox(
                gt_id=int(box["gt_id"]),
                cx=float(box["cx"]),
                cy=float(box["cy"]),
                w=float(box["w"]),
                h=float(box["h"]),
                angle=float(box["angle"]),
            )
            for box in boxes
        ]
    return boxes_by_frame


def _save_work(path: Path, boxes_by_frame: Dict[int, List[RotatedGTBox]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "format": "rotated_gt_work_v1",
        "frames": {
            str(frame): [asdict(box) for box in boxes]
            for frame, boxes in sorted(boxes_by_frame.items())
            if boxes
        },
    }
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def _export_gt_txt(path: Path, boxes_by_frame: Dict[int, List[RotatedGTBox]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for frame, boxes in sorted(boxes_by_frame.items()):
            for box in boxes:
                if box.gt_id <= 0 or box.w <= 0 or box.h <= 0:
                    continue
                x, y, w, h = box.axis_tlwh()
                f.write(f"{frame},{box.gt_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,1,1\n")


def _export_rotated_jsonl(path: Path, boxes_by_frame: Dict[int, List[RotatedGTBox]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for frame, boxes in sorted(boxes_by_frame.items()):
            for box in boxes:
                f.write(
                    json.dumps(
                        {
                            "frame": frame,
                            "gt_id": box.gt_id,
                            "center_x": box.cx,
                            "center_y": box.cy,
                            "width": box.w,
                            "height": box.h,
                            "angle": box.angle,
                            "corners": box.corners().tolist(),
                            "axis_tlwh": list(box.axis_tlwh()),
                        }
                    )
                    + "\n"
                )


def _color_for_id(gt_id: int) -> Tuple[int, int, int]:
    # BGR color for OpenCV.
    rng = np.random.default_rng(int(gt_id) * 9973)
    return tuple(int(x) for x in rng.integers(40, 255, size=3))


class RotatedGTViewer:
    def __init__(
        self,
        root: tk.Tk,
        sequence_dir: Path,
        work_file: Path,
        gt_output_file: Path,
        rotated_output_file: Path,
        seed_file: Optional[Path],
        seed_gt_file: Optional[Path],
        prefer_gt_seed: bool,
    ) -> None:
        self.root = root
        self.sequence_dir = sequence_dir
        self.images = _load_images(sequence_dir)
        self.frames = sorted(self.images)
        self.frame_pos = 0
        self.work_file = work_file
        self.gt_output_file = gt_output_file
        self.rotated_output_file = rotated_output_file

        work = {} if prefer_gt_seed else _load_work(work_file)
        gt_seed = _load_mot_gt(seed_gt_file) if seed_gt_file else {}
        if work:
            self.boxes_by_frame = work
        elif gt_seed:
            self.boxes_by_frame = gt_seed
        else:
            self.boxes_by_frame = _load_rotated_sidecar(seed_file) if seed_file else {}

        self.selected_idx: Optional[int] = None
        self.drag_start: Optional[Tuple[int, int, float, float, float, float]] = None
        self.drag_mode: Optional[str] = None
        self.drag_corner_idx: Optional[int] = None
        self.drag_fixed_corner: Optional[Tuple[float, float]] = None
        self.drag_axis: Optional[Tuple[float, float, float, float]] = None
        self.scale = 1.0
        self.image_offset = (0, 0)
        self.show_labels = True
        self._resize_after_id = None
        self._slider_updating = False
        self.undo_stack: List[Tuple[Dict[int, List[RotatedGTBox]], int, Optional[int]]] = []
        self.max_undo = 100

        self.root.title("Rotated GT Editor")
        self.root.geometry("1800x1050")
        try:
            self.root.state("zoomed")
        except tk.TclError:
            pass
        self._build_ui()
        self._bind_events()
        self.draw()

    @property
    def frame(self) -> int:
        return self.frames[self.frame_pos]

    @property
    def boxes(self) -> List[RotatedGTBox]:
        return self.boxes_by_frame.setdefault(self.frame, [])

    def _build_ui(self) -> None:
        default_font = ("Sans", 16)
        mono_font = ("Monospace", 15)
        style = ttk.Style(self.root)
        style.configure("TButton", font=default_font, padding=(12, 8))
        style.configure("TLabel", font=default_font)
        style.configure("TEntry", font=default_font)

        top = ttk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(top, text="<< Prev", command=self.prev_frame).pack(side=tk.LEFT)
        ttk.Button(top, text="Next >>", command=self.next_frame).pack(side=tk.LEFT)
        ttk.Button(top, text="-10", command=lambda: self.step_frame(-10)).pack(side=tk.LEFT)
        ttk.Button(top, text="+10", command=lambda: self.step_frame(10)).pack(side=tk.LEFT)
        ttk.Label(top, text="Frame").pack(side=tk.LEFT, padx=(12, 2))
        self.frame_entry = ttk.Entry(top, width=8)
        self.frame_entry.pack(side=tk.LEFT)
        ttk.Button(top, text="Go", command=self.jump_to_entry).pack(side=tk.LEFT)
        ttk.Button(top, text="Add (g)", command=self.add_box).pack(side=tk.LEFT)
        ttk.Button(top, text="Set ID (i)", command=self.set_id).pack(side=tk.LEFT)
        ttk.Button(top, text="Delete", command=self.delete_box).pack(side=tk.LEFT)
        ttk.Button(top, text="Rename ID (r)", command=self.open_rename_dialog).pack(side=tk.LEFT)
        ttk.Button(top, text="Delete ID (b)", command=self.open_delete_id_dialog).pack(side=tk.LEFT)
        ttk.Button(top, text="Save (s)", command=self.save).pack(side=tk.LEFT)
        ttk.Button(top, text="Export (e)", command=self.export).pack(side=tk.LEFT)
        ttk.Button(top, text="Labels (l)", command=self.toggle_labels).pack(side=tk.LEFT)
        self.status = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.status).pack(side=tk.LEFT, padx=12)

        edit = ttk.Frame(self.root)
        edit.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(4, 4))
        ttk.Label(edit, text="Edit selected").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(edit, text="Rot -5", command=lambda: self.rotate_selected(-5.0)).pack(side=tk.LEFT)
        ttk.Button(edit, text="Rot +5", command=lambda: self.rotate_selected(5.0)).pack(side=tk.LEFT)
        ttk.Button(edit, text="Rot -1", command=lambda: self.rotate_selected(-1.0)).pack(side=tk.LEFT)
        ttk.Button(edit, text="Rot +1", command=lambda: self.rotate_selected(1.0)).pack(side=tk.LEFT)
        ttk.Button(edit, text="W -", command=lambda: self.resize_selected(dw=-4.0)).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Button(edit, text="W +", command=lambda: self.resize_selected(dw=4.0)).pack(side=tk.LEFT)
        ttk.Button(edit, text="H -", command=lambda: self.resize_selected(dh=-4.0)).pack(side=tk.LEFT)
        ttk.Button(edit, text="H +", command=lambda: self.resize_selected(dh=4.0)).pack(side=tk.LEFT)
        ttk.Button(edit, text="Scale -", command=lambda: self.scale_selected(0.95)).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Button(edit, text="Scale +", command=lambda: self.scale_selected(1.05)).pack(side=tk.LEFT)
        self.selected_var = tk.StringVar(value="No selected box")
        ttk.Label(edit, textvariable=self.selected_var).pack(side=tk.LEFT, padx=16)

        slider_frame = ttk.Frame(self.root)
        slider_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(4, 6))
        ttk.Label(slider_frame, text="Timeline").pack(side=tk.LEFT, padx=(0, 8))
        self.frame_slider = ttk.Scale(
            slider_frame,
            from_=0,
            to=max(0, len(self.frames) - 1),
            orient=tk.HORIZONTAL,
            command=self.on_slider_changed,
        )
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        main = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        canvas_frame = ttk.Frame(main)
        self.canvas = tk.Canvas(canvas_frame, bg="black", highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        main.add(canvas_frame, weight=5)

        info_frame = ttk.Frame(main)
        self.info_text = tk.Text(info_frame, height=10, wrap=tk.NONE, font=mono_font)
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        yscroll = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_text.configure(yscrollcommand=yscroll.set)
        main.add(info_frame, weight=1)

    def _bind_events(self) -> None:
        self.root.bind_all("<Left>", lambda _e: self.prev_frame())
        self.root.bind_all("<Right>", lambda _e: self.next_frame())
        self.root.bind_all("g", lambda _e: self.add_box())
        self.root.bind_all("i", lambda _e: self.set_id())
        self.root.bind_all("s", lambda _e: self.save())
        self.root.bind_all("e", lambda _e: self.export())
        self.root.bind_all("r", lambda _e: self.open_rename_dialog())
        self.root.bind_all("b", lambda _e: self.open_delete_id_dialog())
        self.root.bind_all("<Control-z>", self.undo)
        self.root.bind_all("<Control-Z>", self.undo)
        self.root.bind_all("l", lambda _e: self.toggle_labels())
        self.root.bind_all("<Return>", lambda _e: self.jump_to_entry())
        self.root.bind_all("<Delete>", lambda _e: self.delete_box())
        self.root.bind_all("q", lambda _e: self.rotate_selected(-5.0))
        self.root.bind_all("w", lambda _e: self.rotate_selected(5.0))
        self.root.bind_all("a", lambda _e: self.rotate_selected(-1.0))
        self.root.bind_all("d", lambda _e: self.rotate_selected(1.0))
        self.root.bind_all("z", lambda _e: self.scale_selected(0.95))
        self.root.bind_all("x", lambda _e: self.scale_selected(1.05))
        self.root.bind_all("<Shift-Left>", lambda _e: self.resize_selected(dw=-4.0))
        self.root.bind_all("<Shift-Right>", lambda _e: self.resize_selected(dw=4.0))
        self.root.bind_all("<Shift-Up>", lambda _e: self.resize_selected(dh=-4.0))
        self.root.bind_all("<Shift-Down>", lambda _e: self.resize_selected(dh=4.0))
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

    def _push_undo(self) -> None:
        snapshot = copy.deepcopy(self.boxes_by_frame)
        self.undo_stack.append((snapshot, self.frame_pos, self.selected_idx))
        if len(self.undo_stack) > self.max_undo:
            self.undo_stack.pop(0)

    def undo(self, _event=None):
        if not self.undo_stack:
            self.status.set("Nothing to undo.")
            return "break"
        self.boxes_by_frame, self.frame_pos, self.selected_idx = self.undo_stack.pop()
        self.drag_start = None
        self.drag_mode = None
        self.drag_corner_idx = None
        self.drag_fixed_corner = None
        self.drag_axis = None
        self.status.set("Undid last GT edit.")
        self.draw()
        return "break"

    def on_canvas_resize(self, _event) -> None:
        if self._resize_after_id is not None:
            self.root.after_cancel(self._resize_after_id)
        self._resize_after_id = self.root.after(100, self.draw)

    def on_slider_changed(self, value) -> None:
        if self._slider_updating:
            return
        pos = int(round(float(value)))
        pos = max(0, min(len(self.frames) - 1, pos))
        if pos == self.frame_pos:
            return
        self.frame_pos = pos
        self.selected_idx = None
        self.draw()

    def _load_frame_image(self) -> np.ndarray:
        image = cv2.imread(str(self.images[self.frame]), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read frame image: {self.images[self.frame]}")
        return image

    def _screen_to_image(self, x: int, y: int) -> Tuple[float, float]:
        ox, oy = self.image_offset
        return (x - ox) / self.scale, (y - oy) / self.scale

    @staticmethod
    def _rotated_local_coords(box: RotatedGTBox, x: float, y: float) -> Tuple[float, float]:
        theta = math.radians(box.angle)
        dx, dy = x - box.cx, y - box.cy
        local_x = math.cos(theta) * dx + math.sin(theta) * dy
        local_y = -math.sin(theta) * dx + math.cos(theta) * dy
        return local_x, local_y

    @staticmethod
    def _rotated_world_coords(box: RotatedGTBox, local_x: float, local_y: float) -> Tuple[float, float]:
        theta = math.radians(box.angle)
        x = box.cx + math.cos(theta) * local_x - math.sin(theta) * local_y
        y = box.cy + math.sin(theta) * local_x + math.cos(theta) * local_y
        return x, y

    def _update_info_panel(self) -> None:
        lines = [
            f"Frame: {self.frame} ({self.frame_pos + 1}/{len(self.frames)})",
            f"Boxes: {len(self.boxes)}",
            "",
            "Controls:",
            "  Left/Right: previous/next frame    -10/+10 buttons: jump frames",
            "  Click: select nearest box           Drag: move selected box",
            "  q/w: rotate -5/+5 deg               a/d: rotate -1/+1 deg",
            "  Drag corner: resize rectangle        z/x: scale selected box",
            "  Shift+arrows: width/height edit      g: add box",
            "  i: set selected GT ID                Delete: delete selected",
            "  r: rename GT ID by range             b: delete GT ID by range",
            "  s: save work JSON                    e: export gt.txt + gt_rotated.jsonl",
            "",
            "GT boxes:",
        ]
        if not self.boxes:
            lines.append("  none")
        for idx, box in enumerate(self.boxes):
            marker = "*" if idx == self.selected_idx else " "
            x, y, w, h = box.axis_tlwh()
            lines.append(
                f"{marker} idx={idx:02d} GT{box.gt_id:<3d} "
                f"center=({box.cx:.1f},{box.cy:.1f}) size=({box.w:.1f},{box.h:.1f}) "
                f"angle={box.angle:.1f} axis_tlwh=[{x:.1f},{y:.1f},{w:.1f},{h:.1f}]"
            )
        if self.selected_idx is not None and self.selected_idx < len(self.boxes):
            box = self.boxes[self.selected_idx]
            self.selected_var.set(
                f"Selected idx={self.selected_idx} GT{box.gt_id} "
                f"angle={box.angle:.1f} size=({box.w:.1f},{box.h:.1f})"
            )
            lines.extend(
                [
                    "",
                    "Selected:",
                    json.dumps(asdict(box), indent=2),
                ]
            )
        else:
            self.selected_var.set("No selected box")
        self.info_text.configure(state=tk.NORMAL)
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert(tk.END, "\n".join(lines))
        self.info_text.configure(state=tk.DISABLED)

    def draw(self) -> None:
        image = self._load_frame_image()
        h, w = image.shape[:2]
        self.root.update_idletasks()
        canvas_w = max(900, self.canvas.winfo_width())
        canvas_h = max(650, self.canvas.winfo_height())
        self.scale = min(canvas_w / w, canvas_h / h)
        vis = image.copy()

        for idx, box in enumerate(self.boxes):
            color = _color_for_id(box.gt_id)
            corners = box.corners().astype(np.int32)
            thickness = 4 if idx == self.selected_idx else 2
            cv2.polylines(vis, [corners], isClosed=True, color=color, thickness=thickness)
            cv2.circle(vis, (int(round(box.cx)), int(round(box.cy))), 3, color, -1)
            if idx == self.selected_idx:
                for corner in corners:
                    cv2.circle(vis, (int(corner[0]), int(corner[1])), 7, (0, 255, 255), -1)
                    cv2.circle(vis, (int(corner[0]), int(corner[1])), 8, color, 2)
            if self.show_labels:
                label = f"GT{box.gt_id}"
                p = corners[corners[:, 1].argmin()]
                cv2.putText(vis, label, (int(p[0]), int(p[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 3)

        if abs(self.scale - 1.0) > 1e-6:
            interpolation = cv2.INTER_LINEAR if self.scale > 1.0 else cv2.INTER_AREA
            vis = cv2.resize(vis, (int(w * self.scale), int(h * self.scale)), interpolation=interpolation)
        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.config(scrollregion=(0, 0, rgb.shape[1], rgb.shape[0]))
        self.canvas.delete("all")
        x0 = max(0, (self.canvas.winfo_width() - rgb.shape[1]) // 2)
        y0 = max(0, (self.canvas.winfo_height() - rgb.shape[0]) // 2)
        self.canvas.create_image(x0, y0, image=self.photo, anchor=tk.NW)
        self.image_offset = (x0, y0)
        self.frame_entry.delete(0, tk.END)
        self.frame_entry.insert(0, str(self.frame))
        self._slider_updating = True
        self.frame_slider.set(self.frame_pos)
        self._slider_updating = False
        self.status.set(
            f"Frame {self.frame} ({self.frame_pos + 1}/{len(self.frames)})"
        )
        self._update_info_panel()

    def prev_frame(self) -> None:
        if self.frame_pos > 0:
            self.frame_pos -= 1
            self.selected_idx = None
            self.draw()

    def next_frame(self) -> None:
        if self.frame_pos < len(self.frames) - 1:
            self.frame_pos += 1
            self.selected_idx = None
            self.draw()

    def step_frame(self, delta: int) -> None:
        self.frame_pos = max(0, min(len(self.frames) - 1, self.frame_pos + int(delta)))
        self.selected_idx = None
        self.draw()

    def jump_to_entry(self) -> None:
        try:
            target = int(self.frame_entry.get())
        except ValueError:
            return
        if target in self.images:
            self.frame_pos = self.frames.index(target)
        else:
            nearest = min(range(len(self.frames)), key=lambda i: abs(self.frames[i] - target))
            self.frame_pos = nearest
        self.selected_idx = None
        self.draw()

    def toggle_labels(self) -> None:
        self.show_labels = not self.show_labels
        self.draw()

    def _nearest_box_idx(self, x: float, y: float) -> Optional[int]:
        if not self.boxes:
            return None
        distances = [math.hypot(box.cx - x, box.cy - y) for box in self.boxes]
        idx = int(np.argmin(distances))
        return idx if distances[idx] < 100.0 else None

    def _nearest_corner(self, x: float, y: float) -> Tuple[Optional[int], Optional[int]]:
        best = (None, None, float("inf"))
        for box_idx, box in enumerate(self.boxes):
            corners = box.corners()
            for corner_idx, corner in enumerate(corners):
                dist = math.hypot(float(corner[0]) - x, float(corner[1]) - y)
                if dist < best[2]:
                    best = (box_idx, corner_idx, dist)
        if best[2] <= max(12.0 / max(self.scale, 1e-6), 8.0):
            return best[0], best[1]
        return None, None

    @staticmethod
    def _unit_axes(angle: float) -> Tuple[float, float, float, float]:
        theta = math.radians(angle)
        return math.cos(theta), math.sin(theta), -math.sin(theta), math.cos(theta)

    def on_mouse_down(self, event) -> None:
        x, y = self._screen_to_image(event.x, event.y)
        corner_box_idx, corner_idx = self._nearest_corner(x, y)
        idx = corner_box_idx if corner_box_idx is not None else self._nearest_box_idx(x, y)
        self.selected_idx = idx
        if idx is not None:
            self._push_undo()
            box = self.boxes[idx]
            self.drag_mode = "resize" if corner_box_idx is not None else "move"
            self.drag_corner_idx = corner_idx
            self.drag_start = (event.x, event.y, box.cx, box.cy, box.w, box.h)
            self.drag_fixed_corner = None
            self.drag_axis = None
            if self.drag_mode == "resize" and corner_idx is not None:
                corners = box.corners()
                fixed = corners[(corner_idx + 2) % 4]
                self.drag_fixed_corner = (float(fixed[0]), float(fixed[1]))
                self.drag_axis = self._unit_axes(box.angle)
        else:
            self.drag_start = None
            self.drag_mode = None
            self.drag_corner_idx = None
            self.drag_fixed_corner = None
            self.drag_axis = None
        self.draw()

    def on_mouse_drag(self, event) -> None:
        if self.selected_idx is None or self.drag_start is None:
            return
        box = self.boxes[self.selected_idx]
        sx, sy, cx, cy, start_w, start_h = self.drag_start
        if self.drag_mode == "resize" and self.drag_fixed_corner is not None and self.drag_axis is not None:
            x, y = self._screen_to_image(event.x, event.y)
            fx, fy = self.drag_fixed_corner
            ux, uy, vx, vy = self.drag_axis
            dx, dy = x - fx, y - fy
            width_component = dx * ux + dy * uy
            height_component = dx * vx + dy * vy
            width_sign = 1.0 if width_component >= 0.0 else -1.0
            height_sign = 1.0 if height_component >= 0.0 else -1.0
            new_w = max(4.0, abs(width_component))
            new_h = max(4.0, abs(height_component))
            moved_x = fx + width_sign * new_w * ux + height_sign * new_h * vx
            moved_y = fy + width_sign * new_w * uy + height_sign * new_h * vy
            box.cx = (fx + moved_x) / 2.0
            box.cy = (fy + moved_y) / 2.0
            box.w = new_w
            box.h = new_h
        else:
            box.cx = cx + (event.x - sx) / self.scale
            box.cy = cy + (event.y - sy) / self.scale
        self.draw()

    def on_mouse_up(self, _event) -> None:
        self.drag_start = None
        self.drag_mode = None
        self.drag_corner_idx = None
        self.drag_fixed_corner = None
        self.drag_axis = None

    def add_box(self) -> None:
        image = self._load_frame_image()
        h, w = image.shape[:2]
        gt_id = simpledialog.askinteger("GT ID", "GT ID:", minvalue=1)
        if gt_id is None:
            return
        self._push_undo()
        self.boxes.append(RotatedGTBox(gt_id, w / 2.0, h / 2.0, w * 0.08, h * 0.18, 0.0))
        self.selected_idx = len(self.boxes) - 1
        self.draw()

    def set_id(self) -> None:
        if self.selected_idx is None:
            return
        current = self.boxes[self.selected_idx].gt_id
        gt_id = simpledialog.askinteger("Set GT ID", "GT ID:", initialvalue=current, minvalue=1)
        if gt_id is not None:
            self._push_undo()
            self.boxes[self.selected_idx].gt_id = int(gt_id)
            self.draw()

    def delete_box(self) -> None:
        if self.selected_idx is None:
            return
        self._push_undo()
        del self.boxes[self.selected_idx]
        self.selected_idx = None
        self.draw()

    def rotate_selected(self, delta: float) -> None:
        if self.selected_idx is None:
            self.status.set("No selected box. Click a box first.")
            return
        self._push_undo()
        self.boxes[self.selected_idx].angle += float(delta)
        self.draw()

    def scale_selected(self, factor: float) -> None:
        if self.selected_idx is None:
            self.status.set("No selected box. Click a box first.")
            return
        self._push_undo()
        box = self.boxes[self.selected_idx]
        box.w *= float(factor)
        box.h *= float(factor)
        self.draw()

    def resize_selected(self, dw: float = 0.0, dh: float = 0.0) -> None:
        if self.selected_idx is None:
            self.status.set("No selected box. Click a box first.")
            return
        self._push_undo()
        box = self.boxes[self.selected_idx]
        box.w = max(4.0, box.w + float(dw))
        box.h = max(4.0, box.h + float(dh))
        self.draw()

    def _frame_range_from_entries(self, start_entry: ttk.Entry, end_entry: ttk.Entry) -> Tuple[int, int]:
        start = int(start_entry.get())
        end = int(end_entry.get())
        if start > end:
            start, end = end, start
        return start, end

    def _rename_gt_id(self, old_id: int, new_id: int, start_frame: int, end_frame: int) -> int:
        count = 0
        for frame, boxes in self.boxes_by_frame.items():
            if start_frame <= frame <= end_frame:
                for box in boxes:
                    if box.gt_id == old_id:
                        box.gt_id = new_id
                        count += 1
        return count

    @staticmethod
    def _parse_gt_ids(text: str) -> List[int]:
        ids = []
        for part in text.split():
            ids.append(int(part))
        return sorted(set(ids))

    def _delete_gt_ids(self, gt_ids: List[int], start_frame: int, end_frame: int) -> int:
        gt_id_set = set(gt_ids)
        count = 0
        for frame in list(self.boxes_by_frame):
            if not (start_frame <= frame <= end_frame):
                continue
            before = len(self.boxes_by_frame[frame])
            self.boxes_by_frame[frame] = [
                box for box in self.boxes_by_frame[frame] if box.gt_id not in gt_id_set
            ]
            count += before - len(self.boxes_by_frame[frame])
        return count

    def open_rename_dialog(self) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Rename GT ID")
        dialog.transient(self.root)
        dialog.grab_set()

        current_id = self.boxes[self.selected_idx].gt_id if self.selected_idx is not None else 1
        min_frame, max_frame = self.frames[0], self.frames[-1]

        ttk.Label(dialog, text="GT ID to replace").grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(10, 2))
        old_entry = ttk.Entry(dialog, width=14)
        old_entry.insert(0, str(current_id))
        old_entry.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10)

        ttk.Label(dialog, text="Replace with GT ID").grid(row=2, column=0, columnspan=2, sticky="w", padx=10, pady=(10, 2))
        new_entry = ttk.Entry(dialog, width=14)
        new_entry.insert(0, str(current_id))
        new_entry.grid(row=3, column=0, columnspan=2, sticky="ew", padx=10)

        ttk.Label(dialog, text="Start frame").grid(row=4, column=0, sticky="w", padx=10, pady=(10, 2))
        ttk.Label(dialog, text="End frame").grid(row=4, column=1, sticky="w", padx=10, pady=(10, 2))
        start_entry = ttk.Entry(dialog, width=14)
        start_entry.insert(0, str(self.frame))
        start_entry.grid(row=5, column=0, sticky="ew", padx=10)
        end_entry = ttk.Entry(dialog, width=14)
        end_entry.insert(0, str(self.frame))
        end_entry.grid(row=5, column=1, sticky="ew", padx=10)

        def apply_range() -> None:
            try:
                old_id = int(old_entry.get())
                new_id = int(new_entry.get())
                start, end = self._frame_range_from_entries(start_entry, end_entry)
            except ValueError:
                messagebox.showerror("Rename GT ID", "Please enter valid integer values.", parent=dialog)
                return
            self._push_undo()
            count = self._rename_gt_id(old_id, new_id, start, end)
            dialog.destroy()
            self.selected_idx = None
            self.status.set(f"Renamed {count} boxes: GT{old_id} -> GT{new_id}, frames {start}-{end}")
            self.draw()

        def replace_all() -> None:
            start_entry.delete(0, tk.END)
            start_entry.insert(0, str(min_frame))
            end_entry.delete(0, tk.END)
            end_entry.insert(0, str(max_frame))
            apply_range()

        buttons = ttk.Frame(dialog)
        buttons.grid(row=6, column=0, columnspan=2, sticky="ew", padx=10, pady=12)
        ttk.Button(buttons, text="Apply Range", command=apply_range).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Replace All", command=replace_all).pack(side=tk.LEFT, padx=8)
        ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT)
        old_entry.focus_set()

    def open_delete_id_dialog(self) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Delete GT ID")
        dialog.transient(self.root)
        dialog.grab_set()

        current_id = self.boxes[self.selected_idx].gt_id if self.selected_idx is not None else 1
        min_frame, max_frame = self.frames[0], self.frames[-1]

        ttk.Label(dialog, text="GT IDs to delete, separated by spaces").grid(
            row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(10, 2)
        )
        id_entry = ttk.Entry(dialog, width=28)
        id_entry.insert(0, str(current_id))
        id_entry.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10)

        ttk.Label(dialog, text="Start frame").grid(row=2, column=0, sticky="w", padx=10, pady=(10, 2))
        ttk.Label(dialog, text="End frame").grid(row=2, column=1, sticky="w", padx=10, pady=(10, 2))
        start_entry = ttk.Entry(dialog, width=14)
        start_entry.insert(0, str(self.frame))
        start_entry.grid(row=3, column=0, sticky="ew", padx=10)
        end_entry = ttk.Entry(dialog, width=14)
        end_entry.insert(0, str(self.frame))
        end_entry.grid(row=3, column=1, sticky="ew", padx=10)

        def apply_range() -> None:
            try:
                gt_ids = self._parse_gt_ids(id_entry.get())
                start, end = self._frame_range_from_entries(start_entry, end_entry)
            except ValueError:
                messagebox.showerror("Delete GT ID", "Please enter valid integer GT IDs.", parent=dialog)
                return
            if not gt_ids:
                messagebox.showerror("Delete GT ID", "Please enter at least one GT ID.", parent=dialog)
                return
            self._push_undo()
            count = self._delete_gt_ids(gt_ids, start, end)
            dialog.destroy()
            self.selected_idx = None
            gt_text = " ".join(f"GT{gt_id}" for gt_id in gt_ids)
            self.status.set(f"Deleted {count} boxes: {gt_text}, frames {start}-{end}")
            self.draw()

        def delete_all() -> None:
            start_entry.delete(0, tk.END)
            start_entry.insert(0, str(min_frame))
            end_entry.delete(0, tk.END)
            end_entry.insert(0, str(max_frame))
            apply_range()

        buttons = ttk.Frame(dialog)
        buttons.grid(row=4, column=0, columnspan=2, sticky="ew", padx=10, pady=12)
        ttk.Button(buttons, text="Apply Range", command=apply_range).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Delete All", command=delete_all).pack(side=tk.LEFT, padx=8)
        ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT)
        id_entry.focus_set()

    def save(self) -> None:
        _save_work(self.work_file, self.boxes_by_frame)
        self.status.set(f"Saved: {self.work_file}")

    def export(self) -> None:
        self.save()
        _export_gt_txt(self.gt_output_file, self.boxes_by_frame)
        _export_rotated_jsonl(self.rotated_output_file, self.boxes_by_frame)
        self.status.set(f"Exported: {self.gt_output_file} and {self.rotated_output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Edit rotated GT boxes for gnomonic viewer sequences.")
    parser.add_argument("--sequence-dir", required=True, help="MOT-like viewer_sequence directory.")
    parser.add_argument("--work-file", default=None, help="Defaults to <sequence-dir>/gt/gt_work_rotated.json")
    parser.add_argument("--gt-output-file", default=None, help="Defaults to <sequence-dir>/gt/gt.txt")
    parser.add_argument("--rotated-output-file", default=None, help="Defaults to <sequence-dir>/gt/gt_rotated.jsonl")
    parser.add_argument("--seed-rotated-file", default=None, help="Defaults to <sequence-dir>/rotated_boxes.jsonl")
    parser.add_argument("--seed-gt-file", default=None, help="Defaults to <sequence-dir>/gt/gt.txt")
    parser.add_argument("--prefer-gt-seed", action="store_true", help="Load seed GT txt even if gt_work_rotated.json exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequence_dir = Path(args.sequence_dir)
    work_file = Path(args.work_file) if args.work_file else sequence_dir / "gt" / "gt_work_rotated.json"
    gt_output_file = Path(args.gt_output_file) if args.gt_output_file else sequence_dir / "gt" / "gt.txt"
    rotated_output_file = (
        Path(args.rotated_output_file) if args.rotated_output_file else sequence_dir / "gt" / "gt_rotated.jsonl"
    )
    seed_file = Path(args.seed_rotated_file) if args.seed_rotated_file else sequence_dir / "rotated_boxes.jsonl"
    seed_gt_file = Path(args.seed_gt_file) if args.seed_gt_file else sequence_dir / "gt" / "gt.txt"

    root = tk.Tk()
    RotatedGTViewer(
        root,
        sequence_dir,
        work_file,
        gt_output_file,
        rotated_output_file,
        seed_file,
        seed_gt_file,
        args.prefer_gt_seed,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
