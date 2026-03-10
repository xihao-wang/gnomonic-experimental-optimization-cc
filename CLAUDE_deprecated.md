# CLAUDE.md - AI Assistant Context & Progress Tracker

**Purpose**: Helps Claude resume work across sessions without re-explanation.
**Last Updated**: 2026-03-03

---

## Essential Files to Read on Session Start

1. **CLAUDE.md** (this file)
2. **evaluation/README.md** — dataset workflow and module overview
3. **evaluation/lib/config.py** — metrics evaluation configuration
4. **datasets/lib/base_manager.py** — dataset manager architecture

---

## Project Overview

**Goal**: Find optimal gnomonic projection configuration for pedestrian detection in fisheye images.

**Pipeline**: fisheye → composite (gnomonic projections) → YOLO detection → backproject to fisheye → Soft-NMS → evaluate vs GT

**Research question**: Which configuration (number of projections, FOV, grid layout) gives best mAP?

---

## Phase Status

| Phase | Status | Notes |
|-------|--------|-------|
| 2 — Detection Pipeline | COMPLETE | fisheye→composite→YOLO→backproject→Soft-NMS; video demo |
| 3 — Dataset Infrastructure | COMPLETE | BOMNI, PIROPO, CEPDOF managers; standard JSON format |
| 4 — Evaluation Framework | COMPLETE | backprojection visual test; versioned sessions |
| 5 — Evaluation Metrics | COMPLETE | mAP, PR curves, timing; BOMNI + PIROPO + CEPDOF |
| 6 — Configuration Search | NOT STARTED | |

---

## Current State

### Dataset Status
- **BOMNI-production**: 245 frames, 834 annotations — evaluation ready
- **PIROPO-production**: 3,004 frames — evaluation ready
- **CEPDOF-production**: 25,358 frames — evaluation ready (use spread sampling)

### Key File Locations
- Metrics evaluation entry point: `evaluation/run_metrics_evaluation.py`
- Metrics config: `evaluation/lib/config.py` → `METRICS_EVALUATION` section
- Projection configs to compare: `evaluation/projection_configs_for_metrics.py`
- Dataset loaders: `evaluation/lib/{bomni,piropo,cepdof}_dataset.py`
- Pipeline entry: `detection_pipeline/pipeline.py`
- YOLO models: `models/` (project root, shared)

### Evaluation Output Structure
```
evaluation/proj-conf-comparison/
└── metrics_eval_session_N/
    ├── metadata.txt
    ├── projection_configs_snapshot.py
    └── {dataset}/
        ├── metrics.json
        ├── timing.json
        ├── summary.txt
        ├── comparison_table.txt
        ├── pr_curves/
        └── visuals/{config_id}/{idx:05d}_{composite|fisheye}.jpg
```

---

## Immediate Next Steps

1. **Run full evaluation on all datasets**
   - Edit `evaluation/lib/config.py`: set `MAX_IMAGES = None` for BOMNI/PIROPO; keep ~3000 for CEPDOF
   - Run: `python evaluation/run_metrics_evaluation.py`
   - Outputs: metrics.json, timing.json, summary.txt, comparison_table.txt, PR curves, visuals

2. **Analyze results**
   - Review `comparison_table.txt` per dataset
   - Compare AP@0.50, Precision, Recall across 3 configurations (2×2, 3×3, Chiang baseline)

3. **Phase 6 — Configuration Search** (after evaluation complete)
   - Files to create: `config_search/searcher.py`, `comparator.py`, `best_config_selector.py`
   - Configurations to test: projections {4,8,9,16}, FOV {60°,90°,106°}, grids {2×2,3×3,2×4,4×2,4×4}
   - Baseline: Chiang et al. (8 projections, 60° FOV)

---

## Key Architecture Decisions

### Standard JSON Annotation Format (universal across all datasets)
```json
{ "center_x": float, "center_y": float, "width": float, "height": float, "angle": float, "class_name": string }
```

### Dataset Manager Pattern
- `datasets/lib/base_manager.py` — abstract base (visualize, validate, statistics)
- Dataset-specific managers in `datasets/lib/` implement `convert_to_standard_format()`
- Two-step prep workflow: step1 (extract + convert) → manual review → step2 (cleanup incorrect)

### Backprojection (5-point geometric)
- Backproject bbox center, left/right midpoints (→ width), top/bottom midpoints (→ height)
- Draw radially-aligned rotated box (height points toward fisheye center)
- `cv2.boxPoints()` convention: width on x-axis, height on y-axis

### Two-Stage NMS
- Stage 1 (composite): standard NMS, IoU 0.8 (keeps overlapping for multi-view)
- Stage 2 (fisheye): Gaussian Soft-NMS on rotated boxes after backprojection
  - Formula: `score ← score × exp(-IoU²/σ)`, σ ∈ {0.1, 0.2, 0.4}

### Spread Sampling
- `step = total / max_images`; `indices = [int(i * step) for i in range(max_images)]`
- Ensures sampled frames distributed evenly across full dataset
- Controlled by `SPREAD_SAMPLES` flag in config; `MAX_IMAGES` applied per dataset independently

### Visual Output
- `ENABLE_VISUALS` flag in `evaluation/lib/config.py`
- `pipeline.run(return_visuals=True)` → `(bboxes, composite_image, raw_detections)`
- Saves: `{idx:05d}_composite.jpg` (YOLO detections, green) + `{idx:05d}_fisheye.jpg` (GT green, predictions yellow)

---

## Rules (Mandatory)

1. **NEVER use argparse/click** — all config in class attributes or `config.py`
2. **All runnable scripts must add project root to sys.path** at top (before imports):
   ```python
   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path(__file__).parent.parent))
   ```
3. **Module folder structure**: entry point scripts at top level only; all library code in `lib/` or `utils/`
4. **ALWAYS update evaluation/README.md** when completing any phase
5. **Delete temporary test scripts** after use
6. **No Unicode symbols** in OpenCV-rendered text — use "deg" not "°"
7. **NEVER mention internal workflow files** (CLAUDE.md, commit messages, etc.) in README or user-facing docs
8. **Commit messages** → written to `cmsg.txt`; user commits manually
9. **Never hardcode user-editable values** — put them in `config.py`
10. **Author identity**: Yassir Zardoua — y.zardoua@caplogy.com | yassirzardoua@gmail.com

---

## Known Issues

- **BOMNI dataset limitations**: single room, constant lighting, 640×480 resolution, repetitive frames, approximative annotations (~1-2% mAP noise)
- **CEPDOF bboxes**: NOT radially aligned (free body orientation) — gives opportunity to compare radial vs free-orientation backprojection
