from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_WEPDTOF_SEQS = [
    "convenience_store",
    "empty_store",
    "exhibition",
    "it_office",
    "large_office",
    "large_office_2",
    "warehouse",
]


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TrackEval on MOT-format tracking results.")
    parser.add_argument("--tracker-root", type=Path, required=True, help="Root containing <seq>/*/result.txt or <seq>/result.txt.")
    parser.add_argument("--exp-name", required=True, help="Tracker name shown by TrackEval.")
    parser.add_argument(
        "--work-dir",
        "--tmp",
        dest="work_dir",
        type=Path,
        default=Path("tracker_pipeline/trackeval_runs/default"),
        help="Working/output directory used to assemble TrackEval GT and tracker files.",
    )
    parser.add_argument("--dataset", default="wepdtof", choices=["wepdtof", "mot"], help="GT preparation mode.")
    parser.add_argument("--benchmark", default="WEPDTOF")
    parser.add_argument("--split", default="test")
    parser.add_argument("--class-name", default="pedestrian")
    parser.add_argument("--sequences", nargs="+", default=DEFAULT_WEPDTOF_SEQS)
    parser.add_argument("--trackeval-root", type=Path, default=Path("/home/wang/桌面/TrackEval"))
    parser.add_argument("--wepdtof-root", type=Path, default=Path("/home/wang/下载/WEPDTOF/WEPDTOF"))
    parser.add_argument("--gt-folder", type=Path, default=None, help="Existing TrackEval GT folder for dataset=mot.")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    project = _project_root()
    return path if path.is_absolute() else project / path


def _find_result(tracker_root: Path, seq: str) -> Path:
    direct = tracker_root / seq / "result.txt"
    if direct.exists():
        return direct
    matches = sorted((tracker_root / seq).glob("*/result.txt"))
    if matches:
        return matches[-1]
    raise FileNotFoundError(f"No result.txt found for {seq} under {tracker_root}")


def _prepare_wepdtof_gt(args: argparse.Namespace, gt_root: Path) -> None:
    project = _project_root()
    for seq in args.sequences:
        subprocess.run(
            [
                sys.executable,
                str(project / "tracker_pipeline/wepdtof_gt_to_viewer.py"),
                "--frames-dir",
                str(args.wepdtof_root / "frames" / seq),
                "--annotation-json",
                str(args.wepdtof_root / "annotations" / f"{seq}.json"),
                "--output-dir",
                str(gt_root / f"{args.benchmark}-{args.split}" / seq),
                "--image-mode",
                "symlink",
            ],
            check=True,
        )


def main() -> None:
    args = parse_args()
    tracker_root = _resolve(args.tracker_root)

    work_dir = _resolve(args.work_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    gt_root = work_dir / "gt"
    trackers_root = work_dir / "trackers"
    seqmap_dir = gt_root / "seqmaps"
    tracker_data = trackers_root / f"{args.benchmark}-{args.split}" / args.exp_name / "data"
    seqmap_dir.mkdir(parents=True, exist_ok=True)
    tracker_data.mkdir(parents=True, exist_ok=True)

    if args.dataset == "wepdtof":
        _prepare_wepdtof_gt(args, gt_root)
    else:
        if args.gt_folder is None:
            raise ValueError("--gt-folder is required for --dataset mot")
        shutil.copytree(args.gt_folder, gt_root, dirs_exist_ok=True)

    for seq in args.sequences:
        shutil.copy2(_find_result(tracker_root, seq), tracker_data / f"{seq}.txt")

    seqmap = seqmap_dir / f"{args.benchmark}-{args.split}.txt"
    seqmap.write_text("name\n" + "\n".join(args.sequences) + "\n")

    cmd = [
        sys.executable,
        str(args.trackeval_root / "scripts/run_mot_challenge.py"),
        "--GT_FOLDER",
        str(gt_root),
        "--TRACKERS_FOLDER",
        str(trackers_root),
        "--BENCHMARK",
        args.benchmark,
        "--SPLIT_TO_EVAL",
        args.split,
        "--TRACKERS_TO_EVAL",
        args.exp_name,
        "--CLASSES_TO_EVAL",
        args.class_name,
        "--METRICS",
        "HOTA",
        "CLEAR",
        "Identity",
        "Count",
        "--USE_PARALLEL",
        "False",
        "--PRINT_RESULTS",
        "True",
        "--PRINT_ONLY_COMBINED",
        "False",
        "--OUTPUT_SUMMARY",
        "True",
        "--OUTPUT_DETAILED",
        "True",
        "--PLOT_CURVES",
        "False",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.trackeval_root)
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
