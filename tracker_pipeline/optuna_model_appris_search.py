#!/usr/bin/env python3
"""Optuna search for learned temporal tracker inference parameters."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULT_SEQS = [
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


def _resolve(path: Path) -> Path:
    project = _project_root()
    return path if path.is_absolute() else project / path


def parse_args() -> argparse.Namespace:
    root = _project_root()
    parser = argparse.ArgumentParser(description="Tune model_appris tracker parameters with Optuna.")
    parser.add_argument("--output-root", type=Path, default=root / "tracker_pipeline/results/optuna_model_appris")
    parser.add_argument("--work-root", type=Path, default=root / "tracker_pipeline/optuna_runs/model_appris")
    parser.add_argument("--study-name", default="model_appris_idf1")
    parser.add_argument("--storage", default=None, help="Optuna storage URL. Defaults to sqlite:///<work-root>/<study-name>.db")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--sequences", nargs="+", default=DEFAULT_SEQS)
    parser.add_argument(
        "--metric",
        choices=["IDF1", "HOTA", "AssA", "MOTA", "custom", "balanced_penalty", "balanced_symmetric"],
        default="IDF1",
    )
    parser.add_argument("--checkpoint", type=Path, default=root / "tracker_pipeline/learnable_model/wepdtof_dedup_hardneg5_jtp_valkindergarten.pt")
    parser.add_argument("--cache-root", type=Path, default=None, help="Optional root containing {sequence}.npz detection/ReID caches.")
    parser.add_argument("--wepdtof-root", type=Path, default=Path("/home/wang/下载/WEPDTOF/WEPDTOF"))
    parser.add_argument("--trackeval-root", type=Path, default=Path("/home/wang/桌面/TrackEval"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fastreid-device", default="cuda")
    parser.add_argument("--temporal-device", default="cuda")
    parser.add_argument("--cuda-visible-devices", default="0,1,2,3")
    parser.add_argument("--keep-all-trials", action="store_true", help="Deprecated compatibility flag; trial outputs are kept by default.")
    parser.add_argument("--render-video", action="store_true", help="Render tracked_fisheye.mp4 during tracker runs.")
    parser.add_argument("--narrow", action="store_true", help="Use narrowed ranges around the best previous Optuna trials.")
    parser.add_argument("--parallel-sequences", type=int, default=1, help="Run up to this many sequences concurrently within each trial.")
    return parser.parse_args()


def _run(cmd: Iterable[str], env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    log_path.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nLog: {log_path}\n{proc.stdout[-4000:]}")


def _run_parallel(jobs: List[Tuple[List[str], dict[str, str], Path]], max_workers: int) -> None:
    """Run subprocess jobs in small batches and keep one log per job."""
    max_workers = max(1, int(max_workers))
    for start in range(0, len(jobs), max_workers):
        batch = jobs[start:start + max_workers]
        running = []
        for cmd, env, log_path in batch:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            running.append((proc, cmd, log_path))

        failures = []
        for proc, cmd, log_path in running:
            stdout, _ = proc.communicate()
            log_path.write_text(stdout, encoding="utf-8")
            if proc.returncode != 0:
                failures.append((cmd, log_path, stdout[-4000:]))

        if failures:
            details = "\n\n".join(
                f"Command failed: {' '.join(cmd)}\nLog: {log_path}\n{tail}"
                for cmd, log_path, tail in failures
            )
            raise RuntimeError(details)


def _find_summary(work_dir: Path) -> Path:
    matches = sorted(work_dir.glob("trackers/**/*pedestrian_summary.txt"))
    if not matches:
        raise FileNotFoundError(f"No pedestrian_summary.txt found under {work_dir}")
    return matches[-1]


def _parse_summary(summary_path: Path) -> Dict[str, float]:
    lines = [line.strip() for line in summary_path.read_text().splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError(f"Invalid TrackEval summary: {summary_path}")
    header = lines[0].split()
    values = lines[1].split()
    if len(header) != len(values):
        raise ValueError(f"Summary column mismatch: {summary_path}")
    return {key: float(value) for key, value in zip(header, values)}


def _score(metrics: Dict[str, float], metric_name: str) -> float:
    if metric_name == "balanced_penalty":
        return (
            (float(metrics["IDF1"]) - 49.929) / 0.5
            + (float(metrics["AssA"]) - 55.614) / 0.5
            - max(float(metrics["IDSW"]) - 22.0, 0.0) / 3.0
        )
    if metric_name == "balanced_symmetric":
        return (
            (float(metrics["IDF1"]) - 49.929) / 0.5
            + (float(metrics["AssA"]) - 55.614) / 0.5
            + (22.0 - float(metrics["IDSW"])) / 3.0
        )
    if metric_name != "custom":
        return float(metrics[metric_name])
    return (
        float(metrics["HOTA"])
        + 0.5 * float(metrics["IDF1"])
        + 0.2 * float(metrics["AssA"])
        - 0.02 * float(metrics["IDSW"])
    )


def _sample_params(trial, narrow: bool) -> dict:
    if narrow:
        return {
            "learned_temporal_alpha": trial.suggest_float("learned_temporal_alpha", 0.40, 0.75),
            "learned_temporal_min_scale": trial.suggest_float("learned_temporal_min_scale", 0.006, 0.014),
            "learned_temporal_max_correction": trial.suggest_float("learned_temporal_max_correction", 0.012, 0.028),
            "matching_threshold": trial.suggest_float("matching_threshold", 0.30, 0.43),
            "max_iou_distance": trial.suggest_float("max_iou_distance", 0.62, 0.72),
            "max_age": trial.suggest_int("max_age", 105, 140),
            "n_init": 5,
        }
    return {
        "learned_temporal_alpha": trial.suggest_float("learned_temporal_alpha", 0.2, 2.0),
        "learned_temporal_min_scale": trial.suggest_float("learned_temporal_min_scale", 0.005, 0.05),
        "learned_temporal_max_correction": trial.suggest_float("learned_temporal_max_correction", 0.005, 0.08),
        "matching_threshold": trial.suggest_float("matching_threshold", 0.10, 0.50),
        "max_iou_distance": trial.suggest_float("max_iou_distance", 0.50, 0.90),
        "max_age": trial.suggest_int("max_age", 30, 150),
        "n_init": trial.suggest_int("n_init", 1, 5),
    }


def _gpu_ids(cuda_visible_devices: str) -> List[str]:
    ids = [item.strip() for item in cuda_visible_devices.split(",") if item.strip()]
    return ids or ["0"]


def _available_cuda_count() -> int:
    try:
        import torch
    except Exception:
        return 0
    try:
        return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:
        return 0


def main() -> None:
    args = parse_args()
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit(
            "Optuna is not installed in this Python environment. Install it first:\n"
            "  /home/wang/miniconda3/envs/gnomonic/bin/python3 -m pip install optuna"
        ) from exc

    project = _project_root()
    output_root = _resolve(args.output_root)
    work_root = _resolve(args.work_root)
    checkpoint = _resolve(args.checkpoint)
    cache_root = _resolve(args.cache_root) if args.cache_root is not None else None
    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    storage = args.storage or f"sqlite:///{work_root / (args.study_name + '.db')}"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    gpu_ids = _gpu_ids(args.cuda_visible_devices)
    available_cuda = _available_cuda_count()
    if available_cuda == 1 and len(gpu_ids) > 1:
        print(
            "Only one CUDA device is visible; sharing GPU 0 across parallel sequence workers "
            f"instead of requested CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}."
        )
        gpu_ids = ["0"]

    def objective(trial) -> float:
        params = _sample_params(trial, args.narrow)
        exp_name = f"trial_{trial.number:04d}"
        trial_out = output_root / exp_name
        trial_work = work_root / exp_name

        if trial_out.exists():
            shutil.rmtree(trial_out)
        if trial_work.exists():
            shutil.rmtree(trial_work)
        trial_out.mkdir(parents=True, exist_ok=True)

        tracker_jobs = []
        for idx, seq in enumerate(args.sequences):
            seq_out = trial_out / seq
            if cache_root is None:
                cmd = [
                    sys.executable,
                    str(project / "tracker_pipeline/process_video_strongsort.py"),
                    "--input",
                    str(args.wepdtof_root / "frames" / seq),
                    "--output-dir",
                    str(seq_out),
                    "--learned-temporal",
                    "--fuse-learned-temporal",
                    "--ltm-stm",
                    "--memory-aware",
                    "--topk",
                    "--tracker-model",
                    str(checkpoint),
                    "--learned-temporal-veto-cost",
                    "-1",
                    "--learned-temporal-alpha",
                    str(params["learned_temporal_alpha"]),
                    "--learned-temporal-min-scale",
                    str(params["learned_temporal_min_scale"]),
                    "--learned-temporal-max-correction",
                    str(params["learned_temporal_max_correction"]),
                    "--matching-threshold",
                    str(params["matching_threshold"]),
                    "--max-iou-distance",
                    str(params["max_iou_distance"]),
                    "--max-age",
                    str(params["max_age"]),
                    "--n-init",
                    str(params["n_init"]),
                    "--device",
                    args.device,
                    "--fastreid-device",
                    args.fastreid_device,
                    "--temporal-device",
                    args.temporal_device,
                ]
                if not args.render_video:
                    cmd.append("--no-video")
                cmd.append("--no-h264-copy")
            else:
                cache_path = cache_root / f"{seq}.npz"
                cmd = [
                    sys.executable,
                    str(project / "tracker_pipeline/run_strongsort_from_cache.py"),
                    "--cache",
                    str(cache_path),
                    "--output-dir",
                    str(seq_out),
                    "--sequence-name",
                    seq,
                    "--learned-temporal",
                    "--fuse-learned-temporal",
                    "--ltm-stm",
                    "--memory-aware",
                    "--topk",
                    "--tracker-model",
                    str(checkpoint),
                    "--learned-temporal-veto-cost",
                    "-1",
                    "--learned-temporal-alpha",
                    str(params["learned_temporal_alpha"]),
                    "--learned-temporal-min-scale",
                    str(params["learned_temporal_min_scale"]),
                    "--learned-temporal-max-correction",
                    str(params["learned_temporal_max_correction"]),
                    "--matching-threshold",
                    str(params["matching_threshold"]),
                    "--max-iou-distance",
                    str(params["max_iou_distance"]),
                    "--max-age",
                    str(params["max_age"]),
                    "--n-init",
                    str(params["n_init"]),
                    "--temporal-device",
                    args.temporal_device,
                ]

            job_env = env.copy()
            if args.parallel_sequences > 1:
                job_env["CUDA_VISIBLE_DEVICES"] = gpu_ids[idx % len(gpu_ids)]
            tracker_jobs.append((cmd, job_env, work_root / "logs" / f"{exp_name}_{seq}_tracker.log"))

        if args.parallel_sequences > 1:
            _run_parallel(tracker_jobs, args.parallel_sequences)
        else:
            for cmd, job_env, log_path in tracker_jobs:
                _run(cmd, job_env, log_path)

        _run(
            [
                sys.executable,
                str(project / "tracker_pipeline/run_trackeval.py"),
                "--tracker-root",
                str(trial_out),
                "--exp-name",
                exp_name,
                "--work-dir",
                str(trial_work),
                "--trackeval-root",
                str(args.trackeval_root),
                "--wepdtof-root",
                str(args.wepdtof_root),
                "--sequences",
                *args.sequences,
            ],
            env,
            work_root / "logs" / f"{exp_name}_trackeval.log",
        )

        metrics = _parse_summary(_find_summary(trial_work))
        value = _score(metrics, args.metric)
        for key in ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW", "FP", "FN", "IDs"):
            if key in metrics:
                trial.set_user_attr(key, metrics[key])
        trial.set_user_attr("output_root", str(trial_out))
        trial.set_user_attr("work_dir", str(trial_work))

        record = {
            "trial": trial.number,
            "value": value,
            "params": dict(trial.params),
            "metrics": metrics,
            "output_root": str(trial_out),
            "work_dir": str(trial_work),
        }
        with (work_root / "trials.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(
            f"trial={trial.number} value={value:.3f} "
            f"HOTA={metrics.get('HOTA', -1):.3f} IDF1={metrics.get('IDF1', -1):.3f} "
            f"AssA={metrics.get('AssA', -1):.3f} MOTA={metrics.get('MOTA', -1):.3f} "
            f"IDSW={metrics.get('IDSW', -1):.0f} params={trial.params}"
        )
        return value

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
    )
    study.optimize(objective, n_trials=args.n_trials)

    best = study.best_trial
    print("=" * 80)
    print(f"best_trial={best.number}")
    print(f"best_value={best.value:.6f}")
    print(f"best_params={best.params}")
    print(f"best_metrics={best.user_attrs}")
    print(f"storage={storage}")


if __name__ == "__main__":
    main()
