#!/usr/bin/env python3
"""Optuna search for AFLink post-processing parameters on fixed tracker outputs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable


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
    parser = argparse.ArgumentParser(description="Tune AFLink parameters with Optuna.")
    parser.add_argument("--input-root", type=Path, default=root / "tracker_pipeline/results/dedup/new_model_no_veto")
    parser.add_argument("--output-root", type=Path, default=root / "tracker_pipeline/results/optuna_aflink")
    parser.add_argument("--work-root", type=Path, default=root / "tracker_pipeline/optuna_runs/aflink")
    parser.add_argument("--study-name", default="aflink_model_appris_idf1")
    parser.add_argument("--storage", default=None, help="Optuna storage URL. Defaults to sqlite:///<work-root>/<study-name>.db")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--sequences", nargs="+", default=DEFAULT_SEQS)
    parser.add_argument("--metric", choices=["IDF1", "HOTA", "AssA", "MOTA", "custom"], default="IDF1")
    parser.add_argument("--aflink-code", type=Path, default=Path("/home/wang/桌面/suivi-changement-apparence"))
    parser.add_argument("--weights", type=Path, default=root / "tracker_pipeline/data/aflink/AFLink_WEPTDOF_default.pth")
    parser.add_argument("--trackeval-root", type=Path, default=Path("/home/wang/桌面/TrackEval"))
    parser.add_argument("--wepdtof-root", type=Path, default=Path("/home/wang/下载/WEPDTOF/WEPDTOF"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuda-visible-devices", default="0,1,2,3")
    parser.add_argument("--thr-t-max-low", type=int, default=20)
    parser.add_argument("--thr-t-max-high", type=int, default=150)
    parser.add_argument("--thr-s-low", type=int, default=50)
    parser.add_argument("--thr-s-high", type=int, default=600)
    parser.add_argument("--thr-p-low", type=float, default=0.02)
    parser.add_argument("--thr-p-high", type=float, default=0.50)
    parser.add_argument("--keep-all-trials", action="store_true", help="Keep every trial output. By default only summaries and best dirs are kept.")
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
    if len(values) != len(header):
        raise ValueError(f"Summary column mismatch: {summary_path}")
    return {key: float(value) for key, value in zip(header, values)}


def _score(metrics: Dict[str, float], metric_name: str) -> float:
    if metric_name != "custom":
        return float(metrics[metric_name])
    return (
        float(metrics["HOTA"])
        + 0.5 * float(metrics["IDF1"])
        + 0.2 * float(metrics["AssA"])
        - 0.02 * float(metrics["IDSW"])
    )


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
    input_root = _resolve(args.input_root)
    output_root = _resolve(args.output_root)
    work_root = _resolve(args.work_root)
    weights = _resolve(args.weights)

    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    storage = args.storage or f"sqlite:///{work_root / (args.study_name + '.db')}"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    def objective(trial) -> float:
        thr_t_max = trial.suggest_int("thr_t_max", args.thr_t_max_low, args.thr_t_max_high)
        thr_s = trial.suggest_int("thr_s", args.thr_s_low, args.thr_s_high)
        thr_p = trial.suggest_float("thr_p", args.thr_p_low, args.thr_p_high)

        exp_name = f"trial_{trial.number:04d}_t{thr_t_max}_s{thr_s}_p{thr_p:.3f}".replace(".", "p")
        trial_out = output_root / exp_name
        trial_work = work_root / exp_name

        if trial_out.exists():
            shutil.rmtree(trial_out)
        if trial_work.exists():
            shutil.rmtree(trial_work)

        _run(
            [
                sys.executable,
                str(project / "tracker_pipeline/run_aflink_postprocess.py"),
                "--input-root",
                str(input_root),
                "--output-root",
                str(trial_out),
                "--aflink-code",
                str(args.aflink_code),
                "--weights",
                str(weights),
                "--thr-t-max",
                str(thr_t_max),
                "--thr-s",
                str(thr_s),
                "--thr-p",
                str(thr_p),
                "--device",
                args.device,
            ],
            env,
            work_root / "logs" / f"{exp_name}_aflink.log",
        )

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
        for key in ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW", "FP", "FN"):
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

        if not args.keep_all_trials and trial.number > 0:
            best_number = trial.study.best_trial.number
            if trial.number != best_number:
                shutil.rmtree(trial_out, ignore_errors=True)
                shutil.rmtree(trial_work, ignore_errors=True)

        print(
            f"trial={trial.number} value={value:.3f} "
            f"HOTA={metrics.get('HOTA', -1):.3f} IDF1={metrics.get('IDF1', -1):.3f} "
            f"AssA={metrics.get('AssA', -1):.3f} IDSW={metrics.get('IDSW', -1):.0f} "
            f"params={trial.params}"
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
