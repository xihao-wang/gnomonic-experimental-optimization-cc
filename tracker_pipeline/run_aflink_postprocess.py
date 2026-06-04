from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    root = _project_root()
    parser = argparse.ArgumentParser(description="Apply AFLink post-processing to MOT result.txt files.")
    parser.add_argument("--input-root", type=Path, default=root / "tracker_pipeline/results/dedup/new_model_no_veto")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--aflink-code", type=Path, default=Path("/home/wang/桌面/suivi-changement-apparence"))
    parser.add_argument("--weights", type=Path, default=root / "tracker_pipeline/data/aflink/AFLink_WEPTDOF_default.pth")
    parser.add_argument("--thr-t-min", type=int, default=0)
    parser.add_argument("--thr-t-max", type=int, default=80)
    parser.add_argument("--thr-s", type=int, default=250)
    parser.add_argument("--thr-p", type=float, default=0.20)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = _project_root()
    input_root = args.input_root if args.input_root.is_absolute() else project / args.input_root
    output_root = args.output_root if args.output_root.is_absolute() else project / args.output_root
    weights = args.weights if args.weights.is_absolute() else project / args.weights

    sys.path.insert(0, str(args.aflink_code))
    from AFLink.AppFreeLink import AFLink, LinkData, PostLinker

    output_root.mkdir(parents=True, exist_ok=True)
    src_files = sorted(input_root.glob("*/*/result.txt"))
    print(f"found {len(src_files)} result files")
    print(f"thrT=({args.thr_t_min}, {args.thr_t_max}) thrS={args.thr_s} thrP={args.thr_p}")

    model = PostLinker()
    state = torch.load(weights, map_location="cpu")
    model.load_state_dict(state)
    if args.device == "cuda" and torch.cuda.is_available():
        model.cuda()
    dataset = LinkData("", "")

    for src in src_files:
        seq = src.parents[1].name
        run_name = src.parent.name
        out_dir = output_root / seq / run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / "result.txt"
        shutil.copy2(src, dst)

        linker = AFLink(
            path_in=str(dst),
            path_out=str(dst),
            model=model,
            dataset=dataset,
            thrT=(args.thr_t_min, args.thr_t_max),
            thrS=args.thr_s,
            thrP=args.thr_p,
        )
        linker.link()
        print(dst)


if __name__ == "__main__":
    main()
