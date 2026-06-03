#!/usr/bin/env python3
"""Summarize learned-temporal vs Kalman-gated matching debug JSONL."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument(
        "--stage",
        default="appearance_metric_cascade",
        help="Metric stage to summarize. Use 'all' to include pre-ambiguity calls too.",
    )
    parser.add_argument(
        "--high-confidence-cost",
        type=float,
        default=0.10,
        help="Temporal cost threshold for high-confidence learned matches. cost < threshold means prob > 1-threshold.",
    )
    parser.add_argument("--top-frames", type=int, default=20)
    return parser.parse_args()


def pct(value: int, total: int) -> str:
    if total <= 0:
        return "0.00%"
    return f"{100.0 * value / total:.2f}%"


def main() -> None:
    args = parse_args()
    if not args.jsonl.exists():
        raise FileNotFoundError(args.jsonl)

    total_rows = 0
    learned_best_gated = 0
    learned_best_over_gate = 0
    learned_best_not_final_best = 0
    learned_best_not_assignment_cost = 0
    learned_best_gated_and_not_final = 0
    learned_best_gated_and_not_assignment_cost = 0
    records_with_temporal = 0
    assignment_records = 0
    assigned_pairs = {}
    gated_by_frame = Counter()
    mismatch_by_frame = Counter()
    examples = []
    high_conf_total = 0
    high_conf_gated = 0
    high_conf_gated_and_not_assignment = 0

    with args.jsonl.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    for record in records:
        if record.get("stage") == "appearance_assignments":
            assignment_records += 1
            frame = record.get("frame")
            for item in record.get("assignments", []):
                assigned_pairs[(frame, item["track_id"])] = item["detection_index"]

    for record in records:
        if args.stage != "all" and record.get("stage") != args.stage:
            continue
        if args.stage == "all" and not str(record.get("stage", "")).startswith("appearance_metric"):
            continue
        if record.get("temporal_cost") is None:
            continue
        records_with_temporal += 1
        frame = record.get("frame")
        for row in record.get("row_summary", []):
            if row.get("learned_best_detection_index") is None:
                continue
            total_rows += 1
            track_id = row["track_id"]
            learned_det = row["learned_best_detection_index"]
            final_det = row.get("final_best_detection_index")
            assigned_cost_det = row.get("assigned_by_final_cost_detection_index")
            actual_assigned_det = assigned_pairs.get((frame, track_id), assigned_cost_det)
            gated = bool(row.get("learned_best_gated_to_infty"))
            over_gate = bool(row.get("learned_best_over_gate_threshold"))
            not_final = final_det is not None and learned_det != final_det
            not_assignment = actual_assigned_det is not None and learned_det != actual_assigned_det
            temporal_cost = row.get("learned_best_temporal_cost")
            high_conf = (
                temporal_cost is not None
                and float(temporal_cost) < float(args.high_confidence_cost)
            )

            learned_best_gated += int(gated)
            learned_best_over_gate += int(over_gate)
            learned_best_not_final_best += int(not_final)
            learned_best_not_assignment_cost += int(not_assignment)
            learned_best_gated_and_not_final += int(gated and not_final)
            learned_best_gated_and_not_assignment_cost += int(gated and not_assignment)
            high_conf_total += int(high_conf)
            high_conf_gated += int(high_conf and gated)
            high_conf_gated_and_not_assignment += int(high_conf and gated and not_assignment)
            if gated:
                gated_by_frame[frame] += 1
            if not_assignment:
                mismatch_by_frame[frame] += 1
            if (gated or not_assignment) and len(examples) < 20:
                examples.append({
                    "frame": frame,
                    "track_id": track_id,
                    "learned_det": learned_det,
                    "final_det": final_det,
                    "assigned_det": actual_assigned_det,
                    "gated": gated,
                    "gating_distance": row.get("learned_best_gating_distance"),
                    "temporal_cost": row.get("learned_best_temporal_cost"),
                    "final_cost": row.get("learned_best_final_cost"),
                })

    print(f"file: {args.jsonl}")
    print(f"stage: {args.stage}")
    print(f"appearance metric records with temporal: {records_with_temporal}")
    print(f"appearance assignment records: {assignment_records}")
    print(f"track rows with learned best: {total_rows}")
    print()
    print("Kalman pressure summary")
    print(f"learned best over Kalman threshold: {learned_best_over_gate} / {total_rows} ({pct(learned_best_over_gate, total_rows)})")
    print(f"learned best gated to INFTY_COST: {learned_best_gated} / {total_rows} ({pct(learned_best_gated, total_rows)})")
    print(f"learned best != final-cost best: {learned_best_not_final_best} / {total_rows} ({pct(learned_best_not_final_best, total_rows)})")
    print(f"learned best != actual/final assignment: {learned_best_not_assignment_cost} / {total_rows} ({pct(learned_best_not_assignment_cost, total_rows)})")
    print(f"learned best gated and != final-cost best: {learned_best_gated_and_not_final} / {total_rows} ({pct(learned_best_gated_and_not_final, total_rows)})")
    print(f"learned best gated and != actual/final assignment: {learned_best_gated_and_not_assignment_cost} / {total_rows} ({pct(learned_best_gated_and_not_assignment_cost, total_rows)})")
    print()
    print(f"High-confidence learned best: temporal_cost < {args.high_confidence_cost}")
    print(f"high-confidence learned best rows: {high_conf_total} / {total_rows} ({pct(high_conf_total, total_rows)})")
    print(f"high-confidence learned best gated: {high_conf_gated} / {high_conf_total} ({pct(high_conf_gated, high_conf_total)})")
    print(f"high-confidence learned best gated and != actual/final assignment: {high_conf_gated_and_not_assignment} / {high_conf_total} ({pct(high_conf_gated_and_not_assignment, high_conf_total)})")

    print()
    print("Top frames by gated learned-best count")
    for frame, count in gated_by_frame.most_common(args.top_frames):
        print(f"frame {frame}: {count}")

    print()
    print("Top frames by learned-best assignment mismatch")
    for frame, count in mismatch_by_frame.most_common(args.top_frames):
        print(f"frame {frame}: {count}")

    print()
    print("Examples")
    for item in examples:
        print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    main()
