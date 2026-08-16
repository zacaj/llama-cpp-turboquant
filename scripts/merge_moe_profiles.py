#!/usr/bin/env python3
"""Merge multiple llama-moe-weights expert-selection profile CSVs into one.

Each input is the CSV written by `llama-moe-weights -o` (or by an older
prompt-traffic profiler): rank,expert,total_count,pct,layer_0,...,layer_N.
Per-expert/per-layer counts are summed across all inputs, rank/pct
recomputed from the merged totals, and the result written in the same
format -- consumable directly by lowest_experts_from_profile.py.

Usage:
  python scripts/merge_moe_profiles.py profile1.csv profile2.csv ... output.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path


def fmt(v: float) -> str:
    """Integral counts print clean (e.g. '153330'); weight sums keep precision."""
    return str(int(v)) if v.is_integer() else f"{v:.6f}"


def load(path: Path) -> tuple[list[int], dict[int, dict[int, float]]]:
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is empty")
        layer_cols = sorted(
            (c for c in reader.fieldnames if c.startswith('layer_')),
            key=lambda c: int(c.removeprefix('layer_')),
        )
        rows = list(reader)

    layers = [int(c.removeprefix('layer_')) for c in layer_cols]
    counts: dict[int, dict[int, float]] = {l: {} for l in layers}
    for row in rows:
        expert = int(row['expert'])
        for c, l in zip(layer_cols, layers):
            counts[l][expert] = float(row[c])
    return layers, counts


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)

    inputs = [Path(p) for p in sys.argv[1:-1]]
    output = Path(sys.argv[-1])

    merged: dict[int, dict[int, float]] = {}
    for path in inputs:
        layers, counts = load(path)
        for l in layers:
            dst = merged.setdefault(l, {})
            for expert, cnt in counts[l].items():
                dst[expert] = dst.get(expert, 0) + cnt

    if not merged:
        raise ValueError("no input profiles given")

    all_layers = sorted(merged)
    all_experts = sorted({e for l in all_layers for e in merged[l]})

    expert_totals = {
        e: sum(merged[l].get(e, 0) for l in all_layers)
        for e in all_experts
    }
    grand_total = sum(expert_totals.values())
    ranked = sorted(all_experts, key=lambda e: expert_totals[e], reverse=True)

    with open(output, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['rank', 'expert', 'total_count', 'pct'] + [f'layer_{l}' for l in all_layers])
        for rank, e in enumerate(ranked):
            count = expert_totals[e]
            pct = 100.0 * count / grand_total if grand_total > 0 else 0.0
            per_layer = [fmt(merged[l].get(e, 0.0)) for l in all_layers]
            w.writerow([rank, e, fmt(count), f"{pct:.4f}"] + per_layer)

    print(f"Merged {len(inputs)} profile(s) ({len(all_layers)} layers, {len(all_experts)} experts) -> {output}")


if __name__ == '__main__':
    main()
