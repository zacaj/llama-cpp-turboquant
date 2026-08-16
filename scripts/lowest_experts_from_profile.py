#!/usr/bin/env python3
"""Convert a wide per-layer expert activation profile into a pruning CSV.

Input format (as produced by the MoE profiler, either its plain count CSV or
its sibling .weighted.csv -- see below):
  rank,expert,total_count,pct,layer_0,layer_1,...,layer_N
  0,22,1982310,2.1406,84072,19216,...

For each layer_* column, selects the N experts with the lowest value in that
layer and writes them out in the format expected by prune_moe_experts.py:
  layer,e0,e1,...,e(N-1)

Prefer moe-expert-profile.sh's .weighted.csv (summed router weight-mass per
expert) over its plain count CSV as input here: measured on Qwen3.6-35B-A3B
at 40% expert removal, weight-mass-based pruning beat count-based on PPL
across every corpus and expert_used_count tested. Raw-count pruning can
discard an expert that's rarely selected but dominant (high-weight, rank-1)
whenever it is -- weight-mass pruning doesn't make that mistake.

Layers with no profiling data (e.g. an MTP/draft head that the calibration
run never exercised) can be filled in with --fill-missing-layers, which
synthesizes a value for the missing layer by summing values across all
present layers.

Usage:
  python scripts/lowest_experts_from_profile.py summary.weighted.csv pruning.csv -n 51
  python scripts/lowest_experts_from_profile.py summary.weighted.csv pruning.csv -n 51 --fill-missing-layers 40
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pick the N lowest-activation experts per layer from a profile CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", type=Path, help="Wide-format profile CSV")
    parser.add_argument("output", type=Path, help="Output CSV in layer,e0,e1,... format")
    parser.add_argument("-n", "--num-experts", type=int, required=True,
                         help="Number of lowest-activation experts to remove per layer")
    parser.add_argument("--fill-missing-layers", type=int, metavar="MAX_LAYER",
                         help="Synthesize any missing layer in 0..MAX_LAYER (inclusive) by "
                              "summing counts across all present layers")
    args = parser.parse_args()

    with open(args.input, newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{args.input} is empty")
        layer_cols = [c for c in reader.fieldnames if c.startswith('layer_')]
        rows = list(reader)

    if not layer_cols:
        raise ValueError(f"No layer_* columns found in {args.input}")
    if 'expert' not in (reader.fieldnames or []):
        raise ValueError(f"No 'expert' column found in {args.input}")

    layer_cols.sort(key=lambda c: int(c.removeprefix('layer_')))

    experts = [int(row['expert']) for row in rows]
    layer_counts: dict[int, dict[int, float]] = {}
    for c in layer_cols:
        layer_idx = int(c.removeprefix('layer_'))
        layer_counts[layer_idx] = {e: float(row[c]) for e, row in zip(experts, rows)}

    if args.num_experts > len(rows):
        raise ValueError(f"--num-experts {args.num_experts} exceeds expert count {len(rows)} in profile")

    if args.fill_missing_layers is not None:
        missing = [l for l in range(args.fill_missing_layers + 1) if l not in layer_counts]
        if missing:
            summed = {e: sum(lc[e] for lc in layer_counts.values()) for e in experts}
            for l in missing:
                layer_counts[l] = summed
            print(f"Synthesized layer(s) {missing} by summing counts across "
                  f"{len(layer_cols)} present layers (no profiling data for these)")

    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        for layer_idx in sorted(layer_counts):
            lowest = sorted((cnt, e) for e, cnt in layer_counts[layer_idx].items())[:args.num_experts]
            row_experts = [e for _, e in lowest]
            writer.writerow([layer_idx, *row_experts])

    print(f"{len(layer_counts)} layers, {args.num_experts} lowest-activation experts each -> {args.output}")


if __name__ == '__main__':
    main()
