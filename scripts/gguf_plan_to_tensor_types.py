#!/usr/bin/env python3
"""Convert a per-tensor quantization plan TSV (the format gguf_layer_quants.py dumps:
layer, tensor, type, shape, n_bytes) into a --tensor-type-file for llama-quantize.

Meant for a dump-edit-requantize loop: dump an existing model's recipe with
gguf-layer-quants.sh, edit the "type" column for whichever tensors you want to
change, then feed the edited TSV here. Every row becomes an exact override -- the
plan is the sole source of truth for every tensor it lists, not a diff against the
base ftype's usual mix.

Usage:
  python scripts/gguf_plan_to_tensor_types.py plan.tsv output.types
"""
from __future__ import annotations

import re
import sys
import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plan", type=Path, help="Plan TSV (header + layer/tensor/type/shape/n_bytes rows)")
    parser.add_argument("output", type=Path, help="Where to write the --tensor-type-file")
    args = parser.parse_args()

    lines = args.plan.read_text().splitlines()
    if not lines:
        print(f"ERROR: {args.plan} is empty", file=sys.stderr)
        sys.exit(1)

    header = lines[0].split("\t")
    try:
        tensor_col = header.index("tensor")
        type_col = header.index("type")
    except ValueError:
        print(f"ERROR: expected a header row with 'tensor' and 'type' columns, got: {lines[0]!r}", file=sys.stderr)
        sys.exit(1)

    written = 0
    with open(args.output, "w") as f:
        for line in lines[1:]:
            if not line.strip():
                continue
            cols = line.split("\t")
            name = cols[tensor_col]
            ttype = cols[type_col]
            # anchored + escaped so each line matches exactly one tensor, never a
            # substring of another (llama-quantize treats --tensor-type-file lines
            # as regexes)
            f.write(f"^{re.escape(name)}$={ttype.lower()}\n")
            written += 1

    print(f"Wrote {written} tensor-type override(s) -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
