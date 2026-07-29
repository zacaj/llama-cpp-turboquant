#!/usr/bin/env python3
"""Dump every tensor in a GGUF as a TSV of layer index, tensor name, quant type,
shape and size, plus a per-layer summary of which quant type(s) it mixes.

Only reads tensor-info headers (via mmap), not tensor data, so this is cheap
even against multi-GB models.

Usage:
  python scripts/gguf_layer_quants.py model.gguf
"""
from __future__ import annotations

import re
import sys
import argparse
from pathlib import Path

if (Path(__file__).parent.parent / "gguf-py").exists():
    sys.path.insert(0, str(Path(__file__).parent.parent / "gguf-py"))

import gguf

LAYER_RE = re.compile(r"^blk\.(\d+)\.")


def layer_of(name: str) -> str:
    m = LAYER_RE.match(name)
    return m.group(1) if m else "-"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path, help="GGUF file to inspect")
    args = parser.parse_args()

    print(f"Reading {args.model} ...", file=sys.stderr)
    reader = gguf.GGUFReader(str(args.model), 'r')

    print("layer\ttensor\ttype\tshape\tn_bytes")
    layer_types: dict[str, set[str]] = {}
    for t in reader.tensors:
        layer = layer_of(t.name)
        ttype = t.tensor_type.name
        shape = "x".join(str(d) for d in t.shape.tolist())
        print(f"{layer}\t{t.name}\t{ttype}\t{shape}\t{t.n_bytes}")
        layer_types.setdefault(layer, set()).add(ttype)

    def layer_sort_key(l: str) -> tuple[int, int]:
        return (0, int(l)) if l != "-" else (1, 0)

    print("\n# per-layer summary", file=sys.stderr)
    for layer in sorted(layer_types, key=layer_sort_key):
        print(f"#   layer {layer}: {', '.join(sorted(layer_types[layer]))}", file=sys.stderr)


if __name__ == "__main__":
    main()
