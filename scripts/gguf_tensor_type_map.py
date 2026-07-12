#!/usr/bin/env python3
"""Build a --tensor-type-file mapping every tensor in a reference (already-quantized)
GGUF to its exact ggml type, so the same per-tensor quantization recipe can be
reproduced on a different (same-architecture) GGUF via llama-quantize's
--tensor-type-file.

Only reads tensor-info headers (via mmap), not tensor data, so this is cheap even
against multi-GB models.

Usage:
  python scripts/gguf_tensor_type_map.py reference.gguf target.gguf output.types
"""
from __future__ import annotations

import re
import sys
import argparse
from pathlib import Path

if (Path(__file__).parent.parent / "gguf-py").exists():
    sys.path.insert(0, str(Path(__file__).parent.parent / "gguf-py"))

import gguf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference", type=Path, help="Already-quantized GGUF to copy the per-tensor type recipe from")
    parser.add_argument("target", type=Path, help="F16/BF16 GGUF the recipe will be applied to (only used to check tensor names line up)")
    parser.add_argument("output", type=Path, help="Where to write the --tensor-type-file")
    args = parser.parse_args()

    print(f"Reading reference tensor types from {args.reference} ...", file=sys.stderr)
    ref = gguf.GGUFReader(str(args.reference), 'r')
    print(f"Reading target tensor names from {args.target} ...", file=sys.stderr)
    tgt = gguf.GGUFReader(str(args.target), 'r')

    tgt_names = {t.name for t in tgt.tensors}
    ref_by_name = {t.name: t.tensor_type for t in ref.tensors}

    matched = [(name, ttype) for name, ttype in ref_by_name.items() if name in tgt_names]
    ref_only = sorted(set(ref_by_name) - tgt_names)
    tgt_only = sorted(tgt_names - set(ref_by_name))

    with open(args.output, "w") as f:
        for name, ttype in matched:
            # anchored + escaped so each line matches exactly one tensor, never a
            # substring of another (llama-quantize treats --tensor-type-file lines
            # as regexes)
            f.write(f"^{re.escape(name)}$={ttype.name.lower()}\n")

    print(f"Matched {len(matched)}/{len(ref_by_name)} reference tensors; wrote {args.output}", file=sys.stderr)

    def _warn_list(label: str, names: list[str]) -> None:
        print(f"WARNING: {len(names)} tensor(s) {label}:", file=sys.stderr)
        for n in names[:20]:
            print(f"  {n}", file=sys.stderr)
        if len(names) > 20:
            print(f"  ... and {len(names) - 20} more", file=sys.stderr)

    if ref_only:
        _warn_list("in the reference but not in the target (skipped, no override written)", ref_only)
    if tgt_only:
        _warn_list("in the target but not in the reference (will fall back to the base ftype mix)", tgt_only)


if __name__ == "__main__":
    main()
