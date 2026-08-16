#!/usr/bin/env python3
"""Prune specific experts from MoE layers in a GGUF file.

Removes expert weight slices directly from quantized tensors without
dequantization, producing a smaller GGUF with a reduced expert_count.

All pruned layers must remove the same NUMBER of experts (llama.cpp requires
a uniform expert_count across layers). The specific experts removed per layer
can differ.

Usage:
  python scripts/prune_moe_experts.py input.gguf output.gguf --csv pruning.csv
  python scripts/prune_moe_experts.py input.gguf output.gguf --experts "3:167,204  5-10:254,255,256"
  python scripts/prune_moe_experts.py input.gguf output.gguf --csv p.csv --experts "*:0"
  python scripts/prune_moe_experts.py input.gguf output.gguf --csv p.csv \
      --note pruning.profile=summary.weighted.csv --note pruning.source=input.gguf

--note KEY=VALUE embeds arbitrary string metadata in the output GGUF (repeatable) -- e.g. record
which profile CSV and source model a pruned GGUF came from, so that's recoverable from the file
itself (via gguf_dump.py) even if it gets renamed or moved away from a filename that encoded it.

CSV format (first column = layer spec, remaining columns = expert indices to remove):
  layer,e0,e1,...
  3,167,204
  5-10,254,255,256
  *,100

CLI spec format:  "LAYER_SPEC:e0,e1,...  LAYER_SPEC:e0,e1,..."
  LAYER_SPEC: integer, N-M range (inclusive), or * for all MoE layers
"""

from __future__ import annotations

import sys
import os
import re
import csv
import argparse
from pathlib import Path

import numpy as np

if (Path(__file__).parent.parent / "gguf-py").exists():
    sys.path.insert(0, str(Path(__file__).parent.parent / "gguf-py"))

import gguf
from gguf.constants import GGML_QUANT_SIZES


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

def parse_layer_spec(spec: str, moe_layers: list[int]) -> list[int]:
    spec = spec.strip()
    if spec == '*':
        return list(moe_layers)
    m = re.fullmatch(r'(\d+)-(\d+)', spec)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return [l for l in moe_layers if lo <= l <= hi]
    if re.fullmatch(r'\d+', spec):
        return [int(spec)]
    raise ValueError(f"Invalid layer spec {spec!r}: expected integer, N-M range, or *")


def parse_csv(path: str) -> dict[str, set[int]]:
    raw: dict[str, set[int]] = {}
    with open(path, newline='') as f:
        for row in csv.reader(f):
            if not row or not row[0].strip():
                continue
            layer_spec = row[0].strip()
            if layer_spec.lower() == 'layer':
                continue
            experts = {int(e.strip()) for e in row[1:] if e.strip()}
            if experts:
                raw.setdefault(layer_spec, set()).update(experts)
    return raw


def parse_cli(spec: str) -> dict[str, set[int]]:
    raw: dict[str, set[int]] = {}
    for token in spec.split():
        if ':' not in token:
            raise ValueError(f"Invalid expert spec {token!r}: expected 'LAYER_SPEC:e0,e1,...'")
        layer_spec, expert_str = token.split(':', 1)
        experts = {int(e) for e in expert_str.split(',') if e.strip()}
        if experts:
            raw.setdefault(layer_spec.strip(), set()).update(experts)
    return raw


def expand_prune_map(
    raw_maps: list[dict[str, set[int]]],
    moe_layers: list[int],
    n_expert: int,
) -> dict[int, set[int]]:
    prune_map: dict[int, set[int]] = {}
    for raw in raw_maps:
        for layer_spec, experts in raw.items():
            for l in parse_layer_spec(layer_spec, moe_layers):
                invalid = {e for e in experts if e >= n_expert}
                if invalid:
                    raise ValueError(
                        f"Expert indices {sorted(invalid)} are out of range for layer {l} "
                        f"(n_expert={n_expert}, valid: 0–{n_expert-1})"
                    )
                prune_map.setdefault(l, set()).update(experts)
    return prune_map


def validate_uniform_count(prune_map: dict[int, set[int]]) -> int:
    """All layers with a non-empty prune set must remove the same count."""
    active = {l: len(e) for l, e in prune_map.items() if e}
    if not active:
        raise ValueError("No experts specified to prune")
    unique = set(active.values())
    if len(unique) > 1:
        by_count: dict[int, list[int]] = {}
        for l, c in active.items():
            by_count.setdefault(c, []).append(l)
        detail = '; '.join(
            f"remove {c} on layers {sorted(ls)}"
            for c, ls in sorted(by_count.items())
        )
        raise ValueError(
            "All MoE layers must remove the same number of experts "
            "(llama.cpp uses a single global expert_count).\n"
            f"Inconsistent counts: {detail}\n"
            "Tip: add the missing layers to the spec, or use * to apply uniformly."
        )
    return unique.pop()


# ---------------------------------------------------------------------------
# Tensor classification
# ---------------------------------------------------------------------------

_LAYER_RE = re.compile(r'^blk\.(\d+)\.')

def get_layer(name: str) -> int | None:
    m = _LAYER_RE.match(name)
    return int(m.group(1)) if m else None


def is_expert_3d(name: str) -> bool:
    """Packed 3D weight tensor: blk.X.ffn_{gate,down,up}_exps.weight"""
    return bool(re.search(r'\.(ffn_gate_exps|ffn_down_exps|ffn_up_exps)\.weight$', name))


def is_router(name: str) -> bool:
    """2D router: blk.X.ffn_gate_inp.weight"""
    return name.endswith('.ffn_gate_inp.weight')


def is_expert_1d(name: str) -> bool:
    """1D per-expert scale tensors: blk.X.ffn_*_exps.{scale,input_scale}"""
    return bool(re.search(r'\.(ffn_gate_exps|ffn_down_exps|ffn_up_exps)\.(scale|input_scale)$', name))


def needs_pruning(name: str) -> bool:
    return is_expert_3d(name) or is_router(name) or is_expert_1d(name)


# ---------------------------------------------------------------------------
# Core pruning logic
# ---------------------------------------------------------------------------

def pruned_data(tensor: gguf.ReaderTensor, kept: list[int]) -> np.ndarray:
    """Return a contiguous copy of tensor.data with only the kept expert slices.

    tensor.data is shaped in C (slowest-first) order, so the expert dimension
    is always axis 0 (it is the outermost / slowest-varying GGML dimension).
    Slicing axis 0 gives contiguous per-expert byte ranges for quantized types
    because each expert's blocks are laid out contiguously in memory.
    """
    return np.ascontiguousarray(tensor.data[kept])


def pruned_shape_and_nbytes(tensor: gguf.ReaderTensor, kept: list[int]) -> tuple[tuple, np.dtype, int]:
    """Compute new shape/nbytes without allocating the full data array."""
    new_shape = list(tensor.data.shape)
    new_shape[0] = len(kept)
    nbytes = tensor.data[0].nbytes * len(kept)
    return tuple(new_shape), tensor.data.dtype, nbytes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prune MoE experts from a GGUF file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input",     type=Path, help="Input GGUF file")
    parser.add_argument("output",    type=Path, help="Output GGUF file")
    parser.add_argument("--csv",     type=Path, help="CSV file with pruning spec")
    parser.add_argument("--experts", type=str,  help='CLI spec e.g. "3:167,204  5-10:254,255"')
    parser.add_argument("--force",   action="store_true", help="Overwrite output without prompting")
    parser.add_argument("--note", action="append", default=[], metavar="KEY=VALUE",
                         help="extra string metadata to embed in the output GGUF (e.g. pruning "
                              "provenance -- which profile/source model this was pruned from), "
                              "repeatable. Overrides an identically-named key already present in "
                              "the input, so re-pruning an already-annotated GGUF doesn't fork it.")
    args = parser.parse_args()

    notes: dict[str, str] = {}
    for n in args.note:
        if '=' not in n:
            parser.error(f"Invalid --note {n!r}: expected KEY=VALUE")
        k, v = n.split('=', 1)
        notes[k] = v

    if not args.csv and not args.experts:
        parser.error("Provide at least one of --csv or --experts")

    if args.output.exists() and not args.force:
        print(f"Output {args.output} already exists. Use --force to overwrite.")
        sys.exit(1)

    # Parse inputs
    raw_maps: list[dict[str, set[int]]] = []
    if args.csv:
        raw_maps.append(parse_csv(str(args.csv)))
    if args.experts:
        raw_maps.append(parse_cli(args.experts))

    # Open reader
    print(f"Reading {args.input} ...")
    reader = gguf.GGUFReader(str(args.input), 'r')

    # Detect architecture
    arch_field = reader.get_field(gguf.Keys.General.ARCHITECTURE)
    if arch_field is None:
        raise ValueError("Missing general.architecture — is this a valid GGUF?")
    arch = arch_field.contents()

    # Read expert count
    expert_count_key = gguf.Keys.LLM.EXPERT_COUNT.replace('{arch}', arch)
    ec_field = reader.get_field(expert_count_key)
    if ec_field is None:
        raise ValueError(f"Missing {expert_count_key} — does this model have MoE layers?")
    n_expert_total = int(ec_field.contents())
    print(f"Architecture: {arch}  |  experts per layer: {n_expert_total}")

    # Discover MoE layers from expert weight tensors
    moe_layers = sorted({
        get_layer(t.name)
        for t in reader.tensors
        if is_expert_3d(t.name) and get_layer(t.name) is not None
    })
    if not moe_layers:
        raise ValueError("No MoE expert tensors found in this GGUF")
    print(f"MoE layers: {len(moe_layers)}  ({moe_layers[0]}–{moe_layers[-1]})")

    # Build and validate prune map
    prune_map = expand_prune_map(raw_maps, moe_layers, n_expert_total)
    n_remove = validate_uniform_count(prune_map)
    n_keep = n_expert_total - n_remove

    layers_affected = sorted(l for l, e in prune_map.items() if e)
    print(f"Removing {n_remove} experts from {len(layers_affected)} layer(s) → {n_keep} remaining")

    # Default: layers not in spec keep all experts
    for l in moe_layers:
        prune_map.setdefault(l, set())

    # Build kept-index lists per layer (computed once, reused)
    kept_map: dict[int, list[int]] = {
        l: [i for i in range(n_expert_total) if i not in prune_map[l]]
        for l in moe_layers
    }

    # Open writer
    writer = gguf.GGUFWriter(str(args.output), arch=arch, endianess=reader.endianess)
    alignment_field = reader.get_field(gguf.Keys.General.ALIGNMENT)
    if alignment_field is not None:
        writer.data_alignment = alignment_field.contents()

    # Copy KV metadata, updating expert_count. --note keys are suppressed here and written
    # separately below so they always win over an identically-named key already in the input
    # (relevant when re-pruning an already-annotated GGUF).
    suppress = {gguf.Keys.General.ARCHITECTURE} | set(notes)
    for field in reader.fields.values():
        if field.name in suppress or field.name.startswith('GGUF.'):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        val = field.contents()
        if field.name == expert_count_key:
            print(f"  {expert_count_key}: {n_expert_total} → {n_keep}")
            val = n_keep
        writer.add_key_value(field.name, val, val_type, sub_type=sub_type)

    for key, val in notes.items():
        writer.add_key_value(key, val, gguf.GGUFValueType.STRING)

    # Pass 1: register tensor info (shape/size only, no data copy)
    for tensor in reader.tensors:
        layer = get_layer(tensor.name)
        if layer is not None and layer in kept_map and needs_pruning(tensor.name) and prune_map[layer]:
            kept = kept_map[layer]
            new_shape, dtype, nbytes = pruned_shape_and_nbytes(tensor, kept)
            writer.add_tensor_info(tensor.name, new_shape, dtype, nbytes, tensor.tensor_type)
        else:
            writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype, tensor.data.nbytes, tensor.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    # Pass 2: write tensor data
    pruned_tensors = 0
    for tensor in reader.tensors:
        layer = get_layer(tensor.name)
        if layer is not None and layer in kept_map and needs_pruning(tensor.name) and prune_map[layer]:
            kept = kept_map[layer]
            writer.write_tensor_data(pruned_data(tensor, kept), tensor_endianess=reader.endianess)
            pruned_tensors += 1
        else:
            writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)

    writer.close()

    in_size  = args.input.stat().st_size
    out_size = args.output.stat().st_size
    saved    = in_size - out_size
    print(f"\nModified {pruned_tensors} tensors.")
    print(f"Size: {in_size/1e9:.2f} GB → {out_size/1e9:.2f} GB  ({saved/1e9:.2f} GB saved, {100*saved/in_size:.1f}%)")


if __name__ == '__main__':
    main()
