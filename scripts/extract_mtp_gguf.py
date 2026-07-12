#!/usr/bin/env python3
"""Extract the NextN/MTP block(s) from a combined GGUF into a standalone draft file.

Keeps every non-`blk.N.*` (global) tensor — token_embd, output, output_norm,
rope_freqs, etc. — plus the trailing `nextn_predict_layers` transformer block(s)
(both their plain decoder tensors and their `blk.N.nextn.*` glue tensors).
Drops every other trunk block. All KV metadata (including block_count) is
copied unchanged, since llama.cpp's loader decides "mtp_only" purely by
probing for `blk.0.attn_norm.weight` and treats missing trunk-block tensors
as optional in that mode.

The output is meant to be loaded as the draft file via -md/--model-draft
with --spec-type mtp, alongside a *different* checkpoint that shares the
same architecture, hidden size, and vocab (e.g. a fine-tune of the same base
the MTP block was trained on) — not necessarily the original combined file's
own trunk.

Usage:
  python scripts/extract_mtp_gguf.py input.gguf output-mtp.gguf
"""
from __future__ import annotations

import re
import sys
import argparse
from pathlib import Path

from tqdm import tqdm

if (Path(__file__).parent.parent / "gguf-py").exists():
    sys.path.insert(0, str(Path(__file__).parent.parent / "gguf-py"))

import gguf

_LAYER_RE = re.compile(r'^blk\.(\d+)\.')


def get_layer(name: str) -> int | None:
    m = _LAYER_RE.match(name)
    return int(m.group(1)) if m else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input",  type=Path, help="Input combined GGUF file")
    parser.add_argument("output", type=Path, help="Output MTP-only GGUF file")
    parser.add_argument("--force", action="store_true", help="Overwrite output without prompting")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        print(f"Output {args.output} already exists. Use --force to overwrite.")
        sys.exit(1)

    print(f"Reading {args.input} ...")
    reader = gguf.GGUFReader(str(args.input), 'r')

    arch_field = reader.get_field(gguf.Keys.General.ARCHITECTURE)
    if arch_field is None:
        raise ValueError("Missing general.architecture — is this a valid GGUF?")
    arch = arch_field.contents()

    block_count_key = gguf.Keys.LLM.BLOCK_COUNT.replace('{arch}', arch)
    nextn_key = gguf.Keys.LLM.NEXTN_PREDICT_LAYERS.replace('{arch}', arch)

    bc_field = reader.get_field(block_count_key)
    nextn_field = reader.get_field(nextn_key)
    if bc_field is None or nextn_field is None:
        raise ValueError(f"Missing {block_count_key} or {nextn_key} — does this model have an MTP/NextN block?")

    block_count = int(bc_field.contents())
    n_layer_nextn = int(nextn_field.contents())
    if n_layer_nextn <= 0:
        raise ValueError(f"{nextn_key} = {n_layer_nextn}: nothing to extract")

    nextn_layers = set(range(block_count - n_layer_nextn, block_count))
    print(f"Architecture: {arch}  |  block_count={block_count}  |  nextn layers: {sorted(nextn_layers)}")

    def keep(name: str) -> bool:
        layer = get_layer(name)
        return layer is None or layer in nextn_layers

    kept_tensors = [t for t in reader.tensors if keep(t.name)]
    dropped = len(reader.tensors) - len(kept_tensors)
    print(f"Keeping {len(kept_tensors)} tensors ({', '.join(sorted({t.name for t in kept_tensors if get_layer(t.name) is None}))} + nextn blocks), dropping {dropped} trunk tensors")

    writer = gguf.GGUFWriter(str(args.output), arch=arch, endianess=reader.endianess)
    alignment_field = reader.get_field(gguf.Keys.General.ALIGNMENT)
    if alignment_field is not None:
        writer.data_alignment = alignment_field.contents()

    suppress = {gguf.Keys.General.ARCHITECTURE}
    for field in reader.fields.values():
        if field.name in suppress or field.name.startswith('GGUF.'):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)

    total_bytes = sum(t.n_bytes for t in kept_tensors)
    for tensor in kept_tensors:
        writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype, tensor.data.nbytes, tensor.tensor_type)

    bar = tqdm(desc="Writing", total=total_bytes, unit="byte", unit_scale=True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in kept_tensors:
        writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
        bar.update(tensor.n_bytes)
    writer.close()

    in_size = args.input.stat().st_size
    out_size = args.output.stat().st_size
    print(f"\nSize: {in_size/1e9:.2f} GB -> {out_size/1e9:.2f} GB ({100*out_size/in_size:.1f}%)")


if __name__ == '__main__':
    main()
