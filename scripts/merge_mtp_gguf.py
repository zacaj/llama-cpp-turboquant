#!/usr/bin/env python3
"""Add or replace the NextN/MTP block(s) of a GGUF with those from another GGUF.

The source can be either a full combined checkpoint (trunk + MTP, e.g. an
upstream release) or an already-extracted MTP-only file (see
extract_mtp_gguf.py) — only its trailing `nextn_predict_layers` block(s) are
read either way, so there is no need to pre-extract.

The target's own token_embd/output/output_norm and all other metadata are
kept untouched; only the block count and the injected block's tensors
change. If the target already has its own MTP block(s), they are dropped
and replaced by the source's.

Usage:
  python scripts/merge_mtp_gguf.py target.gguf source.gguf output.gguf
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

_LAYER_RE = re.compile(r'^blk\.(\d+)\.(.*)$')


def get_layer(name: str) -> tuple[int, str] | None:
    m = _LAYER_RE.match(name)
    return (int(m.group(1)), m.group(2)) if m else None


def get_arch(reader: gguf.GGUFReader) -> str:
    field = reader.get_field(gguf.Keys.General.ARCHITECTURE)
    if field is None:
        raise ValueError("Missing general.architecture — is this a valid GGUF?")
    return field.contents()


def get_int_kv(reader: gguf.GGUFReader, key: str, default: int | None = None) -> int:
    field = reader.get_field(key)
    if field is None:
        if default is not None:
            return default
        raise ValueError(f"Missing {key}")
    return int(field.contents())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", type=Path, help="GGUF to add/replace the MTP block in")
    parser.add_argument("source", type=Path, help="GGUF to take the MTP block(s) from (full checkpoint or MTP-only)")
    parser.add_argument("output", type=Path, help="Output GGUF file")
    parser.add_argument("--force", action="store_true", help="Overwrite output without prompting")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        print(f"Output {args.output} already exists. Use --force to overwrite.")
        sys.exit(1)

    print(f"Reading target {args.target} ...")
    tgt = gguf.GGUFReader(str(args.target), 'r')
    tgt_arch = get_arch(tgt)
    tgt_block_count = get_int_kv(tgt, gguf.Keys.LLM.BLOCK_COUNT.replace('{arch}', tgt_arch))
    tgt_nextn = get_int_kv(tgt, gguf.Keys.LLM.NEXTN_PREDICT_LAYERS.replace('{arch}', tgt_arch), default=0)
    tgt_trunk_count = tgt_block_count - tgt_nextn
    tgt_embd = get_int_kv(tgt, gguf.Keys.LLM.EMBEDDING_LENGTH.replace('{arch}', tgt_arch))

    print(f"Reading source {args.source} ...")
    src = gguf.GGUFReader(str(args.source), 'r')
    src_arch = get_arch(src)
    src_block_count = get_int_kv(src, gguf.Keys.LLM.BLOCK_COUNT.replace('{arch}', src_arch))
    src_nextn = get_int_kv(src, gguf.Keys.LLM.NEXTN_PREDICT_LAYERS.replace('{arch}', src_arch), default=0)
    src_embd = get_int_kv(src, gguf.Keys.LLM.EMBEDDING_LENGTH.replace('{arch}', src_arch))

    if src_nextn <= 0:
        raise ValueError(f"{args.source} has no MTP/NextN block(s) to take (nextn_predict_layers=0)")
    if src_arch != tgt_arch:
        raise ValueError(f"Architecture mismatch: target={tgt_arch} source={src_arch}")
    if src_embd != tgt_embd:
        raise ValueError(f"Hidden size mismatch: target n_embd={tgt_embd} source n_embd={src_embd}")

    src_nextn_layers = list(range(src_block_count - src_nextn, src_block_count))
    print(f"Target: {tgt_arch}, {tgt_trunk_count} trunk block(s)"
          + (f" + {tgt_nextn} existing MTP block(s) (will be replaced)" if tgt_nextn else " (no existing MTP)"))
    print(f"Source MTP block(s): {src_nextn_layers} (of {src_block_count} total)")

    # target tensors to keep: everything except its own existing nextn block(s)
    old_nextn_range = range(tgt_trunk_count, tgt_block_count)
    kept_tgt = [t for t in tgt.tensors if (get_layer(t.name) or (None,))[0] not in old_nextn_range]

    # source tensors to inject, renumbered to sit right after the target's trunk
    injected = []
    for t in src.tensors:
        layer = get_layer(t.name)
        if layer is None or layer[0] not in src_nextn_layers:
            continue
        offset = layer[0] - src_nextn_layers[0]
        new_name = f"blk.{tgt_trunk_count + offset}.{layer[1]}"
        injected.append((new_name, t))

    # sanity-check shapes of injected tensors against the target's own trunk blocks
    # (attn/ffn weights share names across blocks; the nextn.* glue tensors are new
    # and have no target-side counterpart to check against). Compare the logical
    # ne[] shape (tensor.shape), not tensor.data.shape -- the latter is the raw
    # quantized-storage view, which differs by quant type even for identically
    # shaped tensors and would produce false-positive mismatches here.
    tgt_shape_by_suffix: dict[str, tuple] = {}
    for t in tgt.tensors:
        layer = get_layer(t.name)
        if layer is not None and layer[0] < tgt_trunk_count:
            tgt_shape_by_suffix.setdefault(layer[1], tuple(t.shape))
    mismatches = []
    for new_name, t in injected:
        suffix = get_layer(new_name)[1]
        expected = tgt_shape_by_suffix.get(suffix)
        got = tuple(t.shape)
        if expected is not None and expected != got:
            mismatches.append((suffix, expected, got))
    if mismatches:
        print("ERROR: shape mismatch between target trunk and injected MTP block:")
        for suffix, expected, got in mismatches:
            print(f"  {suffix}: target={expected} source={got}")
        sys.exit(1)

    print(f"Keeping {len(kept_tgt)} target tensors, injecting {len(injected)} MTP tensors "
          f"as blk.{tgt_trunk_count}..blk.{tgt_trunk_count + src_nextn - 1}")

    writer = gguf.GGUFWriter(str(args.output), arch=tgt_arch, endianess=tgt.endianess)
    alignment_field = tgt.get_field(gguf.Keys.General.ALIGNMENT)
    if alignment_field is not None:
        writer.data_alignment = alignment_field.contents()

    block_count_key = gguf.Keys.LLM.BLOCK_COUNT.replace('{arch}', tgt_arch)
    nextn_key = gguf.Keys.LLM.NEXTN_PREDICT_LAYERS.replace('{arch}', tgt_arch)
    bc_field = tgt.get_field(block_count_key)
    bc_type = bc_field.types[0] if bc_field is not None else gguf.GGUFValueType.UINT32
    nx_field = tgt.get_field(nextn_key) or src.get_field(nextn_key)
    nx_type = nx_field.types[0] if nx_field is not None else gguf.GGUFValueType.UINT32

    suppress = {gguf.Keys.General.ARCHITECTURE, block_count_key, nextn_key}
    for field in tgt.fields.values():
        if field.name in suppress or field.name.startswith('GGUF.'):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)
    writer.add_key_value(block_count_key, tgt_trunk_count + src_nextn, bc_type)
    writer.add_key_value(nextn_key, src_nextn, nx_type)

    total_bytes = sum(t.n_bytes for t in kept_tgt) + sum(t.n_bytes for _, t in injected)
    for t in kept_tgt:
        writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    for new_name, t in injected:
        writer.add_tensor_info(new_name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)

    bar = tqdm(desc="Writing", total=total_bytes, unit="byte", unit_scale=True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in kept_tgt:
        writer.write_tensor_data(t.data, tensor_endianess=tgt.endianess)
        bar.update(t.n_bytes)
    for new_name, t in injected:
        writer.write_tensor_data(t.data, tensor_endianess=src.endianess)
        bar.update(t.n_bytes)
    writer.close()

    out_size = args.output.stat().st_size
    print(f"\nWrote {args.output} ({out_size/1e9:.2f} GB)")


if __name__ == '__main__':
    main()
