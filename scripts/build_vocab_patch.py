#!/usr/bin/env python3
"""Build a small vocab-patch GGUF: the pruned tokenizer KV + the two
vocab-dimensioned tensors only, none of the other ~860 unchanged tensors.

Meant to be loaded alongside the original (unmodified) base GGUF via
llama.cpp's --vocab-patch flag, which overlays this file's KV and tensors
onto the base at load time -- same effect as prune_vocab.py's full rewrite,
without duplicating the whole model on disk. Uses the exact same corpus-
seeding + merge-closure selection as prune_vocab.py (see vocab_prune_lib.py)
so the two never disagree on which tokens survive.

Only supports GPT2-style byte-level BPE tokenizers (tokenizer.ggml.model ==
"gpt2"), which is what convert_hf_to_gguf.py uses for the Qwen family.

Usage:
  python scripts/build_vocab_patch.py base.gguf patch.gguf --corpus corpus1.txt corpus2.txt
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np

if (Path(__file__).parent.parent / "gguf-py").exists():
    sys.path.insert(0, str(Path(__file__).parent.parent / "gguf-py"))

import gguf
from vocab_prune_lib import SPECIAL_ID_KEYS, select_vocab

VOCAB_TENSOR_NAMES = ("token_embd.weight", "output.weight")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a vocab-patch GGUF overlay (KV + two tensors only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input",  type=Path, help="Base GGUF file (read-only, stays unmodified)")
    parser.add_argument("output", type=Path, help="Output patch GGUF file")
    parser.add_argument("--corpus", type=Path, nargs='+', required=True,
                         help="Plain text file(s) representative of what you want the pruned vocab to cover well")
    parser.add_argument("--force", action="store_true", help="Overwrite output without prompting")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        print(f"Output {args.output} already exists. Use --force to overwrite.")
        sys.exit(1)

    print(f"Reading {args.input} ...")
    reader = gguf.GGUFReader(str(args.input), 'r')

    arch_field = reader.get_field(gguf.Keys.General.ARCHITECTURE)
    if arch_field is None:
        raise ValueError("Missing general.architecture -- is this a valid GGUF?")
    arch = arch_field.contents()
    print(f"Architecture: {arch}")

    kept_ids, old_to_new, new_tokens, new_token_type, new_merges = select_vocab(
        reader, args.input, args.corpus,
    )
    n_keep = len(kept_ids)

    vocab_tensors = {t.name: t for t in reader.tensors if t.name in VOCAB_TENSOR_NAMES}
    missing = set(VOCAB_TENSOR_NAMES) - vocab_tensors.keys()
    if missing:
        raise ValueError(f"Base file is missing expected vocab tensor(s): {sorted(missing)}")
    print(f"Vocab-dimensioned tensors: {sorted(vocab_tensors)}")

    writer = gguf.GGUFWriter(str(args.output), arch=arch, endianess=reader.endianess)

    writer.add_token_list(new_tokens)
    writer.add_token_types(new_token_type)
    writer.add_token_merges(new_merges)

    # Remap scalar special-token ids that exist in the base file
    for key in SPECIAL_ID_KEYS:
        field = reader.get_field(key)
        if field is not None:
            writer.add_uint32(key, old_to_new[int(field.contents())])

    # Remap the suppress-tokens array, if present, dropping any pruned entries
    suppress_field = reader.get_field(gguf.Keys.Tokenizer.SUPPRESS_TOKENS)
    if suppress_field is not None:
        remapped = [old_to_new[v] for v in suppress_field.contents() if v in old_to_new]
        writer.add_array(gguf.Keys.Tokenizer.SUPPRESS_TOKENS, remapped)

    # Remap the scalar vocab-size KV, if this arch declares one (this model
    # doesn't -- vocab size is derived purely from the tokens array length --
    # but other architectures do, so handle it for portability)
    vocab_size_key = gguf.Keys.LLM.VOCAB_SIZE.replace('{arch}', arch)
    vocab_size_field = reader.get_field(vocab_size_key)
    if vocab_size_field is not None:
        writer.add_uint32(vocab_size_key, n_keep)

    for name, tensor in vocab_tensors.items():
        new_shape = list(tensor.data.shape)
        new_shape[0] = n_keep
        nbytes = tensor.data[0].nbytes * n_keep
        writer.add_tensor_info(name, tuple(new_shape), tensor.data.dtype, nbytes, tensor.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    for name, tensor in vocab_tensors.items():
        data = np.ascontiguousarray(tensor.data[kept_ids])
        writer.write_tensor_data(data, tensor_endianess=reader.endianess)
    writer.close()

    out_size = args.output.stat().st_size
    print(f"\nPatch file: {out_size/1e6:.1f} MB -> {args.output}")


if __name__ == '__main__':
    main()
