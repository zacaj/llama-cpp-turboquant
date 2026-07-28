#!/usr/bin/env python3
"""Prune unused vocabulary entries from a GGUF file's BPE tokenizer.

Tokenizes one or more corpus files with the model's own tokenizer (via
llama-tokenize, vocab-only load) to find which tokens are actually used,
then expands that set through the BPE merge chain so intermediate subwords
needed mid-merge survive too, even if they never appear as a final output
token. Control/user-defined tokens (chat template, tool-call markers, etc.)
and the 256 base byte-fallback tokens are always kept, so encoding of text
outside the corpus still works -- just possibly less efficiently.

Only supports GPT2-style byte-level BPE tokenizers (tokenizer.ggml.model ==
"gpt2"), which is what convert_hf_to_gguf.py uses for the Qwen family.

Usage:
  python scripts/prune_vocab.py input.gguf output.gguf --corpus corpus1.txt corpus2.txt
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prune unused vocabulary from a GGUF BPE tokenizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input",  type=Path, help="Input GGUF file")
    parser.add_argument("output", type=Path, help="Output GGUF file")
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

    # Tensors whose vocab-sized axis0 needs the same row selection
    vocab_tensor_names = {t.name for t in reader.tensors if t.name in ("token_embd.weight", "output.weight")}
    print(f"Vocab-dimensioned tensors: {sorted(vocab_tensor_names)}")

    writer = gguf.GGUFWriter(str(args.output), arch=arch, endianess=reader.endianess)
    alignment_field = reader.get_field(gguf.Keys.General.ALIGNMENT)
    if alignment_field is not None:
        writer.data_alignment = alignment_field.contents()

    # Copy KV metadata, remapping/dropping vocab-indexed keys
    suppress = {
        gguf.Keys.General.ARCHITECTURE, gguf.Keys.Tokenizer.LIST,
        gguf.Keys.Tokenizer.TOKEN_TYPE, gguf.Keys.Tokenizer.MERGES,
    }
    id_array_keys = {gguf.Keys.Tokenizer.SUPPRESS_TOKENS}
    scalar_id_keys = set(SPECIAL_ID_KEYS)
    for field in reader.fields.values():
        if field.name in suppress or field.name.startswith('GGUF.'):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        val = field.contents()
        if field.name in scalar_id_keys:
            val = old_to_new[val]
        elif field.name in id_array_keys:
            val = [old_to_new[v] for v in val if v in old_to_new]
        writer.add_key_value(field.name, val, val_type, sub_type=sub_type)

    writer.add_token_list(new_tokens)
    writer.add_token_types(new_token_type)
    writer.add_token_merges(new_merges)

    # Pass 1: register tensor info (shape/size only, no data copy)
    for tensor in reader.tensors:
        if tensor.name in vocab_tensor_names:
            new_shape = list(tensor.data.shape)
            new_shape[0] = n_keep
            nbytes = tensor.data[0].nbytes * n_keep
            writer.add_tensor_info(tensor.name, tuple(new_shape), tensor.data.dtype, nbytes, tensor.tensor_type)
        else:
            writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype, tensor.data.nbytes, tensor.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    # Pass 2: write tensor data
    for tensor in reader.tensors:
        if tensor.name in vocab_tensor_names:
            data = np.ascontiguousarray(tensor.data[kept_ids])
            writer.write_tensor_data(data, tensor_endianess=reader.endianess)
        else:
            writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
    writer.close()

    in_size = args.input.stat().st_size
    out_size = args.output.stat().st_size
    saved = in_size - out_size
    print(f"\nSize: {in_size/1e9:.2f} GB -> {out_size/1e9:.2f} GB  ({saved/1e9:.2f} GB saved, {100*saved/in_size:.1f}%)")


if __name__ == '__main__':
    main()
