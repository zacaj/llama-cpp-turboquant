"""Shared corpus-seeding + BPE merge-closure logic for vocab pruning.

Used by both prune_vocab.py (full-file rewrite) and build_vocab_patch.py
(KV+tensor overlay only). Not meant to be run directly -- the two scripts
must never disagree on which tokens survive a given corpus, so this is the
single source of truth for that computation.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import gguf
from gguf.constants import TokenType
from gguf.vocab import bytes_to_unicode

LLAMA_TOKENIZE = "/app/llama-tokenize"

SPECIAL_ID_KEYS = [
    gguf.Keys.Tokenizer.BOS_ID, gguf.Keys.Tokenizer.EOS_ID,
    gguf.Keys.Tokenizer.EOT_ID, gguf.Keys.Tokenizer.EOM_ID,
    gguf.Keys.Tokenizer.UNK_ID, gguf.Keys.Tokenizer.SEP_ID,
    gguf.Keys.Tokenizer.PAD_ID, gguf.Keys.Tokenizer.MASK_ID,
    gguf.Keys.Tokenizer.FIM_PRE_ID, gguf.Keys.Tokenizer.FIM_SUF_ID,
    gguf.Keys.Tokenizer.FIM_MID_ID, gguf.Keys.Tokenizer.FIM_PAD_ID,
    gguf.Keys.Tokenizer.FIM_REP_ID, gguf.Keys.Tokenizer.FIM_SEP_ID,
    gguf.Keys.Tokenizer.PREFIX_ID, gguf.Keys.Tokenizer.SUFFIX_ID,
    gguf.Keys.Tokenizer.MIDDLE_ID,
]


def tokenize_ids(model_path: Path, corpus_path: Path) -> set[int]:
    if not Path(LLAMA_TOKENIZE).exists():
        raise RuntimeError(
            f"{LLAMA_TOKENIZE} not found -- this script must run inside the "
            "turboquant image (use the matching .sh wrapper)"
        )
    out = subprocess.run(
        [LLAMA_TOKENIZE, "-m", str(model_path), "-f", str(corpus_path), "--ids", "--log-disable"],
        capture_output=True, text=True, check=True,
    ).stdout
    return set(json.loads(out))


def merge_closure(
    seed_strings: set[str],
    merges: list[tuple[str, str]],
) -> tuple[set[str], set[int]]:
    """Expand seed token strings to every intermediate token needed to build
    them via the merge chain. Returns (kept token strings, kept merge indices).
    """
    recipe: dict[str, tuple[int, str, str]] = {}
    for i, (l, r) in enumerate(merges):
        result = l + r
        if result not in recipe:
            recipe[result] = (i, l, r)

    keep_tokens = set(seed_strings)
    keep_merge_idx: set[int] = set()
    worklist = list(seed_strings)
    while worklist:
        s = worklist.pop()
        rule = recipe.get(s)
        if rule is None:
            continue
        idx, l, r = rule
        keep_merge_idx.add(idx)
        for part in (l, r):
            if part not in keep_tokens:
                keep_tokens.add(part)
                worklist.append(part)
    return keep_tokens, keep_merge_idx


def select_vocab(reader: gguf.GGUFReader, model_path: Path, corpus_paths: list[Path]):
    """Run corpus-seeding + merge-closure against an open GGUFReader.

    Returns (kept_ids, old_to_new, new_tokens, new_token_type, new_merges),
    where kept_ids/new_tokens/new_token_type are index-aligned and in the
    original vocab's relative order, and new_merges is already re-rendered
    as "left right" strings ready to write back out.
    """
    model_field = reader.get_field(gguf.Keys.Tokenizer.MODEL)
    if model_field is None or model_field.contents() != "gpt2":
        raise ValueError(
            f"tokenizer.ggml.model = {model_field.contents() if model_field else None!r}, "
            "only gpt2-style byte-level BPE is supported"
        )

    tokens = reader.get_field(gguf.Keys.Tokenizer.LIST).contents()
    token_type = reader.get_field(gguf.Keys.Tokenizer.TOKEN_TYPE).contents()
    merges_raw = reader.get_field(gguf.Keys.Tokenizer.MERGES).contents()
    merges = [tuple(m.split(' ', 1)) for m in merges_raw]
    n_vocab = len(tokens)
    print(f"vocab: {n_vocab}  |  merges: {len(merges)}")

    seed_ids: set[int] = set()
    for corpus in corpus_paths:
        print(f"Tokenizing {corpus} ...")
        ids = tokenize_ids(model_path, corpus)
        seed_ids |= ids
        print(f"  {len(ids)} tokens, {len(seed_ids)} unique so far")

    # Always keep control/user-defined tokens (chat template, tool-call, etc.)
    # and the 256 base byte-fallback tokens, so arbitrary input outside the
    # corpus still encodes -- just possibly less efficiently.
    byte_encoder_chars = set(bytes_to_unicode().values())
    forced_ids = {
        i for i, t in enumerate(token_type)
        if t in (TokenType.CONTROL, TokenType.USER_DEFINED)
    }
    forced_ids |= {i for i, s in enumerate(tokens) if s in byte_encoder_chars}
    for key in SPECIAL_ID_KEYS:
        field = reader.get_field(key)
        if field is not None:
            forced_ids.add(int(field.contents()))

    seed_ids |= forced_ids
    seed_strings = {tokens[i] for i in seed_ids}
    print(f"Seed tokens (corpus + control/byte/special): {len(seed_strings)}")

    keep_strings, keep_merge_idx = merge_closure(seed_strings, merges)
    print(f"After merge-chain closure: {len(keep_strings)} tokens, {len(keep_merge_idx)}/{len(merges)} merges")

    kept_ids = [i for i, s in enumerate(tokens) if s in keep_strings]
    old_to_new = {old: new for new, old in enumerate(kept_ids)}
    n_keep = len(kept_ids)
    print(f"Vocab: {n_vocab} -> {n_keep} ({100*(n_vocab-n_keep)/n_vocab:.1f}% removed)")

    new_tokens = [tokens[i] for i in kept_ids]
    new_token_type = [int(token_type[i]) for i in kept_ids]
    new_merges = [f"{merges[i][0]} {merges[i][1]}" for i in sorted(keep_merge_idx)]

    return kept_ids, old_to_new, new_tokens, new_token_type, new_merges
