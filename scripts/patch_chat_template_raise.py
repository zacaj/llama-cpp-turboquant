#!/usr/bin/env python3
"""
Neutralize the brittle raise_exception() assertions in a GGUF's embedded chat
template, TRUE IN PLACE -- zero extra disk space, no tensor data touched.

Background: the whole Qwen3.5/3.6 chat-template lineage (and every finetune
that copies its tokenizer_config.json chat_template verbatim -- e.g.
Ornith-1.0-35B) contains two hard asserts:

    {%- if ns.multi_step_tool %}
        {{- raise_exception('No user query found in messages.') }}
    {%- endif %}

    {%- if message.role == "system" %}
        {%- if not loop.first %}
            {{- raise_exception('System message must be at the beginning.') }}
        {%- endif %}

Runtimes that probe the template with synthetic messages, or that inject
their own system/reminder messages mid-conversation (Claude Code does this),
trip these and the whole request dies with a 400 before generating a token.
See:
  https://github.com/ggml-org/llama.cpp/issues/20733
  https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B/discussions/10

This targets only those two textually-distinctive asserts. It is NOT a
general Jinja rewriter: templates from other lineages (Gemma, Mistral, GLM,
Llama, ...) don't contain this pattern and are left untouched -- the script
says so plainly instead of guessing.

How the in-place trick works: rather than deleting the offending lines
(which would shrink the chat_template string and shift every byte after it
in the file -- metadata, tensor index, and all tensor data), each guard
condition ("not loop.first", "ns.multi_step_tool") is rewritten to a
same-length, always-false expression ("false" padded with trailing spaces).
Because the replacement is byte-for-byte identical in length, the string's
length prefix never changes, so nothing else in the GGUF needs to move.
GGUFReader opened in 'r+' mode memory-maps the file, and the chat_template
field's byte array is a direct view into that mapping -- writing into it
patches the bytes on disk immediately, in place. No copy of the file is
ever made, and no tensor is read.

Usage:
    ./scripts/patch_chat_template_raise.py model.gguf --dry-run
    ./scripts/patch_chat_template_raise.py model.gguf
    ./scripts/patch_chat_template_raise.py model.gguf --force
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path

if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent / "gguf-py").exists():
    sys.path.insert(0, str(Path(__file__).parent.parent / "gguf-py"))

import numpy as np  # noqa: E402
import gguf  # noqa: E402

logger = logging.getLogger("patch-chat-template-raise")

# Each pattern captures the guard condition (group 1) inside a distinctive,
# fully-anchored raise_exception block, so we only ever touch a condition
# that is actually guarding one of these two known-bad asserts.
BLOCK_PATCHES: list[tuple[str, re.Pattern]] = [
    (
        "no-user-query assert",
        re.compile(
            r"\{%-\s*if\s+(ns\.multi_step_tool)\s*%\}\s*"
            r"\{\{-\s*raise_exception\(\s*['\"]No user query found in messages\.['\"]\s*\)\s*\}\}\s*"
            r"\{%-\s*endif\s*%\}",
            re.DOTALL,
        ),
    ),
    (
        "system-message-order assert",
        re.compile(
            r"\{%-\s*if\s+message\.role\s*==\s*[\"']system[\"']\s*%\}\s*"
            r"\{%-\s*if\s+(not\s+loop\.first)\s*%\}\s*"
            r"\{\{-\s*raise_exception\(\s*['\"]System message must be at the beginning\.['\"]\s*\)\s*\}\}\s*"
            r"\{%-\s*endif\s*%\}",
            re.DOTALL,
        ),
    ),
]


def falsify(text: str) -> str:
    """Same-length replacement for a Jinja boolean expression that always evaluates false."""
    n = len(text)
    if n >= 5:
        return "false" + " " * (n - 5)
    return "0" * n  # degenerate fallback for an implausibly short condition, still falsy


def neutralize(template: str) -> tuple[str, list[str]]:
    applied: list[str] = []
    out = template
    for label, pattern in BLOCK_PATCHES:
        def repl(m: re.Match, label: str = label) -> str:
            applied.append(label)
            cond = m.group(1)
            return m.group(0).replace(cond, falsify(cond), 1)
        out = pattern.sub(repl, out)
    return out, applied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path, help="GGUF file to patch in place")
    parser.add_argument("--dry-run", action="store_true", help="Only report what would change, write nothing")
    parser.add_argument("--force", action="store_true", help="Patch without confirmation")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    mode = "r" if args.dry_run else "r+"
    logger.info(f"* Opening ({mode}): {args.model}")
    reader = gguf.GGUFReader(args.model, mode)

    field = reader.get_field(gguf.Keys.Tokenizer.CHAT_TEMPLATE)
    if field is None:
        logger.error(f"! No {gguf.Keys.Tokenizer.CHAT_TEMPLATE} field found in {args.model}, nothing to patch")
        sys.exit(1)

    byte_view = field.parts[field.data[0]]
    old_template = bytes(byte_view).decode("utf-8")

    new_template, applied = neutralize(old_template)

    if not applied:
        logger.info("- No known-bad pattern found in this template. Not touching the file.")
        logger.info("  (This only targets the Qwen3.5/3.6-lineage raise_exception asserts;")
        logger.info("   other template families are expected to hit this message.)")
        sys.exit(0)

    new_bytes = new_template.encode("utf-8")
    if len(new_bytes) != len(byte_view):
        # Should be unreachable given falsify() always preserves length, but
        # this is the one invariant that makes the in-place write safe.
        logger.error(
            f"! BUG: replacement is {len(new_bytes)} bytes, original is {len(byte_view)} bytes. "
            "Refusing to write -- would corrupt the file."
        )
        sys.exit(1)

    logger.info(f"* Found and neutralized: {', '.join(applied)}")
    logger.info(f"* Template size unchanged: {len(byte_view)} bytes (safe for in-place write)")

    if args.dry_run:
        logger.info("* --dry-run set, not writing anything")
        sys.exit(0)

    if not args.force:
        logger.warning(f"* About to modify '{args.model}' in place.")
        logger.warning("* Enter exactly YES if you are positive you want to proceed:")
        if input("YES, I am sure> ") != "YES":
            logger.info("Aborted.")
            sys.exit(0)

    byte_view[:] = np.frombuffer(new_bytes, dtype=np.uint8)
    reader.data.flush()
    logger.info("* Patched in place. 0 extra bytes of disk used, no tensor data touched.")


if __name__ == "__main__":
    main()
