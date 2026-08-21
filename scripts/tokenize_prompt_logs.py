#!/usr/bin/env python3
"""Report real token counts for prompt-log JSON files.

Works on either corpus, since they share a schema:
  - data/logs/prompts/       raw API requests captured by data/prompt-logger/proxy.py
  - data/corpus/sessions/    Claude Code segments from extract_session_corpus.py

Why this exists: prompt_chars is a poor size proxy. Measured across 98 session
segments, chars/token has a median of 3.24 and a range of 1.41 to 4.01 -- so a
char threshold cannot reliably select "segments over 100k tokens". Neither can
compactMetadata.preTokens (see extract_session_corpus.py). Only tokenizing
settles it.

It doubles as the segment selector, since chars/token is also the density
signal: prose and code sit near 3-4, and well below that means base64 or
similar, which is not text and would distort a perplexity or copy-mass run.
That filter is not free -- density is inversely correlated with size in this
corpus, so it removes the largest segments first. Of the 98 measured, all 9
above 300k tokens and 9 of the 13 above 200k fall below 2.5 chars/token.

Text is rendered by extract_session_corpus.render_messages, so a count here and
a later llama-perplexity run refer to byte-identical text.

Tokenization runs llama-tokenize inside the running container by default: it is
built in the `full` image but not in build-test/, and it loads with
vocab_only=true, so it allocates no weights and does not touch the GPU. It is
safe to run against a live server. Text is piped over stdin, so the corpus does
not need to be under one of the container's mounts.

Note --no-escape: without it llama-tokenize expands backslash sequences in the
input, which would turn the literal "\\n" inside a captured tool-call argument
into a newline and quietly change the token count.

Usage:
  scripts/tokenize_prompt_logs.py data/corpus/sessions
  scripts/tokenize_prompt_logs.py data/logs/prompts --min-tokens 80000
  scripts/tokenize_prompt_logs.py data/corpus/sessions --tsv counts.tsv

  # select long, text-like segments and feed them to ppl-context-curve.sh
  scripts/tokenize_prompt_logs.py --from-tsv data/corpus/sessions-tokens.tsv \\
      --min-tokens 98304 --min-chars-per-token 2.5 --names-only --name-ext .txt
"""

from __future__ import annotations

import os
import sys
import json
import shutil
import argparse
import statistics
import subprocess
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_session_corpus import render_messages  # noqa: E402

DEFAULT_CONTAINER = "llama-cpp-turboquant-llama-cpp-1"
# Any GGUF with the right vocab works. The *.vocabPatch.gguf files do NOT load
# in vocab_only mode, so do not default to one.
DEFAULT_MODEL = "/models/Qwen3.6-35B-A3B-Claude-Opus-Distilled-MTP-UD-IQ3_XXS.gguf"


def build_argv(args):
    """Command that reads text on stdin and prints the token count last."""
    flags = ["-m", args.model, "--stdin", "--ids", "--show-count",
             "--log-disable", "--no-escape"]
    if args.binary:
        return [args.binary] + flags
    return ["docker", "exec", "-i", args.container, "/app/llama-tokenize"] + flags


def count_tokens(argv, text):
    """Return (count, None), or (None, error) if llama-tokenize failed.

    The error text is propagated because the interesting failures are
    environmental, not per-file: if the container restarts mid-run (it has
    restart: unless-stopped, and the server does exit on its own), every
    remaining file fails identically and a bare "failed" hides why.
    """
    proc = subprocess.run(argv, input=text, capture_output=True, text=True)
    if proc.returncode != 0:
        return None, (proc.stderr.strip().splitlines() or ["exit %d" % proc.returncode])[-1]
    for line in reversed(proc.stdout.strip().splitlines()):
        if "Total number of tokens" in line:
            return int(line.rsplit(":", 1)[1].strip()), None
    return None, "no token count in output"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path, nargs="?", default=None,
                    help="A prompt-log .json file, or a directory (omit when using --from-tsv)")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="GGUF to take the vocab from, as seen by whatever runs "
                         "llama-tokenize (default: %(default)s)")
    ap.add_argument("--container", default=DEFAULT_CONTAINER,
                    help="Running container to exec into (default: %(default)s)")
    ap.add_argument("--binary", default=None,
                    help="Run this local llama-tokenize instead of using Docker")
    ap.add_argument("--min-tokens", type=int, default=0,
                    help="Only list files at or above this count (summary still "
                         "covers everything scanned)")
    ap.add_argument("--min-chars-per-token", type=float, default=0.0,
                    help="Only list files at or above this chars/token. Normal prose and code sit "
                         "near 3-4; well below that means the text is not really text (base64 and "
                         "other blobs tokenize very densely) and would distort a perplexity or "
                         "copy-mass measurement. Note this correlates with size here -- the very "
                         "largest segments are the blob-heavy ones -- so it bites hardest exactly "
                         "where long-context samples are scarcest")
    ap.add_argument("--tsv", type=Path, default=None,
                    help="Write 'tokens<TAB>chars<TAB>file' here for later reuse")
    ap.add_argument("--from-tsv", type=Path, default=None,
                    help="Filter a --tsv written earlier instead of tokenizing again; makes "
                         "selection instant and guarantees it matches the measured counts")
    ap.add_argument("--names-only", action="store_true",
                    help="Print bare filenames of the selected files, for feeding another script")
    ap.add_argument("--name-ext", default=None,
                    help="With --names-only, swap each name's extension to this (e.g. .txt to "
                         "point at the renders from extract_session_corpus.py --text-dir)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Stop after this many files (quick sanity check)")
    args = ap.parse_args()

    rows = []
    failed = []

    if args.from_tsv:
        for line in args.from_tsv.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            n, chars, name = line.split("\t")
            rows.append((int(n), int(chars), None, None, name))
    else:
        if not args.source:
            sys.exit("a source path is required unless --from-tsv is given")
        if not args.binary and not shutil.which("docker"):
            sys.exit("docker not found; pass --binary /path/to/llama-tokenize")

        files = sorted(args.source.glob("*.json")) if args.source.is_dir() else [args.source]
        if args.limit:
            files = files[:args.limit]
        if not files:
            sys.exit(f"no .json files found in {args.source}")

        argv = build_argv(args)
        for i, path in enumerate(files, 1):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as e:
                failed.append((path.name, str(e)))
                continue
            text = render_messages(payload.get("messages", []))
            n, err = count_tokens(argv, text)
            if n is None:
                failed.append((path.name, err))
                continue
            rows.append((n, len(text), payload.get("prompt_chars"),
                         payload.get("_is_continuation"), path.name))
            print(f"\r  tokenizing {i}/{len(files)}", end="", file=sys.stderr, flush=True)
        print("\r" + " " * 40 + "\r", end="", file=sys.stderr)

    rows.sort(reverse=True)
    if args.tsv:
        args.tsv.write_text(
            "".join(f"{n}\t{c}\t{name}\n" for n, c, _pc, _cont, name in rows),
            encoding="utf-8")

    def selected(row):
        n, chars, _pc, _cont, _name = row
        return n >= args.min_tokens and (chars / n if n else 0) >= args.min_chars_per_token

    keep = [r for r in rows if selected(r)]

    if args.names_only:
        for _n, _c, _pc, _cont, name in keep:
            print(Path(name).with_suffix(args.name_ext) if args.name_ext else name)
        print(f"selected {len(keep)}/{len(rows)}", file=sys.stderr)
        return

    # a TSV carries no continuation flag, so show "?" rather than silently reporting every
    # segment as a fresh one
    print(f"{'TOKENS':>9} {'CHARS':>10} {'CH/TOK':>7} {'CONT':>5}  FILE")
    for n, chars, _pc, cont, name in keep:
        cont_s = "?" if cont is None else ("yes" if cont else "no")
        print(f"{n:>9,} {chars:>10,} {chars / n if n else 0:>7.2f} {cont_s:>5}  {name}")

    if len(keep) != len(rows):
        print(f"\nselected {len(keep)}/{len(rows)} "
              f"(min-tokens {args.min_tokens:,}, min-chars-per-token {args.min_chars_per_token})")
    print(f"\nfiles {'loaded' if args.from_tsv else 'tokenized'}: {len(rows)}"
          + (f"  (failed: {len(failed)})" if failed else ""))
    for name, why in failed[:10]:
        print(f"  !! {name}: {why}")
    if rows:
        ratios = [c / n for n, c, _pc, _cont, _f in rows if n]
        print(f"chars/token: median {statistics.median(ratios):.2f}  "
              f"range {min(ratios):.2f}-{max(ratios):.2f}")
        have_cont = any(r[3] is not None for r in rows)
        for t in (80_000, 100_000, 150_000, 200_000, 300_000):
            hit = [r for r in rows if r[0] >= t]
            if not hit:
                continue
            # a dense segment is mostly base64 or similar, not text; count them separately so the
            # usable long-context sample size is visible rather than the raw one
            dense = sum(1 for r in hit if r[0] and r[1] / r[0] < 2.5)
            extra = f"   (non-continuation: {sum(1 for r in hit if not r[3])})" if have_cont else ""
            print(f"  >= {t:>7,} tok: {len(hit):>4}   (below 2.5 ch/tok: {dense}){extra}")


if __name__ == "__main__":
    main()
