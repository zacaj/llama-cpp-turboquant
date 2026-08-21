#!/usr/bin/env python3
"""Extract Claude Code .jsonl session logs into prompt-logger-shaped JSON files.

The prompt-logger corpus (data/logs/prompts/, see data/prompt-logger/proxy.py)
records raw API requests as they went over the wire, which makes it the ground
truth for "what was actually in the context window" -- but it is capped by what
this machine can serve, so nothing in it exceeds ~150k tokens, and after
grouping away prefix-extension duplicates it yields only ~13 distinct sessions.

Claude Code's own session logs reach much further (Sonnet/Opus run to 1M), so
this script converts them into the same schema, letting both sources feed the
same downstream tooling.

THE FILE IS A TREE, NOT A LOG
-----------------------------
Reading a .jsonl linearly is wrong, and wrong in the worst possible way for
this purpose. Resuming a session replays earlier records verbatim into the same
file: one measured session had 6984 records carrying only 3649 distinct uuids,
so ~48% of a linear read is duplicated text. Rewinding forks the history, and
both branches persist. Concatenating all that would manufacture enormous fake
long-range repetition -- precisely the signal a long-context probe is trying to
measure.

Records form a tree via uuid/parentUuid. This script dedupes by uuid, walks the
tree, and reconstructs each root-to-leaf path. Where one path is a prefix of
another (a rewind and its continuation) only the longest is kept.

SEGMENTS, NOT SESSIONS
----------------------
A .jsonl file is also not one context window. Claude Code compacts when the
context fills: the model's context is replaced by a summary and the
conversation continues. Conveniently this falls out of the tree walk for free,
because each compact_boundary record has a null parentUuid and therefore roots
its own subtree -- so every root-to-leaf path IS exactly one context window.

Segments rooted at a boundary are flagged `is_continuation`: they open with a
dense summary rather than an organic session start, a systematically different
prefix distribution that should be analysed separately, and their topic overlap
with the parent segment means the two are not independent samples.

A boundary record's compactMetadata.preTokens is linked back to the segment
that ended there (via the boundary's logicalParentUuid) and reported as
`pre_tokens`. DO NOT use it to size or select segments: measured against real
llama-tokenize counts over 43 segments, real/preTokens has a median of 0.49 and
a range of 0.34-11.62. It disagrees in both directions with no usable
correction factor, so the two are evidently not counting the same thing. Use
--text-dir and tokenize the rendered text instead (see
scripts/tokenize_prompt_logs.py).

KNOWN FIDELITY LIMITS
---------------------
This is a faithful reconstruction, not a byte-exact replay. Specifically:
  - The system prompt is not stored in the .jsonl at all. Tool definitions and
    CLAUDE.md -- 10-20k tokens of prefix on a real request -- are missing.
  - Claude Code clears old tool results mid-segment without leaving a boundary
    record, so a rebuilt segment can contain text the model no longer had. The
    rendered size therefore OVERSHOOTS pre_tokens, sometimes by ~2x. Compare
    `prompt_chars` against `pre_tokens` to see the gap on any given segment.
  - Sidechain (subagent) records live in the same file but were never in the
    main thread's context; they are dropped unless --keep-sidechain.
  - Thinking-block signatures are long opaque base64 that carries no
    predictable structure; stripped unless --keep-signatures.

`prompt_chars` replicates proxy.py's _estimate_prompt_chars exactly, so the
field means the same thing in both corpora.

Usage:
  python scripts/extract_session_corpus.py <jsonl-or-dir> <out-dir> [--min-chars N]
"""

from __future__ import annotations

import re
import sys
import json
import argparse
from pathlib import Path

# Only these record types carry a message the model actually saw. Everything
# else in the log (attachment, file-history-snapshot, queue-operation,
# last-prompt, custom-title, mode, permission-mode, ai-title) is harness
# bookkeeping -- some of it enormous, none of it context.
MESSAGE_TYPES = {"user", "assistant"}

# Opening line Claude Code injects as the first user turn of a post-compaction
# segment. Used as a fallback when the boundary record itself is missing.
CONTINUATION_RE = re.compile(
    r"^This session is being continued from a previous conversation that ran out of context",
)


def estimate_prompt_chars(messages):
    """Byte-for-byte port of proxy.py's ProxyHandler._estimate_prompt_chars.

    Kept deliberately identical so prompt_chars is comparable across the two
    corpora; do not "improve" it without re-deriving the prompt-logger data.
    """
    total = 0
    for m in messages:
        if isinstance(m, dict):
            c = m.get("content", "")
            total += len(c) if isinstance(c, str) else len(json.dumps(c))
    return total


def clean_content(content, keep_signatures, strip_thinking):
    """Normalise a message's content, dropping blocks that are noise for PPL."""
    if not isinstance(content, list):
        return content
    out = []
    for block in content:
        if not isinstance(block, dict):
            out.append(block)
            continue
        if strip_thinking and block.get("type") == "thinking":
            continue
        if not keep_signatures and "signature" in block:
            block = {k: v for k, v in block.items() if k != "signature"}
        out.append(block)
    return out


def block_text(block):
    """Flatten one content block to the text a model would have seen."""
    if not isinstance(block, dict):
        return str(block)
    btype = block.get("type")
    if btype == "thinking":
        return str(block.get("thinking", ""))
    if btype == "tool_use":
        return json.dumps(block.get("input", ""), ensure_ascii=False)
    if btype == "tool_result":
        result = block.get("content")
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    if "text" in block:
        return str(block.get("text", ""))
    return ""


def render_messages(messages):
    """Render a prompt-log message list to plain text for tokenize/perplexity.

    Deliberately plain -- role headers and content, no chat template. The point
    is to measure how the model handles this text at depth, not to reproduce a
    specific harness's framing, and a template would inject control tokens that
    perplexity would then score as if they were prose.

    Shared with scripts/tokenize_prompt_logs.py so that a token count and a
    later perplexity run refer to byte-identical text.
    """
    return render_spans(messages)[0]


def render_spans(messages):
    """Render as render_messages does, and report what each character range is.

    Returns (text, spans) where spans is a list of (start, end, kind) with kind one of
    role/blocktype pairs like "assistant/thinking" or "user/tool_result", plus "header" for the
    role lines and separators. render_messages is defined in terms of this so the two cannot
    drift; a span table that disagreed with the scored text by even one character would
    mislabel everything after it.

    This matters because the corpus is not evenly made of these: measured over the session
    segments, tool_result is ~56% of characters and tool_use ~21%, while model-authored
    reasoning is ~8% and prose ~5%. A number averaged over all tokens is therefore mostly a
    statement about predicting command output, not about reasoning.
    """
    parts = []
    spans = []
    pos = 0

    def emit(s, kind):
        nonlocal pos
        if not s:
            return
        parts.append(s)
        spans.append((pos, pos + len(s), kind))
        pos += len(s)

    for i, m in enumerate(messages):
        if i:
            emit("\n\n", "header")
        role = str(m.get("role", "?"))
        emit(f"{role.upper()}:\n", "header")

        content = m.get("content")
        if isinstance(content, str):
            emit(content, f"{role}/str")
        elif isinstance(content, list):
            for j, b in enumerate(content):
                if j:
                    emit("\n", "header")
                btype = b.get("type", "?") if isinstance(b, dict) else "raw"
                emit(block_text(b), f"{role}/{btype}")
        else:
            emit(str(content), f"{role}/str")

    return "".join(parts), spans


def iter_records(path):
    """Yield parsed records in file order, skipping unparseable lines."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def split_segments(path, keep_sidechain, keep_signatures, strip_thinking):
    """Reconstruct one .jsonl's context windows by walking its uuid tree.

    Returns (session_id, segments). Each segment is one root-to-leaf path,
    i.e. one real context window; see the module docstring on why a linear
    read is not an acceptable shortcut here.
    """
    recs = {}
    order = []
    session_id = None
    for rec in iter_records(path):
        session_id = session_id or rec.get("sessionId")
        uuid = rec.get("uuid")
        if not uuid or uuid in recs:
            # First occurrence wins; later ones are resume-replays of it.
            continue
        recs[uuid] = rec
        order.append(uuid)

    children = {}
    for uuid in order:
        children.setdefault(recs[uuid].get("parentUuid"), []).append(uuid)

    # preTokens lives on the boundary that FOLLOWS a segment, and points back
    # at that segment's last record through logicalParentUuid.
    tail_meta = {}
    for uuid in order:
        rec = recs[uuid]
        if rec.get("subtype") != "compact_boundary":
            continue
        tail = rec.get("logicalParentUuid")
        if tail:
            tail_meta[tail] = rec.get("compactMetadata") or {}

    # One context window per ROOT, not per leaf. Rewinds fork the trunk, so a
    # single epoch can end in dozens of sibling leaves that share >95% of their
    # records; emitting each would flood the corpus with near-duplicates. The
    # real final context is the deepest path, tie-broken by whichever leaf was
    # written last.
    #
    # Candidate endpoints are the tree's leaves PLUS every compaction tail. A
    # tail is the exact point where the context hit its limit, so when a root
    # has one it beats any leaf: it is the only endpoint with a known token
    # count, and later forks off that trunk are alternate futures, not context.
    rank = {uuid: i for i, uuid in enumerate(order)}
    endpoints = [u for u in order if u not in children]
    endpoints += [u for u in tail_meta if u in recs]  # tails usually have children

    best = {}
    for uuid in endpoints:
        path_uuids = []
        cur = uuid
        seen = set()
        while cur is not None and cur in recs and cur not in seen:
            seen.add(cur)
            path_uuids.append(cur)
            cur = recs[cur].get("parentUuid")
        path_uuids.reverse()
        root = path_uuids[0]
        key = (uuid in tail_meta, len(path_uuids), rank[uuid])
        if root not in best or key > best[root][0]:
            best[root] = (key, path_uuids)

    segments = []
    for _key, path_uuids in sorted(best.values(), key=lambda b: rank[b[1][0]]):
        messages = []
        timestamp = None
        for uuid in path_uuids:
            rec = recs[uuid]
            if rec.get("type") not in MESSAGE_TYPES:
                continue
            if rec.get("isSidechain") and not keep_sidechain:
                continue
            msg = rec.get("message")
            if not isinstance(msg, dict) or msg.get("role") not in ("user", "assistant"):
                continue
            if timestamp is None:
                timestamp = rec.get("timestamp")
            messages.append({
                "role": msg["role"],
                "content": clean_content(msg.get("content"), keep_signatures, strip_thinking),
            })
        if not messages:
            continue

        meta = tail_meta.get(path_uuids[-1], {})
        root = recs[path_uuids[0]]
        segments.append({
            "messages": messages,
            "pre_tokens": meta.get("preTokens"),
            "trigger": meta.get("trigger"),
            "timestamp": timestamp,
            "is_continuation": root.get("subtype") == "compact_boundary",
        })

    segments.sort(key=lambda s: s["timestamp"] or "")
    return session_id or Path(path).stem, segments


def first_text(messages):
    """Leading text of the first user message, for continuation detection."""
    for m in messages:
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str):
            return c[:200]
        if isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and isinstance(b.get("text"), str):
                    return b["text"][:200]
        return ""
    return ""


def emit(out_dir, session_id, index, seg, model, source, text_dir=None):
    messages = seg["messages"]
    chars = estimate_prompt_chars(messages)
    ts = seg["timestamp"] or ""
    stamp = re.sub(r"[^0-9]", "", ts)[:14] or "00000000000000"
    name = f"{stamp}_{session_id[:8]}_s{index:02d}.json"

    payload = {
        "timestamp": ts,
        "endpoint": "/v1/messages",
        "client": "claude-code-jsonl",
        "prompt_chars": chars,
        "model": model,
        "stream": True,
        "messages": messages,
        # Underscore-prefixed so it cannot collide with real request fields.
        "_source": str(source),
        "_session_id": session_id,
        "_segment": index,
        "_pre_tokens": seg["pre_tokens"],
        "_compact_trigger": seg["trigger"],
        # Tree structure is authoritative; the text marker is a fallback for
        # segments whose boundary record did not survive.
        "_is_continuation": seg["is_continuation"] or bool(CONTINUATION_RE.match(first_text(messages))),
        "_n_messages": len(messages),
    }
    (out_dir / name).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    if text_dir is not None:
        (text_dir / name.replace(".json", ".txt")).write_text(
            render_messages(messages), encoding="utf-8")
    return name, chars, seg["pre_tokens"], payload["_is_continuation"]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path, help="A .jsonl file, or a directory to walk")
    ap.add_argument("out_dir", type=Path, help="Output directory for segment JSON files")
    ap.add_argument("--min-chars", type=int, default=300_000,
                    help="Skip segments smaller than this many prompt_chars "
                         "(default: 300000, roughly 80k tokens)")
    ap.add_argument("--exclude", action="append", default=["-tmp"],
                    help="Skip paths containing this substring (repeatable; "
                         "defaults to '-tmp', which holds compaction jobs)")
    ap.add_argument("--model", default="claude-code-session",
                    help="Value for the 'model' field (default: claude-code-session)")
    ap.add_argument("--keep-sidechain", action="store_true",
                    help="Keep subagent records that were never in the main context")
    ap.add_argument("--keep-signatures", action="store_true",
                    help="Keep opaque base64 thinking-block signatures")
    ap.add_argument("--strip-thinking", action="store_true",
                    help="Drop thinking blocks entirely (they were in context; "
                         "only use this if measuring non-reasoning behaviour)")
    ap.add_argument("--text-dir", type=Path, default=None,
                    help="Also write each segment as plain .txt here, for "
                         "llama-tokenize / llama-perplexity. Must be under a "
                         "path the container mounts if you feed it to Docker.")
    ap.add_argument("--dry-run", action="store_true", help="Report only, write nothing")
    args = ap.parse_args()

    if args.source.is_dir():
        files = sorted(p for p in args.source.rglob("*.jsonl")
                       if not any(x in str(p) for x in args.exclude))
    else:
        files = [args.source]
    if not files:
        sys.exit(f"no .jsonl files found under {args.source}")

    if not args.dry_run:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        if args.text_dir:
            args.text_dir.mkdir(parents=True, exist_ok=True)

    kept = skipped = 0
    rows = []
    for path in files:
        try:
            session_id, segments = split_segments(
                path, args.keep_sidechain, args.keep_signatures, args.strip_thinking)
        except OSError as e:
            print(f"  !! {path}: {e}", file=sys.stderr)
            continue
        for i, seg in enumerate(segments):
            chars = estimate_prompt_chars(seg["messages"])
            if chars < args.min_chars:
                skipped += 1
                continue
            kept += 1
            if args.dry_run:
                stamp = re.sub(r"[^0-9]", "", seg["timestamp"] or "")[:14] or "0" * 14
                rows.append((f"{stamp}_{session_id[:8]}_s{i:02d}.json", chars,
                             seg["pre_tokens"], seg["is_continuation"]))
            else:
                rows.append(emit(args.out_dir, session_id, i, seg, args.model, path,
                                 args.text_dir))

    rows.sort(key=lambda r: -r[1])
    print(f"{'PROMPT_CHARS':>13} {'PRE_TOKENS':>11} {'CONT':>5}  FILE")
    for name, chars, pre, cont in rows[:40]:
        print(f"{chars:>13,} {(pre if pre is not None else -1):>11} "
              f"{'yes' if cont else 'no':>5}  {name}")
    if len(rows) > 40:
        print(f"  ... and {len(rows) - 40} more")

    exact = [r[2] for r in rows if r[2]]
    print(f"\nfiles scanned: {len(files)}")
    print(f"segments kept: {kept}  (skipped below --min-chars: {skipped})")
    print(f"continuations: {sum(1 for r in rows if r[3])} of {kept}")
    if exact:
        for t in (80_000, 100_000, 150_000, 200_000, 300_000):
            print(f"  kept segments with exact pre_tokens >= {t:>7,}: "
                  f"{sum(1 for v in exact if v >= t)}")
    if args.dry_run:
        print("\n(dry run -- nothing written)")


if __name__ == "__main__":
    main()
