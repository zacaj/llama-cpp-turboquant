#!/usr/bin/env python3
"""Summarise entropy and copy mass by position, from --ppl-copy-window token dumps.

Answers the question NLL cannot: not "was the model right" but "what kind of prediction was it
making". Degenerate repetition scores very low perplexity, so a model drifting into copy mode is
invisible to perplexity and obvious here.

Columns reported per position bin:
  ppl        as usual, for reference
  entropy    of the full predicted distribution, in nats
  copy_near  probability mass on tokens present in the last copy_window tokens
  copy_far   mass on tokens seen earlier in the window but not recently
  copy_new   1 - near - far, i.e. mass on tokens not in the context at all

The number to watch is copy_near rising with position. That is the attention-radius collapse
story: as diffuse retrieval degrades, the induction circuit (prefix-match then copy) dominates,
so probability piles onto locally-recent tokens. copy_far rising instead means the model is
retrieving from distant context, which is the healthy pattern.

Use --blocks-dir to restrict to tokens the model actually generates. Without it the numbers are
dominated by tool output, which is ~56% of the corpus and is copy-heavy for legitimate reasons
(it genuinely is repeated file content), so it will show high copy mass whether or not anything
is wrong.

Statistics are clustered by segment, not token, for the same reason as merge_context_curve.py:
tokens within one session are correlated, so the effective sample size is the segment count.

Usage:
  scripts/merge_copy_mass.py data/corpus/copy-mass --bin 8192
  scripts/merge_copy_mass.py data/corpus/copy-mass --blocks-dir data/corpus/context-curve \\
      --kinds assistant/
"""

from __future__ import annotations

import sys
import math
import argparse
import statistics
from pathlib import Path

MIN_PROB = 1e-9


def load_dump(path):
    """Return {pos: (nll, entropy, copy_near, copy_far)}; None if the dump lacks copy columns."""
    rows = {}
    with open(path, encoding="utf-8") as fh:
        header = None
        for line in fh:
            if line.startswith("#"):
                continue
            if line.startswith("chunk"):
                header = line.rstrip("\n").split("\t")
                if "copy_near" not in header:
                    return None
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 8:
                continue
            rows[int(p[1])] = (-math.log(max(float(p[4]), MIN_PROB)),
                               float(p[5]), float(p[6]), float(p[7]))
    return rows


def load_kinds(path):
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith(("#", "pos")):
                continue
            pos, kind = line.rstrip("\n").split("\t")
            out[int(pos)] = kind
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path, help="Directory of *.deep.tsv with copy columns")
    ap.add_argument("--bin", type=int, default=8192, help="Position bin width (default: %(default)s)")
    ap.add_argument("--blocks-dir", type=Path, default=None,
                    help="Directory of *.blocks.tsv from tag_token_blocks.py")
    ap.add_argument("--kinds", default=None,
                    help="Block-kind prefixes to keep, e.g. 'assistant/' (needs --blocks-dir)")
    ap.add_argument("--tsv", type=Path, default=None, help="Write the table here")
    ap.add_argument("--min-segments", type=int, default=3)
    args = ap.parse_args()

    if args.kinds and not args.blocks_dir:
        sys.exit("--kinds needs --blocks-dir")

    files = sorted(args.source.glob("*.deep.tsv"))
    if not files:
        sys.exit(f"no *.deep.tsv in {args.source}")

    # per_seg[bin][segment] = [sum_nll, sum_ent, sum_near, sum_far, n]
    per_seg = {}
    used = skipped = 0

    for path in files:
        name = path.name[:-len(".deep.tsv")]
        rows = load_dump(path)
        if rows is None:
            print(f"  !! {name}: no copy columns, rerun with --ppl-copy-window", file=sys.stderr)
            skipped += 1
            continue
        if not rows:
            print(f"  !! {name}: empty dump", file=sys.stderr)
            skipped += 1
            continue

        keep = rows.keys()
        if args.kinds:
            kpath = args.blocks_dir / f"{name}.blocks.tsv"
            if not kpath.exists():
                print(f"  !! {name}: no .blocks.tsv, skipping (would mix excluded kinds in)",
                      file=sys.stderr)
                skipped += 1
                continue
            kinds = load_kinds(kpath)
            want = tuple(k.strip() for k in args.kinds.split(","))
            keep = [p for p in rows if kinds.get(p, "").startswith(want)]
            if not keep:
                print(f"  !! {name}: no tokens of the requested kinds", file=sys.stderr)
                skipped += 1
                continue

        used += 1
        for p in keep:
            nll, ent, near, far = rows[p]
            acc = per_seg.setdefault(p // args.bin, {}).setdefault(name, [0.0, 0.0, 0.0, 0.0, 0])
            acc[0] += nll
            acc[1] += ent
            acc[2] += near
            acc[3] += far
            acc[4] += 1

    if not used:
        sys.exit("no usable dumps")
    print(f"segments: {used}" + (f" (skipped {skipped})" if skipped else ""), file=sys.stderr)

    out = []
    for b in sorted(per_seg):
        segs = per_seg[b]
        n = sum(v[4] for v in segs.values())
        means = [sum(v[i] for v in segs.values()) / n for i in range(4)]
        # spread across per-segment means, not across tokens
        near_by_seg = [v[2] / v[4] for v in segs.values()]
        se = (statistics.stdev(near_by_seg) / math.sqrt(len(near_by_seg))
              if len(near_by_seg) >= args.min_segments else None)
        out.append((b * args.bin, len(segs), n, math.exp(means[0]), means[1],
                    means[2], means[3], 1.0 - means[2] - means[3], se))

    print(f"{'pos':>9} {'segs':>5} {'tokens':>10} {'ppl':>7} {'entropy':>8} "
          f"{'copy_near':>10} {'copy_far':>9} {'copy_new':>9} {'near_se':>8}")
    for pos, nseg, ntok, ppl, ent, near, far, new, se in out:
        se_s = f"{se:8.4f}" if se is not None else "       -"
        print(f"{pos:>9,} {nseg:>5} {ntok:>10,} {ppl:>7.2f} {ent:>8.3f} "
              f"{near:>10.4f} {far:>9.4f} {new:>9.4f} {se_s}")

    if args.tsv:
        with open(args.tsv, "w", encoding="utf-8") as fh:
            fh.write("pos\tsegments\ttokens\tppl\tentropy\tcopy_near\tcopy_far\tcopy_new\tnear_se\n")
            for pos, nseg, ntok, ppl, ent, near, far, new, se in out:
                fh.write(f"{pos}\t{nseg}\t{ntok}\t{ppl:.6f}\t{ent:.6f}\t{near:.6f}\t"
                         f"{far:.6f}\t{new:.6f}\t{'' if se is None else format(se, '.6f')}\n")
        print(f"\nwrote {args.tsv}", file=sys.stderr)


if __name__ == "__main__":
    main()
