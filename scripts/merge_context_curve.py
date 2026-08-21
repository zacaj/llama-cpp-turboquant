#!/usr/bin/env python3
"""Merge deep/shallow --ppl-token-dump pairs into a context-gain curve.

Input is a directory of pairs written by ppl-context-curve.sh:
  <name>.deep.tsv     one whole window, every token scored, depth grows with position
  <name>.shallow.tsv  sliding window, depth bounded by the shallow context size

Both arms score the same tokens of the same document, so subtracting their NLL cancels content
difficulty and leaves the value of the context the shallow arm did not have. The deep arm's own
curve is not a usable answer on its own: later parts of a real session are more repetitive than
earlier ones, and that alone moves NLL more than context depth does.

Statistics are clustered by segment, not by token. Tokens inside one session are heavily
correlated, so pooling them would report a standard error some orders of magnitude too small; the
effective sample size is the number of segments. Each segment contributes one mean gap per bin,
and the reported stderr is taken across those per-segment means. Bins covered by fewer than 3
segments get no stderr, since there is nothing meaningful to take a spread over.

Only positions present in both arms are used, and only where the deep arm actually had at least as
much context as the shallow one. Two things trim the start of the curve: the shallow arm never
scores the first (ctx - stride) tokens of a document, and before roughly the shallow window size
the deep arm is the shallower of the two, so its "gain" there is negative by construction. Both are
expected, and the number dropped for the second reason is reported.

Usage:
  scripts/merge_context_curve.py data/corpus/context-curve
  scripts/merge_context_curve.py data/corpus/context-curve --bin 8192 --tsv curve.tsv
"""

from __future__ import annotations

import sys
import math
import argparse
import statistics
from pathlib import Path

# -log(prob) blows up if a probability underflowed float32 to exactly zero. clamp rather than drop,
# so a pathological token cannot silently shrink a bin's count, and report how often it happened.
MIN_PROB = 1e-9

# coarse depth bands for the by-kind table. finer bins would split each block type too thin to
# read, and the interesting structure here is which side of zero a kind sits on, not its shape.
BANDS = [("<33k", 0, 32768), ("33-57k", 32768, 57344), (">57k", 57344, 1 << 40)]


def band_of(pos):
    for name, lo, hi in BANDS:
        if lo <= pos < hi:
            return name
    return BANDS[-1][0]


def load_dump(path):
    """Return (meta, {pos: (depth, token, nll)}) for one dump, plus a clipped-token count."""
    rows = {}
    meta = ""
    clipped = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                meta = line[1:].strip()
                continue
            if line.startswith("chunk"):
                continue
            _chunk, pos, depth, token, prob = line.split()
            p = float(prob)
            if p < MIN_PROB:
                p = MIN_PROB
                clipped += 1
            rows[int(pos)] = (int(depth), int(token), -math.log(p))
    return meta, rows, clipped


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path, help="Directory holding *.deep.tsv / *.shallow.tsv pairs")
    ap.add_argument("--bin", type=int, default=4096, help="Position bin width (default: %(default)s)")
    ap.add_argument("--tsv", type=Path, default=None, help="Write the merged curve here")
    ap.add_argument("--kinds", default=None,
                    help="Comma-separated block-kind prefixes to restrict the curve to, e.g. "
                         "'assistant/'. Use this for the headline number: tool_result and user "
                         "text are inputs, prefilled into the context at inference, so the model "
                         "never generates them and its loss on them is never exercised. They are "
                         "also 60%% of tokens and behave oppositely, so leaving them in answers a "
                         "different question than the one usually being asked")
    ap.add_argument("--by-kind", action="store_true",
                    help="Also break the curve down by block type, using the *.blocks.tsv written "
                         "by tag_token_blocks.py. Strongly recommended before drawing any "
                         "conclusion from the pooled curve: tool_result is ~56%% of tokens and "
                         "behaves opposite to everything else, so it dominates the mean")
    ap.add_argument("--min-segments", type=int, default=3,
                    help="Bins below this segment count report no stderr (default: %(default)s)")
    args = ap.parse_args()

    deep_files = sorted(args.source.glob("*.deep.tsv"))
    if not deep_files:
        sys.exit(f"no *.deep.tsv found in {args.source}")

    # per_seg[bin][segment] = (sum_deep, sum_shallow, n)
    per_seg = {}
    depths = []
    zero_deep = []
    zero_shal = []
    total_clipped = 0
    dropped_shallower = 0
    dropped_kind = 0
    used = 0
    by_kind = {}

    for deep_path in deep_files:
        name = deep_path.name[:-len(".deep.tsv")]
        shallow_path = args.source / f"{name}.shallow.tsv"
        if not shallow_path.exists():
            print(f"  !! {name}: no shallow arm, skipping", file=sys.stderr)
            continue

        _dm, deep, c1 = load_dump(deep_path)
        _sm, shallow, c2 = load_dump(shallow_path)
        total_clipped += c1 + c2

        common = deep.keys() & shallow.keys()
        if not common:
            print(f"  !! {name}: arms share no positions, skipping", file=sys.stderr)
            continue

        mismatched = sum(1 for p in common if deep[p][1] != shallow[p][1])
        if mismatched:
            # the arms tokenized the same file, so this can only mean the dumps are misaligned;
            # averaging them would be meaningless rather than merely noisy
            print(f"  !! {name}: {mismatched} token ids disagree between arms, skipping",
                  file=sys.stderr)
            continue

        # near the start of a document the deep arm has fewer tokens behind it than the shallow
        # arm's fixed window, so it is not the deeper of the two and the gap inverts. drop those
        # rather than let them land in the first bin as a large negative that reads like a result.
        shallower = {p for p in common if deep[p][0] < shallow[p][0]}
        dropped_shallower += len(shallower)
        common -= shallower
        if not common:
            print(f"  !! {name}: deep arm is never deeper than the shallow arm, skipping",
                  file=sys.stderr)
            continue

        # zero point: where the two arms happen to have the same depth they saw the same context,
        # so their gap must be ~0. anything else means the join is misaligned. this is checked
        # separately rather than read off the first bin, since a bin usually straddles the boundary
        # and mixes matched positions with genuinely deeper ones.
        equal = {p for p in common if deep[p][0] == shallow[p][0]}
        for p in equal:
            zero_deep.append(deep[p][2])
            zero_shal.append(shallow[p][2])
        # keep those out of every table below. their gap is zero by construction, so leaving them
        # in drags results toward zero -- they are the check, not the measurement.
        common -= equal

        if args.by_kind or args.kinds:
            kpath = args.source / f"{name}.blocks.tsv"
            if not kpath.exists():
                # with --kinds this segment cannot be filtered, and letting it through unfiltered
                # would silently mix the excluded kinds back into the headline curve
                print(f"  !! {name}: no .blocks.tsv, skipping" if args.kinds else
                      f"  !! {name}: no .blocks.tsv, excluded from the by-kind table",
                      file=sys.stderr)
                if args.kinds:
                    continue
            else:
                kinds = {}
                with open(kpath, encoding="utf-8") as fh:
                    for line in fh:
                        if line.startswith(("#", "pos")):
                            continue
                        pos, kind = line.rstrip("\n").split("\t")
                        kinds[int(pos)] = kind
                if args.kinds:
                    want = tuple(k.strip() for k in args.kinds.split(","))
                    keep = {p for p in common if kinds.get(p, "").startswith(want)}
                    dropped_kind += len(common) - len(keep)
                    common = keep
                    if not common:
                        print(f"  !! {name}: no tokens of the requested kinds, skipping",
                              file=sys.stderr)
                        continue
                for p in common:
                    k = kinds.get(p)
                    if k is None:
                        continue
                    acc = by_kind.setdefault(k, {}).setdefault(band_of(p), [0.0, 0.0, 0])
                    acc[0] += deep[p][2]
                    acc[1] += shallow[p][2]
                    acc[2] += 1

        depths += [shallow[p][0] for p in common]
        used += 1

        for p in common:
            b = p // args.bin
            acc = per_seg.setdefault(b, {}).setdefault(name, [0.0, 0.0, 0])
            acc[0] += deep[p][2]
            acc[1] += shallow[p][2]
            acc[2] += 1

    if not used:
        sys.exit("no usable segment pairs")

    print(f"segments merged: {used}/{len(deep_files)}", file=sys.stderr)
    if depths:
        print(f"shallow arm depth: {min(depths)}-{max(depths)} tokens", file=sys.stderr)
    if total_clipped:
        print(f"probabilities clamped to {MIN_PROB:g}: {total_clipped}", file=sys.stderr)
    if dropped_shallower:
        print(f"dropped {dropped_shallower:,} early positions where the deep arm had less "
              f"context than the shallow arm", file=sys.stderr)
    if zero_deep:
        z = statistics.fmean(zero_shal) - statistics.fmean(zero_deep)
        flag = "" if abs(z) < 0.01 else "   <-- SUSPECT, arms may be misaligned"
        print(f"zero point ({len(zero_deep)} equal-depth positions): gap {z:+.4f}{flag}",
              file=sys.stderr)
    else:
        print("zero point: no equal-depth positions to check against", file=sys.stderr)

    out = []
    for b in sorted(per_seg):
        segs = per_seg[b]
        gaps = [(s / n) - (d / n) for d, s, n in segs.values()]
        n_tok = sum(v[2] for v in segs.values())
        deep_m = sum(v[0] for v in segs.values()) / n_tok
        shal_m = sum(v[1] for v in segs.values()) / n_tok
        gap = statistics.fmean(gaps)
        se = (statistics.stdev(gaps) / math.sqrt(len(gaps))
              if len(gaps) >= args.min_segments else None)
        out.append((b * args.bin, len(segs), n_tok, deep_m, shal_m, gap, se))

    header = f"{'pos':>9} {'segs':>5} {'tokens':>9} {'deep':>7} {'shallow':>8} {'gap':>7} {'se':>7}"
    print(header)
    for pos, nseg, ntok, d, s, g, se in out:
        se_s = f"{se:7.3f}" if se is not None else "      -"
        print(f"{pos:>9,} {nseg:>5} {ntok:>9,} {d:>7.3f} {s:>8.3f} {g:>+7.3f} {se_s}")

    if by_kind:
        order = sorted(by_kind, key=lambda k: -sum(v[2] for v in by_kind[k].values()))
        print(f"\n{'kind':<22} {'tokens':>10} " + " ".join(f"{b[0]:>8}" for b in BANDS))
        for k in order:
            total = sum(v[2] for v in by_kind[k].values())
            if total < 5000:
                continue
            cells = []
            for name, _lo, _hi in BANDS:
                v = by_kind[k].get(name)
                cells.append(f"{(v[1] - v[0]) / v[2]:>+8.3f}" if v and v[2] > 200 else f"{'-':>8}")
            print(f"{k:<22} {total:>10,} " + " ".join(cells))

    if args.tsv:
        with open(args.tsv, "w", encoding="utf-8") as fh:
            fh.write(f"# bin={args.bin} segments={used}\n")
            fh.write("pos\tsegments\ttokens\tnll_deep\tnll_shallow\tgap\tstderr\n")
            for pos, nseg, ntok, d, s, g, se in out:
                fh.write(f"{pos}\t{nseg}\t{ntok}\t{d:.6f}\t{s:.6f}\t{g:.6f}\t"
                         f"{'' if se is None else format(se, '.6f')}\n")
        print(f"\nwrote {args.tsv}", file=sys.stderr)


if __name__ == "__main__":
    main()
