#!/bin/bash
# Cut a held-out tail slice out of a corpus file that was previously used (with --chunks N)
# to build an imatrix, so it can be used for a domain-relevant perplexity eval without any
# overlap with what the imatrix actually saw.
#
# llama-imatrix reads a corpus front-to-back as one tokenized stream and stops once it hits
# --chunks; everything after that point in the file was never read. This script skips a
# generous safety margin past the actual consumed fraction (tokenization means "N chunks" of
# ctx=512 tokens doesn't map to an exact byte offset) and takes the rest of the file as the
# held-out set.
#
# Usage: corpus-holdout-slice.sh <corpus-file> [skip-fraction] [output-name]
#
# <corpus-file> looked up under CORPUS_DIR (default /mnt/llm/llama.cpp/models), or pass an
# absolute path.
# <skip-fraction> (default 0.10) is the fraction of lines to skip from the start before taking
# the held-out slice -- pick something comfortably larger than (chunks used / total chunks) for
# the imatrix run this corpus fed. E.g. prompt_corpus.txt used ~3.1% of its chunks, 0.10 gives
# over 3x margin.
#
# Output written to OUTPUT_DIR (default: same dir as input) as <corpus-basename>.holdout.txt.
set -euo pipefail

CORPUS="${1:?Usage: corpus-holdout-slice.sh <corpus-file> [skip-fraction] [output-name]}"
SKIP_FRACTION="${2:-0.10}"

CORPUS_DIR="${CORPUS_DIR:-/mnt/llm/llama.cpp/models}"

if [[ "$CORPUS" = /* ]]; then
    CORPUS_PATH="$CORPUS"
else
    CORPUS_PATH="$CORPUS_DIR/$CORPUS"
fi

if [ ! -f "$CORPUS_PATH" ]; then
    echo "ERROR: corpus file not found: $CORPUS_PATH" >&2
    exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "$CORPUS_PATH")}"
OUT_NAME="${3:-$(basename "$CORPUS_PATH" | sed -E 's/\.[^.]+$//').holdout.txt}"
OUT_PATH="$OUTPUT_DIR/$OUT_NAME"

TOTAL_LINES="$(wc -l < "$CORPUS_PATH")"
SKIP_LINES="$(awk -v t="$TOTAL_LINES" -v f="$SKIP_FRACTION" 'BEGIN { printf "%d", t * f }')"

mkdir -p "$OUTPUT_DIR"
tail -n "+$((SKIP_LINES + 1))" "$CORPUS_PATH" > "$OUT_PATH"

HOLDOUT_LINES="$(wc -l < "$OUT_PATH")"
echo "Skipped first $SKIP_LINES/$TOTAL_LINES lines, held out $HOLDOUT_LINES lines -> $OUT_PATH" >&2
echo "$OUT_PATH"
