#!/bin/bash
# Cut a byte prefix from a corpus file sized to land near a target token count, using a real
# tokenizer rather than a fixed bytes-per-token guess (which varies a lot between prose, code, and
# structured-log corpora -- see docs/moe-expert-count-analysis.md Part 6 for how wrong a flat
# assumption can be).
#
# Usage: corpus-token-sample.sh <corpus-file> <target-tokens> [output-path]
#
# <corpus-file> looked up under CORPUS_DIR (default /mnt/llm/llama.cpp/models), or pass an
# absolute path.
# <output-path> defaults to OUTPUT_DIR (default: same dir as input) / <corpus-basename>.sample<N>.txt
#
# Method: tokenize a generous probe prefix (target-tokens * PROBE_BYTES_PER_TOKEN bytes, default
# 8 -- comfortably above any real corpus's ratio) to measure the actual bytes-per-token ratio for
# this specific file, then cut the exact byte offset that ratio implies for target-tokens. Verifies
# the final cut's real token count and reports it -- expect it to land within a token or two of the
# target, not exact, since bytes-per-token varies slightly within a file.
#
# Env vars:
#   CORPUS_DIR (default /mnt/llm/llama.cpp/models)
#   OUTPUT_DIR (default: same dir as input)
#   MODEL      GGUF used only for its tokenizer (default: a Qwen3.6 quant already on disk -- any
#              model sharing the target corpus's intended tokenizer works equally well here)
#   MODEL_DIR  where MODEL is looked up if not an absolute path (default /mnt/2508/Backup 2)
#   IMAGE      docker image with llama-tokenize (default llama-cpp-turboquant-llama-cpp:latest)
set -euo pipefail

CORPUS="${1:?Usage: corpus-token-sample.sh <corpus-file> <target-tokens> [output-path]}"
TARGET_TOKENS="${2:?target token count required}"

CORPUS_DIR="${CORPUS_DIR:-/mnt/llm/llama.cpp/models}"
MODEL_DIR="${MODEL_DIR:-/mnt/2508/Backup 2}"
MODEL="${MODEL:-Qwen3.6-27B-IQ4_XS-combined-imat-zacaj.gguf}"
IMAGE="${IMAGE:-llama-cpp-turboquant-llama-cpp:latest}"
PROBE_BYTES_PER_TOKEN="${PROBE_BYTES_PER_TOKEN:-8}"

if [[ "$CORPUS" = /* ]]; then
    CORPUS_PATH="$CORPUS"
else
    CORPUS_PATH="$CORPUS_DIR/$CORPUS"
fi
[ -f "$CORPUS_PATH" ] || { echo "ERROR: corpus not found: $CORPUS_PATH" >&2; exit 1; }

if [[ "$MODEL" = /* ]]; then
    MODEL_HOST_DIR="$(dirname "$MODEL")"
    MODEL_PATH="/model/$(basename "$MODEL")"
else
    MODEL_HOST_DIR="$MODEL_DIR"
    MODEL_PATH="/model/$MODEL"
fi
[ -f "$MODEL_HOST_DIR/$(basename "$MODEL")" ] || {
    echo "ERROR: tokenizer model not found: $MODEL_HOST_DIR/$(basename "$MODEL")" >&2; exit 1; }

OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "$CORPUS_PATH")}"
OUT_NAME="${3:-$OUTPUT_DIR/$(basename "$CORPUS_PATH" | sed -E 's/\.[^.]+$//').sample${TARGET_TOKENS}.txt}"
mkdir -p "$(dirname "$OUT_NAME")"

CORPUS_HOST_DIR="$(dirname "$CORPUS_PATH")"
CORPUS_BASENAME="$(basename "$CORPUS_PATH")"

token_count() {
    # $1 = container path to a text file
    docker run --rm --gpus all \
        -v "$CORPUS_HOST_DIR":/corpus:ro \
        -v "$MODEL_HOST_DIR":/model:ro \
        -v "$(dirname "$OUT_NAME")":/out \
        --entrypoint /app/llama-tokenize "$IMAGE" \
        -m "$MODEL_PATH" -f "$1" --show-count --log-disable 2>/dev/null \
        | grep -oP 'Total number of tokens: \K\d+'
}

PROBE_BYTES=$(( TARGET_TOKENS * PROBE_BYTES_PER_TOKEN ))
FILE_BYTES=$(wc -c < "$CORPUS_PATH")
[ "$PROBE_BYTES" -gt "$FILE_BYTES" ] && PROBE_BYTES="$FILE_BYTES"

PROBE_NAME="/out/.probe-$(basename "$OUT_NAME")"
PROBE_HOST="$(dirname "$OUT_NAME")/.probe-$(basename "$OUT_NAME")"
head -c "$PROBE_BYTES" "$CORPUS_PATH" > "$PROBE_HOST"
PROBE_TOKENS=$(token_count "$PROBE_NAME")
[ -n "$PROBE_TOKENS" ] && [ "$PROBE_TOKENS" -gt 0 ] || { echo "ERROR: probe tokenization failed" >&2; rm -f "$PROBE_HOST"; exit 1; }

RATIO=$(awk -v b="$PROBE_BYTES" -v t="$PROBE_TOKENS" 'BEGIN { printf "%.6f", b / t }')
TARGET_BYTES=$(awk -v r="$RATIO" -v t="$TARGET_TOKENS" 'BEGIN { printf "%d", r * t }')
[ "$TARGET_BYTES" -gt "$FILE_BYTES" ] && TARGET_BYTES="$FILE_BYTES"
rm -f "$PROBE_HOST"

head -c "$TARGET_BYTES" "$CORPUS_PATH" > "$OUT_NAME"

FINAL_NAME="/out/$(basename "$OUT_NAME")"
FINAL_TOKENS=$(token_count "$FINAL_NAME")

echo "Probe: $PROBE_BYTES bytes -> $PROBE_TOKENS tokens (ratio $RATIO bytes/token)" >&2
echo "Cut $TARGET_BYTES bytes -> $FINAL_TOKENS tokens (target $TARGET_TOKENS) -> $OUT_NAME" >&2
echo "$OUT_NAME"
