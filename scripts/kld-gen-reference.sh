#!/bin/bash
# Generate a bf16 (f16 KV) reference logits file for KV-cache KL-divergence benchmarking.
#
# Usage: kld-gen-reference.sh <model-filename-in-models-dir> [ctx] [chunks] [output-name]
#
# Model file must already exist under /mnt/llm/llama.cpp/models (mounted as /models in the container).
# Default ctx=24000 was chosen to fit the ~32GB WSL RAM cap: the fork's kl_divergence path
# reserves a single n_ctx*n_vocab float32 buffer up front (see tools/perplexity/perplexity.cpp),
# so RAM need scales linearly with ctx. At vocab=248320 that's ~22GiB at ctx=24000; 50000 needs ~46GiB and OOMs.
#
# CORPUS_FILE env var (path relative to data/corpus/) overrides the default wikitext-2 test file --
# e.g. CORPUS_FILE=wikitext-2-raw/wiki.test.shifted.raw for a non-overlapping resample.
# OUTPUT_DIR env var overrides where the (large) reference file gets written -- e.g. a bigger
# external drive when /mnt/llm is too tight for multi-chunk references.
set -euo pipefail

MODEL="${1:?Usage: kld-gen-reference.sh <model-filename> [ctx] [chunks] [output-name]}"
CTX="${2:-24000}"
CHUNKS="${3:-1}"

if [ "$CTX" -lt 1000 ]; then
    echo "ERROR: ctx=$CTX looks like a mistake -- did you mean to put this in the chunks slot?" >&2
    echo "Usage: kld-gen-reference.sh <model-filename> [ctx] [chunks] [output-name]" >&2
    exit 1
fi
OUT_NAME="${4:-$(basename "$MODEL" .gguf)-bf16ref-${CTX}ctx.kld}"
CORPUS_FILE="${CORPUS_FILE:-wikitext-2-raw/wiki.test.raw}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR=/mnt/llm/llama.cpp/models
CORPUS_DIR="$REPO_DIR/data/corpus"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/llm/models/kld-reference}"
IMAGE=llama-cpp-turboquant-llama-cpp:latest

mkdir -p "$OUTPUT_DIR"

if [ ! -f "$CORPUS_DIR/wikitext-2-raw/wiki.test.raw" ]; then
    mkdir -p "$CORPUS_DIR"
    (cd "$CORPUS_DIR" && sh "$REPO_DIR/scripts/get-wikitext-2.sh")
fi

echo "Generating reference: model=$MODEL ctx=$CTX chunks=$CHUNKS corpus=$CORPUS_FILE -> $OUT_NAME" >&2

docker run --rm --gpus all \
    -v "$MODELS_DIR":/models \
    -v "$CORPUS_DIR":/corpus:ro \
    -v "$OUTPUT_DIR":/output \
    "$IMAGE" \
    --perplexity -m "/models/$MODEL" \
    -f "/corpus/$CORPUS_FILE" \
    -c "$CTX" --chunks "$CHUNKS" \
    --n-cpu-moe 0 --no-mmap -dio -fit off \
    -b 2048 -ub 128 --flash-attn on \
    --kl-divergence-base "/output/$OUT_NAME" \
    --n-gpu-layers 99 -lv 3 --log-timestamps

echo "Reference written to $OUTPUT_DIR/$OUT_NAME" >&2
echo "$OUTPUT_DIR/$OUT_NAME"
