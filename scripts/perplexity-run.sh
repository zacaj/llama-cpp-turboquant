#!/bin/bash
# Run llama-perplexity for a model against a text corpus and print the final PPL estimate.
#
# Usage: perplexity-run.sh <model-filename-in-models-dir> <corpus-file> [chunks]
#
# <model-filename> looked up under /mnt/llm/llama.cpp/models (mounted /models) first, then
# under MODELS2_DIR (mounted /models2, default "/mnt/2508/Backup 2") -- same lookup as
# imatrix-gen.sh.
#
# <corpus-file> looked up under CORPUS_DIR (default: /mnt/llm/llama.cpp/models), or pass an
# absolute path.
#
# chunks defaults to 200, matching the wiki.test.raw baseline runs used elsewhere in this repo.
# Pass -1 to run to exhaustion -- see imatrix-gen.sh's comment for why that's risky on large
# real-world corpora (hours, not minutes).
set -euo pipefail

MODEL="${1:?Usage: perplexity-run.sh <model-filename> <corpus-file> [chunks]}"
CORPUS="${2:?corpus file required}"
CHUNKS="${3:-200}"

MODELS_DIR=/mnt/llm/llama.cpp/models
MODELS2_DIR="${MODELS2_DIR:-/mnt/2508/Backup 2}"
CORPUS_DIR="${CORPUS_DIR:-/mnt/llm/llama.cpp/models}"
IMAGE=llama-cpp-turboquant-llama-cpp:latest

if [ -f "$MODELS_DIR/$MODEL" ]; then
    MODEL_MOUNT_ARG=(-v "$MODELS_DIR":/models)
    MODEL_PATH="/models/$MODEL"
elif [ -f "$MODELS2_DIR/$MODEL" ]; then
    MODEL_MOUNT_ARG=(-v "$MODELS2_DIR":/models2)
    MODEL_PATH="/models2/$MODEL"
else
    echo "ERROR: model '$MODEL' not found under $MODELS_DIR or $MODELS2_DIR" >&2
    exit 1
fi

if [[ "$CORPUS" = /* ]]; then
    CORPUS_MOUNT_DIR="$(dirname "$CORPUS")"
    CORPUS_PATH="/corpus-abs/$(basename "$CORPUS")"
    CORPUS_MOUNT_ARG=(-v "$CORPUS_MOUNT_DIR":/corpus-abs:ro)
else
    CORPUS_MOUNT_ARG=(-v "$CORPUS_DIR":/corpus:ro)
    CORPUS_PATH="/corpus/$CORPUS"
fi

echo "Running perplexity: model=$MODEL corpus=$CORPUS chunks=$CHUNKS" >&2

docker run --rm --gpus all \
    "${MODEL_MOUNT_ARG[@]}" \
    "${CORPUS_MOUNT_ARG[@]}" \
    --entrypoint /app/llama-perplexity \
    "$IMAGE" \
    -m "$MODEL_PATH" \
    -f "$CORPUS_PATH" \
    -ngl 99 -c 512 -b 512 --chunks "$CHUNKS" -fa on
