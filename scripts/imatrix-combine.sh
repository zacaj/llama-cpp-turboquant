#!/bin/bash
# Combine multiple precomputed imatrix files into one, weighted by each file's token counts
# (not a naive per-file average) -- llama-imatrix's combine mode sums raw sums/counts per tensor
# across inputs, so a corpus with more tokens contributes proportionally more to the result.
#
# Usage: imatrix-combine.sh <model-filename-in-models-dir> <output-name> <imatrix-file> [imatrix-file ...]
#
# <model-filename> is only used to validate tensor names against; it is looked up the same way as
# in imatrix-gen.sh (checked under /models then /models2/MODELS2_DIR).
#
# All imatrix files must be readable from /mnt/llm/llama.cpp/models (default IMATRIX_DIR) --
# pass filenames relative to that dir, or absolute paths.
set -euo pipefail

MODEL="${1:?Usage: imatrix-combine.sh <model-filename> <output-name> <imatrix-file> [imatrix-file ...]}"
OUT_NAME="${2:?output name required}"
shift 2
IN_FILES=("$@")

if [ "${#IN_FILES[@]}" -lt 1 ]; then
    echo "ERROR: need at least one input imatrix file to combine" >&2
    exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR=/mnt/llm/llama.cpp/models
MODELS2_DIR="${MODELS2_DIR:-/mnt/2508/Backup 2}"
IMATRIX_DIR="${IMATRIX_DIR:-/mnt/llm/llama.cpp/models}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/llm/llama.cpp/models}"
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

mkdir -p "$OUTPUT_DIR"

IN_FILE_CSV=""
for f in "${IN_FILES[@]}"; do
    if [[ "$f" = /* ]]; then
        echo "ERROR: absolute paths for imatrix inputs not supported yet -- copy into $IMATRIX_DIR" >&2
        exit 1
    fi
    if [ ! -f "$IMATRIX_DIR/$f" ]; then
        echo "ERROR: imatrix file not found: $IMATRIX_DIR/$f" >&2
        exit 1
    fi
    IN_FILE_CSV="${IN_FILE_CSV:+$IN_FILE_CSV,}/imatrix/$f"
done

echo "Combining ${#IN_FILES[@]} imatrix file(s) -> $OUT_NAME: ${IN_FILES[*]}" >&2

docker run --rm --gpus all \
    "${MODEL_MOUNT_ARG[@]}" \
    -v "$IMATRIX_DIR":/imatrix \
    -v "$OUTPUT_DIR":/output \
    --entrypoint /app/llama-imatrix \
    "$IMAGE" \
    -m "$MODEL_PATH" \
    --in-file "$IN_FILE_CSV" \
    -o "/output/$OUT_NAME"

echo "Combined imatrix written to $OUTPUT_DIR/$OUT_NAME" >&2
echo "$OUTPUT_DIR/$OUT_NAME"
