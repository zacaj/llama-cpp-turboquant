#!/bin/bash
# Wrapper for build_vocab_patch.py: builds a small vocab-patch GGUF (pruned
# tokenizer KV + the two vocab-dimensioned tensors only) to load alongside
# an unmodified base GGUF via llama.cpp's --vocab-patch flag, running inside
# the turboquant docker image since it needs both gguf-py and the compiled
# llama-tokenize binary.
#
# Usage: build-vocab-patch.sh <base.gguf> <patch.gguf> --corpus <file1> [file2 ...] [--force]
#
# Filenames with no directory component resolve under MODELS_DIR. Absolute
# paths must live under MODELS_DIR or ARCHIVE_DIR (both get mounted into the
# container); anywhere else, move/symlink the file into one of those first.
set -euo pipefail

INPUT="${1:?Usage: build-vocab-patch.sh <base.gguf> <patch.gguf> --corpus <file1> [file2 ...] [--force]}"
OUTPUT="${2:?output patch gguf required}"
shift 2

MODELS_DIR="${MODELS_DIR:-/mnt/llm/llama.cpp/models}"
ARCHIVE_DIR="${ARCHIVE_DIR:-/mnt/2508/Archive}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE=llama-cpp-turboquant-llama-cpp:latest

resolve() {
    local p="$1"
    case "$p" in
        "$MODELS_DIR"/*)  echo "/models/${p#"$MODELS_DIR"/}" ;;
        "$ARCHIVE_DIR"/*) echo "/archive/${p#"$ARCHIVE_DIR"/}" ;;
        /*)
            echo "ERROR: $p is outside MODELS_DIR ($MODELS_DIR) and ARCHIVE_DIR ($ARCHIVE_DIR) -- not visible in the container" >&2
            exit 1
            ;;
        *) echo "/models/$p" ;;
    esac
}

INPUT_ARG="$(resolve "$INPUT")"
OUTPUT_ARG="$(resolve "$OUTPUT")"

CORPUS_ARGS=()
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --corpus)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                CORPUS_ARGS+=("$(resolve "$1")")
                shift
            done
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ ${#CORPUS_ARGS[@]} -eq 0 ]]; then
    echo "ERROR: at least one --corpus file is required" >&2
    exit 1
fi

docker run --rm -i \
    --gpus all \
    --entrypoint python3 \
    -e PYTHONPATH=/app/gguf-py \
    -v "$MODELS_DIR":/models \
    -v "$ARCHIVE_DIR":/archive \
    -v "$REPO_DIR/scripts":/host-scripts:ro \
    "$IMAGE" \
    /host-scripts/build_vocab_patch.py "$INPUT_ARG" "$OUTPUT_ARG" --corpus "${CORPUS_ARGS[@]}" "${EXTRA_ARGS[@]}"
