#!/bin/bash
# Wrapper for extract_mtp_gguf.py: pulls a GGUF's NextN/MTP block(s) out into
# a small standalone draft file, running the python side inside the
# turboquant docker image so no local python/gguf-py install is required.
#
# Usage: extract-mtp-gguf.sh <input.gguf> <output.gguf> [--force]
#
# Filenames with no directory component resolve under MODELS_DIR. Absolute
# paths must live under MODELS_DIR or ARCHIVE_DIR (both get mounted into the
# container); anywhere else, move/symlink the file into one of those first.
set -euo pipefail

INPUT="${1:?Usage: extract-mtp-gguf.sh <input.gguf> <output.gguf> [--force]}"
OUTPUT="${2:?output gguf required}"
shift 2
EXTRA_ARGS=("$@")

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

docker run --rm \
    --entrypoint python3 \
    -e PYTHONPATH=/app/gguf-py \
    -v "$MODELS_DIR":/models \
    -v "$ARCHIVE_DIR":/archive \
    -v "$REPO_DIR/scripts":/host-scripts:ro \
    "$IMAGE" \
    /host-scripts/extract_mtp_gguf.py "$INPUT_ARG" "$OUTPUT_ARG" "${EXTRA_ARGS[@]}"
