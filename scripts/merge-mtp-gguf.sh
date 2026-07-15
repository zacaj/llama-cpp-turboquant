#!/bin/bash
# Wrapper for merge_mtp_gguf.py: adds/replaces a GGUF's NextN/MTP block(s)
# with those from another GGUF (full checkpoint or MTP-only), running the
# python side inside the turboquant docker image so no local python/gguf-py
# install is required.
#
# Usage: merge-mtp-gguf.sh <target.gguf> <source.gguf> <output.gguf> [--force]
#
# Filenames with no directory component resolve under MODELS_DIR. Absolute
# paths must live under MODELS_DIR or ARCHIVE_DIR (both get mounted into the
# container); anywhere else, move/symlink the file into one of those first.
set -euo pipefail

TARGET="${1:?Usage: merge-mtp-gguf.sh <target.gguf> <source.gguf> <output.gguf> [--force]}"
SOURCE="${2:?source gguf required}"
OUTPUT="${3:?output gguf required}"
shift 3
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
        /*) echo "$p" ;;
        *) echo "/models/$p" ;;
    esac
}

TARGET_ARG="$(resolve "$TARGET")"
SOURCE_ARG="$(resolve "$SOURCE")"
OUTPUT_ARG="$(resolve "$OUTPUT")"

docker run --rm \
    --entrypoint python3 \
    -e PYTHONPATH=/app/gguf-py \
    -v "$MODELS_DIR":/models \
    -v "$ARCHIVE_DIR":/archive \
    -v "$TARGET":"$TARGET:ro" \
    -v "$SOURCE":"$SOURCE:ro" \
    -v "$REPO_DIR/scripts":/host-scripts:ro \
    "$IMAGE" \
    /host-scripts/merge_mtp_gguf.py "$TARGET_ARG" "$SOURCE_ARG" "$OUTPUT_ARG" "${EXTRA_ARGS[@]}"
