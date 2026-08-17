#!/bin/bash
# Wrapper for patch_chat_template_raise.py: neutralizes the Qwen3.5/3.6/3.8-lineage
# raise_exception() asserts in a GGUF's embedded chat template, true in place
# (no copy, no tensor data touched), running the python side inside the
# turboquant docker image so no local python/gguf-py install is required.
#
# Usage: patch-chat-template-raise.sh <model.gguf> [--dry-run] [--force] [--verbose]
#
# Filenames with no directory component resolve under MODELS_DIR. Absolute
# paths must live under MODELS_DIR or ARCHIVE_DIR (both get mounted into the
# container); anywhere else, move/symlink the file into one of those first.
set -euo pipefail

MODEL="${1:?Usage: patch-chat-template-raise.sh <model.gguf> [--dry-run] [--force] [--verbose]}"
shift
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

MODEL_ARG="$(resolve "$MODEL")"

docker run --rm -i \
    --entrypoint python3 \
    -e PYTHONPATH=/app/gguf-py \
    -v "$MODELS_DIR":/models \
    -v "$ARCHIVE_DIR":/archive \
    -v "$REPO_DIR/scripts":/host-scripts:ro \
    "$IMAGE" \
    /host-scripts/patch_chat_template_raise.py "$MODEL_ARG" "${EXTRA_ARGS[@]}"
