#!/bin/bash
# Dump a GGUF's tensors as a TSV (layer, tensor, type, shape, n_bytes) to stdout,
# using gguf_layer_quants.py inside the turboquant docker image so no local
# python/gguf-py install is required (same pattern as requant-from-reference.sh).
#
# Usage: gguf-layer-quants.sh <model.gguf>
#
# Filenames with no directory component resolve under MODELS_DIR. Absolute paths
# must live under MODELS_DIR or DUMP_DIR (both get mounted into the container);
# anywhere else, move/symlink the file into one of those first.
set -euo pipefail

MODEL="${1:?Usage: gguf-layer-quants.sh <model.gguf>}"

MODELS_DIR="${MODELS_DIR:-/mnt/llm/llama.cpp/models}"
DUMP_DIR="${DUMP_DIR:-/mnt/2508/Archive}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE=llama-cpp-turboquant-llama-cpp:latest

resolve() {
    local p="$1"
    case "$p" in
        "$MODELS_DIR"/*) echo "/models/${p#"$MODELS_DIR"/}" ;;
        "$DUMP_DIR"/*)   echo "/dumpout/${p#"$DUMP_DIR"/}" ;;
        /*)
            echo "ERROR: $p is outside MODELS_DIR ($MODELS_DIR) and DUMP_DIR ($DUMP_DIR) -- not visible in the container" >&2
            exit 1
            ;;
        *) echo "/models/$p" ;;
    esac
}

MODEL_ARG="$(resolve "$MODEL")"

docker run --rm \
    --entrypoint python3 \
    -e PYTHONPATH=/app/gguf-py \
    -v "$MODELS_DIR":/models \
    -v "$DUMP_DIR":/dumpout \
    -v "$REPO_DIR/scripts":/host-scripts:ro \
    "$IMAGE" \
    /host-scripts/gguf_layer_quants.py "$MODEL_ARG"
