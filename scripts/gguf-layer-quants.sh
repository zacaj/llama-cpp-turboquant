#!/bin/bash
# Dump a GGUF's tensors as a TSV (layer, tensor, type, shape, n_bytes),
# using gguf_layer_quants.py inside the turboquant docker image so no local
# python/gguf-py install is required (same pattern as requant-from-reference.sh).
#
# Usage: gguf-layer-quants.sh <model.gguf> [output.tsv | output-dir/]
#
# Filenames with no directory component resolve under MODELS_DIR. Absolute paths
# are mounted verbatim into the container. If output is a directory, a TSV file
# is auto-generated under that directory using the input filename with a
# .quants.tsv extension.
set -euo pipefail

MODEL="${1:?Usage: gguf-layer-quants.sh <model.gguf> [output.tsv | output-dir/]}"
OUTPUT="${2:-}"

MODELS_DIR="${MODELS_DIR:-/mnt/llm/llama.cpp/models}"
DUMP_DIR="${DUMP_DIR:-/mnt/2508/Archive}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE=llama-cpp-turboquant-llama-cpp:latest

# Track host→container mounts for deduplication
declare -A host_to_container

add_mount() {
    local host_path="$1"
    local container_path="$2"
    host_to_container[$host_path]=$container_path
}

resolve_path() {
    local p="$1"
    case "$p" in
        "$MODELS_DIR"/*)
            add_mount "$MODELS_DIR" "/models"
            echo "/models/${p#"$MODELS_DIR"/}"
            ;;
        "$DUMP_DIR"/*)
            add_mount "$DUMP_DIR" "/dumpout"
            echo "/dumpout/${p#"$DUMP_DIR"/}"
            ;;
        /*)
            # Absolute path: mount verbatim
            add_mount "$p" "$p"
            echo "$p"
            ;;
        *)
            # Relative path: resolve under MODELS_DIR
            add_mount "$MODELS_DIR" "/models"
            echo "/models/$p"
            ;;
    esac
}

MODEL_ARG="$(resolve_path "$MODEL")"

# Handle optional output argument
if [[ -n "$OUTPUT" ]]; then
    # Check if output is a directory
    if [[ -d "$OUTPUT" ]]; then
        # Generate output path: directory/input_basename.quants.tsv
        local input_base="${MODEL##*/}"
        local input_noext="${input_base%.*}"
        OUTPUT="${OUTPUT%/}/${input_noext}.quants.tsv"
    fi
    # Ensure parent directory exists
    mkdir -p "$(dirname "$OUTPUT")"
fi

# Build docker mounts from our tracking map
declare -a docker_mounts
for host in "${!host_to_container[@]}"; do
    docker_mounts+=("-v" "$host:${host_to_container[$host]}")
done

# Run with optional output redirection
if [[ -n "$OUTPUT" ]]; then
    docker run --rm \
        --entrypoint python3 \
        -e PYTHONPATH=/app/gguf-py \
        "${docker_mounts[@]}" \
        -v "$REPO_DIR/scripts":/host-scripts:ro \
        "$IMAGE" \
        /host-scripts/gguf_layer_quants.py "$MODEL_ARG" > "$OUTPUT"
else
    docker run --rm \
        --entrypoint python3 \
        -e PYTHONPATH=/app/gguf-py \
        "${docker_mounts[@]}" \
        -v "$REPO_DIR/scripts":/host-scripts:ro \
        "$IMAGE" \
        /host-scripts/gguf_layer_quants.py "$MODEL_ARG"
fi
