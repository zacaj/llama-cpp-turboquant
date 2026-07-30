#!/bin/bash
# Wrapper for extract_mtp_gguf.py: pulls a GGUF's NextN/MTP block(s) out into
# a small standalone draft file, running the python side inside the
# turboquant docker image so no local python/gguf-py install is required.
#
# Usage: extract-mtp-gguf.sh <input.gguf> <output.gguf> [--force]
#
# Filenames with no directory component resolve under MODELS_DIR. Absolute paths
# are mounted verbatim into the container.
set -euo pipefail

INPUT="${1:?Usage: extract-mtp-gguf.sh <input.gguf> <output.gguf> [--force]}"
OUTPUT="${2:?output gguf required}"
shift 2
EXTRA_ARGS=("$@")

MODELS_DIR="${MODELS_DIR:-/mnt/llm/llama.cpp/models}"
ARCHIVE_DIR="${ARCHIVE_DIR:-/mnt/2508/Archive}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE=llama-cpp-turboquant-llama-cpp:latest

declare -A host_to_container

add_mount() {
    local host_path="$1"
    local container_path="$2"
    host_to_container[$host_path]=$container_path
}

resolve() {
    local p="$1"
    case "$p" in
        "$MODELS_DIR"/*)
            add_mount "$MODELS_DIR" "/models"
            echo "/models/${p#"$MODELS_DIR"/}"
            ;;
        "$ARCHIVE_DIR"/*)
            add_mount "$ARCHIVE_DIR" "/archive"
            echo "/archive/${p#"$ARCHIVE_DIR"/}"
            ;;
        /*)
            add_mount "$p" "$p"
            echo "$p"
            ;;
        *)
            add_mount "$MODELS_DIR" "/models"
            echo "/models/$p"
            ;;
    esac
}

INPUT_ARG="$(resolve "$INPUT")"
OUTPUT_ARG="$(resolve "$OUTPUT")"

# Build docker mounts from tracking map
declare -a docker_mounts
for host in "${!host_to_container[@]}"; do
    docker_mounts+=("-v" "$host:${host_to_container[$host]}")
done

docker run --rm \
    --entrypoint python3 \
    -e PYTHONPATH=/app/gguf-py \
    "${docker_mounts[@]}" \
    -v "$REPO_DIR/scripts":/host-scripts:ro \
    "$IMAGE" \
    /host-scripts/extract_mtp_gguf.py "$INPUT_ARG" "$OUTPUT_ARG" "${EXTRA_ARGS[@]}"
