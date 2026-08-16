#!/bin/bash
# Wrapper for merge_mtp_gguf.py: adds/replaces a GGUF's NextN/MTP block(s)
# with those from another GGUF (full checkpoint or MTP-only), running the
# python side inside the turboquant docker image so no local python/gguf-py
# install is required.
#
# Usage: merge-mtp-gguf.sh <target.gguf> <source.gguf> <output.gguf> [--force]
#
# Filenames with no directory component resolve under MODELS_DIR. Absolute paths
# are mounted verbatim into the container.
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

declare -A host_to_container

add_mount() {
    local host_path="$1"
    local container_path="$2"
    host_to_container[$host_path]=$container_path
}

# Sets RESOLVED as a side effect (not just echoes it) -- must be called as a plain
# statement, not via command substitution, or the add_mount calls run in a subshell
# and their host_to_container mutations never reach the parent shell.
resolve() {
    local p="$1"
    case "$p" in
        "$MODELS_DIR"/*)
            add_mount "$MODELS_DIR" "/models"
            RESOLVED="/models/${p#"$MODELS_DIR"/}"
            ;;
        "$ARCHIVE_DIR"/*)
            add_mount "$ARCHIVE_DIR" "/archive"
            RESOLVED="/archive/${p#"$ARCHIVE_DIR"/}"
            ;;
        /*)
            # Mount the parent dir, not the exact file: the file may not exist yet
            # (e.g. an output path), and docker creates missing bind-mount sources
            # as directories, which would break writing to it.
            local dir; dir="$(dirname "$p")"
            mkdir -p "$dir"
            add_mount "$dir" "$dir"
            RESOLVED="$dir/$(basename "$p")"
            ;;
        *)
            add_mount "$MODELS_DIR" "/models"
            RESOLVED="/models/$p"
            ;;
    esac
}

resolve "$TARGET"; TARGET_ARG="$RESOLVED"
resolve "$SOURCE"; SOURCE_ARG="$RESOLVED"
resolve "$OUTPUT"; OUTPUT_ARG="$RESOLVED"

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
    /host-scripts/merge_mtp_gguf.py "$TARGET_ARG" "$SOURCE_ARG" "$OUTPUT_ARG" "${EXTRA_ARGS[@]}"
