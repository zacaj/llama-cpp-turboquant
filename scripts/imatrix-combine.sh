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
# Imatrix files are looked up under IMATRIX_DIR (default: /mnt/llm/llama.cpp/models);
# absolute paths are mounted verbatim into the container.
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
        "$MODELS2_DIR"/*)
            add_mount "$MODELS2_DIR" "/models2"
            echo "/models2/${p#"$MODELS2_DIR"/}"
            ;;
        "$IMATRIX_DIR"/*)
            add_mount "$IMATRIX_DIR" "/imatrix"
            echo "/imatrix/${p#"$IMATRIX_DIR"/}"
            ;;
        "$OUTPUT_DIR"/*)
            add_mount "$OUTPUT_DIR" "/output"
            echo "/output/${p#"$OUTPUT_DIR"/}"
            ;;
        /*)
            add_mount "$p" "$p"
            echo "$p"
            ;;
        *)
            add_mount "$IMATRIX_DIR" "/imatrix"
            echo "/imatrix/$p"
            ;;
    esac
}

# Resolve model (check both MODELS_DIR and MODELS2_DIR)
if [ -f "$MODELS_DIR/$MODEL" ]; then
    MODEL_PATH="$(resolve_path "$MODELS_DIR/$MODEL")"
elif [ -f "$MODELS2_DIR/$MODEL" ]; then
    MODEL_PATH="$(resolve_path "$MODELS2_DIR/$MODEL")"
else
    echo "ERROR: model '$MODEL' not found under $MODELS_DIR or $MODELS2_DIR" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Ensure OUTPUT_DIR is in mount map
add_mount "$OUTPUT_DIR" "/output"
OUTPUT_PATH="/output/$OUT_NAME"

IN_FILE_CSV=""
for f in "${IN_FILES[@]}"; do
    # Resolve input path, validating it exists
    if [[ "$f" = /* ]]; then
        if [ ! -f "$f" ]; then
            echo "ERROR: imatrix file not found: $f" >&2
            exit 1
        fi
        IN_PATH="$(resolve_path "$f")"
    else
        if [ ! -f "$IMATRIX_DIR/$f" ]; then
            echo "ERROR: imatrix file not found: $IMATRIX_DIR/$f" >&2
            exit 1
        fi
        IN_PATH="$(resolve_path "$IMATRIX_DIR/$f")"
    fi
    IN_FILE_CSV="${IN_FILE_CSV:+$IN_FILE_CSV,}$IN_PATH"
done

# Build docker mounts from tracking map
declare -a docker_mounts
for host in "${!host_to_container[@]}"; do
    docker_mounts+=("-v" "$host:${host_to_container[$host]}")
done

echo "Combining ${#IN_FILES[@]} imatrix file(s) -> $OUT_NAME: ${IN_FILES[*]}" >&2

docker run --rm --gpus all \
    "${docker_mounts[@]}" \
    --entrypoint /app/llama-imatrix \
    "$IMAGE" \
    -m "$MODEL_PATH" \
    --in-file "$IN_FILE_CSV" \
    -o "$OUTPUT_PATH"

echo "Combined imatrix written to $OUTPUT_DIR/$OUT_NAME" >&2
echo "$OUTPUT_DIR/$OUT_NAME"
