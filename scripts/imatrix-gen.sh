#!/bin/bash
# Generate an importance matrix from a text corpus, against a given model.
#
# Usage: imatrix-gen.sh <model-filename-in-models-dir> <corpus-file> [chunks] [output-name]
#
# <model-filename> is looked up under /mnt/llm/llama.cpp/models (mounted /models) first,
# then under MODELS2_DIR (mounted /models2, default "/mnt/2508/Backup 2") if not found there --
# so this works whether the target model lives in the usual models dir or on the big scratch drive.
# The model only needs to run on this GPU; it does not need to be the same quant you eventually
# apply the resulting imatrix to (the tool only reads activations, not weights, per-tensor).
#
# <corpus-file> is looked up under CORPUS_DIR (default: /mnt/llm/llama.cpp/models), so point it at
# e.g. bartowski-calibration_datav5.txt or prompt_corpus.txt directly by filename. Absolute paths
# are mounted verbatim into the container.
#
# chunks defaults to 200 (matches the wikitext-2 baseline runs this repo's KLD tooling uses).
# Pass -1 to run to exhaustion instead -- fine for small corpora (e.g. bartowski's ~6k-line
# calibration set finishes in well under a minute), but a bad idea for large real-world corpora:
# a 446k-line prompt-log corpus was ~12750 chunks at ctx=512, which projected to ~14.5 HOURS at
# ~4s/pass. Cap large corpora explicitly (e.g. 300-500) so multiple sources stay at comparable
# scale for imatrix-combine.sh's token-weighted merge, rather than the biggest file dominating
# the blend just by virtue of being biggest.
#
# Output written to OUTPUT_DIR (default: /mnt/llm/llama.cpp/models) as <output-name>, default
# <corpus-basename>.imatrix.gguf -- combine multiple outputs later with imatrix-combine.sh.
set -euo pipefail

MODEL="${1:?Usage: imatrix-gen.sh <model-filename> <corpus-file> [chunks] [output-name]}"
CORPUS="${2:?corpus file required}"
CHUNKS="${3:-200}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR=/mnt/llm/llama.cpp/models
MODELS2_DIR="${MODELS2_DIR:-/mnt/2508/Backup 2}"
CORPUS_DIR="${CORPUS_DIR:-/mnt/llm/llama.cpp/models}"
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
        "$CORPUS_DIR"/*)
            add_mount "$CORPUS_DIR" "/corpus"
            echo "/corpus/${p#"$CORPUS_DIR"/}"
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
            add_mount "$CORPUS_DIR" "/corpus"
            echo "/corpus/$p"
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

CORPUS_PATH="$(resolve_path "$CORPUS")"

mkdir -p "$OUTPUT_DIR"

OUT_NAME="${4:-$(basename "$CORPUS" | sed -E 's/\.[^.]+$//').imatrix.gguf}"
OUTPUT_PATH="$(resolve_path "$OUTPUT_DIR/$OUT_NAME")"

# Build docker mounts from tracking map
declare -a docker_mounts
for host in "${!host_to_container[@]}"; do
    docker_mounts+=("-v" "$host:${host_to_container[$host]}")
done

echo "Generating imatrix: model=$MODEL corpus=$CORPUS chunks=$CHUNKS -> $OUT_NAME" >&2

docker run --rm --gpus all \
    "${docker_mounts[@]}" \
    --entrypoint /app/llama-imatrix \
    "$IMAGE" \
    -m "$MODEL_PATH" \
    -f "$CORPUS_PATH" \
    -o "$OUTPUT_PATH" \
    --chunks "$CHUNKS" \
    -ngl 99 -c 512 -b 512 --flash-attn on

echo "Imatrix written to $OUTPUT_DIR/$OUT_NAME" >&2
echo "$OUTPUT_DIR/$OUT_NAME"
