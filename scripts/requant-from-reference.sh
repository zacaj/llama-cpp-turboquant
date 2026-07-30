#!/bin/bash
# Quantize an F16/BF16 GGUF using the exact per-tensor quantization recipe read off
# an already-quantized reference GGUF from the same base model (same architecture and
# tensor names -- e.g. a public release you want to reproduce, or tweak a couple
# tensors of and re-quantize).
#
# Usage: requant-from-reference.sh <f16-model> <reference-model> [output.gguf] [imatrix-file]
#
# How it works: gguf_tensor_type_map.py reads every tensor's ggml_type straight out
# of the reference GGUF's header (cheap -- mmap, no tensor data touched) and writes
# an exact tensor_name=type line per tensor. llama-quantize then runs on the F16
# model with --tensor-type-file, so every tensor matched by name gets quantized to
# exactly the type the reference used, regardless of the base ftype's usual mix
# logic. The python side runs inside the turboquant docker image so no local
# python/gguf-py install is required (same pattern as extract-mtp-gguf.sh).
#
# Tensors that exist in only one of the two files are reported to stderr; a target
# tensor with no reference match falls back to BASE_FTYPE's normal mix. That should
# be rare for a real "same base model" pair -- if you see more than a couple, the
# reference probably doesn't actually match this F16 file (different arch, pruned
# layers, MTP present in one but not the other, etc).
#
# Some of the reference's tensor types (mostly sub-4bit IQ* ones) only quantize
# well -- or at all -- with an importance matrix. If llama-quantize errors with
# something like "imatrix required for tensor ... quantization", pass one as the
# 4th arg (generate one first with tools/imatrix if you don't already have one for
# this model).
#
# BASE_FTYPE env var (default Q8_0) sets the fallback mix used ONLY for target
# tensors with no reference match -- it never affects matched tensors, which always
# get the reference's exact type regardless of this setting. Must be a real
# quantized llama-quantize ftype name (not F16/F32/COPY), or the manual per-tensor
# overrides get skipped entirely -- see the --pure guard in llama_tensor_get_type.
#
# Filenames with no directory component resolve under MODELS_DIR. Absolute paths
# are mounted verbatim into the container. Sharded/split GGUF input is not handled
# (no --keep-split).
set -euo pipefail

F16_MODEL="${1:?Usage: requant-from-reference.sh <f16-model> <reference-model> [output.gguf] [imatrix-file]}"
REF_MODEL="${2:?reference (already-quantized) model required}"
OUTPUT="${3:-}"
IMATRIX="${4:-}"

MODELS_DIR="${MODELS_DIR:-/mnt/llm/llama.cpp/models}"
DUMP_DIR="${DUMP_DIR:-/mnt/2508/Archive}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE=llama-cpp-turboquant-llama-cpp:latest
BASE_FTYPE="${BASE_FTYPE:-Q8_0}"

if [ -z "$OUTPUT" ]; then
    OUTPUT="$MODELS_DIR/$(basename "$F16_MODEL" .gguf)-requant-like-$(basename "$REF_MODEL" .gguf).gguf"
fi

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
        "$DUMP_DIR"/*)
            add_mount "$DUMP_DIR" "/dumpout"
            echo "/dumpout/${p#"$DUMP_DIR"/}"
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

F16_ARG="$(resolve "$F16_MODEL")"
REF_ARG="$(resolve "$REF_MODEL")"
OUTPUT_ARG="$(resolve "$OUTPUT")"

mkdir -p "$DUMP_DIR"

TYPES_FILE="$DUMP_DIR/$(basename "$F16_MODEL" .gguf)-tensor-types-from-$(basename "$REF_MODEL" .gguf).txt"
TYPES_ARG="$(resolve "$TYPES_FILE")"

# Build docker mounts from tracking map
declare -a docker_mounts
for host in "${!host_to_container[@]}"; do
    docker_mounts+=("-v" "$host:${host_to_container[$host]}")
done

echo "=== Reading tensor types from reference: $REF_MODEL ===" >&2
docker run --rm \
    --entrypoint python3 \
    -e PYTHONPATH=/app/gguf-py \
    "${docker_mounts[@]}" \
    -v "$REPO_DIR/scripts":/host-scripts:ro \
    "$IMAGE" \
    /host-scripts/gguf_tensor_type_map.py "$REF_ARG" "$F16_ARG" "$TYPES_ARG"

QUANTIZE_ARGS=(--tensor-type-file "$TYPES_ARG")
if [ -n "$IMATRIX" ]; then
    IMATRIX_ARG="$(resolve "$IMATRIX")"
    QUANTIZE_ARGS+=(--imatrix "$IMATRIX_ARG")
fi

echo "=== Quantizing $F16_MODEL -> $OUTPUT (base ftype $BASE_FTYPE for any unmatched tensor) ===" >&2
docker run --rm --gpus all \
    "${docker_mounts[@]}" \
    "$IMAGE" \
    --quantize "${QUANTIZE_ARGS[@]}" "$F16_ARG" "$OUTPUT_ARG" "$BASE_FTYPE"

echo "Done: $OUTPUT" >&2
echo "$OUTPUT"
