#!/bin/bash
# Quantize an F16/BF16 GGUF from an explicit, hand-editable per-tensor plan: a TSV in
# the same format gguf-layer-quants.sh dumps (layer/tensor/type/shape/n_bytes).
#
# Usage: quantize-from-plan.sh <bf16-model-path> <plan-tsv> <output-path> [imatrix-file]
#
# Typical flow: dump an existing model's recipe with gguf-layer-quants.sh, edit the
# "type" column for whichever tensors you want to change, then quantize from that --
# no live reference GGUF needed (see requant-from-reference.sh for that variant,
# which reads the recipe off an already-quantized model instead of a file you can
# preview/edit first).
#
# gguf_plan_to_tensor_types.py turns the plan into a --tensor-type-file: every row
# becomes an exact, anchored override, so the plan is the sole source of truth for
# every tensor it lists, not a diff against the base ftype's usual mix. shape/n_bytes
# columns are ignored -- kept only so gguf-layer-quants.sh output works unmodified as
# a starting point. Tensors in the target model but missing from the plan (e.g.
# quantizing a different variant than what was dumped) fall back to BASE_FTYPE's
# normal per-tensor mix logic.
#
# <bf16-model-path> must be the first shard of a split GGUF named with the standard
# -00001-of-NNNNN.gguf convention (llama.cpp derives the rest from that name), or a
# single non-split GGUF -- same convention as quantize-iq4xs-uniform.sh. Give an
# absolute path; it's mounted read-only into the container.
#
# <plan-tsv> is looked up under PLAN_DIR (default: /mnt/2508/Archive, matching
# gguf-layer-quants.sh's own DUMP_DIR default), or pass an absolute path.
#
# <output-path> is an absolute path for the resulting GGUF; the parent directory is
# mounted read-write into the container.
#
# <imatrix-file> is looked up under IMATRIX_DIR (default: /mnt/llm/llama.cpp/models).
# Some target types (sub-4bit IQ*/IQ3_XXS in particular) only quantize well -- or at
# all -- with one -- see tensor_requires_imatrix() in src/llama-quant.cpp for exactly
# which. If llama-quantize errors with "imatrix required for tensor ... quantization",
# pass one here.
#
# BASE_FTYPE env var (default IQ4_XS) sets the fallback mix used ONLY for tensors
# with no matching row in the plan -- it never affects tensors the plan lists.
#
# Set DRY_RUN=1 to size the output without writing tensor data (fast, ~1s) before
# committing to a full run, which reads/writes the whole model and can take 15-20
# minutes.
set -euo pipefail

BF16_PATH="${1:?Usage: quantize-from-plan.sh <bf16-model-path> <plan-tsv> <output-path> [imatrix-file]}"
PLAN_TSV="${2:?plan tsv required}"
OUT_PATH="${3:?output path required}"
IMATRIX="${4:-}"

PLAN_DIR="${PLAN_DIR:-/mnt/2508/Archive}"
IMATRIX_DIR="${IMATRIX_DIR:-/mnt/llm/llama.cpp/models}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE=llama-cpp-turboquant-llama-cpp:latest
BASE_FTYPE="${BASE_FTYPE:-IQ4_XS}"

if [ ! -f "$BF16_PATH" ]; then
    echo "ERROR: bf16 model not found: $BF16_PATH" >&2
    exit 1
fi

if [[ "$PLAN_TSV" = /* ]]; then
    PLAN_PATH="$PLAN_TSV"
else
    PLAN_PATH="$PLAN_DIR/$PLAN_TSV"
fi
if [ ! -f "$PLAN_PATH" ]; then
    echo "ERROR: plan tsv not found: $PLAN_PATH" >&2
    exit 1
fi

if [ -n "$IMATRIX" ] && [ ! -f "$IMATRIX_DIR/$IMATRIX" ]; then
    echo "ERROR: imatrix not found: $IMATRIX_DIR/$IMATRIX" >&2
    exit 1
fi

BF16_DIR="$(dirname "$BF16_PATH")"
BF16_NAME="$(basename "$BF16_PATH")"
PLAN_DIR_ACTUAL="$(dirname "$PLAN_PATH")"
PLAN_NAME="$(basename "$PLAN_PATH")"
OUT_DIR="$(dirname "$OUT_PATH")"
OUT_NAME="$(basename "$OUT_PATH")"
mkdir -p "$OUT_DIR"

TYPES_NAME="$(basename "$PLAN_PATH" .tsv).types.txt"

echo "=== Converting plan $PLAN_PATH -> tensor-type-file ===" >&2
docker run --rm \
    --entrypoint python3 \
    -v "$PLAN_DIR_ACTUAL":/plan \
    -v "$REPO_DIR/scripts":/host-scripts:ro \
    "$IMAGE" \
    /host-scripts/gguf_plan_to_tensor_types.py "/plan/$PLAN_NAME" "/plan/$TYPES_NAME"

QUANTIZE_ARGS=(--tensor-type-file "/plan/$TYPES_NAME")
IMATRIX_MOUNT_ARG=()
if [ -n "$IMATRIX" ]; then
    QUANTIZE_ARGS+=(--imatrix "/imatrix/$IMATRIX")
    IMATRIX_MOUNT_ARG=(-v "$IMATRIX_DIR":/imatrix:ro)
fi

DRY_RUN_ARG=()
if [ "${DRY_RUN:-0}" = "1" ]; then
    DRY_RUN_ARG=(--dry-run)
    echo "DRY_RUN=1: sizing only, no tensor data will be written" >&2
fi

echo "=== Quantizing $BF16_PATH -> $OUT_PATH (base ftype $BASE_FTYPE for any unmatched tensor) ===" >&2
docker run --rm --gpus all \
    -v "$BF16_DIR":/src:ro \
    -v "$PLAN_DIR_ACTUAL":/plan:ro \
    "${IMATRIX_MOUNT_ARG[@]}" \
    -v "$OUT_DIR":/out \
    --entrypoint /app/llama-quantize \
    "$IMAGE" \
    "${DRY_RUN_ARG[@]}" \
    "${QUANTIZE_ARGS[@]}" \
    "/src/$BF16_NAME" "/out/$OUT_NAME" "$BASE_FTYPE"

if [ "${DRY_RUN:-0}" != "1" ]; then
    echo "Done: $OUT_PATH" >&2
    echo "$OUT_PATH"
fi
