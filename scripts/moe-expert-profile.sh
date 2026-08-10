#!/bin/bash
# Run llama-moe-weights over one or more text corpora and merge the resulting
# per-expert profiles into two summary CSVs -- the input format
# scripts/lowest_experts_from_profile.py expects, by two different rankings:
#   <output.csv>              raw selection count (rank-agnostic)
#   <output>.weighted.csv     summed pre-norm router weight (low-rank picks
#                             count for less -- see moe-weights.cpp's header
#                             comment on moe_expert_usage_collector)
#
# This is the perplexity-corpus alternative to instrumenting live traffic:
# each corpus file gets its own moe-weights pass (own context, no cross-file
# KV state), then merge_moe_profiles.py sums the per-expert/per-layer values
# across all of them into one profile per ranking.
#
# Usage: moe-expert-profile.sh <model-filename> <output.csv> <corpus-file> [corpus-file...]
#
# <model-filename> looked up under MODELS_DIR (default /mnt/llm/llama.cpp/models), then
# MODELS2_DIR (default "/mnt/2508/Backup 2") -- same lookup as perplexity-run.sh.
# <corpus-file>(s) may be bare filenames (looked up under CORPUS_DIR, default
# MODELS_DIR) or absolute paths.
# <output.csv> is where the merged count-ranked summary CSV is written (host path).
#
# Env: MODELS_DIR, MODELS2_DIR, CORPUS_DIR, CTX_SIZE (default 32768).
set -euo pipefail

MODEL="${1:?Usage: moe-expert-profile.sh <model-filename> <output.csv> <corpus-file...>}"
OUTPUT="${2:?output.csv path required}"
shift 2
CORPORA=("$@")
if [ "${#CORPORA[@]}" -eq 0 ]; then
    echo "ERROR: at least one corpus file required" >&2
    exit 1
fi

MODELS_DIR="${MODELS_DIR:-/mnt/llm/llama.cpp/models}"
MODELS2_DIR="${MODELS2_DIR:-/mnt/2508/Backup 2}"
CORPUS_DIR="${CORPUS_DIR:-$MODELS_DIR}"
CTX_SIZE="${CTX_SIZE:-32768}"
IMAGE=llama-cpp-turboquant-llama-cpp:latest

if [ -f "$MODELS_DIR/$MODEL" ]; then
    MODEL_MOUNT_ARG=(-v "$MODELS_DIR":/models)
    MODEL_PATH="/models/$MODEL"
elif [ -f "$MODELS2_DIR/$MODEL" ]; then
    MODEL_MOUNT_ARG=(-v "$MODELS2_DIR":/models2)
    MODEL_PATH="/models2/$MODEL"
else
    echo "ERROR: model '$MODEL' not found under $MODELS_DIR or $MODELS2_DIR" >&2
    exit 1
fi

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT
PROFILES=()
PROFILES_WEIGHTED=()

for CORPUS in "${CORPORA[@]}"; do
    if [[ "$CORPUS" = /* ]]; then
        CORPUS_MOUNT_DIR="$(dirname "$CORPUS")"
        CORPUS_MOUNT_ARG=(-v "$CORPUS_MOUNT_DIR":/corpus-abs:ro)
        CORPUS_PATH="/corpus-abs/$(basename "$CORPUS")"
        NAME="$(basename "$CORPUS")"
    else
        CORPUS_MOUNT_ARG=(-v "$CORPUS_DIR":/corpus:ro)
        CORPUS_PATH="/corpus/$CORPUS"
        NAME="$CORPUS"
    fi

    PROFILE_CSV="$WORKDIR/$NAME.profile.csv"
    PROFILE_WEIGHTED_CSV="$WORKDIR/$NAME.profile.weighted.csv"
    echo "Profiling: model=$MODEL corpus=$CORPUS" >&2

    docker run --rm --gpus all \
        "${MODEL_MOUNT_ARG[@]}" \
        "${CORPUS_MOUNT_ARG[@]}" \
        -v "$WORKDIR":/out \
        --entrypoint /app/llama-moe-weights \
        "$IMAGE" \
        -m "$MODEL_PATH" \
        -f "$CORPUS_PATH" \
        -ngl 999 -c "$CTX_SIZE" --no-mmap \
        -o "/out/$NAME.profile.csv" \
        >&2

    PROFILES+=("$PROFILE_CSV")
    PROFILES_WEIGHTED+=("$PROFILE_WEIGHTED_CSV")
done

OUT_DIR="$(dirname "$OUTPUT")"
mkdir -p "$OUT_DIR"
WEIGHTED_OUTPUT="${OUTPUT%.csv}.weighted.csv"
python3 "$(dirname "$0")/merge_moe_profiles.py" "${PROFILES[@]}" "$OUTPUT"
python3 "$(dirname "$0")/merge_moe_profiles.py" "${PROFILES_WEIGHTED[@]}" "$WEIGHTED_OUTPUT"
echo "Merged $(( ${#CORPORA[@]} )) corpus profile(s) -> $OUTPUT and $WEIGHTED_OUTPUT" >&2
