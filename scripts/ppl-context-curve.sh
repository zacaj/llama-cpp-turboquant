#!/bin/bash
# Measure how much a model actually gains from distant context, as a function of position.
#
# Usage: ppl-context-curve.sh [options] <model-filename> <segment.txt> [segment.txt ...]
#
# Runs each segment twice and compares the two per-token dumps:
#   deep arm     -c $DEEP_CTX --chunks 1 --ppl-first 0
#                every token sees everything before it, so context depth grows with position
#   shallow arm  -c $SHALLOW_CTX --ppl-stride $STRIDE
#                a sliding window recomputed from scratch, so depth stays bounded
#
# Per position, NLL(shallow) - NLL(deep) is what the distant context was worth there. Reading the
# deep arm alone does not work: later parts of a real session are more repetitive than earlier
# ones, so its curve mostly tracks content difficulty. Both arms score the identical tokens, so
# that difficulty cancels in the gap and what is left is attributable to context depth.
#
# Expect the gap to be ~0 at the earliest joined positions, where the shallow window still covers
# everything the deep arm has. That zero point is a free correctness check on any given run.
#
# <model-filename> looked up under /mnt/llm/llama.cpp/models (mounted /models) first, then under
# MODELS2_DIR (mounted /models2, default "/mnt/2508/Backup 2") -- same lookup as perplexity-run.sh.
#
# <segment.txt> looked up under CORPUS_DIR (default: /mnt/llm/llama-cpp-turboquant/data/logs/segtext,
# as written by extract_session_corpus.py --text-dir), or pass an absolute path. Segments shorter
# than DEEP_CTX tokens are skipped: a short one would leave the tail of the window unfilled and its
# positions would not be comparable with the rest.
#
# Options (all may also be set as env vars):
#   --deep-ctx N      context for the deep arm      (default 98304)
#   --shallow-ctx N   context for the shallow arm   (default 4096)
#   --stride N        stride for the shallow arm    (default 2048)
#   --bin N           position bin width in the merged curve (default 4096)
#   --batch N         batch size for the deep arm  (default 2048)
#   --out-dir DIR     where dumps and the curve land (default data/corpus/context-curve)
#   --extra "FLAGS"   extra llama-perplexity flags for both arms (offload, KV types, ...)
#
# Note the shallow arm's real window is SHALLOW_CTX + STRIDE/2: llama-perplexity pads n_ctx by
# half a stride when --ppl-stride is set. The dumps record the depth they actually achieved, and
# merge_context_curve.py reports it, so read that rather than assuming from the flags.
set -euo pipefail

DEEP_CTX="${DEEP_CTX:-98304}"
SHALLOW_CTX="${SHALLOW_CTX:-4096}"
STRIDE="${STRIDE:-2048}"
BIN="${BIN:-4096}"
# deep-arm batch. lower it to cut peak host memory: the deep arm holds n_batch * n_vocab floats
# for scoring, ~1.2 GB at 2048 on a 152K vocab.
BATCH="${BATCH:-2048}"
OUT_DIR="${OUT_DIR:-/mnt/llm/llama-cpp-turboquant/data/corpus/context-curve}"
EXTRA="${EXTRA:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --deep-ctx)    DEEP_CTX="$2";    shift 2 ;;
        --shallow-ctx) SHALLOW_CTX="$2"; shift 2 ;;
        --stride)      STRIDE="$2";      shift 2 ;;
        --bin)         BIN="$2";         shift 2 ;;
        --batch)       BATCH="$2";       shift 2 ;;
        --out-dir)     OUT_DIR="$2";     shift 2 ;;
        --extra)       EXTRA="$2";       shift 2 ;;
        -h|--help)     sed -n '2,40p' "$0"; exit 0 ;;
        --) shift; break ;;
        -*) echo "ERROR: unknown option $1" >&2; exit 1 ;;
        *) break ;;
    esac
done

MODEL="${1:?Usage: ppl-context-curve.sh [options] <model-filename> <segment.txt> [...]}"
shift
[ $# -gt 0 ] || { echo "ERROR: at least one segment file required" >&2; exit 1; }

MODELS_DIR=/mnt/llm/llama.cpp/models
MODELS2_DIR="${MODELS2_DIR:-/mnt/2508/Backup 2}"
CORPUS_DIR="${CORPUS_DIR:-/mnt/llm/llama-cpp-turboquant/data/logs/segtext}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE=llama-cpp-turboquant-llama-cpp:latest

# keyed by host path so a repeated directory is only mounted once
declare -A MOUNTS=()

if [ -f "$MODELS_DIR/$MODEL" ]; then
    MOUNTS["$MODELS_DIR"]=/models
    MODEL_PATH="/models/$MODEL"
elif [ -f "$MODELS2_DIR/$MODEL" ]; then
    MOUNTS["$MODELS2_DIR"]=/models2
    MODEL_PATH="/models2/$MODEL"
else
    echo "ERROR: model '$MODEL' not found under $MODELS_DIR or $MODELS2_DIR" >&2
    exit 1
fi

SEG_HOST=()
SEG_NAME=()
for seg in "$@"; do
    if [[ "$seg" = /* ]]; then host="$seg"; else host="$CORPUS_DIR/$seg"; fi
    [ -f "$host" ] || { echo "ERROR: segment '$seg' not found (looked at $host)" >&2; exit 1; }
    host="$(cd "$(dirname "$host")" && pwd)/$(basename "$host")"   # same absolute-path rule
    MOUNTS["$(dirname "$host")"]=""
    SEG_HOST+=("$host")
    SEG_NAME+=("$(basename "${host%.txt}")")
done

# corpus dirs get numbered container paths, assigned after dedup so each is stable
i=0
for host in "${!MOUNTS[@]}"; do
    if [ -z "${MOUNTS[$host]}" ]; then MOUNTS["$host"]="/corpus$i"; i=$((i+1)); fi
done

MOUNT_ARGS=()
for host in "${!MOUNTS[@]}"; do MOUNT_ARGS+=(-v "$host:${MOUNTS[$host]}"); done

# cap the shallow arm at the deep arm's reach. left uncapped it walks the whole segment, which is
# hundreds of extra full-window recomputes past the last position the two arms can be compared at.
# EFF is llama-perplexity's actual window: it pads n_ctx by half a stride under --ppl-stride.
# this only holds because the shallow arm below passes -b equal to its context. left at the
# default 2048, a shallow context under 2048 makes llama-perplexity set n_parallel = n_batch/n_ctx
# and multiply the context by it, so the window silently comes out wider than asked for -- and the
# resulting multi-sequence batch then fails to find a memory slot.
EFF=$((SHALLOW_CTX + STRIDE / 2))
if [ "$EFF" -ge "$DEEP_CTX" ]; then
    echo "ERROR: shallow window ($EFF) is not smaller than deep context ($DEEP_CTX);" >&2
    echo "       there would be no depth difference to measure" >&2
    exit 1
fi
SHALLOW_CHUNKS=$(( (DEEP_CTX - EFF + STRIDE - 1) / STRIDE + 1 ))
echo "shallow arm: window $EFF, stride $STRIDE, $SHALLOW_CHUNKS chunks to reach $DEEP_CTX" >&2

mkdir -p "$OUT_DIR"
# docker -v rejects a relative source as an invalid volume name, so resolve after creating
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
MOUNTS["$OUT_DIR"]=/out
MOUNT_ARGS+=(-v "$OUT_DIR:/out")

run_ppl() {
    # shellcheck disable=SC2086  # EXTRA is deliberately word-split into flags
    docker run --rm --gpus all "${MOUNT_ARGS[@]}" \
        --entrypoint /app/llama-perplexity "$IMAGE" \
        -m "$MODEL_PATH" -ngl 99 -fa on $EXTRA "$@"
}

for idx in "${!SEG_HOST[@]}"; do
    host="${SEG_HOST[$idx]}"
    name="${SEG_NAME[$idx]}"
    cpath="${MOUNTS[$(dirname "$host")]}/$(basename "$host")"

    echo "=== $name ===" >&2

    echo "  deep arm (-c $DEEP_CTX, whole window)" >&2
    if ! run_ppl -f "$cpath" -c "$DEEP_CTX" -b "$BATCH" --chunks 1 --ppl-first 0 \
            --ppl-token-dump "/out/$name.deep.tsv" >"$OUT_DIR/$name.deep.log" 2>&1; then
        echo "  !! deep arm failed, see $OUT_DIR/$name.deep.log -- skipping segment" >&2
        continue
    fi

    echo "  shallow arm (-c $SHALLOW_CTX, stride $STRIDE)" >&2
    if ! run_ppl -f "$cpath" -c "$SHALLOW_CTX" -b "$SHALLOW_CTX" \
            --ppl-stride "$STRIDE" --chunks "$SHALLOW_CHUNKS" \
            --ppl-token-dump "/out/$name.shallow.tsv" >"$OUT_DIR/$name.shallow.log" 2>&1; then
        echo "  !! shallow arm failed, see $OUT_DIR/$name.shallow.log -- skipping segment" >&2
        continue
    fi
done

echo "=== merging ===" >&2
python3 "$SCRIPT_DIR/merge_context_curve.py" "$OUT_DIR" --bin "$BIN" \
    --tsv "$OUT_DIR/context-curve.tsv"
