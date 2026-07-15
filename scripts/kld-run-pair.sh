#!/bin/bash
# Run one KV cache type pairing against a bf16 reference file and report KL-divergence stats.
#
# Usage: kld-run-pair.sh <model-filename> <reference-file-basename-or-path> <ctk> <ctv> [ctx] [chunks]
#
# Prints one TSV line to stdout:
#   ctk  ctv  bpw_k  bpw_v  total_bpw  pct_f16  mean_kld  mean_kld_stderr  kld_99.9  tail_n
#   chunks_used  converged_early  precision_pct  ppl_q  ppl_base  same_top_p_pct  kld_per_ppl
#   dump_file  log_file
# "precision" is 100*exp(-kld_99.9), per the beellama README formula, with the bf16-vs-bf16
# noise floor term dropped (it's ~1e-4-1e-3, negligible next to real KV-quant KLD in this range).
# mean_kld_stderr is the tool's own Gaussian-error-propagation stderr on the MEAN (stable, ~thousands
# of samples) -- it says nothing about the 99.9th percentile's noise, which comes from the extreme
# tail of a much smaller effective sample. tail_n = chunks_used * count_per_chunk * 0.001 is that
# tail sample count; treat 99.9%-precision differences between combos as noise until they're large
# relative to it (with tail_n ~12, expect single-digit-percentage-point swings from resampling alone).
#
# ppl_base is the MODEL's own bf16 baseline perplexity on this corpus -- it's what makes mean_kld
# comparable ACROSS different models. Two models can rank differently on raw mean_kld simply because
# one had a harder time on this text to begin with (higher ppl_base = more of its own uncertainty,
# less margin to lose to KV quantization noise); ppl_base tells you whether a low KLD reflects real
# quantization robustness or just a model with headroom to spare. kld_per_ppl = mean_kld / ppl_base is
# a basic normalization of the same idea into a single number for at-a-glance cross-model comparison --
# it is NOT a rigorously derived quantity (mean_kld is in nats, ppl_base is not), just a quick ratio to
# flag "does this model's KLD look large or small relative to how hard the text is for it".
#
# By default this runs with --kld-early-stop (see KLD_EARLY_STOP=0 to disable), so chunks_used may be
# less than the reference's full chunk count -- that's expected, not an error; converged_early=1 flags it.
# A quiet-streak counter increments on chunks where stderr of the running mean KLD is under
# max(KLD_EARLY_STOP_REL_STDERR * |mean|, KLD_EARLY_STOP_ABS_FLOOR), and decrements (floor 0, not a
# hard reset) otherwise -- converged once the streak hits KLD_EARLY_STOP_MIN_QUIET, so one isolated
# hard chunk costs one step, not the whole streak. The abs floor (default 0.0002) matters for
# near-lossless combos: as mean_kld -> 0 a pure relative bar becomes unreachable even though the
# absolute noise is already tiny and well-resolved.
# --kld-dump-values is always passed, writing raw per-token (kld, p_diff) values + a metadata header
# (model, ctk/ctv, reference file, chunk counts, computed results) to dump_file, so results stay
# self-describing and poolable/resumable later without needing to regenerate anything.
#
# The docker log is written to a persistent path under log_file (like dump_file, not a /tmp scratch
# file), and that path is printed to stderr BEFORE the run starts so you can `tail -f` it live.
#
# On failure (unsupported combo, crash, OOM) all numeric fields print as ERROR and the log path
# is printed to stderr instead of aborting, so a caller can sweep a whole matrix unattended.
set -uo pipefail

MODEL="${1:?Usage: kld-run-pair.sh <model-filename> <reference-file> <ctk> <ctv> [ctx] [chunks]}"
REF="${2:?reference file required}"
CTK="${3:?ctk required}"
CTV="${4:?ctv required}"
CTX="${5:-24000}"
CHUNKS="${6:-1}"

if [ "$CTX" -lt 1000 ]; then
    echo "ERROR: ctx=$CTX looks like a mistake -- did you mean to put this in the chunks slot?" >&2
    echo "Usage: kld-run-pair.sh <model-filename> <reference-file> <ctk> <ctv> [ctx] [chunks]" >&2
    exit 1
fi
CORPUS_FILE="${CORPUS_FILE:-wikitext-2-raw/wiki.test.raw}"

MODELS_DIR=/mnt/llm/llama.cpp/models
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORPUS_DIR="$REPO_DIR/data/corpus"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/llm/models/kld-reference}"
DUMP_DIR="${KLD_DUMP_DIR:-$OUTPUT_DIR/dumps}"
LOG_DIR="${KLD_LOG_DIR:-$OUTPUT_DIR/logs}"
IMAGE=llama-cpp-turboquant-llama-cpp:latest

KLD_EARLY_STOP="${KLD_EARLY_STOP:-1}"
KLD_EARLY_STOP_REL_STDERR="${KLD_EARLY_STOP_REL_STDERR:-0.03}"
KLD_EARLY_STOP_ABS_FLOOR="${KLD_EARLY_STOP_ABS_FLOOR:-0.0002}"
KLD_EARLY_STOP_MIN_QUIET="${KLD_EARLY_STOP_MIN_QUIET:-3}"

mkdir -p "$DUMP_DIR" "$LOG_DIR"

REF_BASENAME="$(basename "$REF")"
MODEL_BASE="$(basename "$MODEL" .gguf)"
DUMP_NAME="${MODEL_BASE}-${CTK}-${CTV}-${CTX}ctx.kld-dump.tsv"
DUMP_PATH="$DUMP_DIR/$DUMP_NAME"
LOG_NAME="${MODEL_BASE}-${CTK}-${CTV}-${CTX}ctx.log"
LOG="$LOG_DIR/$LOG_NAME"

declare -A BPW=( [q8_0]=8.5 [q6_0]=6.5 [q5_1]=6 [q5_0]=5.5 [q4_1]=5 [q4_0]=4.5 [turbo4]=4.125 [turbo3]=3.125 [turbo2]=2.125 )
BPW_K=${BPW[$CTK]:-NA}
BPW_V=${BPW[$CTV]:-NA}
if [ "$BPW_K" != "NA" ] && [ "$BPW_V" != "NA" ]; then
    TOTAL_BPW=$(echo "scale=3; $BPW_K + $BPW_V" | bc)
    PCT_F16=$(echo "scale=1; 100 * $TOTAL_BPW / 32" | bc)
else
    TOTAL_BPW=NA
    PCT_F16=NA
fi

fail() {
    echo -e "${CTK}\t${CTV}\t${BPW_K}\t${BPW_V}\t${TOTAL_BPW}\t${PCT_F16}\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR\t${LOG}"
    echo "=== $CTK/$CTV failed, log: $LOG ===" >&2
}

EARLY_STOP_ARGS=()
if [ "$KLD_EARLY_STOP" = "1" ]; then
    EARLY_STOP_ARGS=(--kld-early-stop --kld-early-stop-rel-stderr "$KLD_EARLY_STOP_REL_STDERR" --kld-early-stop-abs-floor "$KLD_EARLY_STOP_ABS_FLOOR" --kld-early-stop-min-quiet "$KLD_EARLY_STOP_MIN_QUIET")
fi

echo "=== $CTK/$CTV: starting, log -> $LOG ===" >&2

if ! docker run --rm --gpus all \
    -e "TURBO_AUTO_ASYMMETRIC=0" \
    -e "TURBO_LAYER_ADAPTIVE=0" \
    -v "$MODELS_DIR":/models \
    -v "$CORPUS_DIR":/corpus:ro \
    -v "$OUTPUT_DIR":/output \
    -v "$REF":"$REF" \
    -v "$DUMP_DIR":/dumpout \
    "$IMAGE" \
    --perplexity -m "/models/$MODEL" \
    -f "/corpus/$CORPUS_FILE" \
    -c "$CTX" --chunks "$CHUNKS" \
    --n-cpu-moe 0 --no-mmap -dio \
    -b 2048 -ub 128 --flash-attn on \
    -ctk "$CTK" -ctv "$CTV" -fit off \
    "${EARLY_STOP_ARGS[@]}" \
    --kld-dump-values "/dumpout/$DUMP_NAME" \
    --kl-divergence-base "/output/$REF_BASENAME" --kl-divergence \
    --n-gpu-layers 99 -lv 4 --log-timestamps > "$LOG" 2>&1; then
    fail
    exit 0
fi

MEAN_KLD=$(grep -oP 'Mean\s+KLD:\s+\K[0-9.eE+-]+' "$LOG" | head -1)
MEAN_KLD_STDERR=$(grep -oP 'Mean\s+KLD:\s+[0-9.eE+-]+\s+\S+\s+\K[0-9.eE+-]+' "$LOG" | head -1)
KLD999=$(grep -oP '99\.9%\s+KLD:\s+\K[0-9.eE+-]+' "$LOG" | head -1)
PPL_Q=$(grep -oP 'Mean PPL\(Q\)\s*:\s+\K[0-9.]+' "$LOG" | head -1)
PPL_BASE=$(grep -oP 'Mean PPL\(base\)\s*:\s+\K[0-9.]+' "$LOG" | head -1)
SAME_TOP=$(grep -oP 'Same top p:\s+\K[0-9.]+' "$LOG" | head -1)

if [ -z "$KLD999" ]; then
    fail
    exit 0
fi

PRECISION=$(echo "scale=10; 100 * e(-1 * ($KLD999))" | bc -l 2>/dev/null | awk '{printf "%.2f", $1}')
PRECISION=${PRECISION:-NA}

KLD_PER_PPL=NA
if [ -n "${PPL_BASE:-}" ] && [ "$PPL_BASE" != "0" ]; then
    KLD_PER_PPL=$(echo "scale=8; $MEAN_KLD / $PPL_BASE" | bc -l 2>/dev/null | awk '{printf "%.6f", $1}')
    KLD_PER_PPL=${KLD_PER_PPL:-NA}
fi

# tail_n / chunks_used / converged_early come from the dump file's own metadata header now --
# more accurate than estimating from reference file size, and correct whether or not early-stop fired.
CHUNKS_USED=NA
CONVERGED=NA
TAIL_N=NA
if [ -f "$DUMP_PATH" ]; then
    N_CHUNK_ACTUAL=$(grep -m1 '^n_chunk_actual=' "$DUMP_PATH" | cut -d= -f2)
    COUNT_PER_CHUNK=$(grep -m1 '^count_per_chunk=' "$DUMP_PATH" | cut -d= -f2)
    CONVERGED=$(grep -m1 '^converged_early=' "$DUMP_PATH" | cut -d= -f2)
    if [ -n "$N_CHUNK_ACTUAL" ] && [ -n "$COUNT_PER_CHUNK" ]; then
        CHUNKS_USED="$N_CHUNK_ACTUAL"
        TAIL_N=$(echo "scale=1; $N_CHUNK_ACTUAL * $COUNT_PER_CHUNK * 0.001" | bc -l | awk '{printf "%.1f", $1}')
    fi
fi

echo -e "${CTK}\t${CTV}\t${BPW_K}\t${BPW_V}\t${TOTAL_BPW}\t${PCT_F16}\t${MEAN_KLD}\t${MEAN_KLD_STDERR:-NA}\t${KLD999}\t${TAIL_N}\t${CHUNKS_USED}\t${CONVERGED}\t${PRECISION}\t${PPL_Q}\t${PPL_BASE}\t${SAME_TOP}\t${KLD_PER_PPL}\t${DUMP_PATH}\t${LOG}"
