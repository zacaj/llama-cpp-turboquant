#!/bin/bash
# Run the full K/V cache-type matrix supported by this image's CUDA build against a bf16
# reference, and print a beellama-README-style report table.
#
# The pair list below is exactly the fattn-vec-instance-*.cu set in
# ggml/src/ggml-cuda/CMakeLists.txt (this image is built with -DGGML_CUDA_FA_ALL_QUANTS=OFF,
# so ONLY these pairs are compiled; anything else fails at kernel dispatch). Keep this list in
# sync with that CMakeLists if the build config changes.
#
# Usage: kld-run-matrix.sh <model-filename> <reference-file> [ctx] [chunks] [results-tsv]
#
# Resumable: if results-tsv already exists, SUCCESSFUL pairs are skipped and appended to; pairs
# that previously ERRORed are retried (a transient crash and a genuinely-unsupported combo look
# the same in the TSV, so we'd rather re-pay a few seconds re-failing a known-bad combo than
# silently skip one that failed only because of the crash). Set RESUME=0 to force a fresh start
# (overwrites results-tsv). This is the default since this box has a history of WSL/GPU crashes
# mid-sweep.
#
# Safety: if the first 2 pairs actually attempted (not skipped) both fail, the sweep aborts rather
# than burning through all 38 -- two failures right at the start almost always means something
# global is wrong (bad OUTPUT_DIR, missing/corrupt reference file, GPU unavailable), not two
# unrelated per-combo issues.
set -uo pipefail

MODEL="${1:?Usage: kld-run-matrix.sh <model-filename> <reference-file> [ctx] [chunks] [results-file]}"
REF="${2:?reference file required}"
CTX="${3:-24000}"
CHUNKS="${4:-1}"

if [ "$CTX" -lt 1000 ]; then
    echo "ERROR: ctx=$CTX looks like a mistake -- did you mean to put this in the chunks slot?" >&2
    echo "Usage: kld-run-matrix.sh <model-filename> <reference-file> [ctx] [chunks] [results-file]" >&2
    exit 1
fi
RESULTS="${5:-${OUTPUT_DIR}/$(basename "$MODEL" .gguf)-matrix-${CTX}ctx.tsv}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# PAIRS=(
#     "q8_0 q8_0" "q8_0 q6_0" "q8_0 q5_1" "q8_0 q5_0" "q8_0 q4_1" "q8_0 turbo4" "q8_0 q4_0" "q8_0 turbo3" "q8_0 turbo2"
#     "q6_0 q6_0" "q6_0 q5_1" "q6_0 q5_0" "q6_0 q4_1" "q6_0 turbo4" "q6_0 q4" "q6_0 turbo3" "q6_0 turbo2" 
#     "q5_1 q5_1" "q5_1 q5_0" "q5_1 q4_1" "q5_1 turbo4" "q5_1 q4_0" "q5_1 turbo3" "q5_1 turbo2"
#     "q5_0 q5_0" "q5_0 q4_1" "q5_0 turbo4" "q5_0 q4_0" "q5_0 turbo3" "q5_0 turbo2"
#     "q4_1 q4_1" "q4_1 turbo4" "q4_1 q4_0" "q4_1 turbo3" "q4_1 turbo2"
#     "turbo4 turbo4" "turbo4 q4_1" "turbo4 q4_0" "turbo4 turbo3" "turbo4 turbo2"
#     "q4_0 q4_0" "q4_0 turbo3" "q4_0 turbo2"
#     "turbo3 turbo3" "turbo3 turbo2"
#     "turbo2 turbo2" "turbo2 q8_0"
# )
# higher value pairs for quicker runs
PAIRS=(
    "q8_0 q8_0" "q8_0 q6_0" "q8_0 q5_1" "q8_0 q5_0" "q8_0 q4_1" "q8_0 turbo4" #"q8_0 turbo3" "q8_0 turbo2" #"q8_0 q4_0" 
    "q6_0 q6_0" "q6_0 q5_1" "q6_0 q5_0"  "q6_0 turbo4" "q6_0 q4_0" #"q6_0 turbo3" #"q6_0 turbo2" "q6_0 q4_1"
    "q5_1 q5_1" "q5_1 q5_0"  "q5_1 turbo4" #"q5_1 q4_0" #"q5_1 turbo3" #"q5_1 turbo2""q5_1 q4_1"
    "q5_0 q5_0" "q5_0 q4_1" "q5_0 turbo4"  #"q5_0 turbo3" # "q5_0 turbo2" "q5_0 q4_0"
    #"q4_1 q4_1"  "q4_1 q4_0" #"q4_1 turbo3" #"q4_1 turbo2" "q4_1 turbo4"
    "turbo4 turbo4" "turbo4 q4_0" #"turbo4 turbo3" #"turbo4 turbo2""turbo4 q4_1" 
    #"q4_0 q4_0" #"q4_0 turbo3" #"q4_0 turbo2"
    "turbo3 turbo3" # "turbo3 turbo2"
    # "turbo2 turbo2" "turbo2 q8_0"
)

TSV_HEADER="ctk\tctv\tbpw_k\tbpw_v\ttotal_bpw\tpct_f16\tmean_kld\tmean_kld_stderr\tkld_999\ttail_n\tchunks_used\tconverged_early\tprecision\tppl_q\tppl_base\tsame_top_p\tkld_per_ppl\tdump_file\tlog_file"

RESUME="${RESUME:-1}"

DONE_PAIRS=()
if [ "$RESUME" = "1" ] && [ -f "$RESULTS" ]; then
    echo "Resuming: found existing $RESULTS -- keeping successes, retrying any errored pairs" >&2
    TMP_RESULTS=$(mktemp)
    awk -F'\t' 'NR==1 || $7 != "ERROR"' "$RESULTS" > "$TMP_RESULTS"
    mv "$TMP_RESULTS" "$RESULTS"
    while IFS=$'\t' read -r ctk ctv _rest; do
        DONE_PAIRS+=("$ctk $ctv")
    done < <(tail -n +2 "$RESULTS")
else
    echo -e "$TSV_HEADER" > "$RESULTS"
fi

is_done() {
    local pair="$1"
    for d in "${DONE_PAIRS[@]-}"; do
        [ "$d" = "$pair" ] && return 0
    done
    return 1
}

print_summary() {
    local row="$1"
    IFS=$'\t' read -r ctk ctv bpw_k bpw_v total_bpw pct_f16 mean_kld mean_kld_stderr kld999 tail_n chunks_used converged precision ppl_q ppl_base same_top kld_per_ppl dump_file log_file <<< "$row"
    if [ "$mean_kld" = "ERROR" ]; then
        echo "  -> FAILED  ($ctk/$ctv)  log: $log_file" >&2
    else
        local conv_note=""
        [ "$converged" = "1" ] && conv_note=" (early-stopped)"
        echo "  -> ${ctk}/${ctv}: precision=${precision}%  mean_kld=${mean_kld}+/-${mean_kld_stderr}  chunks=${chunks_used}${conv_note}  size=${pct_f16}% of f16  ppl_base=${ppl_base}  kld/ppl=${kld_per_ppl}" >&2
    fi
}

TOTAL=${#PAIRS[@]}
i=0
ATTEMPTED=0
FAILED=0
for pair in "${PAIRS[@]}"; do
    i=$((i+1))
    read -r ctk ctv <<< "$pair"
    if is_done "$pair"; then
        echo "[$i/$TOTAL] $MODEL: $ctk / $ctv ... already in $RESULTS, skipping" >&2
        continue
    fi
    echo "[$i/$TOTAL] $MODEL: $ctk / $ctv ..." >&2
    ROW=$(bash "$SCRIPT_DIR/kld-run-pair.sh" "$MODEL" "$REF" "$ctk" "$ctv" "$CTX" "$CHUNKS")
    echo "$ROW" >> "$RESULTS"
    print_summary "$ROW"

    ATTEMPTED=$((ATTEMPTED+1))
    if [ "$(cut -f7 <<< "$ROW")" = "ERROR" ]; then
        FAILED=$((FAILED+1))
    fi
    if [ "$ATTEMPTED" -eq 2 ] && [ "$FAILED" -eq 2 ]; then
        echo "ABORT: the first 2 attempted pairs both failed -- this looks like a global problem" >&2
        echo "(wrong OUTPUT_DIR, missing/corrupt reference file, GPU unavailable), not two" >&2
        echo "unrelated per-combo issues. Stopping the rest of the sweep; check the log paths above." >&2
        break
    fi
done

echo "Results (tsv): $RESULTS" >&2

# --- beellama-style markdown report ---
# Always written to MD_PATH (not just printed to stdout) -- relying on the caller to redirect
# stdout to the right place has already lost a report once when output got mixed into a scratch log.
MD_PATH="${RESULTS%.tsv}.md"

{
echo ""
echo "## KV cache precision matrix -- $MODEL (ctx=$CTX)"
echo ""
echo "Sampling-noise note: the 99.9% precision column comes from the 99.9th-percentile KLD, an"
echo "extreme order statistic drawn from only ~tail_n tail tokens (see column) -- with tail_n in"
echo "the 10-15 range, treat precision differences of a few points between neighboring rows as"
echo "noise, not a reliable ranking. Mean KLD (+/- its stderr, both from the tool's own Gaussian"
echo "error propagation over ALL scored tokens) is the much more stable statistic and is a better"
echo "basis for ranking close combos."
echo ""
echo "Each combo runs with --kld-early-stop by default, so chunks_used may be less than the"
echo "reference's full chunk count (converged column marks it) -- that's expected, not a partial"
echo "failure. Raw per-token values + full metadata for every combo are saved under dump_file,"
echo "so results here can be re-pooled or extended later without rerunning. Full docker logs are"
echo "kept under log_file (not a /tmp scratch path) for later reference."
echo ""
echo "ppl_base is this MODEL's own bf16 baseline perplexity on this corpus -- it's constant across"
echo "every row here (same model, same reference), so within one model's matrix it's only useful"
echo "for comparing this report against a DIFFERENT model's matrix: a lower ppl_base means the model"
echo "found this text easier to begin with, which by itself can produce lower mean_kld even with no"
echo "real quantization-robustness difference. kld_per_ppl (mean_kld / ppl_base) is a basic, not"
echo "rigorously derived, normalization of that same idea into one number -- use it as a rough cue"
echo "when eyeballing this model's numbers against another model's, not as a precise ranking metric."
echo ""
echo "| K / V | bpw (K/V) | % of f16 size | 99.9% precision | tail n | chunks used | Mean KLD (+/- stderr) | ppl_base | kld_per_ppl |"
echo "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"

tail -n +2 "$RESULTS" | while IFS=$'\t' read -r ctk ctv bpw_k bpw_v total_bpw pct_f16 mean_kld mean_kld_stderr kld999 tail_n chunks_used converged precision ppl_q ppl_base same_top kld_per_ppl dump_file log_file; do
    if [ "$mean_kld" = "ERROR" ]; then
        echo "| ${ctk} / ${ctv} | ${bpw_k}/${bpw_v} | ${pct_f16}% | FAILED | -- | -- | -- | -- | -- |"
    else
        chunks_disp="$chunks_used"
        if [ "$converged" = "1" ]; then
            chunks_disp="${chunks_used} (early-stopped)"
        fi
        echo "| ${ctk} / ${ctv} | ${bpw_k}/${bpw_v} | ${pct_f16}% | ${precision}% | ${tail_n} | ${chunks_disp} | ${mean_kld} +/- ${mean_kld_stderr} | ${ppl_base} | ${kld_per_ppl} |"
    fi
done
} | tee "$MD_PATH"

echo "Markdown report: $MD_PATH" >&2
