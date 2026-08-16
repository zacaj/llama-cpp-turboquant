#!/bin/bash
# Run llama-perplexity across the cartesian product of one or more models, one or more
# expert_used_count (k) overrides, and one or more corpora, keeping the full per-chunk log for
# every combination and writing a combined summary table (TSV + markdown).
#
# Usage:
#   CORPUS=<file[,file2,...]> [K_VALUES="8 12 16" K_KEY=<arch>] \
#     [PRUNE_LEVELS="0 64 102 128" PROFILE_CSV=<profile.csv>] \
#     perplexity-sweep.sh MODEL[.gguf] [MODEL2 ...]
#
# Why per-chunk logs, not just the final PPL: the final estimate is a geometric mean dominated by
# outliers -- see docs/moe-expert-count-analysis.md for cases where it pointed the wrong way until
# decomposed per chunk. Compare the full traces, not just summary.tsv's PPL column.
#
# Resumable: each log's first line is a marker recording exactly what produced it (model, k,
# K_KEY, corpus, ctx, chunks). A combo is skipped only if its log's marker matches this exact
# invocation AND the log contains a "Final estimate: PPL" line -- so a log left over from a
# different invocation (a since-fixed K_KEY typo, a different ctx, etc.) is never mistaken for
# "already complete" just because it happens to contain a valid-looking result. A partial/truncated
# log has no matching marker+estimate, so it re-runs cleanly and overwrites the stale log --
# safe to re-run after a crash, a reboot, or a manually Ctrl-C'd sweep without losing finished work.
# Set FORCE=1 to ignore existing logs and re-run everything regardless.
#
# Two summary outputs, to avoid ever re-deriving "what changed this run" vs "everything so far"
# from the same append-only file (that's what used to duplicate rows on resume):
#   - $OUT_DIR/summary.tsv           the all-time table. Fully REGENERATED every invocation by
#                                     scanning every *.log in OUT_DIR and re-reading its marker
#                                     line -- never appended to -- so it can never accumulate
#                                     duplicate or stale rows no matter how many times you resume.
#   - $OUT_DIR/summary-run-<ts>.tsv  just this invocation's combos (skipped or freshly run), named
#                                     with this run's timestamp so successive invocations never
#                                     collide or need merging.
#
# Env vars:
#   CORPUS      REQUIRED. One or more corpus files, comma-separated, e.g.
#               CORPUS=ppl_sample.txt or CORPUS=a.txt,b.txt. Looked up under CORPUS_DIR, or pass
#               absolute paths.
#   K_VALUES    space/comma-separated expert_used_count values to sweep, e.g. "8 12 16".
#               Unset/empty = run each model/corpus once with no override (trained k).
#   K_KEY       GGUF metadata key PREFIX for the override, e.g. "qwen35moe" (not the full key --
#               the script appends ".expert_used_count" itself) ->
#               --override-kv qwen35moe.expert_used_count=int:$K. Required if K_VALUES is set.
#               (Different model architectures use different key prefixes.) Checked against each
#               model's real metadata before the sweep starts, since --override-kv silently no-ops
#               on a key that doesn't match anything -- a wrong K_KEY would otherwise produce
#               plausible-looking numbers that are secretly all the same untouched trained k. A
#               model missing the key is treated as non-MoE (or just a different arch): it runs
#               once per corpus with no override, as a reference point, rather than erroring --
#               unless *every* model is missing it, which is almost certainly a K_KEY typo. Results
#               are cached (see KKEY_CACHE_FILE) so re-running the same models doesn't repeat the
#               ~15-20s-per-model metadata dump every time.
#   KKEY_CACHE_FILE   where the K_KEY check cache lives; default ~/.cache/perplexity-sweep-kkey.tsv.
#                     Keyed by (model path, size, mtime, K_KEY), so replacing a file at the same
#                     path invalidates its cache entry automatically.
#   PRUNE_LEVELS      space/comma-separated expert-removal counts to sweep per MoE model, e.g.
#               "0 64 102 128" (0 = the unpruned model). Unset/empty = today's behavior, one entry
#               per input model. Most prior sweeps used a single hand-picked n=102 with no evidence
#               that's actually the best tradeoff point -- this compares several levels in one
#               invocation, the same way K_VALUES compares several expert_used_count overrides. Only
#               a model that is MoE *and* whose expert_count matches PROFILE_CSV's total expert
#               count (i.e. it's the unpruned model the profile was built from) is expanded; a model
#               that's already pruned, or isn't MoE at all, can't be mapped onto these levels with
#               this profile and runs once, unchanged -- exactly like a model missing K_KEY, not an
#               error (sweeping a base model alongside an already-pruned one in one invocation is
#               normal). For level N>0, a cached GGUF at PRUNE_DIR/<model>.pruned-<profile-tag>-n<N>.gguf
#               is reused if present, else generated via scripts/lowest_experts_from_profile.py
#               (profile -> per-layer pruning spec) + scripts/prune_moe_experts.py (spec -> pruned
#               GGUF) -- the same two steps you'd run by hand, see CLAUDE.md's entry for them. The
#               generated GGUF's own metadata records which profile CSV, expert-removal count, and
#               source model produced it (prune_moe_experts.py's --note flag), so that's recoverable
#               via gguf_dump.py even if the file is later renamed or moved off of PRUNE_DIR.
#   PROFILE_CSV       REQUIRED if PRUNE_LEVELS is set. A moe-expert-profile.sh output CSV (prefer
#               its .weighted.csv sibling -- see that script's header and CLAUDE.md for why
#               weight-mass pruning beats raw-count pruning on PPL).
#   PRUNE_DIR         where generated per-level pruning-spec CSVs and pruned GGUFs are cached;
#               default $MODELS3_DIR/pruned. Point this at a directory of already-pruned GGUFs
#               (named per the convention above) to reuse them instead of regenerating. Pruned
#               GGUFs run several GB each -- pick a disk with room; MODELS3_DIR's default
#               (/mnt/2508/Archive) has it, $OUT_DIR's default (under /mnt/llm) usually doesn't.
#   PRUNE_FILL_MISSING_LAYERS   --fill-missing-layers value forwarded to
#               lowest_experts_from_profile.py; default is each model's own last block index
#               (from its GGUF metadata), so an MoE layer the profiling run never reached (e.g. an
#               MTP/draft head -- common, since a normal forward-pass profiling corpus doesn't
#               exercise it) still gets a synthesized prune spec instead of being silently left
#               unpruned while every other layer's router shrinks -- llama.cpp then refuses to load
#               on the resulting per-layer shape mismatch. Set only to override that default.
#   PRUNE_CACHE_FILE  where the MoE/expert_count check cache lives; default
#                     ~/.cache/perplexity-sweep-prune.tsv. Same (path, size, mtime) keying as
#                     KKEY_CACHE_FILE.
#   CTX=4096          context size (-c). Also the chunk size perplexity reports against.
#   CHUNKS=-1         --chunks value; -1 runs to exhaustion (floor(tokens/CTX) chunks).
#   OUT_DIR=...       where to write logs/summary; default ./ppl-sweep-<first-corpus-basename>
#   WAIT_FOR_GPU=0    if 1, block until no other llama-* container is running before starting.
#   EXTRA_ARGS=""     extra flags appended to every llama-perplexity invocation.
#   FORCE=0           if 1, re-run every combo even if a completed log already exists. Also forces
#                     regenerating a pruned GGUF even if PRUNED_PATH already exists (e.g. after
#                     fixing PROFILE_CSV, a pruning bug, or picking different PRUNE_LEVELS logic).
#   OUTLIER_RATIO=3   a chunk counts as an outlier if its own PPL is more than this many times the
#                      run's median chunk PPL (see "Per-run variability" below).
#
# Per-run variability: PPL is a geometric mean, so a run dominated by a few bad chunks can report
# the same headline number as one that's uniformly mediocre -- see docs/moe-expert-count-analysis.md
# for several cases where that distinction mattered a lot and was easy to miss. Each summary row
# also reports "gsd" (geometric standard deviation of the per-chunk PPLs: exp(stdev(ln(chunk_ppl))))
# and "outliers" (chunks more than OUTLIER_RATIO x the median chunk PPL, as a fraction of the total).
# Both are scale-invariant -- unlike the +/- from llama-perplexity, they don't just grow with PPL
# itself, so a gsd of e.g. 3x means the same thing whether the run's PPL is 5 or 50. gsd close to
# 1.0x with 0 outliers means the chunks are all in the same ballpark; a high gsd or nonzero outlier
# count is the signal to go decompose the per-chunk trace before trusting the headline PPL.
#
# Model lookup: bare filenames resolved against MODELS_DIR / MODELS2_DIR / MODELS3_DIR (default
# /mnt/llm/llama.cpp/models, "/mnt/2508/Backup 2", /mnt/2508/Archive), in that order. Corpus lookup
# is against CORPUS_DIR (default /mnt/llm/llama.cpp/models). Absolute paths mounted verbatim.
# Every distinct host directory referenced by any model or corpus is mounted exactly once
# (deduplicated, and each gets its own container mount point -- no name collisions between
# differently-sourced absolute paths).
#
# Offload: -ngl is deliberately NOT set, and `-fit on` is passed instead, so models larger than
# available VRAM degrade to a partial offload rather than OOMing. Perplexity is unaffected by
# where layers run (only speed is), which keeps differently-sized models/quants comparable.
set -euo pipefail

# Without this, Ctrl-C during a `docker run` only kills that one container: docker run returns
# non-zero, the `|| echo "(run failed ...)"` below swallows it, and the for-loop just moves on to
# the next combination. Trap SIGINT/SIGTERM explicitly so an interrupt aborts the whole sweep.
# Already-completed logs are untouched; re-run the same command to resume (see header comment).
trap 'echo; echo "Interrupted -- aborting sweep. Re-run the same command to resume." >&2; exit 130' INT TERM

[ "$#" -ge 1 ] || { echo "ERROR: at least one model required" >&2; exit 1; }
MODELS=("$@")

MODELS_DIR="${MODELS_DIR:-/mnt/llm/llama.cpp/models}"
MODELS2_DIR="${MODELS2_DIR:-/mnt/2508/Backup 2}"
MODELS3_DIR="${MODELS3_DIR:-/mnt/2508/Archive}"
CORPUS_DIR="${CORPUS_DIR:-/mnt/llm/llama.cpp/models}"
IMAGE="${IMAGE:-llama-cpp-turboquant-llama-cpp:latest}"

CTX="${CTX:-4096}"
CHUNKS="${CHUNKS:--1}"
WAIT_FOR_GPU="${WAIT_FOR_GPU:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
K_KEY="${K_KEY:-}"

: "${CORPUS:?Usage: CORPUS=file1.txt[,file2.txt,...] perplexity-sweep.sh MODEL [MODEL2 ...]}"
IFS=',' read -ra CORPUS_LIST <<< "$CORPUS"

if [ -n "${K_VALUES:-}" ]; then
    K_VALUES_NORM="${K_VALUES//,/ }"
    read -ra K_LIST <<< "$K_VALUES_NORM"
    [ -n "$K_KEY" ] || { echo "ERROR: K_KEY required when K_VALUES is set (e.g. K_KEY=qwen35moe)" >&2; exit 1; }
else
    K_LIST=("")   # single empty-string sentinel = no override, run at trained k
fi

OUT_DIR="${OUT_DIR:-./ppl-sweep}"
mkdir -p "$OUT_DIR"
echo "Logs and summary -> $OUT_DIR" >&2

# Seconds -> "1h23m", "4m05s", or "37s", for progress/ETA output.
fmt_duration() {
    local s="$1"
    if [ "$s" -ge 3600 ]; then
        printf '%dh%02dm' $((s / 3600)) $(((s % 3600) / 60))
    elif [ "$s" -ge 60 ]; then
        printf '%dm%02ds' $((s / 60)) $((s % 60))
    else
        printf '%ds' "$s"
    fi
}

# Read a completed log's per-chunk cumulative-PPL trace and print
# "n_chunks<TAB>gsd<TAB>outliers" -- see "Per-run variability" in the header comment.
chunk_stats() {
    local log="$1" ratio="${OUTLIER_RATIO:-3}"
    local cum
    cum="$(grep -oP '\[\d+\]\K[0-9.]+' "$log" | tr '\n' ' ')"
    [ -n "$cum" ] || { printf '0\t-\t-\n'; return; }
    awk -v ratio="$ratio" '
        {
            n = NF
            prev = 0
            for (i = 1; i <= n; i++) {
                if ($i !~ /^[0-9.]+$/) { err = 1; break }   # not a plain number -- bail out
                cur = log($i) * i
                if (cur - prev > 700) { err = 1; break }    # would overflow exp() below
                p = exp(cur - prev)
                chunk[i] = p
                logsum += log(p)
                prev = cur
            }
        }
        END {
            if (err || n < 1) { print "0\t-\t-"; exit }
            if (n < 2) { printf "%d\t1.00\t0/%d\n", n, n; exit }
            mean_log = logsum / n
            ss = 0
            for (i = 1; i <= n; i++) { d = log(chunk[i]) - mean_log; ss += d * d }
            gsd = exp(sqrt(ss / n))
            for (i = 1; i <= n; i++) sorted[i] = chunk[i]
            for (i = 2; i <= n; i++) {
                key = sorted[i]; j = i - 1
                while (j >= 1 && sorted[j] > key) { sorted[j + 1] = sorted[j]; j-- }
                sorted[j + 1] = key
            }
            median = (n % 2 == 1) ? sorted[(n + 1) / 2] : (sorted[n / 2] + sorted[n / 2 + 1]) / 2
            outliers = 0
            for (i = 1; i <= n; i++) if (chunk[i] > median * ratio) outliers++
            printf "%d\t%.2f\t%d/%d\n", n, gsd, outliers, n
        }
    ' <<< "$cum"
}

# Build one tab-separated summary row for a completed log, given its tags.
summary_row() {
    local log="$1" model_tag="$2" k_tag="$3" corpus_tag="$4"
    local ppl err gsd outliers
    ppl="$(grep -oP 'Final estimate: PPL = \K[0-9.]+' "$log" | tail -1 || true)"
    err="$(grep -oP 'Final estimate: PPL = [0-9.]+ \+/- \K[0-9.]+' "$log" | tail -1 || true)"
    IFS=$'\t' read -r _ gsd outliers < <(chunk_stats "$log") || true
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$model_tag" "$k_tag" "$corpus_tag" "${ppl:-FAILED}" "${err:-}" "$gsd" "$outliers" "$log"
}

# Rebuild the all-time summary.tsv from every completed log currently in OUT_DIR, keyed off each
# log's own marker line rather than what this invocation happened to touch. A log is the single
# source of truth for its combo, so there's nothing to deduplicate against a prior summary --
# only to regenerate from scratch, which is immune to the old append-duplicate bug by construction.
regenerate_summary() {
    printf 'model\tk\tcorpus\tppl\tstderr\tgsd\toutliers\tlog\n' > "$SUMMARY"
    local log marker model_tag k_tag corpus_tag
    for log in "$OUT_DIR"/*.log; do
        [ -e "$log" ] || continue
        marker="$(head -n 1 "$log")"
        [[ "$marker" == '# perplexity-sweep:'* ]] || continue
        grep -q 'Final estimate: PPL' "$log" || continue
        model_tag="$(grep -oP 'model=\K\S+' <<< "$marker")"
        k_tag="$(grep -oP ' k=\K\S+' <<< "$marker")"
        corpus_tag="$(grep -oP 'corpus=\K\S+' <<< "$marker")"
        summary_row "$log" "$model_tag" "$k_tag" "$corpus_tag" >> "$SUMMARY"
    done
}

# Resolve a bare filename against a search-dir list, or pass an absolute path through.
# Echoes "host_dir<TAB>basename".
resolve() {
    local name="$1"; shift
    if [[ "$name" = /* ]]; then
        echo -e "$(dirname "$name")\t$(basename "$name")"
        return 0
    fi
    local d
    for d in "$@"; do
        [ -f "$d/$name" ] && { echo -e "$d\t$name"; return 0; }
    done
    return 1
}

# Every distinct host directory gets its own sequential container mount point, so absolute paths
# from different sources can never collide on a shared bucket name.
declare -A MOUNT_OF=()
NEXT_MOUNT=0
mount_for() {
    local host="$1"
    if [ -z "${MOUNT_OF[$host]+x}" ]; then
        MOUNT_OF["$host"]="/mnt$NEXT_MOUNT"
        NEXT_MOUNT=$((NEXT_MOUNT + 1))
    fi
}

# Validate every model/corpus up front -- a typo partway through a long sweep is expensive.
declare -A MODEL_HOST=() MODEL_BASE=() MODEL_TAG_OF=()
for m in "${MODELS[@]}"; do
    IFS=$'\t' read -r h b < <(resolve "$m" "$MODELS_DIR" "$MODELS2_DIR" "$MODELS3_DIR") \
        || { echo "ERROR: model not found: $m" >&2; exit 1; }
    MODEL_HOST["$m"]="$h"; MODEL_BASE["$m"]="$b"
    mount_for "$h"
done

declare -A CORPUS_HOST=() CORPUS_BASE=()
for c in "${CORPUS_LIST[@]}"; do
    IFS=$'\t' read -r h b < <(resolve "$c" "$CORPUS_DIR") \
        || { echo "ERROR: corpus not found: $c" >&2; exit 1; }
    CORPUS_HOST["$c"]="$h"; CORPUS_BASE["$c"]="$b"
    mount_for "$h"
done

# PRUNE_LEVELS expands each MoE input model (whose expert_count matches PROFILE_CSV) into one
# effective model per "experts removed" level -- see the header comment for the full rationale.
# A model that doesn't qualify (not MoE, or expert_count doesn't match -- most likely already
# pruned) passes through as a single unchanged entry, same as the K_KEY-missing case below.
# Runs after model/corpus resolution (needs MODEL_HOST/MODEL_BASE) but before the K_KEY check and
# main loop, both of which iterate over MODELS -- rewriting MODELS here (plus MODEL_HOST/MODEL_BASE
# for the new synthetic per-level entries, and MODEL_TAG_OF for their display tags) means neither
# of those needs to know prune expansion happened at all.
if [ -n "${PRUNE_LEVELS:-}" ]; then
    : "${PROFILE_CSV:?PROFILE_CSV required when PRUNE_LEVELS is set (a moe-expert-profile.sh output CSV)}"
    [ -f "$PROFILE_CSV" ] || { echo "ERROR: PROFILE_CSV not found: $PROFILE_CSV" >&2; exit 1; }
    PROFILE_CSV="$(realpath "$PROFILE_CSV")"
    PROFILE_HOST_DIR="$(dirname "$PROFILE_CSV")"
    # Row count minus the header = total experts the profile covers; a model whose real
    # expert_count matches this is the unpruned model the profile was built from.
    PROFILE_N_EXPERT=$(( $(wc -l < "$PROFILE_CSV") - 1 ))

    PRUNE_LEVELS_NORM="${PRUNE_LEVELS//,/ }"
    read -ra PRUNE_LIST <<< "$PRUNE_LEVELS_NORM"

    PRUNE_DIR="${PRUNE_DIR:-$MODELS3_DIR/pruned}"
    mkdir -p "$PRUNE_DIR"
    mount_for "$PRUNE_DIR"
    REPO_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    PRUNE_CACHE_FILE="${PRUNE_CACHE_FILE:-$HOME/.cache/perplexity-sweep-prune.tsv}"
    mkdir -p "$(dirname "$PRUNE_CACHE_FILE")"
    touch "$PRUNE_CACHE_FILE"

    # Echoes "arch<TAB>expert_count<TAB>block_count" for a model (expert_count empty = not MoE).
    # Cached by (path, size, mtime), same rationale as model_has_k_key below. block_count (the
    # highest block index + 1) is used to fill --fill-missing-layers for lowest_experts_from_profile.py
    # below -- an MoE layer the profiling corpus never exercised (e.g. an MTP/draft head, which a
    # normal forward-pass profiling run doesn't reach) would otherwise be silently left unpruned by
    # lowest_experts_from_profile.py (no CSV row -> prune_moe_experts.py treats "not in spec" as
    # "keep all experts" for that layer), while every other layer's router shrinks to the new
    # expert_count -- llama.cpp then refuses to load the model on a per-layer router shape mismatch.
    model_arch_and_expert_count() {
        local host="$1" base="$2" path size mtime cached meta arch n_expert n_block
        path="$host/$base"
        size="$(stat -c %s "$path" 2>/dev/null || echo 0)"
        mtime="$(stat -c %Y "$path" 2>/dev/null || echo 0)"
        cached="$(awk -F'\t' -v p="$path" -v s="$size" -v m="$mtime" \
            '$1==p && $2==s && $3==m { print $4"\t"$5"\t"$6; exit }' "$PRUNE_CACHE_FILE")"
        if [ -n "$cached" ]; then
            echo "$cached"
            return
        fi
        meta="$(docker run --rm -v "$host":/check:ro --entrypoint python3 "$IMAGE" \
            /app/gguf-py/gguf/scripts/gguf_dump.py --no-tensors --json "/check/$base" 2>/dev/null)"
        arch="$(grep -oP '"general\.architecture":\s*\{[^}]*"value":\s*"\K[^"]+' <<< "$meta" || true)"
        n_expert=""
        n_block=""
        if [ -n "$arch" ]; then
            n_expert="$(grep -oP "\"${arch}\.expert_count\":\s*\{[^}]*\"value\":\s*\K[0-9]+" <<< "$meta" || true)"
            n_block="$(grep -oP "\"${arch}\.block_count\":\s*\{[^}]*\"value\":\s*\K[0-9]+" <<< "$meta" || true)"
        fi
        echo -e "$path\t$size\t$mtime\t$arch\t$n_expert\t$n_block" >> "$PRUNE_CACHE_FILE"
        echo -e "$arch\t$n_expert\t$n_block"
    }

    echo "Checking MoE/expert_count for prune-level expansion against $PROFILE_CSV ($PROFILE_N_EXPERT experts) ..." >&2
    NEW_MODELS=()
    for m in "${MODELS[@]}"; do
        h="${MODEL_HOST[$m]}"; b="${MODEL_BASE[$m]}"
        TAG="$(basename "$m" .gguf)"

        IFS=$'\t' read -r ARCH N_EXPERT N_BLOCK < <(model_arch_and_expert_count "$h" "$b")
        if [ -z "$N_EXPERT" ]; then
            echo "  $m: not a MoE model -- running once, no prune-level sweep" >&2
            NEW_MODELS+=("$m")
            continue
        fi
        if [ "$N_EXPERT" != "$PROFILE_N_EXPERT" ]; then
            echo "  $m: expert_count=$N_EXPERT != profile's $PROFILE_N_EXPERT experts (already pruned?) -- running once, no prune-level sweep" >&2
            NEW_MODELS+=("$m")
            continue
        fi

        # Profile CSVs are conventionally named "<model-basename>.<ranking-tag>.csv" -- strip the
        # model prefix when present so generated filenames don't repeat the (long) model name
        # twice; fall back to the full profile basename otherwise.
        profile_base="$(basename "$PROFILE_CSV" .csv)"
        if [[ "$profile_base" == "$TAG."* ]]; then
            PROFILE_TAG="${profile_base#"$TAG".}"
        else
            PROFILE_TAG="$profile_base"
        fi

        echo "  $m: expert_count=$N_EXPERT matches profile -- expanding to levels: ${PRUNE_LIST[*]}" >&2
        for LEVEL in "${PRUNE_LIST[@]}"; do
            KEY="$m::prune$LEVEL"
            MODEL_TAG_OF["$KEY"]="$TAG.prune$LEVEL"
            if [ "$LEVEL" = "0" ]; then
                MODEL_HOST["$KEY"]="$h"; MODEL_BASE["$KEY"]="$b"
                NEW_MODELS+=("$KEY")
                continue
            fi

            PRUNED_BASE="$TAG.pruned-$PROFILE_TAG-n$LEVEL.gguf"
            PRUNED_PATH="$PRUNE_DIR/$PRUNED_BASE"
            if [ -f "$PRUNED_PATH" ] && [ "${FORCE:-0}" != "1" ]; then
                echo "    n=$LEVEL: found cached $PRUNED_PATH" >&2
            else
                SPEC_CSV="$TAG.pruning-$PROFILE_TAG.n$LEVEL.csv"
                # Default fill-up-to-layer is this model's own last block index, so any MoE layer
                # the profiler didn't reach (see comment on model_arch_and_expert_count above) still
                # gets a synthesized prune spec instead of being silently left unpruned.
                FILL_MAX="${PRUNE_FILL_MISSING_LAYERS:-$((N_BLOCK - 1))}"
                FILL_ARGS=(--fill-missing-layers "$FILL_MAX")
                echo "    n=$LEVEL: generating pruning spec -> $PRUNE_DIR/$SPEC_CSV" >&2
                docker run --rm --entrypoint python3 \
                    -v "$PROFILE_HOST_DIR":/profile:ro \
                    -v "$PRUNE_DIR":/pruned \
                    -v "$REPO_SCRIPTS_DIR":/host-scripts:ro \
                    "$IMAGE" \
                    /host-scripts/lowest_experts_from_profile.py \
                    "/profile/$(basename "$PROFILE_CSV")" "/pruned/$SPEC_CSV" \
                    -n "$LEVEL" "${FILL_ARGS[@]}" >&2

                echo "    n=$LEVEL: pruning -> $PRUNED_PATH" >&2
                docker run --rm --entrypoint python3 \
                    -e PYTHONPATH=/app/gguf-py \
                    -v "$h":/src:ro \
                    -v "$PRUNE_DIR":/pruned \
                    -v "$REPO_SCRIPTS_DIR":/host-scripts:ro \
                    "$IMAGE" \
                    /host-scripts/prune_moe_experts.py \
                    "/src/$b" "/pruned/$PRUNED_BASE" --csv "/pruned/$SPEC_CSV" --force \
                    --note "pruning.profile_csv=$(basename "$PROFILE_CSV")" \
                    --note "pruning.experts_removed=$LEVEL" \
                    --note "pruning.source_model=$b" >&2
            fi
            MODEL_HOST["$KEY"]="$PRUNE_DIR"; MODEL_BASE["$KEY"]="$PRUNED_BASE"
            NEW_MODELS+=("$KEY")
        done
    done
    MODELS=("${NEW_MODELS[@]}")
fi

# --override-kv silently does nothing if the key doesn't match anything in the model's actual
# metadata -- llama.cpp gives no warning, so a wrong K_KEY produces plausible-looking numbers that
# are secretly all the model's untouched trained k. Check the real metadata (via gguf_dump.py,
# metadata-only, no weight load) for each model, cached by (path, size, mtime, K_KEY) so repeat
# invocations skip the ~15-20s-per-model dump. A model missing the key is treated as non-MoE (or
# just not using this K_KEY) rather than an error -- it runs once per corpus with no override, as
# a reference point -- UNLESS every model is missing it, which is almost certainly a typo in K_KEY.
KKEY_CACHE_FILE="${KKEY_CACHE_FILE:-$HOME/.cache/perplexity-sweep-kkey.tsv}"
mkdir -p "$(dirname "$KKEY_CACHE_FILE")"
touch "$KKEY_CACHE_FILE"

# Echoes "yes" or "no": does this model's metadata have "$3.expert_used_count"?
model_has_k_key() {
    local host="$1" base="$2" key="$3" path size mtime cached meta
    path="$host/$base"
    size="$(stat -c %s "$path" 2>/dev/null || echo 0)"
    mtime="$(stat -c %Y "$path" 2>/dev/null || echo 0)"
    cached="$(awk -F'\t' -v p="$path" -v s="$size" -v m="$mtime" -v k="$key" \
        '$1==p && $2==s && $3==m && $4==k { print $5; exit }' "$KKEY_CACHE_FILE")"
    if [ -n "$cached" ]; then
        echo "$cached"
        return
    fi
    meta="$(docker run --rm -v "$host":/check:ro --entrypoint python3 "$IMAGE" \
        /app/gguf-py/gguf/scripts/gguf_dump.py --no-tensors --json "/check/$base" 2>/dev/null)"
    if echo "$meta" | grep -q "\"${key}.expert_used_count\""; then
        echo -e "$path\t$size\t$mtime\t$key\tyes" >> "$KKEY_CACHE_FILE"
        echo "yes"
    else
        echo -e "$path\t$size\t$mtime\t$key\tno" >> "$KKEY_CACHE_FILE"
        echo "no"
    fi
}

declare -A MODEL_HAS_K=()
if [ -n "${K_VALUES:-}" ]; then
    echo "Checking which models have K_KEY='$K_KEY' (cache: $KKEY_CACHE_FILE) ..." >&2
    ANY_HAS_KEY=0
    for m in "${MODELS[@]}"; do
        HAS="$(model_has_k_key "${MODEL_HOST[$m]}" "${MODEL_BASE[$m]}" "$K_KEY")"
        MODEL_HAS_K["$m"]="$HAS"
        if [ "$HAS" = "yes" ]; then
            ANY_HAS_KEY=1
        else
            echo "  $m: no '$K_KEY.expert_used_count' key -- running once per corpus, no k override" >&2
        fi
    done
    if [ "$ANY_HAS_KEY" != "1" ]; then
        echo "ERROR: none of the given models have a '$K_KEY.expert_used_count' metadata key." >&2
        echo "       K_KEY should be just the architecture prefix (e.g. 'qwen35moe'), not the" >&2
        echo "       full key -- this script appends '.expert_used_count' itself." >&2
        exit 1
    fi
fi

if [ "$WAIT_FOR_GPU" = "1" ]; then
    echo "Waiting for other llama containers to exit ..." >&2
    while docker ps --format '{{.Names}}' | grep -q '^llama-cpp-turboquant-llama-cpp-run-'; do
        sleep 60
    done
    echo "GPU free, starting sweep." >&2
fi

MOUNT_ARGS=()
for host in "${!MOUNT_OF[@]}"; do
    MOUNT_ARGS+=(-v "$host:${MOUNT_OF[$host]}:ro")
done

TS="$(date +%Y%m%dT%H%M%S)"
SUMMARY="$OUT_DIR/summary.tsv"
RUN_SUMMARY="$OUT_DIR/summary-run-$TS.tsv"
printf 'model\tk\tcorpus\tppl\tstderr\tgsd\toutliers\tlog\n' > "$RUN_SUMMARY"

# Models without the K_KEY (non-MoE, or just a different arch) get a single no-override run per
# corpus instead of the full K_LIST -- see the K_KEY validation block above.
TOTAL=0
for m in "${MODELS[@]}"; do
    if [ -n "${K_VALUES:-}" ] && [ "${MODEL_HAS_K[$m]:-}" = "yes" ]; then
        TOTAL=$((TOTAL + ${#K_LIST[@]} * ${#CORPUS_LIST[@]}))
    else
        TOTAL=$((TOTAL + ${#CORPUS_LIST[@]}))
    fi
done
DONE=0
RUN_COUNT=0        # combos actually executed this invocation (excludes skipped/resumed ones)
RUN_TIME_TOTAL=0   # seconds, summed over RUN_COUNT -- basis for the ETA estimate

for MODEL in "${MODELS[@]}"; do
    MODEL_PATH="${MOUNT_OF[${MODEL_HOST[$MODEL]}]}/${MODEL_BASE[$MODEL]}"
    MODEL_TAG="${MODEL_TAG_OF[$MODEL]:-$(basename "$MODEL" .gguf)}"
    if [ -n "${K_VALUES:-}" ] && [ "${MODEL_HAS_K[$MODEL]:-}" = "yes" ]; then
        MODEL_K_LIST=("${K_LIST[@]}")
    else
        MODEL_K_LIST=("")
    fi
    for K in "${MODEL_K_LIST[@]}"; do
        K_TAG="${K:-default}"
        K_ARGS=()
        [ -n "$K" ] && K_ARGS=(--override-kv "${K_KEY}.expert_used_count=int:$K")
        for CORPUS_ITEM in "${CORPUS_LIST[@]}"; do
            CORPUS_PATH="${MOUNT_OF[${CORPUS_HOST[$CORPUS_ITEM]}]}/${CORPUS_BASE[$CORPUS_ITEM]}"
            CORPUS_TAG="$(basename "$CORPUS_ITEM" | sed -E 's/\.[^.]+$//')"

            TAG="$MODEL_TAG"
            [ "${#MODEL_K_LIST[@]}" -gt 1 ] && TAG="$TAG.k$K_TAG"
            [ "${#CORPUS_LIST[@]}" -gt 1 ] && TAG="$TAG.$CORPUS_TAG"
            LOG="$OUT_DIR/$TAG.log"

            DONE=$((DONE + 1))
            # Stamped as the log's first line so a log from a *different* invocation (different k,
            # k_key, ctx, or chunks -- e.g. left over from a run with a since-fixed K_KEY typo) is
            # never mistaken for "already complete" just because it happens to contain a valid
            # "Final estimate" line. Resume only skips a combo whose marker matches this exact one.
            MARKER="# perplexity-sweep: model=$MODEL_TAG k=$K_TAG k_key=${K_KEY:-} corpus=$CORPUS_TAG ctx=$CTX chunks=$CHUNKS"
            if [ "${FORCE:-0}" != "1" ] && [ -f "$LOG" ] && grep -qF "$MARKER" "$LOG" \
                && grep -q 'Final estimate: PPL' "$LOG"; then
                echo "=== [$DONE/$TOTAL] $MODEL_TAG  k=$K_TAG  corpus=$CORPUS_TAG (skipped, already complete) ===" >&2
            else
                echo "=== [$DONE/$TOTAL] $MODEL_TAG  k=$K_TAG  corpus=$CORPUS_TAG ===" >&2
                COMBO_START=$(date +%s)
                echo "$MARKER" > "$LOG"
                # shellcheck disable=SC2086
                docker run --rm --gpus all "${MOUNT_ARGS[@]}" \
                    --entrypoint /app/llama-perplexity "$IMAGE" \
                    -m "$MODEL_PATH" -f "$CORPUS_PATH" \
                    -c "$CTX" --chunks "$CHUNKS" -fit on "${K_ARGS[@]}" $EXTRA_ARGS \
                    >> "$LOG" 2>&1 || echo "  (run failed -- see $LOG)" >&2
                COMBO_ELAPSED=$(( $(date +%s) - COMBO_START ))
                RUN_COUNT=$((RUN_COUNT + 1))
                RUN_TIME_TOTAL=$((RUN_TIME_TOTAL + COMBO_ELAPSED))
                AVG=$((RUN_TIME_TOTAL / RUN_COUNT))
                REMAINING=$((TOTAL - DONE))
                echo "  took $(fmt_duration "$COMBO_ELAPSED") (avg $(fmt_duration "$AVG")/run, ~$(fmt_duration $((AVG * REMAINING))) left for $REMAINING more)" >&2
            fi

            ROW="$(summary_row "$LOG" "$MODEL_TAG" "$K_TAG" "$CORPUS_TAG")"
            echo "$ROW" >> "$RUN_SUMMARY"
            IFS=$'\t' read -r _ _ _ PPL ERR GSD OUTLIERS _ <<< "$ROW"
            echo "  PPL = ${PPL:-FAILED} +/- ${ERR:-?}  |  gsd=${GSD}x  outliers=${OUTLIERS}" >&2
        done
    done
done

regenerate_summary

MD="$OUT_DIR/summary-$TS.md"
{
    echo "| model | k | corpus | ppl | stderr | gsd | outliers |"
    echo "|---|---|---|---|---|---|---|"
    tail -n +2 "$SUMMARY" | awk -F'\t' \
        '{printf "| %s | %s | %s | %s | %s | %s | %s |\n", $1, $2, $3, $4, $5, $6, $7}'
} > "$MD"

echo >&2
echo "All-time summary regenerated -> $SUMMARY and $MD" >&2
echo "This run's combos -> $RUN_SUMMARY" >&2
column -t -s $'\t' "$SUMMARY" >&2
