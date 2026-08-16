#!/bin/bash
# Thin wrapper over perplexity-sweep.sh for the common single-corpus, no-k-sweep case: run
# llama-perplexity across several models against one corpus, keeping the full per-chunk log per
# model. See perplexity-sweep.sh for the general form (multiple corpora, expert_used_count sweep).
#
# Usage: perplexity-quant-sweep.sh <corpus-file> <model-filename> [<model-filename> ...]
#
# All perplexity-sweep.sh env vars (CTX, CHUNKS, OUT_DIR, WAIT_FOR_GPU, EXTRA_ARGS, MODELS_DIR,
# MODELS2_DIR, MODELS3_DIR, CORPUS_DIR, IMAGE) apply unchanged; this script only fixes CORPUS to
# the single positional argument.
set -euo pipefail

CORPUS="${1:?Usage: perplexity-quant-sweep.sh <corpus-file> <model-filename> [...]}"
shift
[ "$#" -ge 1 ] || { echo "ERROR: at least one model required" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CORPUS
exec "$SCRIPT_DIR/perplexity-sweep.sh" "$@"
