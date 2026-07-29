# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Read [AGENTS.md](AGENTS.md) before doing any work here.** It sets hard rules for AI-assisted
> contributions in this repo (no fully-AI-generated PRs, ASCII-only punctuation, comment style,
> and — critically — never `git commit`/`push`/`gh pr create` without the user explicitly asking
> for that specific action each time). Those rules override generic instincts elsewhere in this file.

## What this is

A fork of `ggml-org/llama.cpp` integrating **TurboQuant+**: Walsh-Hadamard-rotated
polar-codebook quantization for the KV cache (`turbo2`/`turbo3`/`turbo4` types) and for weights
(`TQ3_1S`/`TQ4_1S`). The codec papers (design rationale, benchmarks, per-model tuning guidance)
live in the sibling repo `TheTom/turboquant_plus`, linked throughout the README — read the
specific paper before changing turbo-type behavior, not just the code.

Remote/branch layout (this is a fork of a fork):
- `upstream` remote -> the original TurboQuant fork (`TheTom/llama-cpp-turboquant`); its default
  branch is `feature/turboquant-kv-cache`, which carries all the TurboQuant integration work.
- `origin` (this repo's default remote) -> the user's own fork off of that; `zac` is this repo's
  main/default branch, sitting on top of `feature/turboquant-kv-cache` with day-to-day
  experiment/tooling commits.
- Plain `ggml-org/llama.cpp` upstream-of-upstream is only reachable indirectly (via merges into
  `feature/turboquant-kv-cache`) — there's no direct remote for it in this repo.

Core rule baked into the whole codec design, repeated everywhere (README, docker-compose
comments): **V tolerates aggressive compression, K does not.** Never make K more compressed than
V; `q8_0`-K / `turbo3`-V is the default recommended pairing. See
`README.md` ("KV-cache quantization" section) before touching K/V type selection logic.

## Always work inside Docker

Everything in this repo — building, running the server, and the Python-based utility scripts —
is meant to run inside the `llama-cpp` docker-compose service (or a container built from the
same Dockerfile), not directly on the host. The host is not guaranteed to have a matching
compiler/CUDA toolchain or the Python deps (`gguf-py`, etc.) that these scripts need; several
`scripts/*.sh` wrappers make this explicit by shelling into the docker image themselves rather
than assuming a local install. Don't run bare `cmake`, `python3 scripts/foo.py`, etc. against the
host environment — go through `docker compose run`/`exec` against `llama-cpp`, or use/extend one
of the existing `scripts/*.sh` wrappers that already does this.

## Build

Building happens via the `llama-cpp` docker-compose service (`.devops/cuda.Dockerfile`):

```bash
docker compose build llama-cpp
```

CMakePresets.json documents the underlying per-platform CMake presets
(`x64-linux-gcc-release`, `arm64-apple-clang-release`, etc.) that the Dockerfile/CI drive, and
raw `cmake -B build -DGGML_CUDA=ON && cmake --build build -j`-style commands exist in
`docs/build.md` for reference — but they're not verified to work directly on the host here, since
the host isn't guaranteed to have the right CUDA/compiler versions installed. TurboQuant types
compile in automatically once the matching backend is enabled, no separate flag needed.

## Test

- `ctest` (run inside the container/build image, not on the host) runs the suite under `tests/`
  (chat parsers, GGUF I/O, backend ops, grammar, tokenizer, etc). Run a single test with
  `ctest -R <name>` or invoke the compiled `tests/test-*` binary directly.
- `ci/run.sh <results-dir> <mnt-dir>` runs the full upstream CI matrix (`GG_BUILD_CUDA=1`,
  `GG_BUILD_VULKAN=1`, etc. env vars select backends).
- `scripts/turbo-quality-gate.sh` — fork-specific quality gate for turbo KV types (perplexity /
  correctness check), separate from CTest.
- Perplexity/quality regression checks against real models go through
  `scripts/perplexity-run.sh` and the imatrix scripts below, not CTest — those need real GGUFs
  and a GPU, so they're driven through the docker-compose `llama-cpp` service.

## Primary dev workflow: docker-compose

`docker-compose.yml` defines the actual way this fork gets run day to day:

- **`llama-cpp`** — the real service. Builds from `.devops/cuda.Dockerfile` and runs
  `llama-server`. Its `command:` block is normally left commented-out except for one active
  config; when changing server flags, edit that block rather than passing flags externally.
  A large comment block above it is a `-ctk`/`-ctv` quick-reference table (bpw, size vs f16,
  accuracy, speed deltas) — keep it in sync with the README's compression-ladder guidance if you
  change defaults.
- **`prompt-logger`** (port 3002) and **`prompt-router`** (port 3003) — thin Python reverse
  proxies in front of `llama-cpp`'s port 8080. `prompt-logger` records every request/response to
  `data/logs/prompts/` (used later by `dedup-prompt-logs.py` / `extract_prompt_corpus.py` to
  build calibration corpora from real traffic); `prompt-router` is the same proxy with logging
  off, for routing without the disk cost. Direct access to `llama-cpp` on 8080 bypasses both.
- Models are expected under `/mnt/llm/llama.cpp/models` (mounted `/models`), with a secondary
  `/models2` mount for overflow storage — most scripts below default to searching both, in that
  order.
- `scripts/fit-print.sh` previews the memory-fit decisions (`llama_params_fit`) for the
  `llama-cpp` service's *current* `docker-compose.yml` command, without loading model weights —
  reads flags via `docker compose config`, so it never goes stale relative to the compose file.

## Fork-specific tooling (scripts/)

Beyond upstream's `scripts/`, this fork adds a GGUF/quantization/corpus toolchain. Common
conventions across these: bash wrapper scripts run the heavy lifting inside the `llama-cpp`
docker image (so no local Python/gguf-py install is needed) and take a bare filename that's
resolved against `MODELS_DIR`/`CORPUS_DIR`/`IMATRIX_DIR` env vars (defaulting to the models
mount) rather than requiring absolute paths.

For any general-purpose operation (not a true one-off), add a script under `scripts/` following
this pattern rather than running the equivalent as an ad hoc shell/docker command. It should be
reusable the next time the same kind of task comes up, documented with a header comment like the
existing scripts (purpose, usage, path-resolution rules), and go through Docker per the rule
above rather than assuming host tooling.

- `imatrix-gen.sh`, `imatrix-combine.sh` — generate/merge importance matrices from a text corpus
  for a model; combine is token-count-weighted, not a naive average.
- `perplexity-run.sh` — run `llama-perplexity` against a corpus, print final PPL.
- `corpus-holdout-slice.sh` — cut an untouched tail slice out of a corpus already consumed (by
  `--chunks N`) for imatrix generation, for a no-overlap PPL eval set.
- `quantize-iq4xs-uniform.sh` — requantize to a uniform IQ4_XS (forces attn_qkv/ffn_down/attn_v to
  iq4_xs instead of llama.cpp's default q5_K bumps) for a smaller file at a measured quality cost.
- `gguf-layer-quants.sh` / `gguf_layer_quants.py` — dump a GGUF's per-tensor type/shape/size as
  TSV, for auditing quant mix after a custom quantize.
- `prune_moe_experts.py` + `lowest_experts_from_profile.py` — remove specific experts from MoE
  GGUFs directly on quantized tensors (no dequant round-trip); the latter turns a per-layer expert
  activation profile into the CSV the former expects. All layers must drop the same expert count.
- `dedup-prompt-logs.py` / `extract_prompt_corpus.py` — turn `prompt-logger`'s JSON dumps into a
  deduplicated plaintext calibration corpus (whole-file dedup, then leaf-string dedup, since
  real traffic replays near-identical system prompts/tool defs on almost every request).
- `prune_vocab.py` / `build_vocab_patch.py` — BPE vocab pruning and patching tools.
- `extract_mtp_gguf.py` / `merge_mtp_gguf.py` — split out or merge in a model's MTP/draft head as
  its own GGUF.
- `kld-run-pair.sh` / `kld-run-matrix.sh` / `kld-gen-reference.sh` — KL-divergence quality
  comparison between quantizations/configs (pairwise or full matrix), against a reference.
- `compare-llama-bench.py` / `bench-models.sh` / `bench-smem-m5.sh` — perf comparison/benchmarking
  harnesses.

## Architecture notes specific to this fork

- **`common/fit.h` / `common/fit.cpp`** — `common_fit_params()` auto-adjusts
  `llama_model_params`/`llama_context_params` (context size, tensor split, per-tensor buffer-type
  overrides for MoE expert offload) to fit free device memory, driven by the `-fit`/`--fit` CLI
  flags. Only touches parameters still at their default value. `tools/fit-params/` is a thin CLI
  that runs the same fit logic standalone and prints the resulting flags (pipe into `xargs` ahead
  of a real `llama-server`/`llama-cli` invocation — see `tools/fit-params/README.md`).
- **`src/llama-ext.h`** — explicitly a staging ground for new/experimental `llama.h` API surface
  (memory-breakdown queries, MTP/nextn embedding accessors, quantization internals exposed for
  testing). Comment at the top says not to include it outside this staging use — treat additions
  here as unstable, and prefer promoting something to `llama.h` once it settles rather than
  growing this file indefinitely.
- **`common/kv-cache-lru.{h,cpp}`** — a session-aware LRU manager layered on top of `llama_memory_t`
  for multi-session prompt caching (keeps multiple sessions' KV alive, evicts the LRU session's
  tail under pressure instead of dropping everything on any cache miss). Design doc/test plan:
  `docs/test-plan-kv-cache-lru.md`.
- **TurboQuant kernel spread** — each backend implements the turbo types independently; there is
  no single shared reference implementation to grep for. When changing turbo K/V semantics, expect
  to touch the CPU vec_dot path plus whichever of CUDA (`ggml/src/ggml-cuda`), Metal
  (`ggml/src/ggml-metal`), HIP, Vulkan, and SYCL backends are in scope, per the per-backend support
  matrix in `README.md`. `TURBOQUANT_UPSTREAM_MERGE.md` documents which contributor work lives
  where post-merge, and calls out the Vulkan turbo3 flash-attention path as a known-deferred re-port.
- **Server slot save/restore across restarts** (`--slot-save-path` + `--ctx-checkpoints`) — fork
  addition on top of upstream's slot save/restore; writes a `.ckpt` sidecar carrying in-memory
  context checkpoints that upstream's serialization doesn't persist. Only matters for SWA/hybrid
  models (Gemma-style sliding window, Mamba/hybrid); dense-attention models never need it. See
  README "Server slot save/restore" section before changing this path.
- `docs/test-plan-kv-cache-lru.md`, `docs/speculative.md`, `docs/autoparser.md` — fork-authored
  design/test docs worth reading before modifying the corresponding subsystem; not just API
  reference like the rest of `docs/`.
