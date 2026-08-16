# MoE Router Weight Distribution & Expert Count Sweep (qwen35moe)

## Summary

Measured how the MoE router distributes gate weight across its selected experts, and how model
quality responds to overriding `expert_used_count` (top-k) away from the trained value of 8.

Eight findings:

1. **The router's weight-by-rank curve has no knee.** Past rank ~6 it decays almost linearly and
   very slowly - rank 12 still carries 80% of rank 8's weight. There is nothing in the
   distribution that marks 8 as a natural cutoff; it is purely a training-time compute budget.
2. **Quality peaks at k approx 12, not at the trained k=8, but collapses beyond ~16.** PPL improves
   1.21% from k=8 to k=12, is already worse by k=16, and is 2.4x worse at k=256 (all experts).
   Raising k slightly is safe and mildly beneficial; raising it a lot is destructive, because
   renormalization dilutes the top expert by up to 4.9x and lets the unselected tail dominate.
3. **The top-8 cut captures only 20.55% of the router's probability mass** (the other 79.45% is
   discarded). Measured from the pre-normalization tensor, which unlike the post-norm weights is
   directly comparable across values of k.
4. **Decode cost is linear in k but with a large fixed term**, so k is a weak cost lever: only
   22.5% of decode time depends on the expert count (17.8% at 40k context). Halving k to 4 buys
   just ~10% throughput while costing more quality than k=6's +3.50%. Quality moves ~3x faster
   than speed in both directions. So while k=12 is the *quality* optimum (finding 2), k=8 remains
   a reasonable *cost/quality* default - the 1.21% is not free.
5. **VRAM scales linearly with k at prefill** (~+117 MB from k=8 to k=12 at `n_ubatch` 2048) via
   the compute buffer, not the weights. This is the cost of raising k that bites hardest on a
   memory-constrained fit.
6. **The stock (non-distilled) 35B's worst corpus content is routing-recoverable, not
   capacity-limited.** Raising k from 8 to 64 drops its worst chunk's PPL 7.5x (27.2 -> 3.6) even
   though nothing changed about the model's weights - the needed experts were already in the
   model, just outside the trained top-8 for this content. This is a distinct failure mode from
   finding 2's dilution and can hide inside it: on a heavy-tailed corpus, admitting more experts
   can pull the headline PPL *down* past the point where 85% of chunks have already started
   degrading, producing a misleading "optimal k" if the per-chunk trace is never decomposed.
7. **Quantizing a dense 27B cannot fix this content gap.** Across 3.0-4.25 bpw and seven
   independent imatrix/mix strategies of the *same* training run (unsloth's stock instruction
   tune), the worst chunks are pinned within a narrow band (no trend with bits or calibration) at
   65-80x a same-family MoE reference.
8. **Most of that gap was training-data coverage, not architecture - but not all of it.** A
   differently-trained dense 27B (same base, distilled from Claude Opus reasoning traces by a
   third party) drops the worst chunk 7x (1795-2141 -> 262) with no architecture change and no
   MoE-style routing lever available to it, landing statistically level with the stock-35B MoE's
   aggregate PPL. This rules out "dense architecture has no relevant capacity" as an explanation
   for finding 7. It does not fully close the gap: a **6-10x residual** remains against the
   content-matched distilled 35B (262 vs 2.68 on chunk 17), consistent with either leftover
   dense-capacity shortfall or the reasoning-trace distillation being a partial, not exact, match
   to this content's actual register (tool-transcripts, terminal capture, structured logs) -
   not distinguishable from this comparison alone.

## Test Environment

| Component | Value |
|-----------|-------|
| Model | `Qwen3.6-35B-A3B-Claude-Opus-Distilled-MTP-UD-IQ3_XXS.gguf` |
| Arch | `qwen35moe` - 256 experts, `expert_used_count` 8, 41 layers (40 MoE + MTP) |
| `n_embd` / expert FFN | 2048 / 512 (fine-grained MoE) |
| Gating | `SOFTMAX` over all 256, then top-k, then renormalize (`norm_w = true`) |
| GPU | Blackwell, `CUDA_DOCKER_ARCH=120a`, `-ngl 999` (full offload) |
| Runner | `docker compose run --rm --entrypoint ... llama-cpp` |

### Gating config is hardcoded, not metadata

`expert_gating_func` / `expert_weights_norm` / `expert_weights_scale` do **not** appear in this
model's GGUF KV dump. They are passed as literals at the `build_moe_ffn` call site in
`src/models/qwen35moe.cpp` (~line 496):

```cpp
build_moe_ffn(cur, model.layers[il].ffn_gate_inp, ...,
    nullptr,                                    // exp_probs_b - no selection/weight decoupling
    n_expert, n_expert_used,
    LLM_FFN_SILU, true,                         // norm_w = true
    hparams.expert_weights_scale,               // 0.0f (unset) -> scale skipped
    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il, ...);
```

To determine a model's gating path, **grep the arch builder, not the KV dump.** Consequently
`--override-kv qwen35moe.expert_gating_func=...` has no effect, while
`--override-kv qwen35moe.expert_used_count=int:N` works because that one really is read from
metadata.

Because `exp_probs_b == nullptr`, selection order and gate magnitude are the *same tensor* for this
arch - rank 1 is by definition the highest-weighted expert. (DeepSeek-V3, LLAMA4 and GROVEMOE
decouple these via `selection_probs`; qwen35moe does not.)

## Part 1: Router weight distribution

### Tool

`examples/moe-weights/` - registers a `ggml_backend_sched_eval_callback`, filters for tensors named
`ffn_moe_weights` (pre-norm) and `ffn_moe_weights_norm` (post-norm), and accumulates per-rank
mean/stdev across all tokens and layers. Both tables come from one pass.

Gotcha: at the `cb()` call in `src/llama-graph.cpp` (`build_moe_ffn`), `ffn_moe_weights_norm` is
**2D `[n_expert_used, n_tokens]`**. The `ggml_reshape_3d` to `[1, n_expert_used, n_tokens]` happens
*after* the callback fires. The tool handles both interpretations defensively.

### Commands

```bash
head -c 3400 /mnt/llm/models/prompt_corpus.txt > /mnt/llm/models/moe_sample.txt   # 725 tokens

# baseline
docker compose run --rm --entrypoint /app/llama-moe-weights llama-cpp \
  -m /models/Qwen3.6-35B-A3B-Claude-Opus-Distilled-MTP-UD-IQ3_XXS.gguf \
  -ngl 999 -f /models/moe_sample.txt -c 4096

# with k overridden
docker compose run --rm --entrypoint /app/llama-moe-weights llama-cpp \
  -m /models/Qwen3.6-35B-A3B-Claude-Opus-Distilled-MTP-UD-IQ3_XXS.gguf \
  -ngl 999 -f /models/moe_sample.txt -c 4096 \
  --override-kv qwen35moe.expert_used_count=int:12
```

### Results (mean gate weight by rank, 725 tokens x 40 layers)

k=8:
```
MEAN | 0.255 0.172 0.134 0.111 0.096 0.085 0.077 0.071   | ranks 7-8 = 14.74%
STDEV| 0.079 0.034 0.024 0.020 0.019 0.018 0.017 0.017
```

k=12:
```
MEAN | 0.212 0.141 0.109 0.089 0.076 0.067 0.061 0.055 0.052 0.048 0.046 0.044
STDEV| 0.074 0.033 0.022 0.016 0.014 0.012 0.012 0.011 0.011 0.011 0.011 0.011
```

Reproducibility: an independent 848-token slice gave 14.76% for ranks 7-8 vs 14.74% here. The
distribution is a property of the model, not the sample.

### Pre-normalization weights (the k-independent view)

Post-norm weights cannot be compared across k: dividing by the row sum folds k into every entry,
so rank 8 reads 0.071 at k=8 but 0.055 at k=12 for the same underlying value. The pre-norm tensor
`ffn_moe_weights-<il>` (llama-graph.cpp:1588, the `ggml_get_rows` result before the `ggml_div`)
avoids this entirely - it is the raw softmax prob, and for SOFTMAX gating the softmax runs over
all 256 experts *before* the top-k cut, so rank r is computed identically at any k.

The tool collects both tensors in a single pass and prints a table for each.

| | rank1 | rank2 | rank3 | rank4 | rank5 | rank6 | rank7 | rank8 | rank9 | r10 | r11 | r12 | captured |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pre-norm, k=8 | 0.055 | 0.036 | 0.028 | 0.022 | 0.019 | 0.017 | 0.015 | 0.014 | | | | | **0.2055** |
| pre-norm, k=12 | 0.058 | 0.037 | 0.028 | 0.023 | 0.019 | 0.017 | 0.015 | 0.014 | 0.013 | 0.012 | 0.011 | 0.010 | **0.2566** |
| post-norm, k=8 | 0.255 | 0.172 | 0.134 | 0.111 | 0.096 | 0.085 | 0.077 | 0.071 | | | | | 1.0 |

**Captured mass is the quantity renormalization destroys.** The top 8 of 256 experts carry only
**20.55%** of the router's total probability; the cut discards the other 79.45%. This is not
recoverable from post-norm data by any rescaling, since those rows sum to 1 by construction.
Raising k to 12 lifts captured mass to 25.66% (+25% relative) - see Part 2 for what that buys.

Ranks 3-12 agree exactly across k. Ranks 1-2 drift slightly, and this is a real systematic effect,
not noise (two k=8 runs are bit-reproducible at 0.2055). Per-layer it resolves as a second-order
feedback effect rather than a k-dependence in the routing math:

```
layer  0   k8=0.071  k12=0.071  diff=+0.0%     <- bit-identical
layer  1   k8=0.060  k12=0.059  diff=-1.7%
layer 18   k8=0.054  k12=0.058  diff=+7.4%
layer 39   k8=0.031  k12=0.030  diff=-3.2%
```

Layer 0's router sees the same embeddings in both runs so it must agree exactly. Once layer 0's
MoE writes a different result to the residual stream, every downstream router sees a different
input. The routing function is k-independent; the activations fed to it are not.

Note bottom-2 mass reads 13.81% pre-norm vs 14.74% post-norm. Normalization is per-token, so
low-captured-mass tokens are scaled up more and the two averages weight tokens differently.
Post-norm is what the FFN sees; pre-norm is what the router meant.

### Interpretation

Weights are renormalized to sum to 1 over the selected k, so cross-k comparison of the *post-norm*
table requires rescaling by retained mass. At k=12, ranks 1-8 sum to **0.810** and ranks 9-12 to
**0.190**.

| | rank 8 | rank 9 | rank 10 | rank 11 | rank 12 |
|---|---|---|---|---|---|
| weight (k=12 basis) | 0.055 | 0.052 | 0.048 | 0.046 | 0.044 |
| ratio to previous rank | - | 95% | 92% | 96% | 96% |

- **No knee.** Rank 12 is 80% of rank 8. 6, 8 or 14 would all look equally arbitrary.
- **Ranking is stable under k.** Rescaling k=12's top 8 to sum to 1 gives rank1 = 0.262 (vs 0.255
  at k=8) and rank8 = 0.068 (vs 0.071). Adding experts appends to the tail; it does not reshuffle.
- **The deep tail is uniform background, not selective.** STDEV at ranks 9-12 (0.011) is *lower*
  than at rank 8 - those experts contribute a consistent small amount on every token.
- **Per-layer trend:** early layers are peaked (layer 2: rank1 = 0.304, ranks 7-8 = 12.04%), late
  layers nearly flat (layers 37-38: rank1 approx  0.187, ranks 7-8 approx  18.7%).

This shape is characteristic of fine-grained MoE (512-wide experts against 2048 `n_embd`): many
small interchangeable fragments behaving like a weighted sum of basis functions. A coarse MoE such
as Mixtral's 8 x large experts would show much steeper decay and a real knee - **do not carry these
conclusions across architectures.**

## Part 2: Perplexity sweep

### Commands

```bash
head -c 500000 /mnt/llm/models/prompt_corpus.holdout.txt > /mnt/llm/models/ppl_sample.txt

for K in 6 9 10 12 16 32 64 128 192 256; do
  docker compose run --rm --entrypoint /app/llama-perplexity llama-cpp \
    -m /models/Qwen3.6-35B-A3B-Claude-Opus-Distilled-MTP-UD-IQ3_XXS.gguf \
    -ngl 999 -f /models/ppl_sample.txt -c 4096 \
    --override-kv qwen35moe.expert_used_count=int:$K
done
```

(k=8 baseline: same command without `--override-kv`.) 33 chunks, identical text across all runs.

### Results

| k | PPL | delta vs k=8 | expert compute |
|---|-----|----------|----------------|
| 6 | 4.2913 +/- 0.0384 | **+3.50%** | 0.75x |
| 8 | 4.1463 +/- 0.0370 | - | 1.0x |
| 9 | 4.1204 +/- 0.0367 | -0.62% | 1.125x |
| 10 | 4.1149 +/- 0.0367 | -0.76% | 1.25x |
| 12 | 4.0962 +/- 0.0363 | **-1.21%** | 1.5x |
| 16 | 4.1365 +/- 0.0366 | -0.24% | 2.0x |
| 32 | 4.5322 +/- 0.0408 | +9.3% | 4.0x |
| 64 | 5.6297 +/- 0.0534 | +35.8% | 8.0x |
| 128 | 7.7676 +/- 0.0792 | +87.3% | 16.0x |
| 192 | 9.2427 +/- 0.0978 | +122.9% | 24.0x |
| 256 | 9.9711 +/- 0.1072 | +140.5% | 32.0x (all experts) |

**The curve turns around.** Quality improves to a minimum at k approx 12, is already past it by
k=16, and then degrades steeply. Using every expert is **2.4x worse** than the trained k=8. Note
the two error bars that matter here are wide apart: the k=32 regression is ~9%, far outside any
plausible interval, so the reversal is not a statistical artifact even though the k=9/10/12 gaps
are marginal.

### Why the tail is destructive, not merely redundant

This is the captured-mass number (Part 1) predicting its own consequence. Renormalization divides
by the captured mass, so rank 1's post-norm weight is `pre_norm_rank1 / captured`:

| k   | captured | rank-1 post-norm weight |
| --- | -------- | ----------------------- |
| 8   | 0.2055   | 0.055 / 0.2055 = 0.268 (measured 0.255) |
| 12  | 0.2566   | 0.058 / 0.2566 = 0.226 (measured 0.212) |
| 256 | 1.0      | 0.055 / 1.0 = **0.055** |

At k=256 the top expert is diluted **4.9x** and 248 low-confidence experts collectively carry 79%
of the output. The router's decision is drowned out by the aggregate of experts it deliberately
did not select. Past roughly k=16 the tail is actively harmful, and the harm scales with the
probability mass admitted.

### The degradation is uniform, not localized

The per-chunk shape is preserved at every k - chunk 11 is easiest and chunk 24 hardest throughout -
and the k=256/k=8 ratio is 2.26 at chunk 11 vs 2.32 at chunk 24. A near-constant multiplier across
all content is the signature of a global dilution effect, consistent with the renormalization
account above.

**This uniformity turned out to be a useful reference shape.** Three models were run on the same
corpus at IQ3_XXS, same tokenizer (all tokenize `ppl_sample.txt` to 135942 tokens):

| model                            | PPL     | +/- (relative) |
| -------------------------------- | ------- | -------------- |
| 35B-A3B Claude-Opus-Distilled    | 4.1463  | 0.89%          |
| 35B-A3B unsloth stock            | 6.9344  | 1.20%          |
| 27B dense unsloth stock          | 14.5936 | 1.77%          |

Decomposing the 3.52x total gap multiplicatively: **1.67x is distillation / domain match** (41% of
the log-gap, since the corpus is Claude traffic and the distilled model was trained to imitate it),
and 2.10x is everything else. Any model outside the Claude-output lineage starts ~1.7x behind on
this corpus for reasons unrelated to capability.

Per-chunk decomposition separates two distinct effects that the final numbers hide:

- **Some chunks are intrinsically hard for non-distilled models.** Chunks 13/17/19/24 spike to
  27-66 on the stock 35B against its ~5 baseline; the distilled 35B is flat there.
- **The 27B's response is disproportionate**: 1.0-2.2x the stock 35B on easy chunks but 4.6-50x on
  hard ones (chunk 17: distilled flat -> stock 35B 27 -> 27B **1357**). Dropping the worst four
  chunks moves its 14.59 to 9.36.

A content-dependent ratio spanning 1.0x to 50x cannot be a capability gap - contrast the k=256
dilution above, where a genuine global degradation held at 2.26x on the easiest chunk and 2.32x on
the hardest. Accurate on common content and catastrophic on rare content is the signature of
**quantization damage on a dense model**: the error only bites where precision was needed. The
fine-grained MoE tolerates 3-bit far better, which is the redundancy of Part 1 seen from the other
side - function spread across 256 experts, top-8 carrying only 20.55% of router mass, so error is
absorbed by the ensemble.

Methodological upshot: **always decompose the per-chunk trace before comparing models.** A single
PPL number cannot distinguish smooth capability loss from structural breakage on specific content,
and the two call for completely different responses.

### Statistical caveat

The reported `+/-` are **unpaired** standard errors, and the k=9/10/12 gaps fall inside them. The
runs share identical chunks, so the per-chunk traces are the stronger evidence: at k=9 all 33
cumulative values sit below their k=8 counterparts with no crossings. A consistent sign across all
chunks is far better support than the unpaired interval implies.

The exception is k=10 vs k=12 - k=12 trails k=10 for the first ~18 chunks and only overtakes in the
back half. That specific ordering is not well established.

### Interpretation

- **Raising k modestly above the trained value is safe; raising it far is not.** Renormalization
  shrinks all eight original weights by 19% at k=12, a substantial perturbation, yet quality still
  improves - the learned ranking stays meaningful somewhat past the training cutoff. But the same
  mechanism reverses the sign by k=16 and dominates completely by k=32. The usable window is
  roughly k=9-14.
- **k=8 is a compute choice, not a quality optimum.** The model would prefer ~12. Training left
  about 1.2% quality on the table, recoverable for 11% decode throughput and ~117 MB of prefill
  VRAM (Parts 4 and 5), with a hard wall immediately past it.
- **Sharp diminishing returns.** 8 -> 9 captures roughly half the total gain available out to k=12,
  for a quarter of the extra compute. If any value other than 8 is worth running, it is 9.
- **Strong asymmetry.** k=6 costs +3.50%; k=12 buys only -1.21%. Dropping experts hurts ~3x more
  than adding them helps. Consistent with Part 1: the 14.7% of mass at ranks 7-8 is load-bearing,
  while the 19% at ranks 9-12 is largely redundant with what the top 8 already supply.
- **Perplexity is the least sensitive quality metric available.** A 1.2% gain is well inside the
  range that shows no movement on instruction-following or reasoning benchmarks. Do not assume this
  translates to usable quality without a task-shaped eval.

## Part 3: Implications for adaptive per-token k

The idea of a router-probability floor (an "`--expert-min-p`") looks unattractive on this model:

- **The flat tail kills it.** Thresholding needs a roughly bimodal distribution to cut cleanly.
  With decay this shallow, any threshold either keeps nearly everything or chops an arbitrary
  count, and behaves near-identically across tokens - i.e. uniform k with extra machinery.
- **Per-token variance is in the wrong place.** It is concentrated in the head (rank-1 STDEV 0.074
  against mean 0.212), not in the tail where skipping would occur.
- **The shape constraint is real but not the true blocker.** `ggml_mul_mat_id` requires a
  rectangular `[n_expert_used, n_tokens]` selection, so *arbitrary gather indices are free but the
  count must be uniform*. At decode (n_tokens = 1) that is trivially satisfiable - the actual
  blockers are host-readback latency (~41 syncs/token) and CUDA-graph invalidation. A workable
  design would keep the static shape and apply a device-side mask with an early-out in
  `mul_mat_id` before expert weights are fetched.

Note that `--min-p` is unrelated: that is token sampling, not expert routing.

## Part 4: Cost side (decode)

Wall clock across the perplexity runs was roughly flat (1:42 / 1:59 / 1:54 / 1:55 / 1:58 for
k=6/8/9/10/12), but those were single unrepeated runs including ~21s of model load, and perplexity
is **prompt processing** - compute-bound at batch 2048, where grouped GEMM amortizes expert count
well. Decode is the regime where k was expected to matter, since each extra expert means fetching
another set of weights from VRAM.

Measured tg, short prompt:

| k  | t/s | ms/token | fit `8.70 + 0.312k` | fit t/s | error |
| -- | --- | -------- | ------------------- | ------- | ----- |
| 4  | 100 | 10.00    | 9.95                | 100.5   | +0.5% |
| 8  | 90  | 11.11    | 11.20               | 89.3    | -0.8% |
| 9  | 87  | 11.49    | 11.51               | 86.9    | -0.1% |
| 12 | 80  | 12.50    | 12.45               | 80.3    | +0.4% |

**Decode cost is linear in k with a large constant term.** Least squares over the four points gives
a fixed cost of **8.70 ms/token** and a marginal cost of **0.312 ms/expert**; every measurement
falls within 0.8% of that line. At k=8 the expert term is 2.50 ms of 11.11 ms, i.e. **22.5%** of
decode time.

Long context (40k tokens) was measured at k=4/k=8 only:

| prompt  | k=4    | k=8    | B (ms/expert) | B*8 (ms) | total (ms) | k-dependent share |
| ------- | ------ | ------ | ------------- | -------- | ---------- | ----------------- |
| short   | 10.00  | 11.11  | 0.31          | 2.50     | 11.11      | 22.5%             |
| 40k tok | 22.22  | 24.39  | 0.54          | 4.34     | 24.39      | 17.8%             |

The other ~4/5 of decode time - attention, KV traffic, shared/dense weights, per-layer launch
overhead - is fixed regardless of k. That the k-dependent share *falls* at 40k context is
consistent: KV traffic grows with context and dilutes the expert term further. This is the
fine-grained-MoE geometry showing up again (`expert_feed_forward_length` 512 against `n_embd`
2048): each expert is small, so the per-expert byte cost is small relative to everything else in
the layer.

### Combined cost/benefit

Short-prompt decode, all four k values measured:

| k  | tg vs k=8 | PPL vs k=8      |
| -- | --------- | --------------- |
| 4  | +11%      | worse than +3.5% (not measured; k=6 alone is +3.50%) |
| 6  | +5% (interpolated) | +3.50% |
| 8  | -         | -               |
| 9  | -3.3%     | -0.62%          |
| 12 | -11%      | -1.21%          |

- **Cutting experts is a bad trade.** k=4 gives up more than 3.5% perplexity for 11% throughput.
  Quality degrades roughly 3x faster than speed improves, because k drives quality directly but
  only drives 1/5 of the time.
- **k=9 is the one defensible change.** -0.62% PPL for -3.3% tg, i.e. half the available quality
  gain out to k=12 for a third of its cost. Still roughly break-even; take it only if the quality
  matters more than the throughput.
- **k=12 is not worth it.** -11% decode for a perplexity gain small enough (1.21%) that it likely
  does not register on a task-shaped eval.
- **k=8 is a well-chosen default.** With cost linear in k and quality returns sharply diminishing
  above 8 while degrading fast below it, the trained value sits near the knee of the tradeoff even
  though the *weight distribution* has no knee there (Part 1).

Caveats: these are single unrepeated background runs. Long-context cost is still a two-point fit at
k=4/k=8, so its 0.54 ms/expert slope is a magnitude estimate, not a precise one. The short-prompt
fit is four points and tight (<1% residuals).

## Part 5: VRAM cost (compute buffer)

Raising k also raises VRAM use. The weights are not the cause - all `n_expert` experts stay
resident regardless of k, and the KV cache is untouched. It is the **compute buffer**: every
intermediate in the expert path carries `n_expert_used` as its middle dimension, and the expert
dimension only collapses at the very end.

From `build_moe_ffn()` in `src/llama-graph.cpp`:

| tensor              | line | shape                                 |
| ------------------- | ---- | ------------------------------------- |
| `ffn_moe_gate_up`   | 1635 | `[n_ff_exp*2, n_expert_used, n_tokens]` |
| `ffn_moe_swiglu`    | 1724 | `[n_ff_exp, n_expert_used, n_tokens]`   |
| `ffn_moe_down`      | 1767 | `[n_embd, n_expert_used, n_tokens]`     |
| `ffn_moe_weighted`  | 1785 | `[n_embd, n_expert_used, n_tokens]`     |
| `ffn_moe_out`       | 1819 | `[n_embd, n_tokens]` - k collapsed      |

The collapse happens in the aggregation loop (lines 1795-1812), which builds `n_expert_used` 2D
views and adds them down.

Summing the f32 intermediates: `1024 + 512 + 2048 + 2048` = 5632 floats = **22.5 KB per token per
expert**. Graph-allocator reuse brings the live peak down to roughly 14 KB effective, giving
approximately `n_ubatch * k * 14 KB`:

| n_ubatch     | k=8     | k=12    | delta      |
| ------------ | ------- | ------- | ---------- |
| 2048         | ~235 MB | ~352 MB | **+117 MB** |
| 512          | ~59 MB  | ~88 MB  | +29 MB     |
| 1 (decode)   | ~0.1 MB | ~0.2 MB | negligible |

Implications:

- **Invisible at decode, significant at prefill.** The cost is proportional to `n_ubatch`, so if
  VRAM is the constraint on raising k, **lowering `-ub` compensates directly** - trading some
  prefill throughput for the headroom.
- **k has three cost curves with different slopes**: decode time (weak - 22.5% of the budget),
  prefill time (weakest - batching amortizes expert count), and prefill VRAM (strong and linear).
  On a memory-constrained fit the VRAM curve is the binding one, which reframes the question from
  "is k=12 worth 11% decode speed" to "is it worth 117 MB that could have gone to context."

### Why the aggregation loop uses `hparams.n_expert_used`

Lines 1795-1812 deliberately use `hparams.n_expert_used` rather than the local `n_expert_used`,
per the comment referencing upstream PR #14753. The local value can vary per-ubatch; building the
add-chain from it would emit a different node count between warmup and steady state, invalidating
the graph cache and forcing CUDA graph recapture. Using the hparam keeps graph topology fixed -
which is also what makes `--override-kv ...expert_used_count` produce one stable graph instead of
reallocating on every batch shape.

## Part 6: Dense capacity vs MoE routing - 27B quant ladder

Part 2 flagged that the 27B dense model's gap over the stock 35B was disproportionate on a handful
of chunks (1.0-2.2x on easy content, 4.6-50x on hard content) and called this the signature of
quantization damage. This part tests that directly: does more bits, or a different imatrix, close
the gap? And separately - what is actually *in* those chunks?

### What is actually in the bad chunks

`ppl_sample.txt` is built from real Claude Code session traffic (see Artifacts). Chunk N covers
tokens `[(N-1)*4096, N*4096)` of the tokenized file; reconstructing exact chunk text just requires
dumping every token with `llama-tokenize` (default output is `id -> 'piece'`, one token per
apparent line) and concatenating the pieces for that range - verified byte-exact against the raw
file's opening bytes. The reconstruction script tolerates pieces that themselves contain a literal
newline (which would otherwise look like a second token-start line) by only treating a line as a
new token if it matches the `<digits> -> '` prefix.

The worst chunks (9, 13, 17, 19, 21, 24) are not "hard" in an ordinary sense - they are a
different register of text entirely:

- **Chunk 17**: dense TypeScript (nested ternaries, template literals, Playwright automation code)
- **Chunk 19**: raw terminal output with ANSI color escape codes (`\x1b[32m` ... `\x1b[39m`) and a
  Playwright stack trace
- **Chunk 9**: structured JSON log lines (pino-style: `{"level":30,"time":1782659018070,...}`),
  full of UUIDs, epoch timestamps, and floating-point response times
- **Chunk 13**: Claude Code's own harness/system-prompt text (tool-use conventions,
  `<system-reminder>` semantics, approval-scope rules)
- **Chunk 24**: `git status`/`git log` output, again with ANSI-colored commit hashes

Two things follow from this. First, some fraction of every bad chunk's elevated PPL is an **entropy
floor** - UUIDs, hashes, and timestamps are close to random from a language model's perspective, so
no model size or quant level will predict them well; this bounds how low any of these chunks can go
regardless of capability. Second, and separately, this diagnosed a **corpus-hygiene problem**, fixed
in this session - see "Corpus filtering fix" below.

### Routing-recoverable: stock 35B, k=8 to k=64

Extending the k-sweep (Part 2) on the *stock* (non-distilled) 35B, which - unlike the distilled
model - is not flat on the bad chunks, splits the corpus into two populations that move in opposite
directions as k rises:

| k  | easy chunks (28), geomean | hard chunks (5: 13/17/19/21/24), geomean | reported PPL |
|----|---------------------------|-------------------------------------------|--------------|
| 8  | 5.289                     | 31.59                                       | 6.9344       |
| 12 | 4.940                     | 27.74                                        | 6.4157       |
| 16 | 4.733                     | 25.70                                        | 6.1161       |
| 24 | **4.569**                 | 19.66                                        | 5.6994       |
| 32 | 4.625                     | 16.09                                        | **5.5867**   |
| 64 | 5.241                     | **13.07**                                    | 6.0189       |

The easy population bottoms out at k approx 24 and turns around, exactly like finding 2's global
curve. The apparent k=32 "optimum" in the reported PPL is 5 chunks out-voting 28: the hard
population is still improving steeply enough at k=32 to drag the geometric mean down past the point
where 85% of the corpus has already started degrading.

Individual chunks make the mechanism vivid - chunk 17 falls **27.2 -> 21.8 -> 18.6 -> 8.2 -> 4.6 ->
3.6** from k=8 to k=64 (7.5x), chunk 19 falls 65.6 -> 10.7 (6.1x), purely by admitting more experts
with no change to the weights. If this chunk were genuinely beyond the model's capability, no
amount of extra experts would fix it - capability lives in the weights, not in how many of them are
consulted. That it drops sharply means the right experts were already in the model, just ranked
below the trained top-8 cutoff for this content: **the router was pointing at the wrong subset**,
not missing the right one.

This reframes what distillation bought in Part 2's 1.67x figure: the distilled 35B is flat on these
chunks at k=8, needing no extra experts, which suggests a real part of the distillation gain is
*router alignment to the domain* (the right experts already rank in the top 8 for Claude-format
content) rather than only new knowledge. Raising k is a brute-force substitute for that alignment -
it recovers misrouted probability mass by widening the net instead of aiming better.

### Capacity-limited: 27B quant ladder, 3.0-4.25 bpw, seven imatrix/mix variants

```bash
CTX=4096 CHUNKS=-1 OUT_DIR=data/ppl-quant-sweep2 \
  ./scripts/perplexity-quant-sweep.sh /mnt/llm/models/ppl_sample.pre-filter.txt \
    Qwen3.6-27B-UD-IQ3_XXS-MTP-unsloth.gguf \
    Qwen3.6-27B-Q3_K_M-MTP-unsloth.gguf \
    Qwen3.6-27B-UD-Q3_K_XL-unsloth.gguf \
    Qwen3.6-27B-IQ4_XS-unsloth.gguf \
    Qwen3.6-27B-IQ4_XS-combined-imat-zacaj.gguf \
    Qwen3.6-27B-IQ4_XS-uniform-imat-zacaj.gguf \
    Qwen3.6-27B-IQ4_XS-qkv5-imat-zacaj.gguf \
    Qwen3.6-27B-IQ4XS-mixed-q3k-zacaj.gguf \
    Qwen3.6-27B-IQ4XS-mixed-iq3-zacaj.gguf \
    Qwen3.6-27B-IQ4XS-mixed-iq3-unslothimat-zacaj.gguf
```

| model (bpw approx)          | overall PPL | chunk 17 | chunk 19 |
|------------------------------|-------------|----------|----------|
| UD-IQ3_XXS (3.0)             | 14.5936     | 1356.6   | 640.2    |
| Q3_K_M (3.4)                 | 13.1866     | 1920.9   | 472.1    |
| UD-Q3_K_XL (3.7)              | 12.7535     | 2091.9   | 482.8    |
| IQ4_XS stock (4.25)           | 12.6702     | 1794.5   | 506.3    |
| IQ4_XS combined-imat          | 13.5374     | 1964.1   | 639.5    |
| IQ4_XS uniform-imat           | 13.6290     | 1867.5   | 635.4    |
| IQ4_XS qkv5-imat               | 13.2441     | 2141.4   | 625.2    |
| IQ4XS mixed-q3k                | 12.2018     | 1795.4   | 596.5    |
| IQ4XS mixed-iq3                | 13.3276     | 1976.4   | 474.4    |
| IQ4XS mixed-iq3-unslothimat     | 12.7726     | 1801.0   | 423.7    |
| *stock 35B, k=8 (reference)*  | *6.9344*    | *27.2*   | *65.6*   |

Across a 40% bpw range and seven independent imatrix/mix strategies at the same bit depth, chunk 17
sits in a tight 1795-2141 band with no trend against bits or calibration method - noise around a
floor roughly 65-80x the same-family MoE's k=8 reference. Chunk 19 is similarly flat (424-640
against a 65.6 reference). `qkv5-imat` - built specifically to weight `attn_qkv` more heavily - is
if anything the *worst* of the seven on chunk 17, ruling out attention-precision as the lever too.

The overall-PPL ranking (mixed-q3k best at 12.20, uniform-imat worst at 13.63) is decided entirely
by the 28 easy chunks (which move steadily with bits, ordinary quantization behavior) and is
uncorrelated with the two pinned chunks - the same single-scalar-hides-two-populations trap as
Parts 2 and this part's k-sweep above, just with quantization level as the varying axis instead of
k.

**Conclusion (quantization only): not a fixable quantization or calibration artifact.** Ten
variants of one training run, spanning 3.0-4.25 bpw and seven imatrix/mix strategies, never move
the pinned chunks. This does not by itself prove a *dense-architecture* capacity limit, though -
every variant shares one training run that never saw this content, and quantizing afterward cannot
inject knowledge training never provided. See the next section for a real test of that distinction.

### Training vs architecture: does a differently-trained dense 27B still fail?

Every model in the ladder above shares one training run (unsloth's stock instruction tune),
varying only how it was quantized afterward. That confounds two different claims: "dense 27B
lacks the capacity to represent this content" (an architecture claim) vs. "this particular
training run never saw this content, and quantization cannot inject knowledge that was never
there" (a training-coverage claim, orthogonal to architecture). The ladder only tests the second
claim; the first requires varying *training*, not bits.

[rico03/Qwen3.6-27B-Claude-Opus-Reasoning-Distilled-GGUF](https://huggingface.co/rico03/Qwen3.6-27B-Claude-Opus-Reasoning-Distilled-GGUF)
is a third-party fine-tune of the same dense 27B base, distilled from ~14k Claude 4.6 Opus
*reasoning traces* - a different register from this corpus's tool-transcripts/terminal-capture
content, but still Claude-Opus-derived, unlike unsloth's stock tune. Tested at `Q3_K_M` (13.3 GB,
closest size match to the existing ladder), same corpus, same chunk boundaries. Downloaded and run
directly for this comparison (`huggingface_hub.hf_hub_download`, then
`scripts/perplexity-quant-sweep.sh`); the distilled-35B and stock-35B rows below were re-verified
in the same session rather than reused from earlier parts, so all four rows come from one
consistent measurement pass:

| model                                                 | overall PPL | chunk 17 | chunk 19 | chunk 9 | chunk 13 | chunk 1 (easy) |
|--------------------------------------------------------|-------------|----------|----------|---------|----------|----------------|
| unsloth-27B stock, best of 10 quant/imatrix variants   | 12.20-14.59 | 1795-2141| 424-640  | -       | -        | -              |
| **rico03 dense-27B, reasoning-distilled (`Q3_K_M`)**    | **6.8705**  | **262.0**| **173.5**| **34.0**| **18.3** | **4.9**        |
| stock-unsloth-35B MoE, k=8 (no distillation)            | 6.9344      | 27.2     | 65.6     | 11.3    | 33.0     | 5.0            |
| Claude-Opus-distilled-35B MoE, k=8                      | 4.1463      | 2.68     | 18.01    | 2.99    | 17.23    | 3.67           |

Training alone - no architecture change, no MoE routing lever available to a dense model - drops
chunk 17 roughly **7x** (1795-2141 -> 262) and chunk 19 roughly **3x** (424-640 -> 173.5), moving
every finding-7 variant's pinned floor by close to an order of magnitude. rico03's overall PPL
(6.87) lands statistically indistinguishable from the stock-35B MoE's (6.93): a 27B dense model
matches a 35B MoE's aggregate quality on this corpus purely via better training data, despite fewer
total parameters and no capacity to route around gaps. **This rules out "dense architecture has
zero relevant capacity" as an explanation** - the earlier reading in finding 7 was too strong.

It does not, however, close the gap entirely, and the residual is the more interesting number.
rico03 still trails the *content-matched* distilled 35B by roughly **6-10x** on the worst chunks
(262 vs 2.68 on chunk 17; 173.5 vs 18.01 on chunk 19), despite landing close to the *non-matched*
stock 35B in aggregate. Two things are true at once here, not one: quantizing one fixed training
run cannot fix this (finding 7, confirmed), and training coverage recovers most - not all - of the
gap even without any architectural change (this section). What the remaining 6-10x is - residual
dense-capacity shortfall, or simply that reasoning-trace distillation is a partial rather than
exact match to tool-transcript/terminal-capture content - is not established by this comparison. A
dense model distilled specifically on this corpus's actual register would be needed to separate
those two remaining explanations.

### Corpus filtering fix

The content-diagnosis step above surfaced a corpus-hygiene issue independent of the quant-ladder
question: `extract_prompt_corpus.py` was including ANSI-escape-laden terminal captures,
JSON-Lines-style structured log dumps, and giant near-duplicate system-prompt/harness blocks (up to
34K chars, ~40 unique variants across 11K logs) in the calibration/eval corpus. These dominate a
small eval sample out of proportion to how informative they are, and the log dumps carry entropy
(UUIDs, timestamps) no model size or quant level will predict.

Fixed by filtering three categories by default in `extract_prompt_corpus.py` (opt back in with
`--keep-ansi` / `--keep-log-dumps` / `--keep-system`), then regenerating the full chain
(`prompt_corpus.txt` -> `corpus-holdout-slice.sh` -> `ppl_sample.txt`). `prompt_corpus.txt` dropped
from 23.6M to 21.5M chars (~9%). **All PPL numbers in this document, including this part's quant
ladder, were measured against the pre-filter corpus** (preserved as `ppl_sample.pre-filter.txt` /
`prompt_corpus.pre-filter.txt`) for internal consistency - `ppl_sample.txt` now refers to the
filtered corpus and will not reproduce these exact figures. Future perplexity work on this repo
should use the filtered file unless deliberately reproducing something in this document.

### Follow-up: the entropy leak the first fix missed

Re-running the same per-chunk decomposition on a later 4-chunk `sample20000` sweep found the
identical failure mode still present, and traced it to a gap in the fix above. The three filters
are all *whole-string* categories; bare session ids and ISO-8601 timestamps arrive from
`walk_strings` as standalone leaf strings (a 32-char id and a microsecond timestamp are both
exactly 32 chars, above the default `--min-len 20`), contain no ANSI, are not JSON lines, and are
not system-role. They passed all three. Measured on `prompt_corpus.txt`:

- 10,714 lines -- 3.1% of all non-blank lines -- were *nothing but* a bare id or timestamp.
- Filtering made the density **worse**, 196 -> 248 id32/MB, because it removed a lot of long
  id-free text (system prompts are huge and contain none) while leaving every short id intact.
- 93% of 32-char ids and 99.7% of microsecond timestamps were bare; conversely 100% of UUIDs were
  embedded, nearly all inside `/tmp/claude-<n>/.../<uuid>/...` paths.

Fixed by a fourth default filter (`--keep-entropy` to opt back in) that drops a string when it is
itself one high-entropy token, or is a multi-line block that is >=50% such tokens. Embedded tokens
are left alone on purpose: masking them would replace unpredictable tokens with a repeated,
trivially predictable placeholder and skew perplexity the other way. Replayed over the existing
`prompt_corpus.txt` this drops 11.6% of leaf strings but only 1.6% of characters, taking id32 from
248/MB to 18/MB and microsecond timestamps from 5,761 to 15.

One caveat on the artifacts: `prompt_corpus.pre-filter.txt` is not a strict before-image of
`prompt_corpus.txt`. Comparing unique bare-id sets, the filter removed zero and the filtered file
contains 507 ids the pre-filter file does not, so the source log set grew between the two runs and
the "23.6M -> 21.5M chars (~9%)" figure above conflates the filter with corpus growth. The
document's own numbers are unaffected -- they were all measured against the pre-filter file.

## Artifacts

- `examples/moe-weights/` - the measurement tool (not upstreamed)
- `scripts/perplexity-quant-sweep.sh` - multi-model PPL sweep with full per-chunk logs retained,
  used for Part 6's quant ladder (now a thin wrapper over `scripts/perplexity-sweep.sh`, which
  generalizes to a models x k-values x corpora sweep - see its header for usage)
- `scripts/extract_prompt_corpus.py` - corpus generator; now filters ANSI/log-dump/system-role
  content by default (Part 6)
- `/mnt/llm/models/moe_sample.txt` - 3.4KB / 725 tokens, from `prompt_corpus.txt`
- `/mnt/llm/models/ppl_sample.pre-filter.txt` - 500KB / 33 chunks, the exact file every PPL number
  in this document was measured against (pre corpus-filtering fix)
- `/mnt/llm/models/ppl_sample.txt` - current (filtered) 500KB sample; not byte-identical to the
  above, see "Corpus filtering fix"
- `/mnt/2508/Backup 2/Qwen3.6-27B-Claude-Opus-Reasoning-Distilled-Q3_K_M.gguf` - third-party
  reasoning-distilled dense 27B (Part 6, "Training vs architecture"), downloaded from
  [rico03/Qwen3.6-27B-Claude-Opus-Reasoning-Distilled-GGUF](https://huggingface.co/rico03/Qwen3.6-27B-Claude-Opus-Reasoning-Distilled-GGUF)

### Build note

Docker BuildKit has cached the `COPY . .` layer despite source changes on this host. If a rebuild
appears not to pick up edits, run `docker compose build --no-cache llama-cpp` once.
