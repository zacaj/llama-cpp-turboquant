// moe-weights: report the distribution of MoE router weights by rank.
//
// Runs a single prompt-processing pass over a text file/prompt and, via the
// ggml_backend_sched_eval_callback mechanism (same one examples/eval-callback
// uses), collects the per-token, per-layer normalized router weights
// router weights, and reports per layer and overall:
//   - mean weight at each rank (1 = highest)
//   - stddev at each rank
//   - cumulative mass carried by the bottom two ranks (what a k-2 reduction
//     would discard)
//
// Two tensors are collected, both produced in build_moe_ffn() in
// src/llama-graph.cpp:
//
//   "ffn_moe_weights-<il>"      pre-normalization: the raw softmax probs of the
//                               selected experts, gathered by ggml_get_rows().
//   "ffn_moe_weights_norm-<il>" post-normalization: the above divided by its
//                               own row sum, i.e. what actually scales the FFN.
//
// The pre-norm numbers are the ones to use when comparing across different
// values of n_expert_used. For gating funcs that softmax over all n_expert
// before the top-k cut (SOFTMAX, which is what qwen35moe uses), the value at
// rank r does not depend on k at all: the softmax is computed over the full
// expert set, and top-k just takes a prefix of the same descending order. The
// post-norm numbers cannot be compared across k, because dividing by the row
// sum folds k into every entry.
//
// The pre-norm row sum is itself the quantity of interest: it is the fraction
// of the router's total probability mass that the top-k cut actually captured.
// That is reported as "captured".
//
// The tensor has shape [1, n_expert_used, n_tokens] and is already sorted
// descending by rank (ggml_argsort_top_k produced the gather indices used to
// build it, and that order is preserved) -- we deliberately do NOT re-sort on
// the host, we just verify the assumption holds.
//
// A third tensor, "ffn_moe_topk-<il>" (the argsort_top_k output itself, i.e.
// which expert IDs were selected -- not their weights), is also collected.
// With -o/--output FNAME, two CSVs are written in scripts/
// lowest_experts_from_profile.py's input format (rank,expert,total_count,
// pct,layer_0,...,layer_N), so pruning candidates can be picked from any
// perplexity-style corpus instead of requiring live traffic instrumentation:
//   FNAME               ranked by raw selection count (rank-agnostic: a
//                       rank-1 pick and a rank-8 pick count the same).
//   FNAME.weighted.csv  ranked by summed pre-norm router weight instead --
//                       an expert hit often but always at a low rank (small
//                       contribution to the FFN mix) can rank very
//                       differently here than by raw count.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "../../src/llama-ext.h" // llama_model_n_expert

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <numeric>
#include <string>
#include <vector>

// Accumulated stats for one layer.
struct layer_stats {
    int64_t n_expert_used = 0;
    int64_t n_rows         = 0;   // number of token-rows accumulated
    std::vector<double> sum;      // sum of weight, per rank
    std::vector<double> sum_sq;   // sum of weight^2, per rank
    double bottom2_mass_sum = 0.0; // sum over rows of (mass in bottom 2 ranks)
    double row_sum_sum      = 0.0; // sum over rows of (total mass in the row)
    bool   warned_nonmono = false;
    bool   warned_sum     = false;

    void ensure(int64_t n) {
        if (sum.empty()) {
            n_expert_used = n;
            sum.assign(n, 0.0);
            sum_sq.assign(n, 0.0);
        }
    }
};

struct moe_weights_collector {
    std::map<int, layer_stats> layers; // key = il

    const char * prefix;      // tensor name prefix this collector claims
    const char * label;       // human-readable name for the report header
    bool         expect_norm; // if true, warn when a row does not sum to 1

    moe_weights_collector(const char * prefix, const char * label, bool expect_norm)
        : prefix(prefix), label(label), expect_norm(expect_norm) {}

    // Parse "<prefix><il>" -> il. Returns false if it doesn't match.
    // Note the prefixes are nested ("ffn_moe_weights-" is a prefix of nothing,
    // but "ffn_moe_weights_norm-" shares a stem with it), so matching is done
    // on the full prefix including the trailing '-' to keep them disjoint.
    static bool parse_layer(const char * prefix, const char * name, int * il_out) {
        const size_t prefix_len = strlen(prefix);
        if (strncmp(name, prefix, prefix_len) != 0) {
            return false;
        }
        const char * suffix = name + prefix_len;
        if (*suffix == '\0') {
            return false;
        }
        char * end = nullptr;
        long il = strtol(suffix, &end, 10);
        if (end == suffix || *end != '\0' || il < 0) {
            return false;
        }
        *il_out = (int) il;
        return true;
    }

    void process(ggml_tensor * t) {
        int il = -1;
        if (!parse_layer(prefix, t->name, &il)) {
            return;
        }

        if (t->type != GGML_TYPE_F32) {
            LOG_ERR("moe-weights: tensor %s has unexpected type %s, skipping\n", t->name, ggml_type_name(t->type));
            return;
        }

        // The task spec (and llama-graph.cpp's comment at the ggml_div() call site) describes this
        // tensor as [1, n_expert_used, n_tokens] (a leading dim-1 axis). In practice, on this build
        // (Qwen3.6 / qwen35moe), the tensor is observed at cb() time as 2D: [n_expert_used, n_tokens]
        // -- ggml_div's cb() fires *before* the subsequent ggml_reshape_3d() call that re-adds the
        // leading 1-dim in build_moe_ffn(). Handle both shapes.
        int64_t n_expert_used;
        int64_t n_tokens;
        size_t  nb_rank;  // byte stride between consecutive ranks within a token row
        size_t  nb_tok;   // byte stride between consecutive tokens
        if (t->ne[0] == 1) {
            // [1, n_expert_used, n_tokens]
            n_expert_used = t->ne[1];
            n_tokens      = t->ne[2];
            nb_rank       = t->nb[1];
            nb_tok        = t->nb[2];
        } else {
            // [n_expert_used, n_tokens]
            n_expert_used = t->ne[0];
            n_tokens      = t->ne[1];
            nb_rank       = t->nb[0];
            nb_tok        = t->nb[1];
        }

        const size_t n_bytes = ggml_nbytes(t);
        std::vector<uint8_t> buf(n_bytes);

        const bool is_host = ggml_backend_buffer_is_host(t->buffer);
        if (is_host) {
            memcpy(buf.data(), t->data, n_bytes);
        } else {
            ggml_backend_tensor_get(t, buf.data(), 0, n_bytes);
        }

        const float * data = (const float *) buf.data();

        auto & ls = layers[il];
        ls.ensure(n_expert_used);
        if (ls.n_expert_used != n_expert_used) {
            LOG_ERR("moe-weights: layer %d n_expert_used changed (%lld -> %lld), skipping tensor\n",
                    il, (long long) ls.n_expert_used, (long long) n_expert_used);
            return;
        }

        const size_t nb1 = nb_rank;
        const size_t nb2 = nb_tok;

        for (int64_t tok = 0; tok < n_tokens; ++tok) {
            const uint8_t * row_base = buf.data() + tok * nb2;

            float row_sum = 0.0f;
            float prev = INFINITY;
            bool nonmono = false;

            for (int64_t r = 0; r < n_expert_used; ++r) {
                const float v = *(const float *) (row_base + r * nb1);
                if (v > prev + 1e-4f) {
                    nonmono = true;
                }
                prev = v;

                ls.sum[r]    += v;
                ls.sum_sq[r] += (double) v * (double) v;
                row_sum += v;
            }

            if (nonmono && !ls.warned_nonmono) {
                LOG_ERR("moe-weights: WARNING layer %d: rank weights are not non-increasing "
                        "(assumption that ggml_argsort_top_k output stays sorted appears to be "
                        "violated) - results may be meaningless\n", il);
                ls.warned_nonmono = true;
            }

            // For the post-norm tensor a row must sum to 1; anything else means we
            // latched onto the wrong tensor. For the pre-norm tensor the row sum is
            // the captured probability mass, which is the point of collecting it.
            if (expect_norm && std::fabs(row_sum - 1.0f) > 1e-2f && !ls.warned_sum) {
                LOG_ERR("moe-weights: WARNING layer %d: row does not sum to 1.0 (got %f) - "
                        "expected post-normalization weights\n", il, row_sum);
                ls.warned_sum = true;
            }
            ls.row_sum_sum += row_sum;

            double bottom2 = 0.0;
            const int64_t n_bottom = n_expert_used >= 2 ? 2 : n_expert_used;
            for (int64_t r = n_expert_used - n_bottom; r < n_expert_used; ++r) {
                bottom2 += *(const float *) (row_base + r * nb1);
            }
            ls.bottom2_mass_sum += bottom2;

            ls.n_rows++;
        }

        (void) data;
    }
};

// Tallies which expert IDs are actually selected, per layer -- the data
// scripts/lowest_experts_from_profile.py needs to pick pruning candidates --
// two ways: raw hit count (rank-agnostic: a rank-1 pick and a rank-8 pick
// count the same), and summed pre-norm router weight (a rank-8 pick, which
// contributes little to the FFN mix, adds little; a rank-1 pick adds a lot).
// An expert hit often but always at a low rank can rank very differently by
// these two criteria.
//
// Fed from two tensors produced back-to-back in the same layer's slice of
// build_moe_ffn() (src/llama-graph.cpp):
//   "ffn_moe_topk-<il>"    the ggml_argsort_top_k() output: which experts,
//                          shape [n_expert_used, n_tokens], dtype I32.
//   "ffn_moe_weights-<il>" = ggml_get_rows(probs, ffn_moe_topk): their
//                          pre-norm softmax weight, same [rank, token]
//                          layout, dtype F32.
// Because the weights tensor is a row-gather of the topk tensor computed
// immediately after it in the same graph, the topk tensor for a layer always
// arrives (via this eval callback) right before that layer's weights tensor,
// within the same eval pass. Buffering the most recent topk tensor per layer
// and consuming it when that layer's weights tensor shows up is therefore
// safe -- no cross-layer or cross-batch aliasing.
struct moe_expert_usage_collector {
    std::map<int, std::vector<int32_t>> topk_buf;  // il -> [tok*n_used + r] = expert id
    std::map<int, int64_t> topk_n_used;
    std::map<int, int64_t> topk_n_tokens;

    std::map<int, std::vector<int64_t>> counts;      // il -> counts[expert]
    std::map<int, std::vector<double>>  weight_sum;  // il -> weight_sum[expert]
    int64_t n_expert = 0; // set by main() from llama_model_n_expert() before run()

    // Shared shape/stride decoding for both tensors: [1, n_expert_used, n_tokens]
    // or [n_expert_used, n_tokens] depending on whether cb() fired pre- or
    // post- the leading dim-1 reshape (see moe_weights_collector::process()).
    static void decode_shape(ggml_tensor * t, int64_t & n_expert_used, int64_t & n_tokens, size_t & nb_rank, size_t & nb_tok) {
        if (t->ne[0] == 1) {
            n_expert_used = t->ne[1];
            n_tokens      = t->ne[2];
            nb_rank       = t->nb[1];
            nb_tok        = t->nb[2];
        } else {
            n_expert_used = t->ne[0];
            n_tokens      = t->ne[1];
            nb_rank       = t->nb[0];
            nb_tok        = t->nb[1];
        }
    }

    static std::vector<uint8_t> read_tensor(ggml_tensor * t) {
        std::vector<uint8_t> buf(ggml_nbytes(t));
        if (ggml_backend_buffer_is_host(t->buffer)) {
            memcpy(buf.data(), t->data, buf.size());
        } else {
            ggml_backend_tensor_get(t, buf.data(), 0, buf.size());
        }
        return buf;
    }

    void process_topk(ggml_tensor * t, int il) {
        if (t->type != GGML_TYPE_I32) {
            LOG_ERR("moe-weights: tensor %s has unexpected type %s, skipping\n", t->name, ggml_type_name(t->type));
            return;
        }

        int64_t n_expert_used, n_tokens;
        size_t  nb_rank, nb_tok;
        decode_shape(t, n_expert_used, n_tokens, nb_rank, nb_tok);
        const std::vector<uint8_t> buf = read_tensor(t);

        auto & flat = topk_buf[il];
        flat.assign(n_expert_used * n_tokens, -1);
        for (int64_t tok = 0; tok < n_tokens; ++tok) {
            const uint8_t * row_base = buf.data() + tok * nb_tok;
            for (int64_t r = 0; r < n_expert_used; ++r) {
                flat[tok * n_expert_used + r] = *(const int32_t *) (row_base + r * nb_rank);
            }
        }
        topk_n_used[il]   = n_expert_used;
        topk_n_tokens[il] = n_tokens;

        auto & c = counts[il];
        if (c.empty()) {
            c.assign(n_expert, 0);
        }
        for (int32_t e : flat) {
            if (e < 0 || e >= n_expert) {
                LOG_ERR("moe-weights: layer %d: expert id %d out of range [0,%lld), skipping\n", il, e, (long long) n_expert);
                continue;
            }
            c[e]++;
        }
    }

    void process_weights(ggml_tensor * t, int il) {
        if (t->type != GGML_TYPE_F32) {
            LOG_ERR("moe-weights: tensor %s has unexpected type %s, skipping\n", t->name, ggml_type_name(t->type));
            return;
        }

        int64_t n_expert_used, n_tokens;
        size_t  nb_rank, nb_tok;
        decode_shape(t, n_expert_used, n_tokens, nb_rank, nb_tok);

        auto it = topk_buf.find(il);
        if (it == topk_buf.end() || topk_n_used[il] != n_expert_used || topk_n_tokens[il] != n_tokens) {
            LOG_ERR("moe-weights: layer %d: ffn_moe_weights arrived without a matching ffn_moe_topk buffer, skipping\n", il);
            return;
        }
        const std::vector<uint8_t> buf = read_tensor(t);

        auto & ws = weight_sum[il];
        if (ws.empty()) {
            ws.assign(n_expert, 0.0);
        }
        const std::vector<int32_t> & ids = it->second;
        for (int64_t tok = 0; tok < n_tokens; ++tok) {
            const uint8_t * row_base = buf.data() + tok * nb_tok;
            for (int64_t r = 0; r < n_expert_used; ++r) {
                const int32_t e = ids[tok * n_expert_used + r];
                if (e < 0 || e >= n_expert) {
                    continue; // already warned in process_topk
                }
                const float w = *(const float *) (row_base + r * nb_rank);
                ws[e] += (double) w;
            }
        }
    }
};

static std::string fmt_value(int64_t v) { char b[32]; snprintf(b, sizeof(b), "%lld", (long long) v); return b; }
static std::string fmt_value(double   v) { char b[32]; snprintf(b, sizeof(b), "%.6f", v);            return b; }

// Writes scripts/lowest_experts_from_profile.py's expected input format:
//   rank,expert,total_count,pct,layer_0,layer_1,...,layer_N
// where rank 0 is the single most-selected/highest-weighted expert (summed
// across layers). Shared by both the raw-count and weight-sum profiles --
// only the value type (int64_t counts vs double weight mass) differs.
template <typename T>
static void write_expert_csv(const std::map<int, std::vector<T>> & layer_values, int64_t n_expert, const std::string & label, const std::string & path) {
    if (layer_values.empty()) {
        LOG_ERR("moe-weights: no %s data collected, not writing %s\n", label.c_str(), path.c_str());
        return;
    }

    std::vector<int> layers;
    for (const auto & kv : layer_values) {
        layers.push_back(kv.first);
    }
    std::sort(layers.begin(), layers.end());

    std::vector<T> expert_totals(n_expert, T(0));
    for (int il : layers) {
        const auto & vals = layer_values.at(il);
        for (int64_t e = 0; e < n_expert; ++e) {
            expert_totals[e] += vals[e];
        }
    }
    const T grand_total = std::accumulate(expert_totals.begin(), expert_totals.end(), T(0));

    std::vector<int64_t> ranked(n_expert);
    for (int64_t e = 0; e < n_expert; ++e) {
        ranked[e] = e;
    }
    std::sort(ranked.begin(), ranked.end(), [&](int64_t a, int64_t b) {
        return expert_totals[a] > expert_totals[b];
    });

    FILE * f = fopen(path.c_str(), "w");
    if (!f) {
        LOG_ERR("moe-weights: failed to open %s for writing\n", path.c_str());
        return;
    }

    fprintf(f, "rank,expert,total_count,pct");
    for (int il : layers) {
        fprintf(f, ",layer_%d", il);
    }
    fprintf(f, "\n");

    for (size_t rank = 0; rank < ranked.size(); ++rank) {
        const int64_t e     = ranked[rank];
        const T       count = expert_totals[e];
        const double  pct   = grand_total > T(0) ? 100.0 * (double) count / (double) grand_total : 0.0;
        fprintf(f, "%zu,%lld,%s,%.4f", rank, (long long) e, fmt_value(count).c_str(), pct);
        for (int il : layers) {
            fprintf(f, ",%s", fmt_value(layer_values.at(il)[e]).c_str());
        }
        fprintf(f, "\n");
    }

    fclose(f);
    LOG("moe-weights: wrote per-expert %s (%zu layers, %lld experts) to %s\n",
        label.c_str(), layers.size(), (long long) n_expert, path.c_str());
}

// Both collectors are fed from the same eval pass.
struct moe_collectors {
    moe_weights_collector pre { "ffn_moe_weights-",      "PRE-NORM (raw softmax probs, k-independent)", false };
    moe_weights_collector post{ "ffn_moe_weights_norm-", "POST-NORM (what scales the FFN)",             true  };
    moe_expert_usage_collector usage;
};

static bool moe_weights_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * c = (moe_collectors *) user_data;

    int il = -1;
    const bool is_pre  = moe_weights_collector::parse_layer(c->pre.prefix,      t->name, &il);
    const bool is_post = moe_weights_collector::parse_layer(c->post.prefix,     t->name, &il);
    const bool is_topk = moe_weights_collector::parse_layer("ffn_moe_topk-",    t->name, &il);

    if (!is_pre && !is_post && !is_topk) {
        return true; // not interested, but must return true to let the graph continue
    }

    if (ask) {
        return true; // yes, please give us the data
    }

    if (is_pre) {
        c->pre.process(t);
        c->usage.process_weights(t, il);
    } else if (is_post) {
        c->post.process(t);
    } else {
        c->usage.process_topk(t, il);
    }

    return true;
}

static void print_report(const moe_weights_collector & collector, int64_t n_expert, int64_t n_expert_used_hparam) {
    if (collector.layers.empty()) {
        LOG("moe-weights: no MoE layers observed (tensor \"%s<il>\" never appeared) "
            "- is this actually a MoE model?\n", collector.prefix);
        return;
    }

    LOG("\n=== %s ===\n", collector.label);

    int64_t n_expert_used = 0;
    int64_t total_tokens = 0;
    for (const auto & kv : collector.layers) {
        n_expert_used = std::max(n_expert_used, kv.second.n_expert_used);
        total_tokens = std::max(total_tokens, kv.second.n_rows);
    }

    LOG("\n");
    LOG("n_expert       = %lld\n", (long long) n_expert);
    LOG("n_expert_used  = %lld\n", (long long) n_expert_used);
    if (n_expert_used_hparam > 0 && n_expert_used_hparam != n_expert_used) {
        LOG("  (note: hparams reported n_expert_used=%lld, tensor shape reported %lld)\n",
            (long long) n_expert_used_hparam, (long long) n_expert_used);
    }
    LOG("layers observed = %zu\n", collector.layers.size());
    LOG("tokens analyzed (max over layers) = %lld\n", (long long) total_tokens);
    LOG("\n");

    // header
    std::string header = " layer |";
    for (int64_t r = 0; r < n_expert_used; ++r) {
        char buf[16];
        snprintf(buf, sizeof(buf), " rank%-2lld", (long long) (r + 1));
        header += buf;
    }
    header += " | bottom2%";
    if (!collector.expect_norm) {
        header += " | captured";
    }
    LOG("%s\n", header.c_str());

    std::string sep(header.size(), '-');
    // put a '+' where the '|' characters are, cosmetic only
    for (size_t i = 0; i < header.size(); ++i) {
        if (header[i] == '|') sep[i] = '+';
    }
    LOG("%s\n", sep.c_str());

    std::vector<double> overall_sum(n_expert_used, 0.0);
    std::vector<double> overall_sum_sq(n_expert_used, 0.0);
    double overall_bottom2 = 0.0;
    double overall_row_sum = 0.0;
    int64_t overall_rows = 0;

    for (const auto & kv : collector.layers) {
        const int il = kv.first;
        const layer_stats & ls = kv.second;
        if (ls.n_rows == 0) continue;

        std::string line;
        char cell[32];
        snprintf(cell, sizeof(cell), " %5d |", il);
        line += cell;

        for (int64_t r = 0; r < ls.n_expert_used; ++r) {
            const double mean = ls.sum[r] / (double) ls.n_rows;
            snprintf(cell, sizeof(cell), " %.3f", mean);
            line += cell;

            overall_sum[r]    += ls.sum[r];
            overall_sum_sq[r] += ls.sum_sq[r];
        }
        overall_rows   += ls.n_rows;
        overall_bottom2 += ls.bottom2_mass_sum;

        // bottom2 is reported as a share of the row's own total in both tables, so
        // the column means the same thing whether or not the row was normalized.
        const double denom = collector.expect_norm ? (double) ls.n_rows : ls.row_sum_sum;
        const double bottom2_pct = denom > 0.0 ? 100.0 * ls.bottom2_mass_sum / denom : 0.0;
        snprintf(cell, sizeof(cell), " | %6.2f%%", bottom2_pct);
        line += cell;

        if (!collector.expect_norm) {
            snprintf(cell, sizeof(cell), " | %7.4f", ls.row_sum_sum / (double) ls.n_rows);
            line += cell;
        }

        overall_row_sum += ls.row_sum_sum;

        LOG("%s\n", line.c_str());
    }

    LOG("%s\n", sep.c_str());

    // MEAN row (mean across layers, weighted by rows, of per-rank mean)
    {
        std::string line = "  MEAN |";
        char cell[32];
        for (int64_t r = 0; r < n_expert_used; ++r) {
            const double mean = overall_rows > 0 ? overall_sum[r] / (double) overall_rows : 0.0;
            snprintf(cell, sizeof(cell), " %.3f", mean);
            line += cell;
        }
        const double denom = collector.expect_norm ? (double) overall_rows : overall_row_sum;
        const double bottom2_pct = denom > 0.0 ? 100.0 * overall_bottom2 / denom : 0.0;
        snprintf(cell, sizeof(cell), " | %6.2f%%", bottom2_pct);
        line += cell;
        if (!collector.expect_norm) {
            snprintf(cell, sizeof(cell), " | %7.4f", overall_rows > 0 ? overall_row_sum / (double) overall_rows : 0.0);
            line += cell;
        }
        LOG("%s\n", line.c_str());
    }

    // STDDEV row (overall, per rank)
    {
        std::string line = " STDEV |";
        char cell[32];
        for (int64_t r = 0; r < n_expert_used; ++r) {
            double mean = overall_rows > 0 ? overall_sum[r] / (double) overall_rows : 0.0;
            double var  = overall_rows > 0 ? overall_sum_sq[r] / (double) overall_rows - mean * mean : 0.0;
            if (var < 0.0) var = 0.0; // guard against fp roundoff
            snprintf(cell, sizeof(cell), " %.3f", std::sqrt(var));
            line += cell;
        }
        line += " |";
        LOG("%s\n", line.c_str());
    }

    LOG("\n");
    {
        const double denom = collector.expect_norm ? (double) overall_rows : overall_row_sum;
        LOG("Overall bottom-2-rank cumulative mass: %.2f%% (i.e. what a k-2 expert reduction would discard, on average)\n",
            denom > 0.0 ? 100.0 * overall_bottom2 / denom : 0.0);
    }
    if (!collector.expect_norm && overall_rows > 0) {
        const double captured = overall_row_sum / (double) overall_rows;
        LOG("Overall captured probability mass: %.4f (top-%lld of %lld experts; the remaining %.4f is discarded by the cut)\n",
            captured, (long long) n_expert_used, (long long) n_expert, 1.0 - captured);
    }
}

static bool run(llama_context * ctx, const common_params & params) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos, true);

    if (tokens.empty()) {
        LOG_ERR("%s : there are no input tokens to process - (try to provide a prompt with '-p' or a file with '-f')\n", __func__);
        return false;
    }

    LOG_INF("number of input tokens = %zu\n", tokens.size());

    const int n_batch = params.n_batch > 0 ? params.n_batch : 512;

    for (size_t i = 0; i < tokens.size(); i += n_batch) {
        const size_t n = std::min<size_t>(n_batch, tokens.size() - i);
        if (llama_decode(ctx, llama_batch_get_one(tokens.data() + i, n))) {
            LOG_ERR("%s : failed to eval\n", __func__);
            return false;
        }
    }

    return true;
}

int main(int argc, char ** argv) {
    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    moe_collectors collectors;

    params.cb_eval           = moe_weights_cb_eval;
    params.cb_eval_user_data = &collectors;
    params.warmup            = false;

    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }

    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
        LOG_INF("\n");
    }

    const int64_t n_expert = llama_model_n_expert(model);

    if (n_expert <= 1) {
        LOG("moe-weights: model reports n_expert=%lld - this does not look like a MoE model\n",
            (long long) n_expert);
    }
    collectors.usage.n_expert = n_expert;

    bool OK = run(ctx, params);
    if (!OK) {
        return 1;
    }

    print_report(collectors.pre,  n_expert, -1);
    print_report(collectors.post, n_expert, -1);

    if (!params.out_file.empty()) {
        write_expert_csv(collectors.usage.counts, n_expert, "activation counts", params.out_file);

        // sibling file: same layer/expert grid, but ranked by summed pre-norm
        // router weight instead of raw hit count (see moe_expert_usage_collector
        // comment -- a rank-8 hit contributes far less to the FFN mix than rank-1).
        std::string weighted_path = params.out_file;
        const std::string suffix = ".csv";
        if (weighted_path.size() >= suffix.size() &&
            weighted_path.compare(weighted_path.size() - suffix.size(), suffix.size(), suffix) == 0) {
            weighted_path.resize(weighted_path.size() - suffix.size());
        }
        weighted_path += ".weighted.csv";
        write_expert_csv(collectors.usage.weight_sum, n_expert, "router weight mass", weighted_path);
    }

    LOG("\n");
    llama_perf_context_print(ctx);

    llama_backend_free();

    return 0;
}
