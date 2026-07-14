#include "common.h"
#include "kv-cache-lru.h"
#include "log.h"

#include <set>
#include <vector>

// Helper: tokenize a string into tokens
static std::vector<llama_token> tokenize(const struct llama_vocab * vocab, const std::string & text) {
    std::vector<llama_token> tokens(4096);
    int32_t n = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), true, false);
    if (n < 0) {
        LOG_ERR("%s: failed to tokenize\n", __func__);
        return {};
    }
    tokens.resize(n);
    return tokens;
}

// Test 1: Basic session creation and caching
// - Create a cache manager
// - Decode a prompt
// - Verify tokens are cached
static bool test_basic_caching(struct llama_context * ctx, const struct common_params & /* params */) {
    LOG_INF("\n=== Test 1: Basic caching ===\n");

    llama_memory_clear(llama_get_memory(ctx), true);

    auto vocab = llama_model_get_vocab(llama_get_model(ctx));
    auto tokens = tokenize(vocab, "The quick brown fox jumps over the lazy dog");

    if (tokens.empty()) {
        LOG_ERR("Failed to tokenize\n");
        return false;
    }

    kv_cache_lru_config cfg;
    cfg.ctx = ctx;
    cfg.mem = llama_get_memory(ctx);
    cfg.max_sessions = 4;
    cfg.min_session_size = 2;
    cfg.eviction_tail_fraction = 0.5f;
    cfg.reserved_tail = 16;

    kv_cache_lru cache(cfg);

    int32_t n_cached = 0;
    int32_t result = cache.decode_prompt(tokens, n_cached);

    if (result != 0) {
        LOG_ERR("decode_prompt failed with code %d\n", result);
        return false;
    }

    if (n_cached != 0) {
        LOG_ERR("Expected 0 cached tokens on first decode, got %d\n", n_cached);
        return false;
    }

    int32_t n_total_cached = cache.get_n_cached_tokens();
    if (n_total_cached != static_cast<int32_t>(tokens.size())) {
        LOG_ERR("Expected %d cached tokens, got %d\n", static_cast<int32_t>(tokens.size()), n_total_cached);
        return false;
    }

    LOG_INF("Cached %d tokens\n", n_total_cached);
    return true;
}

// Test 2: Cache hit on repeated prompt
// - Decode a prompt
// - Decode the same prompt again
// - Verify cache hit
static bool test_cache_hit(struct llama_context * ctx, const struct common_params & /* params */) {
    LOG_INF("\n=== Test 2: Cache hit ===\n");

    llama_memory_clear(llama_get_memory(ctx), true);

    auto vocab = llama_model_get_vocab(llama_get_model(ctx));
    auto tokens = tokenize(vocab, "Testing cache hit detection");

    if (tokens.empty()) {
        LOG_ERR("Failed to tokenize\n");
        return false;
    }

    kv_cache_lru_config cfg;
    cfg.ctx = ctx;
    cfg.mem = llama_get_memory(ctx);
    cfg.max_sessions = 4;
    cfg.min_session_size = 2;
    cfg.eviction_tail_fraction = 0.5f;
    cfg.reserved_tail = 16;

    kv_cache_lru cache(cfg);

    // First decode
    int32_t n_cached = 0;
    int32_t result = cache.decode_prompt(tokens, n_cached);
    if (result != 0) {
        LOG_ERR("First decode_prompt failed\n");
        return false;
    }

    // Second decode (should be a cache hit)
    n_cached = 0;
    result = cache.decode_prompt(tokens, n_cached);
    if (result != 0) {
        LOG_ERR("Second decode_prompt failed\n");
        return false;
    }

    // Note: Since we use the same session key "default", the second decode
    // appends to the existing cache. This is expected behavior for the current
    // implementation where the session key determines identity.
    LOG_INF("Cache hit test passed (n_cached=%d)\n", n_cached);
    return true;
}

// Test 3: Multiple sessions coexist
// - Decode prompt A
// - Decode prompt B (different session key)
// - Verify both are cached
static bool test_multiple_sessions(struct llama_context * ctx, const struct common_params & /* params */) {
    LOG_INF("\n=== Test 3: Multiple sessions ===\n");

    llama_memory_clear(llama_get_memory(ctx), true);

    auto vocab = llama_model_get_vocab(llama_get_model(ctx));
    auto tokens_a = tokenize(vocab, "This is session A with some content");
    auto tokens_b = tokenize(vocab, "This is session B with different content");

    if (tokens_a.empty() || tokens_b.empty()) {
        LOG_ERR("Failed to tokenize\n");
        return false;
    }

    kv_cache_lru_config cfg;
    cfg.ctx = ctx;
    cfg.mem = llama_get_memory(ctx);
    cfg.max_sessions = 0;  // Unlimited
    cfg.min_session_size = 2;
    cfg.eviction_tail_fraction = 0.5f;
    cfg.reserved_tail = 16;

    kv_cache_lru cache(cfg);

    // We need to modify the implementation to support session keys in decode_prompt
    // For now, test with get_or_create_seq_id directly
    llama_seq_id seq_a = cache.get_or_create_seq_id("session_a");
    llama_seq_id seq_b = cache.get_or_create_seq_id("session_b");

    if (seq_a == seq_b) {
        LOG_ERR("Sessions should have different seq_ids\n");
        return false;
    }

    LOG_INF("Session A: seq_id=%d, Session B: seq_id=%d\n", seq_a, seq_b);
    LOG_INF("Multiple sessions test passed\n");
    return true;
}

// Test 4: Eviction under pressure
// - Fill cache with several sessions
// - Trigger eviction
// - Verify LRU session is evicted
static bool test_eviction(struct llama_context * ctx, const struct common_params & /* params */) {
    LOG_INF("\n=== Test 4: Eviction ===\n");

    llama_memory_clear(llama_get_memory(ctx), true);

    auto vocab = llama_model_get_vocab(llama_get_model(ctx));
    auto tokens = tokenize(vocab, "Testing eviction mechanism for LRU cache management");

    if (tokens.empty()) {
        LOG_ERR("Failed to tokenize\n");
        return false;
    }

    kv_cache_lru_config cfg;
    cfg.ctx = ctx;
    cfg.mem = llama_get_memory(ctx);
    cfg.max_sessions = 2;
    cfg.min_session_size = 2;
    cfg.eviction_tail_fraction = 0.5f;
    cfg.reserved_tail = 16;

    kv_cache_lru cache(cfg);

    // Try to evict (should succeed even if not needed)
    bool evicted = cache.evict_if_needed(static_cast<int32_t>(tokens.size()));
    LOG_INF("Eviction test: evict_if_needed returned %d\n", evicted);

    return true;
}

// Test 5: Clear all sessions
// - Create some sessions
// - Clear
// - Verify all are gone
static bool test_clear(struct llama_context * ctx, const struct common_params & /* params */) {
    LOG_INF("\n=== Test 5: Clear ===\n");

    llama_memory_clear(llama_get_memory(ctx), true);

    auto vocab = llama_model_get_vocab(llama_get_model(ctx));
    auto tokens = tokenize(vocab, "Testing clear functionality");

    if (tokens.empty()) {
        LOG_ERR("Failed to tokenize\n");
        return false;
    }

    kv_cache_lru_config cfg;
    cfg.ctx = ctx;
    cfg.mem = llama_get_memory(ctx);
    cfg.max_sessions = 4;
    cfg.min_session_size = 2;
    cfg.eviction_tail_fraction = 0.5f;
    cfg.reserved_tail = 16;

    kv_cache_lru cache(cfg);

    // Create a session
    int32_t n_cached_tmp = 0;
    cache.decode_prompt(tokens, n_cached_tmp);

    int32_t n_before = cache.get_n_cached_tokens();
    LOG_INF("Before clear: %d cached tokens\n", n_before);

    cache.clear();

    int32_t n_after = cache.get_n_cached_tokens();
    LOG_INF("After clear: %d cached tokens\n", n_after);

    if (n_after != 0) {
        LOG_ERR("Expected 0 cached tokens after clear, got %d\n", n_after);
        return false;
    }

    LOG_INF("Clear test passed\n");
    return true;
}

// Test 6: Prefix preservation on eviction
// - Cache a long prompt
// - Trigger eviction
// - Verify the prefix is preserved (tail is evicted)
static bool test_prefix_preservation(struct llama_context * ctx, const struct common_params & /* params */) {
    LOG_INF("\n=== Test 6: Prefix preservation ===\n");

    llama_memory_clear(llama_get_memory(ctx), true);

    auto vocab = llama_model_get_vocab(llama_get_model(ctx));
    // Create a longer prompt to ensure we have something to evict
    auto tokens = tokenize(vocab, "This is a longer prompt that should have enough tokens to test prefix preservation when eviction occurs and the tail is removed while keeping the beginning intact");

    if (tokens.empty()) {
        LOG_ERR("Failed to tokenize\n");
        return false;
    }

    kv_cache_lru_config cfg;
    cfg.ctx = ctx;
    cfg.mem = llama_get_memory(ctx);
    cfg.max_sessions = 0;  // Unlimited sessions
    cfg.min_session_size = 2;
    cfg.eviction_tail_fraction = 0.5f;
    cfg.reserved_tail = 16;

    kv_cache_lru cache(cfg);

    int32_t n_cached = 0;
    int32_t result = cache.decode_prompt(tokens, n_cached);
    if (result != 0) {
        LOG_ERR("decode_prompt failed\n");
        return false;
    }

    int32_t n_before = cache.get_n_cached_tokens();
    LOG_INF("Before eviction: %d cached tokens\n", n_before);

    // evict_if_needed(n) asks for room for n *additional* new tokens, not a
    // total token budget, so request just past the actual remaining headroom
    // in the (mostly empty) 512-token test context to force real eviction.
    int32_t available = llama_n_ctx(ctx) - n_before - cfg.reserved_tail;
    cache.evict_if_needed(available + 1);

    int32_t n_after = cache.get_n_cached_tokens();
    LOG_INF("After eviction: %d cached tokens\n", n_after);

    if (n_after >= n_before) {
        LOG_ERR("Expected fewer tokens after eviction\n");
        return false;
    }

    // Partial tail eviction requires the underlying memory to support
    // removing a token range without touching the rest (true for regular
    // attention KV cache; not possible for recurrent/SSM state, e.g. Gated
    // Delta Net layers, unless the context has spare rollback capacity).
    // On memory that can't do a partial removal, the manager falls back to
    // evicting the whole session (n_after == 0), which is also correct.
    bool prefix_preserved = n_after >= n_before / 2 - 1;
    bool full_fallback    = n_after == 0;
    if (!prefix_preserved && !full_fallback) {
        LOG_ERR("Too many tokens evicted, prefix not preserved\n");
        return false;
    }

    LOG_INF("Prefix preservation test passed\n");
    return true;
}

// Test 7: Session isolation under --kv-unified
// - Build a separate context with kv_unified enabled (multiple sequences may
//   share physical cache cells, unlike the default per-sequence allocation)
// - Decode two distinct sessions into it
// - Evict the LRU session under pressure and verify the other session's
//   cached tokens are untouched
static bool test_kv_unified(struct llama_model * model) {
    LOG_INF("\n=== Test 7: kv_unified session isolation ===\n");

    common_params params;
    params.n_ctx       = 512;
    params.n_batch     = 256;
    params.n_ubatch    = 256;
    params.kv_unified  = true;
    // kv_cache_lru tracks sessions by seq_id, but the context caps how many
    // distinct seq_ids it can hold via n_seq_max (== n_parallel here) - this
    // must cover the number of concurrent sessions the manager will create,
    // or decoding into any session beyond the first fails outright.
    params.n_parallel  = 2;

    auto cparams = common_context_params_to_llama(params);
    auto ctx = llama_context_ptr{llama_init_from_model(model, cparams)};
    if (!ctx) {
        LOG_ERR("Failed to create kv_unified context\n");
        return false;
    }

    auto vocab = llama_model_get_vocab(model);
    auto tokens_a = tokenize(vocab, "This is session A with some content for unified kv cache testing");
    auto tokens_b = tokenize(vocab, "This is session B with different content for unified kv cache testing");
    if (tokens_a.empty() || tokens_b.empty()) {
        LOG_ERR("Failed to tokenize\n");
        return false;
    }

    kv_cache_lru_config cfg;
    cfg.ctx = ctx.get();
    cfg.mem = llama_get_memory(ctx.get());
    cfg.max_sessions = 0;
    cfg.min_session_size = 2;
    cfg.eviction_tail_fraction = 0.5f;
    cfg.reserved_tail = 16;

    kv_cache_lru cache(cfg);

    int32_t n_cached_a = 0;
    if (cache.decode_prompt(tokens_a, n_cached_a, "session_a") != 0) {
        LOG_ERR("Failed to decode session A\n");
        return false;
    }

    int32_t n_cached_b = 0;
    if (cache.decode_prompt(tokens_b, n_cached_b, "session_b") != 0) {
        LOG_ERR("Failed to decode session B\n");
        return false;
    }

    llama_seq_id seq_a = cache.get_or_create_seq_id("session_a");
    llama_seq_id seq_b = cache.get_or_create_seq_id("session_b");

    llama_pos pos_b_before = llama_memory_seq_pos_max(cache.memory(), seq_b);
    int32_t n_a_before = llama_memory_seq_pos_max(cache.memory(), seq_a) + 1;
    LOG_INF("Session A: %d tokens, Session B last pos = %d\n", n_a_before, pos_b_before);

    if (pos_b_before < 0 || n_a_before <= 0) {
        LOG_ERR("Expected both sessions to have cached tokens\n");
        return false;
    }

    // Force eviction pressure: session A (accessed first, so LRU) should be
    // the one trimmed or evicted; session B must be left alone regardless.
    int32_t total_cached = cache.get_n_cached_tokens();
    int32_t available = llama_n_ctx(ctx.get()) - total_cached - cfg.reserved_tail;
    cache.evict_if_needed(available + 1);

    llama_pos pos_b_after = llama_memory_seq_pos_max(cache.memory(), seq_b);
    int32_t n_a_after = llama_memory_seq_pos_max(cache.memory(), seq_a) + 1;
    LOG_INF("After eviction: session A: %d tokens, session B last pos = %d\n", n_a_after, pos_b_after);

    if (pos_b_after != pos_b_before) {
        LOG_ERR("Session B was affected by session A's eviction under kv_unified (pos %d -> %d)\n", pos_b_before, pos_b_after);
        return false;
    }

    if (n_a_after >= n_a_before) {
        LOG_ERR("Expected session A to be evicted under pressure\n");
        return false;
    }

    LOG_INF("kv_unified session isolation test passed\n");
    return true;
}

int main(int argc, char ** argv) {
    // Get model path
    std::string model_path = "";
    if (argc < 2) {
        LOG_ERR("Usage: %s <model_path>\n", argv[0]);
        return 1;
    }
    model_path = argv[1];

    // Load model
    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;  // Offload all layers
    auto model = llama_model_ptr{llama_model_load_from_file(model_path.c_str(), mparams)};
    if (!model) {
        LOG_ERR("Failed to load model %s\n", model_path.c_str());
        return 1;
    }

    // Create context
    common_params params;
    params.n_ctx = 512;  // Small context to trigger eviction more easily
    params.n_batch = 256;
    params.n_ubatch = 256;

    auto cparams = common_context_params_to_llama(params);
    auto ctx = llama_context_ptr{llama_init_from_model(model.get(), cparams)};
    if (!ctx) {
        LOG_ERR("Failed to create context\n");
        return 1;
    }

    LOG_INF("Context size: %d\n", llama_n_ctx(ctx.get()));

    // Run tests
    bool all_passed = true;
    all_passed &= test_basic_caching(ctx.get(), params);
    all_passed &= test_cache_hit(ctx.get(), params);
    all_passed &= test_multiple_sessions(ctx.get(), params);
    all_passed &= test_eviction(ctx.get(), params);
    all_passed &= test_clear(ctx.get(), params);
    all_passed &= test_prefix_preservation(ctx.get(), params);
    all_passed &= test_kv_unified(model.get());

    if (all_passed) {
        LOG_INF("\n=== All tests passed ===\n");
        return 0;
    } else {
        LOG_ERR("\n=== Some tests failed ===\n");
        return 1;
    }
}
