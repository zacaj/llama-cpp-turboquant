#pragma once

#include "llama.h"

#include <chrono>
#include <map>
#include <string>
#include <vector>

//
// LRU-aware KV cache manager for multi-session prompt caching
//
// Instead of discarding all cached tokens when a new prompt doesn't match,
// this manager keeps old sessions in the cache and evicts them under pressure.
//
// Key design:
//   - Each session gets its own seq_id
//   - Sessions are tracked in LRU order (most recently accessed first)
//   - When cache pressure hits, we evict from the TAIL of the LRU session
//     (keeping the prefix, which is most likely to be reused)
//   - If a session is evicted entirely, it must be re-computed from scratch
//
// Usage pattern:
//   1. On new prompt: call `decode_prompt()` - it handles cache pressure
//   2. On continuation: call `decode_continuation()` - resumes from cached pos
//   3. Call `evict_if_needed()` before each decode to ensure space
//   4. Call `record_access()` after successful decode to update LRU order
//
// Prerequisite: `ctx` must be created with n_seq_max (n_parallel) large enough
// to cover the number of concurrent sessions this manager will track - each
// session gets a distinct seq_id, and llama_decode fails outright for any
// seq_id >= n_seq_max ("failed to find a memory slot").

struct kv_cache_lru_config {
    llama_context * ctx;
    llama_memory_t  mem;

    // Maximum number of sessions to keep in cache before eviction
    // (0 = unlimited, cache may fill up and fail)
    int32_t max_sessions = 0;

    // Minimum tokens to keep from a session before evicting the whole thing
    // (prevents pointless fragments that are cheaper to recompute)
    int32_t min_session_size = 4;

    // When evicting, what fraction of the tail to remove (0.0 - 1.0)
    // 0.5 means remove the last half of the LRU session
    float eviction_tail_fraction = 0.5f;

    // Number of tokens to reserve at the end of context for generation
    int32_t reserved_tail = 64;
};

// Metadata about a cached session
struct cached_session {
    llama_seq_id seq_id;
    llama_pos    pos_min;
    llama_pos    pos_max;
    int32_t      n_tokens;  // total tokens in this session (including evicted)
    std::chrono::steady_clock::time_point last_accessed;
};

// LRU KV cache manager
// Maintains sessions in LRU order and handles eviction under cache pressure
class kv_cache_lru {
public:
    explicit kv_cache_lru(kv_cache_lru_config config);

    // Decode a new prompt, handling cache pressure automatically.
    // Returns 0 on success, -1 on failure.
    // `tokens` is the tokenized prompt.
    // `out_n_cached` receives how many tokens were already cached for this
    // session before this call. All of `tokens` is still decoded regardless
    // (prefix reuse against `out_n_cached` is not yet implemented).
    // `session_key` identifies which session to decode into.
    int32_t decode_prompt(const std::vector<llama_token> & tokens, int32_t & out_n_cached, const std::string & session_key = "default");

    // Decode a continuation of the most recent prompt.
    // If the last accessed session is still partially cached, resume from there.
    // Returns the position from which decoding should continue.
    llama_pos decode_continuation(const std::vector<llama_token> & new_tokens);

    // Try to evict enough cache to fit `n_tokens` new tokens.
    // Returns true if enough space was freed (or was already available).
    bool evict_if_needed(int32_t n_tokens);

    // Record that a session was just accessed (updates LRU timestamp).
    void record_access(llama_seq_id seq_id);

    // Get the seq_id for a session key (creates a new one if not found).
    llama_seq_id get_or_create_seq_id(const std::string & session_key);

    // Get the current number of cached tokens.
    int32_t get_n_cached_tokens() const;

    // Get the number of active sessions.
    int32_t get_n_sessions() const;

    // Clear all cached sessions.
    void clear();

    // Get the memory object (for direct access if needed).
    llama_memory_t memory() const { return config.mem; }

    // Get the context (for direct access if needed).
    llama_context * context() const { return config.ctx; }

private:
    kv_cache_lru_config config;

    // Next seq_id to assign (monotonically increasing)
    llama_seq_id next_seq_id;

    // Session key -> metadata, ordered by LRU access time
    std::map<std::string, cached_session> sessions;

    // Helper: find the LRU session (oldest last_accessed)
    cached_session * find_lru_session();

    // Helper: evict tail of a session
    void evict_session_tail(cached_session & session, int32_t n_tokens_to_free);

    // Helper: evict an entire session
    void evict_session(cached_session & session);

    // Helper: get total cached tokens across all sessions
    int32_t total_cached_tokens() const;

    // Helper: get context size
    int32_t ctx_size() const;

    // Helper: check how many tokens are available
    int32_t available_slots() const;
};
