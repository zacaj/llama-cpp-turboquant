#include "kv-cache-lru.h"

#include "log.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <set>
#include <vector>

kv_cache_lru::kv_cache_lru(kv_cache_lru_config cfg)
    : config(cfg)
    , next_seq_id(0)
{
    // Ensure we have a memory object
    if (!config.mem) {
        config.mem = llama_get_memory(config.ctx);
    }
    assert(config.mem != nullptr);
}

llama_seq_id kv_cache_lru::get_or_create_seq_id(const std::string & session_key) {
    auto it = sessions.find(session_key);
    if (it != sessions.end()) {
        return it->second.seq_id;
    }
    // Create new session
    llama_seq_id seq_id = next_seq_id++;
    sessions[session_key] = cached_session{
        .seq_id = seq_id,
        .pos_min = 0,
        .pos_max = -1,
        .n_tokens = 0,
        .last_accessed = std::chrono::steady_clock::now()
    };
    return seq_id;
}

int32_t kv_cache_lru::total_cached_tokens() const {
    int32_t total = 0;
    for (const auto & [key, session] : sessions) {
        if (session.pos_max >= 0) {
            total += (session.pos_max - session.pos_min + 1);
        }
    }
    return total;
}

int32_t kv_cache_lru::ctx_size() const {
    return static_cast<int32_t>(llama_n_ctx(config.ctx));
}

int32_t kv_cache_lru::available_slots() const {
    return ctx_size() - total_cached_tokens() - config.reserved_tail;
}

int32_t kv_cache_lru::get_n_cached_tokens() const {
    return total_cached_tokens();
}

int32_t kv_cache_lru::get_n_sessions() const {
    int32_t count = 0;
    for (const auto & [key, session] : sessions) {
        if (session.pos_max >= 0) {
            count++;
        }
    }
    return count;
}

void kv_cache_lru::clear() {
    for (auto & [key, session] : sessions) {
        if (session.pos_max >= 0) {
            llama_memory_seq_rm(config.mem, session.seq_id, -1, -1);
        }
    }
    sessions.clear();
    next_seq_id = 0;
}

bool kv_cache_lru::evict_if_needed(int32_t n_tokens) {
    int32_t needed = n_tokens - available_slots();
    if (needed <= 0) {
        return true;  // Already enough space
    }

    LOG_INF("%s: need %d more slots, evicting under cache pressure\n", __func__, needed);

    while (needed > 0 && !sessions.empty()) {
        cached_session * lru = find_lru_session();
        if (!lru) break;

        // If session is tiny, just remove it entirely
        int32_t session_size = lru->pos_max - lru->pos_min + 1;
        if (session_size <= config.min_session_size) {
            evict_session(*lru);
            needed -= session_size;
        } else {
            // Evict tail of the session
            int32_t n_to_remove = std::max(static_cast<int32_t>(std::ceil(session_size * config.eviction_tail_fraction)), config.min_session_size);
            evict_session_tail(*lru, n_to_remove);

            int32_t session_size_after = (lru->pos_max >= 0) ? (lru->pos_max - lru->pos_min + 1) : 0;
            int32_t freed = session_size - session_size_after;
            if (freed <= 0) {
                // Partial removal isn't supported for this session's memory type
                // (e.g. recurrent/SSM state without rollback capacity) - fall back
                // to evicting it entirely so eviction always makes progress.
                evict_session(*lru);
                freed = session_size;
            }
            needed -= freed;
        }
    }

    return needed <= 0;
}

cached_session * kv_cache_lru::find_lru_session() {
    cached_session * lru = nullptr;
    auto latest_time = std::chrono::steady_clock::time_point::max();

    for (auto & [key, session] : sessions) {
        if (session.pos_max >= 0 && session.last_accessed < latest_time) {
            latest_time = session.last_accessed;
            lru = &session;
        }
    }
    return lru;
}

void kv_cache_lru::evict_session_tail(cached_session & session, int32_t n_tokens_to_free) {
    // Remove from the tail (latest positions), keeping the prefix
    llama_pos p0 = session.pos_max - n_tokens_to_free + 1;
    if (p0 < session.pos_min) p0 = session.pos_min;

    LOG_INF("%s: evicting tail of session %d, positions [%d, %d)\n",
            __func__, session.seq_id, p0, session.pos_max);

    llama_memory_seq_rm(config.mem, session.seq_id, p0, -1);

    // Update metadata
    session.pos_max = llama_memory_seq_pos_max(config.mem, session.seq_id);
    if (session.pos_max < 0) {
        session.pos_min = 0;
    }
}

void kv_cache_lru::evict_session(cached_session & session) {
    LOG_INF("%s: evicting entire session %d\n", __func__, session.seq_id);
    llama_memory_seq_rm(config.mem, session.seq_id, -1, -1);
    session.pos_min = 0;
    session.pos_max = -1;
    session.n_tokens = 0;
}

void kv_cache_lru::record_access(llama_seq_id seq_id) {
    for (auto & [key, session] : sessions) {
        if (session.seq_id == seq_id) {
            session.last_accessed = std::chrono::steady_clock::now();
            break;
        }
    }
}

int32_t kv_cache_lru::decode_prompt(
    const std::vector<llama_token> & tokens,
    int32_t & out_n_cached,
    const std::string & session_key)
{
    if (tokens.empty()) {
        out_n_cached = 0;
        return 0;
    }

    llama_seq_id seq_id = get_or_create_seq_id(session_key);

    // Check how much is already cached for this session
    cached_session & session = sessions[session_key];
    llama_pos n_cached = (session.pos_max >= 0) ? (session.pos_max - session.pos_min + 1) : 0;

    // If the session is empty or we need to start fresh, evict if needed
    if (n_cached == 0 && !sessions.empty()) {
        // New session, might need to evict old ones
        if (!evict_if_needed(static_cast<int32_t>(tokens.size()))) {
            LOG_ERR("%s: failed to evict enough cache for %d tokens\n", __func__, static_cast<int32_t>(tokens.size()));
            out_n_cached = 0;
            return -1;
        }
    }

    // Decode the prompt tokens
    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    int32_t start_pos = session.pos_max + 1;

    for (int32_t i = 0; i < static_cast<int32_t>(tokens.size()); i++) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = start_pos + i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = seq_id;
        batch.logits[i]    = (i == static_cast<int32_t>(tokens.size()) - 1);  // logits only for last
    }
    batch.n_tokens = static_cast<int32_t>(tokens.size());

    int32_t result = llama_decode(config.ctx, batch);
    llama_batch_free(batch);

    if (result != 0) {
        LOG_ERR("%s: llama_decode failed with code %d\n", __func__, result);
        out_n_cached = 0;
        return -1;
    }

    // Update session metadata
    session.pos_max = llama_memory_seq_pos_max(config.mem, seq_id);
    session.n_tokens = session.pos_max + 1;
    record_access(seq_id);

    out_n_cached = n_cached;
    return 0;
}

llama_pos kv_cache_lru::decode_continuation(
    const std::vector<llama_token> & new_tokens)
{
    // Find the most recently accessed session
    cached_session * most_recent = nullptr;
    auto latest_time = std::chrono::steady_clock::time_point::min();
    for (auto & [key, session] : sessions) {
        if (session.pos_max >= 0 && session.last_accessed > latest_time) {
            latest_time = session.last_accessed;
            most_recent = &session;
        }
    }

    if (!most_recent) {
        return 0;  // No sessions to continue
    }

    llama_pos start_pos = most_recent->pos_max + 1;

    // Check if we have space
    if (!evict_if_needed(static_cast<int32_t>(new_tokens.size()))) {
        LOG_ERR("%s: failed to evict enough cache for continuation\n", __func__);
        return -1;
    }

    // Decode continuation
    llama_batch batch = llama_batch_init(new_tokens.size(), 0, 1);
    for (int32_t i = 0; i < static_cast<int32_t>(new_tokens.size()); i++) {
        batch.token[i]     = new_tokens[i];
        batch.pos[i]       = start_pos + i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = most_recent->seq_id;
        batch.logits[i]    = (i == static_cast<int32_t>(new_tokens.size()) - 1);
    }
    batch.n_tokens = static_cast<int32_t>(new_tokens.size());

    int32_t result = llama_decode(config.ctx, batch);
    llama_batch_free(batch);

    if (result != 0) {
        LOG_ERR("%s: llama_decode failed with code %d\n", __func__, result);
        return -1;
    }

    // Update session metadata
    most_recent->pos_max = llama_memory_seq_pos_max(config.mem, most_recent->seq_id);
    most_recent->n_tokens = most_recent->pos_max + 1;
    record_access(most_recent->seq_id);

    return start_pos;
}
