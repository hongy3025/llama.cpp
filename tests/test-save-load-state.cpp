#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama-cpp.h"

#include <clocale>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>

struct llama_batch_ptr {
    llama_batch batch;

    llama_batch_ptr(int32_t n_tokens, int32_t embd, int32_t n_seq_max)
        : batch{llama_batch_init(n_tokens, embd, n_seq_max)} {}

    ~llama_batch_ptr() { llama_batch_free(batch); }

    llama_batch_ptr(const llama_batch_ptr &) = delete;
    llama_batch_ptr & operator=(const llama_batch_ptr &) = delete;
    llama_batch_ptr(llama_batch_ptr &&) = default;
    llama_batch_ptr & operator=(llama_batch_ptr &&) = default;

    llama_batch & get() { return batch; }
    const llama_batch & get() const { return batch; }
};

// GGSD incremental state tests (see docs/superpowers/specs/2026-07-31-ggsd-incremental-slot-storage-design.md)

static const char * const k_incr_dir = "incr_test";

static std::string session_path(const std::string & name) {
    return std::string(k_incr_dir) + "/" + name;
}

static void incr_test_cleanup() {
    std::filesystem::remove_all(k_incr_dir);
    std::filesystem::create_directories(k_incr_dir);
}

static llama_tokens generate_tokens(llama_context * ctx, llama_sampler * smpl, int & n_past, int32_t n_predict, llama_seq_id seq_id) {
    llama_tokens result;
    llama_batch_ptr batch(1, 0, 1);

    for (int i = 0; i < n_predict; i++) {
        auto next_token = llama_sampler_sample(smpl, ctx, -1);

        LOG("%d ", next_token);
        result.push_back(next_token);

        common_batch_clear(batch.get());
        common_batch_add(batch.get(), next_token, n_past, {seq_id}, true);

        if (llama_decode(ctx, batch.get())) {
            LOG_ERR("\n%s: failed to evaluate\n", __func__);
            return {};
        }
        n_past++;
    }

    return result;
}

static llama_tokens make_random_tokens(struct llama_model * model, size_t n, uint32_t seed) {
    const auto * vocab = llama_model_get_vocab(model);
    const auto n_vocab = llama_vocab_n_tokens(vocab);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<llama_token> dist(0, n_vocab - 1);

    llama_tokens tokens;
    tokens.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        tokens.push_back(dist(rng));
    }

    return tokens;
}

static bool decode_tokens(llama_context * ctx, const llama_tokens & tokens, size_t i_begin, size_t i_end, llama_seq_id seq_id) {
    llama_batch_ptr batch(512, 0, 1);

    for (size_t i = i_begin; i < i_end; ) {
        const size_t n = std::min<size_t>(512, i_end - i);

        common_batch_clear(batch.get());
        for (size_t j = 0; j < n; ++j) {
            common_batch_add(batch.get(), tokens[i + j], (llama_pos) (i + j), {seq_id}, true);
        }

        if (llama_decode(ctx, batch.get())) {
            LOG_ERR("%s: failed to decode tokens [%zu, %zu)\n", __func__, i, i + n);
            return false;
        }

        i += n;
    }

    return true;
}

static size_t count_segment_files() {
    size_t n = 0;
    for (const auto & entry : std::filesystem::directory_iterator(k_incr_dir)) {
        const std::string name = entry.path().filename().string();
        if (name.rfind("seg_", 0) == 0) {
            ++n;
        }
    }
    return n;
}

static std::string segment_file_by_index(uint32_t idx) {
    for (const auto & entry : std::filesystem::directory_iterator(k_incr_dir)) {
        const std::string name = entry.path().filename().string();
        if (name.rfind("seg_", 0) != 0) {
            continue;
        }

        std::ifstream f(entry.path(), std::ios::binary);
        char     magic[4];
        uint32_t version;
        uint32_t seg_index;
        f.read(magic, 4);
        f.read((char *) &version, sizeof(version));
        f.read((char *) &seg_index, sizeof(seg_index));
        if (seg_index == idx) {
            return entry.path().string();
        }
    }
    return "";
}

static bool compare_generation(
        struct llama_model * model,
        const struct common_params & params,
        llama_context * ctx,
        int n_past,
        const llama_tokens & expected_result) {
    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    int n_past_local = n_past;
    auto result = generate_tokens(ctx, smpl.get(), n_past_local, params.n_predict, 0);
    if (result.empty()) {
        return false;
    }

    if (result != expected_result) {
        LOG_ERR("\n%s: error: generation differs from expected\n", __func__);
        return false;
    }

    LOG("\nPASS\n");
    return true;
}

// Test 1: baseline
// - decode all but the last token
// - save state to disk
// - decode the last token
// - generate n_predict tokens
static llama_tokens test_baseline(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens) {
    auto ctx = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    auto n_past = 0;
    if (!common_prompt_batch_decode(ctx.get(), tokens, (int)tokens.size(), n_past, params.n_batch, params.out_file, true)) {
        LOG_ERR("%s: failed to decode prompt\n", __func__);
        return {};
    }

    LOG("\n=== Test 1: baseline ===\n");

    auto result = generate_tokens(ctx.get(), smpl.get(), n_past, params.n_predict, 0);
    if (result.empty()) {
        return {};
    }

    LOG("\n");

    return result;
}


// Test 2: sequence removal isolation
// - decode the same prefix into two sequences
// - remove sequence 0
// - verify that sequence 1 remains unchanged
static bool test_seq_rm_isolated(
        struct llama_model         * model,
        const struct common_params & params,
        const llama_tokens         & tokens) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_ctx      = 256;
    params_ctx.n_seq_max  = 2;
    params_ctx.kv_unified = true;

    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};
    if (!ctx) {
        LOG_ERR("%s: failed to create context\n", __func__);
        return false;
    }

    LOG("\n=== Test 2: sequence removal isolation ===\n");

    const size_t n_tokens = tokens.size() < 128 ? tokens.size() : 128;
    for (llama_seq_id seq_id = 0; seq_id < 2; ++seq_id) {
        llama_batch_ptr batch(n_tokens, 0, 1);
        for (size_t i = 0; i < n_tokens; ++i) {
            common_batch_add(batch.get(), tokens[i], i, { seq_id }, false);
        }

        if (llama_decode(ctx.get(), batch.get())) {
            LOG_ERR("%s: failed to decode prompt for sequence %d\n", __func__, seq_id);
            return false;
        }
    }

    const auto get_seq_state = [&](llama_seq_id seq_id, std::vector<uint8_t> & state) {
        const size_t state_size = llama_state_seq_get_size(ctx.get(), seq_id);
        if (state_size == 0) {
            LOG_ERR("%s: sequence state is empty\n", __func__);
            return false;
        }

        state.resize(state_size);
        const size_t ncopy = llama_state_seq_get_data(ctx.get(), state.data(), state.size(), seq_id);
        if (ncopy != state.size()) {
            LOG_ERR("%s: sequence state length %zu does not match expected length %zu\n",
                    __func__, ncopy, state.size());
            return false;
        }

        return true;
    };

    std::vector<uint8_t> state_before;
    if (!get_seq_state(1, state_before)) {
        return false;
    }

    if (!llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, -1, -1)) {
        LOG_ERR("%s: failed to remove sequence 0\n", __func__);
        return false;
    }

    std::vector<uint8_t> state_after;
    if (!get_seq_state(1, state_after)) {
        return false;
    }

    if (state_before != state_after) {
        LOG_ERR("%s: removing sequence 0 changed sequence 1\n", __func__);
        return false;
    }

    LOG("PASS\n");
    return true;
}


// Test 3: state load
// - create a new context
// - load state from file
// - replay the last prompt token
// - generate n_predict tokens and compare against expected result
static bool test_state_load(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens, const llama_tokens & expected_result) {
    auto ctx = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    LOG("\n=== Test 3: state load ===\n");

    // Load state from file
    llama_tokens unused_sts(tokens.size());
    size_t n_token_count_out = 0;

    if (!llama_state_load_file(ctx.get(), params.out_file.data(), unused_sts.data(), unused_sts.size(), &n_token_count_out)) {
        LOG_ERR("\n%s: failed to load state\n", __func__);
        return false;
    }

    LOG_TRC("%s: loaded state with %zu tokens\n", __func__, n_token_count_out);

    // Replay last token
    int n_past = (int) n_token_count_out - 1;
    if (!common_replay_last_token(ctx.get(), tokens.back(), n_past)) {
        return false;
    }
    n_past++;

    // Generate tokens
    auto result = generate_tokens(ctx.get(), smpl.get(), n_past, params.n_predict, 0);
    if (result.empty()) {
        return false;
    }

    if (result != expected_result) {
        LOG_ERR("\n%s: error: generation differs from expected\n", __func__);
        return false;
    }

    LOG("\nPASS\n");
    return true;
}


// Test 4: seq copy (host)
// - create a multi-seq context
// - load state from file
// - replay the last prompt token
// - migrate KV cache from seq 0 to seq 1 via the CPU path
// - generate n_predict tokens on seq 1 and compare against expected result
static bool test_seq_cp_host(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens, const llama_tokens & expected_result) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_seq_max = 2;
    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    LOG("\n=== Test 4: seq copy (host) ===\n");

    // Load state from file
    llama_tokens unused_sts(tokens.size());
    size_t n_token_count_out = 0;

    if (!llama_state_load_file(ctx.get(), params.out_file.data(), unused_sts.data(), unused_sts.size(), &n_token_count_out)) {
        LOG_ERR("\n%s: failed to load state\n", __func__);
        return false;
    }

    LOG_TRC("%s: loaded state with %zu tokens\n", __func__, n_token_count_out);

    // Replay last token
    int n_past = (int) n_token_count_out - 1;
    if (!common_replay_last_token(ctx.get(), tokens.back(), n_past)) {
        return false;
    }
    n_past++;

    // Migrate KV cache from seq 0 to seq 1 (CPU path)
    {
        std::vector<uint8_t> seq_store(llama_state_seq_get_size(ctx.get(), 0));
        const size_t ncopy = llama_state_seq_get_data(ctx.get(), seq_store.data(), seq_store.size(), 0);
        if (ncopy != seq_store.size()) {
            LOG_ERR("\n%s: seq copy data length %zd does not match expected length %zd\n", __func__, ncopy, seq_store.size());
            return false;
        }
        LOG_TRC("%s: seq 0 copied, %zd bytes\n", __func__, ncopy);

        llama_memory_clear(llama_get_memory(ctx.get()), true);
        LOG_TRC("%s: kv cache cleared\n", __func__);

        const size_t nset = llama_state_seq_set_data(ctx.get(), seq_store.data(), seq_store.size(), 1);
        if (nset != seq_store.size()) {
            LOG_ERR("\n%s: seq set data length %zd does not match expected length %zd\n", __func__, nset, seq_store.size());
            return false;
        }
        LOG_TRC("%s: seq 1 restored, %zd bytes\n", __func__, nset);
    }

    // Generate tokens on seq 1
    auto result = generate_tokens(ctx.get(), smpl.get(), n_past, params.n_predict, 1);
    if (result.empty()) {
        return false;
    }

    if (result != expected_result) {
        LOG_ERR("\n%s: error: generation differs from expected\n", __func__);
        return false;
    }

    LOG("\nPASS\n");
    return true;
}


// Test 5: seq copy (device)
// - create a multi-seq context
// - load state from file
// - replay the last prompt token
// - migrate KV cache from seq 0 to seq 1 via the on-device path
// - generate n_predict tokens on seq 1 and compare against expected result
static bool test_seq_cp_device(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens, const llama_tokens & expected_result) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_seq_max = 2;
    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    LOG("\n=== Test 5: seq copy (device) ===\n");

    // Load state from file
    llama_tokens unused_sts(tokens.size());
    size_t n_token_count_out = 0;

    if (!llama_state_load_file(ctx.get(), params.out_file.data(), unused_sts.data(), unused_sts.size(), &n_token_count_out)) {
        LOG_ERR("\n%s: failed to load state\n", __func__);
        return false;
    }

    LOG_TRC("%s: loaded state with %zu tokens\n", __func__, n_token_count_out);

    // Replay last token
    int n_past = (int) n_token_count_out - 1;
    if (!common_replay_last_token(ctx.get(), tokens.back(), n_past)) {
        return false;
    }
    n_past++;

    // Migrate KV cache from seq 0 to seq 1 (on-device path)
    {
        std::vector<uint8_t> seq_store(llama_state_seq_get_size_ext(ctx.get(), 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE));
        const size_t ncopy = llama_state_seq_get_data_ext(ctx.get(), seq_store.data(), seq_store.size(), 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
        if (ncopy != seq_store.size()) {
            LOG_ERR("\n%s: seq copy data length %zd does not match expected length %zd\n", __func__, ncopy, seq_store.size());
            return false;
        }
        LOG_TRC("%s: seq 0 copied, %zd bytes\n", __func__, ncopy);

        llama_memory_clear(llama_get_memory(ctx.get()), true);
        LOG_TRC("%s: kv cache cleared\n", __func__);

        const size_t nset = llama_state_seq_set_data_ext(ctx.get(), seq_store.data(), seq_store.size(), 1, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
        if (nset != seq_store.size()) {
            LOG_ERR("\n%s: seq set data length %zd does not match expected length %zd\n", __func__, nset, seq_store.size());
            return false;
        }
        LOG_TRC("%s: seq 1 restored, %zd bytes\n", __func__, nset);
    }

    // Generate tokens on seq 1
    auto result = generate_tokens(ctx.get(), smpl.get(), n_past, params.n_predict, 1);
    if (result.empty()) {
        return false;
    }

    if (result != expected_result) {
        LOG_ERR("\n%s: error: generation differs from expected\n", __func__);
        return false;
    }

    LOG("\nPASS\n");
    return true;
}

// Test 5: GGSD save/load roundtrip
// - decode 2500 tokens, save (2 segments)
// - restore 2048 tokens in a fresh context, decode the remainder, generate
// - compare the generation against the reference (same KV -> same logits)
static bool test_incr_roundtrip(struct llama_model * model, const struct common_params & params) {
    incr_test_cleanup();

    const std::string session = session_path("session_roundtrip.bin");
    const llama_tokens tokens = make_random_tokens(model, 2500, 1234);

    auto ctx = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};
    if (!decode_tokens(ctx.get(), tokens, 0, tokens.size(), 0)) {
        return false;
    }

    const int32_t n_segments = llama_state_seq_save_incr(ctx.get(), session.c_str(), 0, tokens.data(), tokens.size());
    if (n_segments != 2) {
        LOG_ERR("\n%s: error: expected 2 segments after first save, got %d\n", __func__, n_segments);
        return false;
    }

    // reference generation
    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    int n_past_ref = (int) tokens.size();
    const llama_tokens expected_result = generate_tokens(ctx.get(), smpl.get(), n_past_ref, params.n_predict, 0);
    if (expected_result.empty()) {
        return false;
    }

    // fresh context: restore the aligned prefix, decode the remainder
    auto ctx2 = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};

    const size_t n_restored = llama_state_seq_load_incr(ctx2.get(), session.c_str(), 0, tokens.data(), tokens.size(), 64);
    if (n_restored != 2048) {
        LOG_ERR("\n%s: error: expected 2048 tokens restored, got %zu\n", __func__, n_restored);
        return false;
    }

    if (!decode_tokens(ctx2.get(), tokens, n_restored, tokens.size(), 0)) {
        return false;
    }

    return compare_generation(model, params, ctx2.get(), (int) tokens.size(), expected_result);
}

// Test 6: GGSD incremental save
// - decode 1500 tokens, save -> 1 segment
// - decode 1100 more, save -> 2 segments, only 1 new file
// - restore -> 2048 tokens
static bool test_incr_append(struct llama_model * model, const struct common_params & params) {
    incr_test_cleanup();

    const std::string session = session_path("session_append.bin");
    const llama_tokens tokens = make_random_tokens(model, 2600, 4321);

    auto ctx = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};
    if (!decode_tokens(ctx.get(), tokens, 0, 1500, 0)) {
        return false;
    }

    int32_t n_segments = llama_state_seq_save_incr(ctx.get(), session.c_str(), 0, tokens.data(), 1500);
    if (n_segments != 1) {
        LOG_ERR("\n%s: error: expected 1 segment after first save, got %d\n", __func__, n_segments);
        return false;
    }
    if (count_segment_files() != 1) {
        LOG_ERR("\n%s: error: expected 1 segment file, got %zu\n", __func__, count_segment_files());
        return false;
    }

    if (!decode_tokens(ctx.get(), tokens, 1500, 2600, 0)) {
        return false;
    }

    n_segments = llama_state_seq_save_incr(ctx.get(), session.c_str(), 0, tokens.data(), 2600);
    if (n_segments != 2) {
        LOG_ERR("\n%s: error: expected 2 segments after append save, got %d\n", __func__, n_segments);
        return false;
    }
    if (count_segment_files() != 2) {
        LOG_ERR("\n%s: error: expected 2 segment files after append, got %zu\n", __func__, count_segment_files());
        return false;
    }

    auto ctx2 = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};

    const size_t n_restored = llama_state_seq_load_incr(ctx2.get(), session.c_str(), 0, tokens.data(), tokens.size(), 64);
    if (n_restored != 2048) {
        LOG_ERR("\n%s: error: expected 2048 tokens restored, got %zu\n", __func__, n_restored);
        return false;
    }

    LOG("\nPASS\n");
    return true;
}

// Test 7: GGSD fork
// - chain A: 3000 tokens, save -> 2 segments
// - chain B: A[0..1500) + 1500 new tokens, save -> 2 segments, 1 shared file
// - restore with B -> 2048 tokens (matched chain)
// - restore with A -> 2048 tokens (orphaned chain still on disk)
static bool test_incr_fork(struct llama_model * model, const struct common_params & params) {
    incr_test_cleanup();

    const std::string session = session_path("session_fork.bin");
    const llama_tokens tokens_a = make_random_tokens(model, 3000, 111);
    const llama_tokens tokens_b_tail = make_random_tokens(model, 1500, 222);

    llama_tokens tokens_b = tokens_a;
    tokens_b.resize(1500);
    tokens_b.insert(tokens_b.end(), tokens_b_tail.begin(), tokens_b_tail.end());

    auto ctx_a = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};
    if (!decode_tokens(ctx_a.get(), tokens_a, 0, tokens_a.size(), 0)) {
        return false;
    }
    const int32_t n_segments_a = llama_state_seq_save_incr(ctx_a.get(), session.c_str(), 0, tokens_a.data(), tokens_a.size());
    if (n_segments_a != 2 || count_segment_files() != 2) {
        LOG_ERR("\n%s: error: chain A expected 2 segments/2 files, got %d/%zu\n", __func__, n_segments_a, count_segment_files());
        return false;
    }

    auto ctx_b = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};
    if (!decode_tokens(ctx_b.get(), tokens_b, 0, tokens_b.size(), 0)) {
        return false;
    }
    const int32_t n_segments_b = llama_state_seq_save_incr(ctx_b.get(), session.c_str(), 0, tokens_b.data(), tokens_b.size());
    if (n_segments_b != 2) {
        LOG_ERR("\n%s: error: chain B expected 2 segments, got %d\n", __func__, n_segments_b);
        return false;
    }
    if (count_segment_files() != 3) {
        LOG_ERR("\n%s: error: expected 3 segment files after fork (1 shared + 2 unique), got %zu\n", __func__, count_segment_files());
        return false;
    }

    // restore chain B (the session tail)
    auto ctx_rb = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};
    const size_t n_restored_b = llama_state_seq_load_incr(ctx_rb.get(), session.c_str(), 0, tokens_b.data(), tokens_b.size(), 64);
    if (n_restored_b != 2048) {
        LOG_ERR("\n%s: error: chain B restore expected 2048 tokens, got %zu\n", __func__, n_restored_b);
        return false;
    }

    // restore chain A (orphaned but still on disk)
    auto ctx_ra = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};
    const size_t n_restored_a = llama_state_seq_load_incr(ctx_ra.get(), session.c_str(), 0, tokens_a.data(), tokens_a.size(), 64);
    if (n_restored_a != 2048) {
        LOG_ERR("\n%s: error: chain A restore expected 2048 tokens, got %zu\n", __func__, n_restored_a);
        return false;
    }

    LOG("\nPASS\n");
    return true;
}

// Test 8: GGSD partial KV
// - decode 3000 tokens, save -> 2 segments
// - evict cells [1024, 3000), save -> chain is preserved (files on disk stay
//   valid regardless of KV eviction; R2) and nothing new is written
// - restore with the full prompt -> 2048 tokens (both segments are on disk)
// - evict the head cells [0, 1024), save to the original session -> still 2
//   (zero writes; the head file on disk keeps the chain alive, R2)
// - evict part of the head cells [512, 1024) in a fresh context, save to a
//   fresh session -> error (-1): the head segment can neither be loaded from
//   disk (no file) nor written (incomplete cells) - the R3 head-missing rule
static bool test_incr_partial(struct llama_model * model, const struct common_params & params) {
    incr_test_cleanup();

    const std::string session = session_path("session_partial.bin");
    const llama_tokens tokens = make_random_tokens(model, 3000, 555);

    auto ctx = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};
    if (!decode_tokens(ctx.get(), tokens, 0, tokens.size(), 0)) {
        return false;
    }

    int32_t n_segments = llama_state_seq_save_incr(ctx.get(), session.c_str(), 0, tokens.data(), tokens.size());
    if (n_segments != 2) {
        LOG_ERR("\n%s: error: expected 2 segments, got %d\n", __func__, n_segments);
        return false;
    }

    // evict the tail of the KV cache
    llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, 1024, 3000);

    n_segments = llama_state_seq_save_incr(ctx.get(), session.c_str(), 0, tokens.data(), tokens.size());
    if (n_segments != 2) {
        LOG_ERR("\n%s: error: expected chain to be preserved after eviction, got %d\n", __func__, n_segments);
        return false;
    }

    auto ctx2 = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};
    const size_t n_restored = llama_state_seq_load_incr(ctx2.get(), session.c_str(), 0, tokens.data(), tokens.size(), 64);
    if (n_restored != 2048) {
        LOG_ERR("\n%s: error: expected 2048 tokens restored, got %zu\n", __func__, n_restored);
        return false;
    }
    // evict the whole head: the chain on disk is untouched, save is a no-op (R2)
    llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, 0, 1024);

    n_segments = llama_state_seq_save_incr(ctx.get(), session.c_str(), 0, tokens.data(), tokens.size());
    if (n_segments != 2) {
        LOG_ERR("\n%s: error: expected chain to be preserved after head eviction, got %d\n", __func__, n_segments);
        return false;
    }

    // R3 head-missing rule: head segment present in the KV cache only
    // partially (pos base still 0), no file on disk, cannot be written
    auto ctx3 = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};
    if (!decode_tokens(ctx3.get(), tokens, 0, tokens.size(), 0)) {
        return false;
    }
    llama_memory_seq_rm(llama_get_memory(ctx3.get()), 0, 512, 1024);

    const std::string session_fresh = session_path("session_partial_fresh.bin");
    const int32_t n_segments_err = llama_state_seq_save_incr(ctx3.get(), session_fresh.c_str(), 0, tokens.data(), tokens.size());
    if (n_segments_err != -1) {
        LOG_ERR("\n%s: error: expected save to fail with head missing, got %d\n", __func__, n_segments_err);
        return false;
    }

    LOG("\nPASS\n");
    return true;
}

// Test 9: GGSD corrupt segment
// - decode 2500 tokens, save -> 2 segments
// - flip a byte in the payload of segment 1
// - corrupt session file -> restore is unaffected (pool semantics, R1)
static bool test_incr_corrupt(struct llama_model * model, const struct common_params & params) {
    incr_test_cleanup();

    const std::string session = session_path("session_corrupt.bin");
    const llama_tokens tokens = make_random_tokens(model, 2500, 777);

    auto ctx = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};
    if (!decode_tokens(ctx.get(), tokens, 0, tokens.size(), 0)) {
        return false;
    }
    if (llama_state_seq_save_incr(ctx.get(), session.c_str(), 0, tokens.data(), tokens.size()) != 2) {
        return false;
    }

    // flip the last byte of segment 1 (inside the payload)
    {
        const std::string fname = segment_file_by_index(1);
        if (fname.empty()) {
            LOG_ERR("\n%s: error: segment 1 file not found\n", __func__);
            return false;
        }

        std::fstream f(fname, std::ios::binary | std::ios::in | std::ios::out);
        f.seekg(-1, std::ios::end);
        char c;
        f.read(&c, 1);
        c ^= 0xFF;
        f.seekp(-1, std::ios::end);
        f.write(&c, 1);
    }

    auto ctx2 = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};
    const size_t n_restored = llama_state_seq_load_incr(ctx2.get(), session.c_str(), 0, tokens.data(), tokens.size(), 64);
    if (n_restored != 1024) {
        LOG_ERR("\n%s: error: expected 1024 tokens restored (stop at corrupt segment), got %zu\n", __func__, n_restored);
        return false;
    }

    // corrupt session file: restore is a no-op
    {
        std::ofstream f(session, std::ios::binary);
        f << "garbage";
    }

    const size_t n_restored2 = llama_state_seq_load_incr(ctx2.get(), session.c_str(), 0, tokens.data(), tokens.size(), 64);
    if (n_restored2 != 1024) {
        LOG_ERR("\n%s: error: expected 1024 tokens restored despite corrupt session (pool semantics), got %zu\n", __func__, n_restored2);
        return false;
    }

    LOG("\nPASS\n");
    return true;
}

// Test 10: GGSD kv params mismatch and min_prefix
// - save with an F32 K cache -> hashes differ from the default F16 cache
// - restore from the default cache -> 0 tokens (cross-config mismatch)
// - min_prefix: restore with a shorter prompt, threshold above the match -> 0
static bool test_incr_mismatch(struct llama_model * model, const struct common_params & params) {
    incr_test_cleanup();

    const std::string session = session_path("session_kv.bin");
    const llama_tokens tokens = make_random_tokens(model, 2048, 999);

    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.type_k = GGML_TYPE_F32;

    auto ctx_f32 = llama_context_ptr{llama_init_from_model(model, params_ctx)};
    if (!decode_tokens(ctx_f32.get(), tokens, 0, tokens.size(), 0)) {
        return false;
    }
    if (llama_state_seq_save_incr(ctx_f32.get(), session.c_str(), 0, tokens.data(), tokens.size()) != 2) {
        return false;
    }

    // restore with the default F16 cache: kv_params is part of the hash, no match
    auto ctx = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};
    const size_t n_restored = llama_state_seq_load_incr(ctx.get(), session.c_str(), 0, tokens.data(), tokens.size(), 64);
    if (n_restored != 0) {
        LOG_ERR("\n%s: error: expected 0 tokens restored (kv params mismatch), got %zu\n", __func__, n_restored);
        return false;
    }

    // restore with the F32 cache and a shorter prompt: match is floor-aligned to 1024
    const size_t n_restored2 = llama_state_seq_load_incr(ctx_f32.get(), session.c_str(), 0, tokens.data(), 1500, 1024);
    if (n_restored2 != 1024) {
        LOG_ERR("\n%s: error: expected 1024 tokens restored, got %zu\n", __func__, n_restored2);
        return false;
    }

    // min_prefix above the match -> no-op
    const size_t n_restored3 = llama_state_seq_load_incr(ctx_f32.get(), session.c_str(), 0, tokens.data(), 1500, 2000);
    if (n_restored3 != 0) {
        LOG_ERR("\n%s: error: expected 0 tokens restored (min_prefix), got %zu\n", __func__, n_restored3);
        return false;
    }

    // nonexistent session -> no-op
    const size_t n_restored4 = llama_state_seq_load_incr(ctx.get(), session_path("session_none.bin").c_str(), 0, tokens.data(), tokens.size(), 64);
    if (n_restored4 != 0) {
        LOG_ERR("\n%s: error: expected 0 tokens restored (no session), got %zu\n", __func__, n_restored4);
        return false;
    }

    LOG("\nPASS\n");
    return true;
}


int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.prompt = "";
    params.n_batch = 100;
    params.out_file = "dump_state.bin";
    params.sampling.seed = 1234;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    if (params.n_parallel == 1) {
        LOG_TRC("%s: n_parallel == 1, enabling unified kv cache\n", __func__);
        params.kv_unified = true;
    }

    if (params.n_predict < 0) {
        params.n_predict = 16;
    }

    ggml_backend_load_all();

    auto llama_init = common_init_from_params(params, true);
    auto * model = llama_init->model();

    if (model == nullptr) {
        LOG_ERR("%s: failed to init\n", __func__);
        return 1;
    }

    GGML_ASSERT(llama_init->context() == nullptr);

    // Tokenize prompt or generate random tokens
    llama_tokens tokens;
    if (params.prompt.empty()) {
        const int n_prompt = params.n_batch;

        // this path is useful for model files that do not have a tokenizer
        LOG_INF("%s: no prompt provided, generating %d (n_batch) random tokens\n", __func__, n_prompt);

        const auto * vocab = llama_model_get_vocab(model);
        const auto n_vocab = llama_vocab_n_tokens(vocab);

        std::mt19937 rng(params.sampling.seed);
        std::uniform_int_distribution<llama_token> dist(0, n_vocab - 1);
        for (int i = 0; i < n_prompt; i++) {
            tokens.push_back(dist(rng));
        }
    } else {
        LOG_INF("%s: tokenizing prompt '%s'\n", __func__, params.prompt.c_str());

        auto ctx = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};
        tokens = common_tokenize(ctx.get(), params.prompt, true);
    }

    LOG_INF("%s: the input prompt is %d tokens\n", __func__, (int)tokens.size());

    // Test 1: baseline (saves state to disk)
    auto result_baseline = test_baseline(model, params, tokens);
    if (result_baseline.empty()) {
        return 1;
    }

    // Test 2: sequence removal isolation
    if (!test_seq_rm_isolated(model, params, tokens)) {
        return 1;
    }

    // Test 3: state load
    if (!test_state_load(model, params, tokens, result_baseline)) {
        return 1;
    }

    // Test 4: seq copy (host)
    if (!test_seq_cp_host(model, params, tokens, result_baseline)) {
        return 1;
    }

    // Test 5: GGSD save/load roundtrip
    if (!test_incr_roundtrip(model, params)) {
        return 1;
    }

    // Test 6: GGSD incremental save
    if (!test_incr_append(model, params)) {
        return 1;
    }

    // Test 7: GGSD fork
    if (!test_incr_fork(model, params)) {
        return 1;
    }

    // Test 8: GGSD partial KV
    if (!test_incr_partial(model, params)) {
        return 1;
    }

    // Test 9: GGSD corrupt segment
    if (!test_incr_corrupt(model, params)) {
        return 1;
    }

    // Test 10: GGSD kv params mismatch
    if (!test_incr_mismatch(model, params)) {
        return 1;
    }

    LOG("\nAll tests passed.\n");

    return 0;
}
