# GGSD Autoload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the GGSD segment pool a third KV-reuse source in the server: at slot selection, arbitrate between slot KV, RAM prompt cache, and a zero-IO GGSD prefix estimate, then execute only the winning path - with differential replay so segments the slot already holds are never read.

**Architecture:** `llama_state_seq_load_incr` gains an `n_prefix_valid` parameter (0 = current full-replay semantics; m > 0 = skip the first `floor(m/1024)` segments entirely). A new `llama_context::state_seq_load_incr_estimate` computes the restorable prefix length with hash math plus one `stat` per segment. `server_prompt_cache::load` is split into `peek`/`consume` so arbitration can price the RAM-cache candidate without consuming it. The decision lives in the existing `update_cache` block of `get_available_slot`, gated behind `--slot-incr-autoload` (default off).

**Tech Stack:** C++17, llama.cpp C API, server task queue, pytest-free manual server smoke (tinyllama-class test models have n_ctx_train < 1024, so segment-scale server behavior cannot be pytest-tested; verified with a documented curl smoke on a SmolLM2-class model).

**Spec:** docs/superpowers/specs/2026-09-03-ggsd-autoload-design.md

## Global Constraints

- `GGSD_AUTOLOAD_MIN_PREFIX = 1024`, `GGSD_AUTOLOAD_MARGIN = 256` - fixed constants, not configurable.
- `--slot-incr-autoload` defaults to off; disabled behavior must be byte-identical to today.
- Do NOT modify the GGSQ v2 code paths or the base GGSD save semantics.
- The C API never truncates the sequence; truncation is the caller's job.
- No new files under `tests/*`; extend `tests/test-save-load-state.cpp` and `docs/ggsd-guide.md` only.
- ASCII only, no emdash, no unicode arrows; comments 1-2 lines, non-obvious invariants only.
- Every task ends with a build + test verification and a commit; never push; `Assisted-by:` only on user-approved commits.
- Environment is Linux; build dir is `build-linux`; test model is `build-linux/tinyllamas/stories15M-q4_0.gguf` (sha256 66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739).

---

### Task 1: Differential load - `n_prefix_valid` parameter

**Files:**
- Modify: `include/llama.h:919` (signature)
- Modify: `src/llama-context.h:184` (member signature)
- Modify: `src/llama-state-incr.cpp:411-560` (implementation)
- Modify: `src/llama-state-incr.cpp:580-595` (C wrapper)
- Modify: `tools/server/server-context.cpp:2638` (endpoint call site)
- Test: `tests/test-save-load-state.cpp` (new Test 11 + existing call sites get `, 0`)

**Interfaces:**
- Consumes: existing `state_seq_load_incr`, `kv->seq_rm`, `state_read_append` replay loop.
- Produces: 7-arg `llama_state_seq_load_incr(ctx, session_path, seq_id, prompt_tokens, n_prompt_tokens, min_prefix_tokens, n_prefix_valid)` used by Tasks 3-4; test helper expectations below.

- [ ] **Step 1: Write the failing test (Test 11)**

Add after `test_incr_mismatch` (test 10) in `tests/test-save-load-state.cpp`:

```cpp
// Test 11: GGSD differential replay
// - decode 2500, save -> 2 segments
// - fresh context decodes the first 1500 tokens, restore with
//   n_prefix_valid = 1024 -> 2048 restored, generation matches reference
// - delete segment 0's file, restore again with n_prefix_valid = 1024 ->
//   still 2048 (skipped segments are never read)
// - same deletion with n_prefix_valid = 0 -> 0 restored (head missing)
static bool test_incr_differential(struct llama_model * model, const struct common_params & params) {
    incr_test_cleanup();

    const std::string session = session_path("session_diff.bin");
    const llama_tokens tokens = make_random_tokens(model, 2500, 314);

    auto ctx = llama_context_ptr{llama_init_from_model(model, incr_context_params(params))};
    if (!decode_tokens(ctx.get(), tokens, 0, tokens.size(), 0)) {
        return false;
    }
    if (llama_state_seq_save_incr(ctx.get(), session.c_str(), 0, tokens.data(), tokens.size()) != 2) {
        return false;
    }

    // reference generation from the full context
    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));
    const llama_tokens expected = generate_tokens(ctx.get(), smpl.get(), (int) tokens.size(), params.n_predict, 0);
    if (expected.empty()) {
        return false;
    }

    // differential restore in a fresh context that already holds [0, 1500)
    auto ctx2 = llama_context_ptr{llama_init_from_model(model, incr_context_params(params))};
    if (!decode_tokens(ctx2.get(), tokens, 0, 1500, 0)) {
        return false;
    }

    const size_t n_restored = llama_state_seq_load_incr(ctx2.get(), session.c_str(), 0,
            tokens.data(), tokens.size(), 64, 1024);
    if (n_restored != 2048) {
        LOG_ERR("\n%s: error: expected 2048 restored (differential), got %zu\n", __func__, n_restored);
        return false;
    }
    if (!decode_tokens(ctx2.get(), tokens, n_restored, tokens.size(), 0)) {
        return false;
    }
    if (!compare_generation(model, params, ctx2.get(), (int) tokens.size(), expected)) {
        return false;
    }

    // delete segment 0: skipped segments must not be read
    const std::string fname0 = segment_file_by_index(0);
    if (fname0.empty()) {
        LOG_ERR("\n%s: error: segment 0 file not found\n", __func__);
        return false;
    }
    std::filesystem::remove(fname0);

    auto ctx3 = llama_context_ptr{llama_init_from_model(model, incr_context_params(params))};
    if (!decode_tokens(ctx3.get(), tokens, 0, 1500, 0)) {
        return false;
    }
    const size_t n_restored2 = llama_state_seq_load_incr(ctx3.get(), session.c_str(), 0,
            tokens.data(), tokens.size(), 64, 1024);
    if (n_restored2 != 2048) {
        LOG_ERR("\n%s: error: expected 2048 restored despite deleted head (differential), got %zu\n", __func__, n_restored2);
        return false;
    }

    // full-replay mode sees the deleted head and restores nothing
    auto ctx4 = llama_context_ptr{llama_init_from_model(model, incr_context_params(params))};
    const size_t n_restored3 = llama_state_seq_load_incr(ctx4.get(), session.c_str(), 0,
            tokens.data(), tokens.size(), 64, 0);
    if (n_restored3 != 0) {
        LOG_ERR("\n%s: error: expected 0 restored (deleted head, full replay), got %zu\n", __func__, n_restored3);
        return false;
    }

    LOG("\nPASS\n");
    return true;
}
```

Register in `main` after the Test 10 call:

```cpp
    // Test 11: GGSD differential replay
    if (!test_incr_differential(model, params)) {
        return 1;
    }
```

- [ ] **Step 2: Update the signature everywhere (mechanical, no behavior change yet)**

`include/llama.h` - replace the `llama_state_seq_load_incr` declaration (line 919) with:

```c
    // n_prefix_valid: if > 0, the caller asserts the sequence already holds
    // valid KV for tokens [0, n_prefix_valid) of prompt_tokens; those segments
    // are skipped (not read). Must be a multiple of 1024 for exact skipping;
    // other values are floored. 0 = full replay from the chain head.
    LLAMA_API size_t llama_state_seq_load_incr(
            struct llama_context * ctx,
                      const char * session_path,
                    llama_seq_id   seq_id,
               const llama_token * prompt_tokens,
                          size_t   n_prompt_tokens,
                          size_t   min_prefix_tokens,
                          size_t   n_prefix_valid);
```

`src/llama-context.h:184` - same parameter appended:

```cpp
    size_t state_seq_load_incr(const char * session_path, llama_seq_id seq_id, const llama_token * prompt_tokens, size_t n_prompt_tokens, size_t min_prefix_tokens, size_t n_prefix_valid);
```

`src/llama-state-incr.cpp` C wrapper (line 580) - add the parameter and forward it.

Update every call site to append `, 0`: `tools/server/server-context.cpp:2638` and the 12 existing calls in `tests/test-save-load-state.cpp` (lines 502, 556, 608, 616, 663, 736, 751, 763, 796, 803, 810, 817).

- [ ] **Step 3: Run the build to verify the red state is gone but Test 11 fails**

```bash
cmake --build build-linux --target test-save-load-state -j16
cd build-linux && ./bin/test-save-load-state -m tinyllamas/stories15M-q4_0.gguf
```

Expected: builds; Tests 1-10 PASS; Test 11 FAILS at the first assertion (the parameter is still ignored, so the differential restore replays from the head and the deleted-head case returns 0).

- [ ] **Step 4: Implement the differential path in `llama_context::state_seq_load_incr`**

In `src/llama-state-incr.cpp`, inside the member function (after the guards, once the hash chain is computed):

```cpp
    // differential mode: skip the first k0' segments entirely - the caller
    // asserts the slot holds valid KV for them (never read, not even stat)
    size_t k0 = 0;
    if (n_prefix_valid > 0) {
        if (kv->seq_pos_min(seq_id) != 0) {
            LLAMA_LOG_ERROR("%s: differential load requires positions starting at 0 (min pos = %d)\n",
                    __func__, (int) kv->seq_pos_min(seq_id));
            return 0;
        }
        k0 = std::min<size_t>(n_prefix_valid / GGSD_SEGMENT_TOKENS, n_seg);
    }
```

Change the existence walk and the replay loop to start at `k0`:

```cpp
    // find the longest chain of segment files matching the prompt
    size_t n_seg_ok = k0;
    for (; n_seg_ok < n_seg; ++n_seg_ok) {
        if (!std::filesystem::exists(ggsd_segment_path(session_path, hashes[n_seg_ok]))) {
            break;
        }
    }
```

Change the coverage and min_prefix logic so skipped segments count toward the restored coverage:

```cpp
    const size_t n_restored = n_seg_ok * GGSD_SEGMENT_TOKENS;
    if (n_restored < min_prefix_tokens) {
        return 0;
    }

    // full replay clears the sequence itself; differential mode relies on
    // the caller having truncated to the aligned boundary
    if (k0 == 0) {
        kv->seq_rm(seq_id, -1, -1);
    }

    size_t n_loaded = 0;
    try {
        for (size_t k = k0; k < n_seg_ok; ++k) {
```

(the rest of the replay loop is unchanged; `n_loaded` counting is unchanged)

- [ ] **Step 5: Run the tests to green**

```bash
cmake --build build-linux --target test-save-load-state -j16
cd build-linux && ./bin/test-save-load-state -m tinyllamas/stories15M-q4_0.gguf
```

Expected: all tests including Test 11 PASS.

- [ ] **Step 6: Commit**

```bash
git add include/llama.h src/llama-context.h src/llama-state-incr.cpp tools/server/server-context.cpp tests/test-save-load-state.cpp
git commit -m "llama : add n_prefix_valid differential mode to GGSD load"
```

---

### Task 2: Zero-IO estimate + hash-chain helper consolidation

**Files:**
- Modify: `src/llama-context.h` (declare `state_seq_load_incr_estimate` after `state_seq_load_incr`, line ~184)
- Modify: `src/llama-state-incr.cpp` (extract shared chain helper; implement estimate)

**Interfaces:**
- Consumes: `ggsd_segment_hash`, `ggsd_segment_path` (anonymous namespace, existing).
- Produces:
  - `namespace { std::vector<std::string> ggsd_hash_chain(const std::string & model_id, const std::string & kv_params, const llama_token * tokens, size_t n_seg); }`
  - `size_t llama_context::state_seq_load_incr_estimate(const char * session_path, llama_seq_id seq_id, const llama_token * prompt_tokens, size_t n_prompt_tokens, size_t min_prefix_tokens) const;`
- Task 4 consumes the estimate member with exactly this signature.

- [ ] **Step 1: Extract the shared hash-chain helper**

In `src/llama-state-incr.cpp` anonymous namespace, add:

```cpp
    // hash chain of the first n_seg 1024-token blocks of tokens
    std::vector<std::string> ggsd_hash_chain(
            const std::string & model_id,
            const std::string & kv_params,
            const llama_token * tokens,
            size_t n_seg) {
        std::vector<std::string> hashes(n_seg);
        for (size_t k = 0; k < n_seg; ++k) {
            uint8_t prev[GGSD_HASH_BYTES];
            if (k == 0) {
                memset(prev, 0, sizeof(prev));
            } else {
                ggsd_hash_hex_to_bytes(hashes[k - 1], prev);
            }

            hashes[k] = ggsd_segment_hash(model_id, kv_params, prev, tokens + k * GGSD_SEGMENT_TOKENS);
        }

        return hashes;
    }
```

Replace the identical inline loops in `state_seq_save_incr` and `state_seq_load_incr` with calls to it. Pure refactor - behavior identical.

- [ ] **Step 2: Implement the estimate**

In `src/llama-context.h` after `state_seq_load_incr`:

```cpp
    // restorable prefix length for this prompt (floor-aligned to 1024) if
    // load_incr ran with an empty sequence; hash math + one stat per segment
    size_t state_seq_load_incr_estimate(const char * session_path, llama_seq_id seq_id, const llama_token * prompt_tokens, size_t n_prompt_tokens, size_t min_prefix_tokens) const;
```

In `src/llama-state-incr.cpp` (after `state_seq_load_incr`):

```cpp
size_t llama_context::state_seq_load_incr_estimate(
        const char * session_path,
              llama_seq_id   seq_id,
        const llama_token * prompt_tokens,
              size_t   n_prompt_tokens,
              size_t   min_prefix_tokens) const {
    if (n_prompt_tokens < GGSD_SEGMENT_TOKENS) {
        return 0;
    }

    auto * kv = dynamic_cast<const llama_kv_cache *>(memory.get());
    if (kv == nullptr || model.hparams.swa_type != LLAMA_SWA_TYPE_NONE || model.hparams.n_pos_per_embd() != 1) {
        return 0;
    }

    GGML_UNUSED(seq_id);

    const uint32_t n_pos_per_embd = model.hparams.n_pos_per_embd();
    const std::string model_id = llm_arch_name(model.arch);
    const std::string kv_params = std::string(ggml_type_name(kv->type_k())) + "|" +
                                  std::string(ggml_type_name(kv->type_v())) + "|" +
                                  std::to_string(n_pos_per_embd);

    const size_t n_seg = n_prompt_tokens / GGSD_SEGMENT_TOKENS;
    const std::vector<std::string> hashes = ggsd_hash_chain(model_id, kv_params, prompt_tokens, n_seg);

    size_t n_seg_ok = 0;
    for (; n_seg_ok < n_seg; ++n_seg_ok) {
        if (!std::filesystem::exists(ggsd_segment_path(session_path, hashes[n_seg_ok]))) {
            break;
        }
    }

    const size_t n_restored = n_seg_ok * GGSD_SEGMENT_TOKENS;
    return n_restored < min_prefix_tokens ? 0 : n_restored;
}
```

Notes: `ggsd_hash_chain`, `ggsd_segment_hash`, `ggsd_segment_path` are file-local and const-safe. `count_cells_range` is not needed here - the estimate assumes an empty sequence, matching the spec.

- [ ] **Step 3: Build and run the existing tests (no new behavior yet)**

```bash
cmake --build build-linux --target test-save-load-state llama-server -j16
cd build-linux && ./bin/test-save-load-state -m tinyllamas/stories15M-q4_0.gguf
```

Expected: builds clean, all tests PASS (refactor must not change behavior).

- [ ] **Step 4: Commit**

```bash
git add src/llama-context.h src/llama-state-incr.cpp
git commit -m "llama : add GGSD load estimate and share the hash chain helper"
```

---

### Task 3: `server_prompt_cache` peek/consume split

**Files:**
- Modify: `tools/server/server-task.h:629-653` (`server_prompt_cache` class)
- Modify: `tools/server/server-task.cpp:1814-1889` (`load` -> `peek` + `consume`)
- Modify: `tools/server/server-context.cpp:1554-1559` (call site)

**Interfaces:**
- Consumes: existing `server_prompt_cache_state`, `get_common_prefix`.
- Produces (Task 4 consumes):
  - `struct server_prompt_cache::peek_result { std::list<server_prompt_cache_state>::iterator it; float f_keep; float f_sim; };`
  - `peek_result peek(const server_tokens & tokens_new, const server_tokens & tokens_slot);` - returns the best entry by the existing f_keep/f_sim criteria without consuming it; `it == states.end()` means no candidate.
  - `bool consume(peek_result & r, server_prompt & prompt, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot);` - current set_data + prompt move + erase logic.

- [ ] **Step 1: Split the class methods**

In `tools/server/server-task.h`, replace the `bool load(...)` declaration with:

```cpp
    struct peek_result {
        std::list<server_prompt_cache_state>::iterator it;
        float f_keep = 0.0f;
        float f_sim  = 0.0f;
    };

    // find the best cached prompt for tokens_new without consuming it;
    // it == states.end() means no candidate beats the slot's own state
    peek_result peek(const server_tokens & tokens_new, const server_tokens & tokens_slot);

    // move the peeked entry into the slot (set_data + prompt + erase)
    bool consume(peek_result & r, server_prompt & prompt, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot);
```

- [ ] **Step 2: Move the logic**

In `tools/server/server-task.cpp`, rewrite `server_prompt_cache::load` as two functions. `peek` keeps lines 1814-1852 (baseline f_keep/f_sim from `tokens_slot`, candidate scan with the `f_keep_cur < 0.25f` skip and the strict-improvement comparison) but returns `it_best` + scores instead of restoring. `consume` keeps lines 1853-1888 (the two `set_data_ext` blocks, `prompt = std::move(...)`, `states.erase(it)`) parameterized on `r.it`. Behavior is identical to today's `load`.

- [ ] **Step 3: Rewrite the call site**

`tools/server/server-context.cpp` around line 1554 currently calls `ret->prompt_load(*prompt_cache, task.tokens)`. Keep `server_slot::prompt_load` as a thin wrapper but reimplement it as peek + consume so non-autoload paths are untouched:

```cpp
    bool prompt_load(server_prompt_cache & prompt_cache, const server_tokens & tokens) {
        auto r = prompt_cache.peek(tokens, prompt.tokens);
        if (r.it == prompt_cache.states.end()) {
            return false;
        }

        const bool res = prompt_cache.consume(r, prompt, ctx_tgt, ctx_dft, id);
        if (!res) {
            SLT_WRN(*this, "%s", "failed to load prompt from cache\n");
        }

        return res;
    }
```

(`prompt_cache.states` must be public or exposed via a begin/end accessor - prefer keeping `states` public as it already is.)

- [ ] **Step 4: Build and verify no behavior change**

```bash
cmake --build build-linux --target llama-server -j16
cmake --build build-linux --target test-save-load-state -j16 && cd build-linux && ./bin/test-save-load-state -m tinyllamas/stories15M-q4_0.gguf
```

Expected: builds clean; C tests PASS (server-side behavior is exercised in Task 5).

- [ ] **Step 5: Commit**

```bash
git add tools/server/server-task.h tools/server/server-task.cpp tools/server/server-context.cpp
git commit -m "server : split prompt cache load into peek and consume"
```

---

### Task 4: `--slot-incr-autoload` flag and arbitration

**Files:**
- Modify: `common/common.h:675` (add flag near `slot_save_path`)
- Modify: `common/arg.cpp` (parser entry, next to `--slot-save-path` at line 3575)
- Modify: `tools/server/server-context.cpp` (startup warning ~line 1269; arbitration in `get_available_slot` ~line 1541)

**Interfaces:**
- Consumes: `state_seq_load_incr_estimate` (Task 2), `peek`/`consume` (Task 3), `llama_state_seq_load_incr` with `n_prefix_valid` (Task 1).
- Produces: server behavior change only; no further interface changes.

- [ ] **Step 1: Add the flag**

`common/common.h`, next to `slot_save_path`:

```cpp
    bool slot_incr_autoload      = false; // restore GGSD prefixes automatically at slot selection
```

`common/arg.cpp`, after the `--slot-save-path` entry:

```cpp
        {"--slot-incr-autoload", "",
            "automatically restore matching GGSD prefixes at slot selection (requires --slot-save-path)",
            params.slot_incr_autoload,
            true},
```

(match the exact boolean-entry form used by other flags in the same section)

- [ ] **Step 2: Startup precondition**

In `tools/server/server-context.cpp` near the prompt-cache init (~line 1269), add:

```cpp
        if (params_base.slot_incr_autoload && params_base.slot_save_path.empty()) {
            SRV_WRN("%s", "--slot-incr-autoload requires --slot-save-path, disabling\n");
            params_base.slot_incr_autoload = false;
        }
```

- [ ] **Step 3: Arbitration in `get_available_slot`**

Replace the body of the `if (update_cache)` block (line 1541) with:

```cpp
        if (ret) {
            update_cache = update_cache && prompt_cache;

            // cache prompts only for completion tasks
            update_cache = update_cache && task.type == SERVER_TASK_TYPE_COMPLETION;

            if (update_cache) {
                SRV_TRC("%s", "updating prompt cache\n");

                const int64_t t_start = ggml_time_us();

                ret->prompt_save(*prompt_cache);

                if (params_base.slot_incr_autoload && !task.tokens.has_mtmd && !task.tokens.get_tokens().empty()) {
                    if (autoload_ggsd(*ret, task)) {
                        prompt_cache->update();
                        SRV_TRC("prompt cache update took %.2f ms\n", (ggml_time_us() - t_start) / 1000.0);
                        return ret;
                    }
                }

                if (!ret->prompt_load(*prompt_cache, task.tokens)) {
                    ret->prompt_clear();
                }

                prompt_cache->update();

                SRV_TRC("prompt cache update took %.2f ms\n", (ggml_time_us() - t_start) / 1000.0);
            }
        }

        return ret;
```

Add a private method next to `get_available_slot` (same class, so it has `params_base`, `slots`, `ctx_tgt` access as needed):

```cpp
    // three-source arbitration for GGSD (spec: 2026-09-03-ggsd-autoload-design.md).
    // returns true if the slot was filled from the segment pool
    bool autoload_ggsd(server_slot & slot, const server_task & task) {
        const auto & task_tokens = task.tokens.get_tokens();

        const size_t n_slot = slot.prompt.tokens.get_common_prefix(task.tokens);

        auto r_cache = prompt_cache->peek(task.tokens, slot.prompt.tokens);
        const size_t n_cache = r_cache.it != prompt_cache->states.end()
            ? (size_t) r_cache.it->prompt.tokens.get_common_prefix(task.tokens) : 0;

        const size_t n_ggsd = ctx_tgt == nullptr ? 0 :
            ctx_tgt->state_seq_load_incr_estimate(params_base.slot_save_path.c_str(), slot.id,
                    task_tokens.data(), task_tokens.size(), GGSD_AUTOLOAD_MIN_PREFIX);

        const size_t n_best = std::max({n_slot, n_cache, n_ggsd});
        if (n_best < GGSD_AUTOLOAD_MIN_PREFIX) {
            return false;
        }

        if (n_ggsd >= n_best && n_ggsd - std::max(n_slot, n_cache) >= GGSD_AUTOLOAD_MARGIN) {
            size_t m_aligned = std::min(n_slot, n_ggsd) / 1024 * 1024;
            const bool lcp_complete = n_slot > 0 && n_slot == slot.prompt.tokens.size()
                && slot.prompt.tokens.get_tokens().size() >= m_aligned;

            size_t n_restored = 0;
            if (lcp_complete && m_aligned >= 1024) {
                llama_memory_seq_rm(llama_get_memory(ctx_tgt), slot.id, m_aligned, -1);
                n_restored = llama_state_seq_load_incr(ctx_tgt, params_base.slot_save_path.c_str(),
                        slot.id, task_tokens.data(), task_tokens.size(), GGSD_AUTOLOAD_MIN_PREFIX, m_aligned);
            } else {
                m_aligned = 0;
                n_restored = llama_state_seq_load_incr(ctx_tgt, params_base.slot_save_path.c_str(),
                        slot.id, task_tokens.data(), task_tokens.size(), GGSD_AUTOLOAD_MIN_PREFIX, 0);
            }

            if (n_restored >= GGSD_AUTOLOAD_MIN_PREFIX) {
                slot.prompt.tokens = server_tokens(
                        llama_tokens(task_tokens.begin(), task_tokens.begin() + n_restored),
                        /* has_mtmd = */ false);
                slot.prompt.checkpoints.clear();
                SRV_INF("slot %d: autoloaded %zu GGSD tokens (slot %zu, cache %zu)\n",
                        slot.id, n_restored, n_slot, n_cache);
                return true;
            }

            if (m_aligned > 0) {
                // differential load fell short: drop the partially restored
                // state so the prefill starts clean
                llama_memory_seq_rm(llama_get_memory(ctx_tgt), slot.id, -1, -1);
            }
        }

        if (n_cache >= n_best) {
            const bool res = prompt_cache->consume(r_cache, slot.prompt, ctx_tgt, ctx_dft, slot.id);
            return res;
        }

        return false;
    }
```

Add the constants at the top of `server-context.cpp` (near other file-local constants):

```cpp
static constexpr size_t GGSD_AUTOLOAD_MIN_PREFIX = 1024; // at least one full segment
static constexpr size_t GGSD_AUTOLOAD_MARGIN     = 256;  // must beat the runner-up by this much
```

Notes:
- `n_cache` is the LCP of the peeked entry (already `f_sim`-checked by peek's criteria); the comparison units are reusable prefix tokens for all three sources.
- On `ggsd` failure after truncation (`n_restored < MIN_PREFIX`), the sequence is cleared so the request falls through to a full prefill - the RAM cache rescue already preserved the slot's previous context.
- The `n_cache >= n_best` tail executes the plain cache path when the cache wins or nothing did; when nothing reached MIN_PREFIX, autoload returns false early and the original `prompt_load`/`prompt_clear` lines run as before.

- [ ] **Step 4: Build and verify the off-path is unchanged**

```bash
cmake --build build-linux --target llama-server -j16
cmake --build build-linux --target test-save-load-state -j16 && cd build-linux && ./bin/test-save-load-state -m tinyllamas/stories15M-q4_0.gguf
```

Expected: builds clean; all tests PASS. With the flag off, `autoload_ggsd` is never called.

- [ ] **Step 5: Commit**

```bash
git add common/common.h common/arg.cpp tools/server/server-context.cpp
git commit -m "server : add --slot-incr-autoload with three-source KV arbitration"
```

---

### Task 5: Smoke verification and documentation

**Files:**
- Modify: `docs/ggsd-guide.md` (config section 1.2, API section 1.3, semantics 1.6)

**Interfaces:**
- Consumes: everything above; no code changes.

- [ ] **Step 1: Download a ctx_train-capable model and run the smoke**

The pytest suite's presets (stories260K) have n_ctx_train < 1024, so this cannot be a committed pytest; verify manually with the model from the earlier GGSD smoke:

```bash
curl -sSL -o smollm2-135m-q8_0.gguf "https://hf-mirror.com/Felladrin/gguf-Q8_0-SmolLM2-135M-Instruct/resolve/main/smollm2-135m-instruct-q8_0.gguf?download=true"
mkdir -p saves
./build-linux/bin/llama-server -m smollm2-135m-q8_0.gguf --slot-save-path saves \
    --slot-incr-autoload -c 4096 --port 8012 -t 4 &

# 1. fill a slot with a ~2000-token prompt
P=$(python3 -c "print('the little dog ran ' * 500)")
curl -s http://127.0.0.1:8012/completion -d "{\"prompt\": \"$P\", \"n_predict\": 4}" > /dev/null

# 2. persist it
curl -s "http://127.0.0.1:8012/slots/0?action=save_incr" -d '{"filename":"auto"}'
# expect n_segments >= 1

# 3. restart the server (same flags) - RAM cache is gone, disk pool remains
# 4. repeat request 1; check the log for "autoloaded ... GGSD tokens" and that
#    timings.prompt_n covers only the tokens after the restored prefix
```

Expected: on the restarted server the repeat request logs the ggsd autoload branch and processes only the remainder. Kill the server afterwards; `rm -rf saves smollm2-135m-q8_0.gguf`.

Also verify the off-path: run the same steps without `--slot-incr-autoload` and confirm no autoload log line appears.

- [ ] **Step 2: Full C test suite**

```bash
cmake --build build-linux --target test-save-load-state test-thread-safety test-state-restore-fragmented -j16
cd build-linux && ./bin/test-save-load-state -m tinyllamas/stories15M-q4_0.gguf && \
./bin/test-state-restore-fragmented -m tinyllamas/stories15M-q4_0.gguf && \
./bin/test-thread-safety -m tinyllamas/stories15M-q4_0.gguf -ngl 99 -p "The meaning of life is" -n 128 -c 256 -ub 32 -np 4 -t 2
```

Expected: all PASS.

- [ ] **Step 3: Update docs/ggsd-guide.md**

Section 1.2: add `--slot-incr-autoload` (flag, default off, requires `--slot-save-path`) and the two fixed constants. Section 1.3: note that with the flag on, the server restores matching GGSD prefixes automatically at slot selection; `restore_incr` semantics are unchanged. Section 1.6: add rows - "Autoload enabled, prefix matches saved segments | restored automatically before prefill"; "Autoload enabled, GGSD wins but a segment went missing | verified prefix restored, remainder prefilled".

- [ ] **Step 4: Commit**

```bash
git add docs/ggsd-guide.md
git commit -m "docs : document --slot-incr-autoload in the GGSD guide"
```

---

## Plan Self-Review Notes

- Spec coverage: C API param (Task 1), estimate (Task 2), peek/consume (Task 3), flag + arbitration + startup precondition + constants (Task 4), all five spec tests mapped (Tasks 1 and 5), documentation (Task 5). The spec's "hash-chain helper consolidation" is Task 2 Step 1.
- Type consistency: `peek_result.it` iterator type matches `states` (`std::list<server_prompt_cache_state>::iterator`); the 7-arg load signature is used identically in Tasks 1 and 4; constants named identically in Task 4 code and spec.
- Known deviation from the spec text: the spec's pseudocode passes `GGSD_AUTOLOAD_MIN_PREFIX` as `min_prefix_tokens` on the autoload load call - kept as written; the manual endpoint keeps its user-supplied min_prefix.
