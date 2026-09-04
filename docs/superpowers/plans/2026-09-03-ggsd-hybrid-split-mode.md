# GGSD Split Mode for Hybrid Caches Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `--prompt-cache-ssd` (GGSD) work for hybrid-architecture models (Qwen3.5) by dispatching to a split mode: attention-layer KV as shared content-addressed segments, recurrent state + non-aligned attention tail as one `rec_*.bin` file per conversation state.

**Architecture:** All changes live in `src/llama-state-incr.cpp` (dispatch by memory class, split save/load/estimate) plus one prerequisite fix in `src/llama-kv-cache.cpp` (restore `cell_ext` on the ubatch append path, an upstream TODO). Server code needs no changes: `slot_autosave` / `autoload_ggsd` call the same three entry points. Attention sub-cache of `llama_memory_hybrid` is accessed via the existing public `get_mem_attn()` / `get_mem_recr()` accessors - the hybrid class is NOT modified.

**Tech Stack:** C++17, existing GGSD infrastructure in `src/llama-state-incr.cpp` (sha256 chain, `llama_file`, `ggsd_io_write_file`/`ggsd_io_read_host`), CMake test `test-save-load-state`.

**Spec:** `docs/superpowers/specs/2026-09-03-ggsd-hybrid-split-mode-design.md`

## Global Constraints

- ASCII only: no emdash, no unicode arrows/ellipsis in code or comments (repo AGENTS.md).
- Comments: 1-2 lines, ASD-STE100 plain wording, no hard column wrap.
- NO new files in `tests/` (maintainer rule). Regression = existing `test-save-load-state`.
- Commits: every `git commit` step requires the user's explicit approval first; use `Assisted-by: <assistant name>`, never `Co-authored-by:`. NEVER `git push`.
- Do not modify `llama-memory-hybrid.*`, `llama-memory-recurrent.*` (accessors already public).
- Verify facts by running: model binary at `build-linux/bin/`, test model at `build-linux/tinyllamas/stories15M-q4_0.gguf` (sha256 `66967fbe...739`).
- Existing GGSD semantics that MUST be preserved: stop-at-first-corrupt load, payload sha256 verification, `.tmp` + rename atomic segment writes, fork-point skip, pool semantics (session file never consulted on load), R2/R3 position guards.

---

### Task 1: Restore cell_ext on the ubatch append path

**Files:**
- Modify: `src/llama-kv-cache.cpp` (inside `llama_kv_cache::state_read_meta`, append branch, around lines 2286-2343)

**Interfaces:**
- Consumes: `llama_kv_cells::ext_set(uint32_t idx, const llama_kv_cell_ext & ext)` (exists, used at lines 530/1136/2366); `sinfo.idxs[0]` cell indices (asserted at line 2338).
- Produces: append-path restore now leaves correct `cell_ext` on restored cells for `n_pos_per_embd() > 1` models. All later tasks rely on this for IMROPE (Qwen3.5, `n_pos_per_embd == 4`).

**Context:** The append branch reads each cell's `ext` and routes only `ext.y`/`ext.x` into `ubatch.pos` (lines 2305-2311), then calls `apply_ubatch` (line 2332) which sets primary positions but drops ext. The whole-cache branch (line 2363-2367) already does `cells.ext_set` - mirror it.

- [ ] **Step 1: Stash ext values during the per-cell read loop**

In the append branch, before the loop (after `llama_batch_allocr balloc...` at line 2287), add:

```cpp
std::vector<llama_kv_cell_ext> exts;
if (hparams.n_pos_per_embd() > 1) {
    exts.reserve(cell_count);
}
```

Inside the loop, extend the existing ext read block (lines 2305-2311) to keep the value:

```cpp
if (hparams.n_pos_per_embd() > 1) {
    llama_kv_cell_ext ext;
    io.read(&ext, sizeof(ext));

    ubatch.pos[i + ubatch.n_tokens]   = ext.y;
    ubatch.pos[i + ubatch.n_tokens*2] = ext.x;
    exts.push_back(ext);
}
```

- [ ] **Step 2: Apply ext after apply_ubatch**

Replace the TODO comment + `apply_ubatch` call (lines 2330-2332) with:

```cpp
apply_ubatch(sinfo, ubatch);

if (hparams.n_pos_per_embd() > 1) {
    for (uint32_t i = 0; i < cell_count; ++i) {
        cells.ext_set(sinfo.idxs[0][i], exts[i]);
    }
}
```

Check what `cells` refers to in this scope first (the whole-cache branch at line 2361 uses `cells.pos_set` directly, so a `cells` reference exists in `state_read_meta`; if the append branch scope names it differently, use the same reference the whole-cache branch uses).

- [ ] **Step 3: Build**

Run: `cmake --build build-linux --target test-save-load-state -j16 2>&1 | tail -3`
Expected: build succeeds, no new warnings.

- [ ] **Step 4: Regression run (n_pos_per_embd == 1 must be unaffected)**

Run: `cd build-linux && ./bin/test-save-load-state -m tinyllamas/stories15M-q4_0.gguf 2>&1 | grep -cE "PASS|FAIL|Assertion"`
Expected: 10 `=== Test ... PASS` lines, no FAIL/Assertion (baseline: all 10 passed pre-change).

- [ ] **Step 5: Commit (after user approval)**

```bash
git add src/llama-kv-cache.cpp
git commit -m "kv-cache : restore cell_ext on the ubatch state-read append path

Assisted-by: <assistant name>"
```

---

### Task 2: Shared identity helpers + remove the latent n_pos_per_embd position bug

**Files:**
- Modify: `src/llama-state-incr.cpp` (anonymous namespace lines 25-224, save body lines 226-421, load body 423-574, estimate 576-611)

**Interfaces:**
- Consumes: `ggsd_hash_chain`, `ggsd_write_str`/`ggsd_read_str`, IO classes (all exist).
- Produces (used by Tasks 3-5):
  - `struct ggsd_rec_header { uint32_t n_tokens; uint32_t n_tail; std::string chain_hash; std::string model_id; std::string kv_params; std::vector<llama_token> tokens; uint64_t payload_size; uint8_t payload_hash[GGSD_HASH_BYTES]; };`
  - `std::string ggsd_prefix_hash(const std::string & model_id, const std::string & kv_params, const llama_token * tokens, size_t n)` - truncated sha256 hex of length-prefixed model_id, kv_params, then `n * sizeof(llama_token)` bytes of tokens.
  - `std::string ggsd_rec_path(const std::string & session_path, const std::string & hash)` - `<dir>/rec_<hash>.bin`.
  - `constexpr char GGSD_REC_MAGIC[4] = { 'G', 'G', 'S', 'R' };`
  - kv_params gains `model.hparams.n_layer()` as a 4th `|`-separated field (via a new `ggsd_kv_params(const llama_model & model, const llama_kv_cache * kv)` helper) - both standard and hybrid paths use it. Note: this invalidates previously persisted segment pools (feature branch, accepted by spec).

- [ ] **Step 1: Add `ggsd_kv_params` helper and use it in all three entry points**

```cpp
std::string ggsd_kv_params(const llama_model & model, const llama_kv_cache * kv) {
    return std::string(ggml_type_name(kv->type_k())) + "|" +
           std::string(ggml_type_name(kv->type_v())) + "|" +
           std::to_string(model.hparams.n_pos_per_embd()) + "|" +
           std::to_string(model.hparams.n_layer());
}
```

Replace the three inline `kv_params = ...` constructions (save line ~256, load ~459, estimate ~595) with calls to it.

- [ ] **Step 2: Fix the pos-range multiplication bug**

In `state_seq_save_incr` (lines 322-323), cell primary positions are token indices for every model (`pos_get(i) == i` verified in `src/llama-kv-cells.h`; ext carries the mrope sections). Remove the multiplication:

```cpp
const llama_pos pos_begin = (llama_pos) ( k       * GGSD_SEGMENT_TOKENS);
const llama_pos pos_end   = (llama_pos) ((k + 1) * GGSD_SEGMENT_TOKENS);
```

Same for the error message at line 402 (drop `* n_pos_per_embd`).

- [ ] **Step 3: Relax the n_pos_per_embd guard on the standard path**

Delete the `n_pos_per_embd != 1` early-return blocks in save (lines 249-252) and load (lines 445-449) and the `n_pos_per_embd() != 1` term in estimate (line 587). Keep `n_pos_per_embd` local variable where the code uses it after removal (it is no longer needed - remove dead locals). Keep both SWA guards (lines 242-245, 440-443, 587) exactly as they are. Task 1 makes mrope restore correct; the existing R2/R3 pos guards (lines 313-317, 472-478) still apply unchanged.

- [ ] **Step 4: Add rec-file helpers (used by Tasks 3-5)**

Append to the anonymous namespace (before its closing brace at line 224):

```cpp
constexpr char GGSD_REC_MAGIC[4] = { 'G', 'G', 'S', 'R' };

struct ggsd_rec_header {
    uint32_t n_tokens = 0;
    uint32_t n_tail = 0;
    std::string chain_hash;
    std::string model_id;
    std::string kv_params;
    std::vector<llama_token> tokens;
    uint64_t payload_size = 0;
    uint8_t payload_hash[GGSD_HASH_BYTES] = {};
};

std::string ggsd_prefix_hash(
        const std::string & model_id,
        const std::string & kv_params,
        const llama_token * tokens,
        size_t n) {
    llama_sha256_t sha;
    llama_sha256_init(&sha);

    const uint32_t len_model = (uint32_t) model_id.size();
    llama_sha256_update(&sha, (const uint8_t *) &len_model, sizeof(len_model));
    llama_sha256_update(&sha, (const uint8_t *) model_id.data(), len_model);

    const uint32_t len_params = (uint32_t) kv_params.size();
    llama_sha256_update(&sha, (const uint8_t *) &len_params, sizeof(len_params));
    llama_sha256_update(&sha, (const uint8_t *) kv_params.data(), len_params);

    llama_sha256_update(&sha, (const uint8_t *) tokens, n * sizeof(llama_token));

    uint8_t digest[32];
    llama_sha256_final(&sha, digest);

    std::string hex(GGSD_HASH_HEX_LEN, '0');
    for (size_t i = 0; i < GGSD_HASH_BYTES; ++i) {
        const uint8_t hi = digest[i] >> 4;
        const uint8_t lo = digest[i] & 0x0F;
        hex[2*i + 0] = hi < 10 ? '0' + hi : 'a' + hi - 10;
        hex[2*i + 1] = lo < 10 ? '0' + lo : 'a' + lo - 10;
    }
    return hex;
}

std::string ggsd_rec_path(const std::string & session_path, const std::string & hash) {
    std::filesystem::path p(session_path);
    return (p.parent_path() / ("rec_" + hash + ".bin")).string();
}

// reads magic, version, n_tokens, n_tail, chain_hash, payload_size, payload_hash,
// model_id, kv_params, tokens. returns false on wrong magic/version or IO error.
bool ggsd_rec_read_header(llama_file & file, ggsd_rec_header & out) {
    try {
        char magic[4];
        file.read_raw(magic, sizeof(magic));
        if (memcmp(magic, GGSD_REC_MAGIC, sizeof(magic)) != 0) {
            return false;
        }
        if (file.read_u32() != GGSD_VERSION) {
            return false;
        }
        out.n_tokens = file.read_u32();
        out.n_tail   = file.read_u32();

        char hash[GGSD_HASH_HEX_LEN];
        file.read_raw(hash, sizeof(hash));
        out.chain_hash.assign(hash, sizeof(hash));

        file.read_raw(&out.payload_size, sizeof(out.payload_size));
        file.read_raw(out.payload_hash, GGSD_HASH_BYTES);

        out.model_id  = ggsd_read_str(file);
        out.kv_params = ggsd_read_str(file);

        out.tokens.resize(out.n_tokens);
        if (out.n_tokens > 0) {
            file.read_raw(out.tokens.data(), out.n_tokens * sizeof(llama_token));
        }
    } catch (...) {
        return false;
    }
    return true;
}
```

Note the field order is the on-disk contract: `magic, version, n_tokens, n_tail, chain_hash[32], payload_size, payload_hash[16], model_id, kv_params, tokens[n_tokens], payload`.

- [ ] **Step 5: Build + regression**

Run: `cmake --build build-linux --target test-save-load-state -j16 2>&1 | tail -3 && cd build-linux && ./bin/test-save-load-state -m tinyllamas/stories15M-q4_0.gguf 2>&1 | grep -E "PASS|FAIL" | head -12`
Expected: build ok, 10 PASS.

- [ ] **Step 6: Commit (after user approval)**

```bash
git add src/llama-state-incr.cpp
git commit -m "ggsd : extend identity with n_layer, fix mrope pos ranges, add rec file helpers

Assisted-by: <assistant name>"
```

---

### Task 3: Hybrid split-mode save

**Files:**
- Modify: `src/llama-state-incr.cpp` (`state_seq_save_incr`, and add a file-static helper `state_seq_save_incr_hybrid`)

**Interfaces:**
- Consumes: Task 2 helpers; `llama_memory_hybrid::get_mem_attn() -> llama_kv_cache *`, `get_mem_recr() -> llama_memory_recurrent *`; `llama_kv_cache::state_write_range(io, seq_id, pos_begin, pos_end)`; `llama_memory_recurrent::state_write(io, seq_id, flags = 0)`.
- Produces: on disk, `seg_*.bin` for aligned attention ranges and `rec_<prefix_hash>.bin` for (attn tail + rec state). Load/estimate (Tasks 4-5) read this layout.

- [ ] **Step 1: Dispatch at the top of `state_seq_save_incr`**

After the existing standard-cache dispatch (keep the `dynamic_cast<llama_kv_cache*>` block as-is but change the error return into a fall-through to hybrid first - reorder so hybrid is tried BEFORE the standard-cast error):

```cpp
if (auto * mem = dynamic_cast<llama_memory_hybrid *>(memory.get())) {
    return state_seq_save_incr_hybrid(session_path, seq_id, tokens, n_token_count, *mem);
}
```

(The standard path continues unchanged below for `llama_kv_cache`.)

- [ ] **Step 2: Implement the hybrid save helper**

```cpp
size_t llama_context::state_seq_save_incr_hybrid(
        const char * session_path,
              llama_seq_id   seq_id,
        const llama_token * tokens,
              size_t   n_token_count,
              llama_memory_hybrid & mem) {
    auto * kv_attn = mem.get_mem_attn();
    auto * mem_recr = mem.get_mem_recr();

    if (model.hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        LLAMA_LOG_ERROR("%s: hybrid split save does not support SWA attention caches\n", __func__);
        return (size_t) -1;
    }

    if (n_token_count == 0) {
        return 0;
    }

    const std::string model_id  = llm_arch_name(model.arch);
    const std::string kv_params = ggsd_kv_params(model, kv_attn);

    const size_t n_seg = n_token_count / GGSD_SEGMENT_TOKENS;

    // hash chain of aligned blocks (same identity as the standard path)
    const std::vector<std::string> hashes = ggsd_hash_chain(model_id, kv_params, tokens, n_seg);

    if (kv_attn->seq_pos_min(seq_id) != 0) {
        LLAMA_LOG_ERROR("%s: sequence positions do not start at 0 (min pos = %d)\n",
                __func__, (int) kv_attn->seq_pos_min(seq_id));
        return (size_t) -1;
    }

    // aligned segments: same write loop as the standard path
    size_t n_written = 0;
    for (size_t k = 0; k < n_seg; ++k) {
        const std::string fname = ggsd_segment_path(session_path, hashes[k]);
        if (std::filesystem::exists(fname)) {
            continue;
        }

        const llama_pos pos_begin = (llama_pos) ( k       * GGSD_SEGMENT_TOKENS);
        const llama_pos pos_end   = (llama_pos) ((k + 1) * GGSD_SEGMENT_TOKENS);

        if (kv_attn->count_cells_range(seq_id, pos_begin, pos_end) != GGSD_SEGMENT_TOKENS) {
            break;
        }

        // [identical header/write/rename body as standard save lines 340-386 -
        //  reuse by extracting the standard segment-write block into a helper
        //  ggsd_write_segment(kv, session_path, seq_id, k, hashes, model_id,
        //  kv_params, tokens, pos_begin, pos_end) and calling it from both paths]
        ...
        ++n_written;
    }

    // longest on-disk chain prefix
    size_t n_chain = 0;
    for (; n_chain < n_seg; ++n_chain) {
        if (!std::filesystem::exists(ggsd_segment_path(session_path, hashes[n_chain]))) {
            break;
        }
    }

    const size_t n_tail = n_token_count - n_chain * GGSD_SEGMENT_TOKENS;

    // rec file: attn tail + recurrent snapshot, coverage = n_token_count
    if (n_token_count >= GGSD_SEGMENT_TOKENS) {
        const std::string chain_hash = ggsd_prefix_hash(model_id, kv_params, tokens, n_token_count);
        const std::string fname = ggsd_rec_path(session_path, chain_hash);

        if (!std::filesystem::exists(fname)) {
            const llama_pos tail_begin = (llama_pos) (n_chain * GGSD_SEGMENT_TOKENS);
            const llama_pos tail_end   = (llama_pos) n_token_count;

            if (kv_attn->count_cells_range(seq_id, tail_begin, tail_end) != (size_t) n_tail) {
                LLAMA_LOG_WARN("%s: attn cache missing tail cells [%d, %d), rec file skipped\n",
                        __func__, (int) tail_begin, (int) tail_end);
            } else {
                ggsd_io_write_dummy io_dummy;
                kv_attn->state_write_range(io_dummy, seq_id, tail_begin, tail_end);
                mem_recr->state_write(io_dummy, seq_id);
                const uint64_t payload_size = io_dummy.n_bytes();

                const std::string fname_tmp = fname + ".tmp";
                {
                    llama_file file(fname_tmp.c_str(), "wb");

                    file.write_raw(GGSD_REC_MAGIC, sizeof(GGSD_REC_MAGIC));
                    file.write_u32(GGSD_VERSION);
                    file.write_u32((uint32_t) n_token_count);
                    file.write_u32((uint32_t) n_tail);
                    file.write_raw(chain_hash.c_str(), GGSD_HASH_HEX_LEN);
                    file.write_raw(&payload_size, sizeof(payload_size));

                    uint8_t digest[32];
                    memset(digest, 0, sizeof(digest));
                    file.write_raw(digest, GGSD_HASH_BYTES); // patched below

                    ggsd_write_str(file, model_id);
                    ggsd_write_str(file, kv_params);
                    file.write_raw(tokens, n_token_count * sizeof(llama_token));

                    llama_sha256_t sha;
                    llama_sha256_init(&sha);
                    {
                        ggsd_io_write_file io(&file, &sha);
                        kv_attn->state_write_range(io, seq_id, tail_begin, tail_end);
                        mem_recr->state_write(io, seq_id);
                    }
                    llama_sha256_final(&sha, digest);

                    // header size up to payload hash: 4 + 4 + 4 + 4 + 32 + 8 = 56
                    file.seek(56, SEEK_SET);
                    file.write_raw(digest, GGSD_HASH_BYTES);
                }
                std::filesystem::rename(fname_tmp, fname);
            }
        }

        // dominated-parent cleanup: drop rec files whose token sequence is a
        // strict prefix of the one just saved
        for (const auto & entry : std::filesystem::directory_iterator(
                std::filesystem::path(session_path).parent_path())) {
            const std::string name = entry.path().filename().string();
            if (name.rfind("rec_", 0) != 0 || name.find(".bin") == std::string::npos ||
                    entry.path() == std::filesystem::path(fname)) {
                continue;
            }
            try {
                llama_file f(entry.path().string().c_str(), "rb");
                ggsd_rec_header h;
                if (!ggsd_rec_read_header(f, h)) {
                    continue;
                }
                if (h.model_id != model_id || h.kv_params != kv_params) {
                    continue;
                }
                if (h.n_tokens < n_token_count &&
                        memcmp(h.tokens.data(), tokens, h.n_tokens * sizeof(llama_token)) == 0) {
                    std::filesystem::remove(entry.path());
                    LLAMA_LOG_INFO("%s: removed dominated rec file %s (%u tokens)\n",
                            __func__, name.c_str(), h.n_tokens);
                }
            } catch (...) {
                // unreadable file: leave it alone
            }
        }
    }

    if (n_seg > 0 && n_chain == 0) {
        LLAMA_LOG_ERROR("%s: attn cache is missing the head of the sequence, nothing usable saved\n", __func__);
        return (size_t) -1;
    }

    LLAMA_LOG_INFO("%s: wrote %zu segment(s), rec coverage %zu tokens (chain %zu)\n",
            __func__, n_written, n_token_count, n_chain);

    return n_chain;
}
```

Notes for the implementer:
- Extract the standard segment-write body (current lines 340-386) into `ggsd_write_segment(...)` and call it from both save paths - do not duplicate it. Signature: `void ggsd_write_segment(llama_kv_cache * kv, const std::string & session_path, llama_seq_id seq_id, size_t k, const std::vector<std::string> & hashes, const std::string & model_id, const std::string & kv_params, const llama_token * tokens, llama_pos pos_begin, llama_pos pos_end, size_t & n_written)`.
- The hybrid path intentionally does NOT rewrite the GGSD session file (pool semantics; load never consults it).
- Declaration goes in `src/llama-context.h` next to the other `state_seq_*` members: `size_t state_seq_save_incr_hybrid(const char * session_path, llama_seq_id seq_id, const llama_token * tokens, size_t n_token_count, llama_memory_hybrid & mem);` plus a forward `class llama_memory_hybrid;` include of `llama-memory-hybrid.h` in `src/llama-context.h` (it already includes llama-memory.h; add the hybrid header).

- [ ] **Step 3: Build**

Run: `cmake --build build-linux --target test-save-load-state llama-server -j16 2>&1 | tail -3`
Expected: build succeeds.

- [ ] **Step 4: Regression (standard path unchanged)**

Run: `cd build-linux && ./bin/test-save-load-state -m tinyllamas/stories15M-q4_0.gguf 2>&1 | grep -E "PASS|FAIL" | head -12`
Expected: 10 PASS.

- [ ] **Step 5: Commit (after user approval)**

```bash
git add src/llama-state-incr.cpp src/llama-context.h
git commit -m "ggsd : hybrid split-mode save (attn segments + rec snapshot file)

Assisted-by: <assistant name>"
```

---

### Task 4: Hybrid split-mode load

**Files:**
- Modify: `src/llama-state-incr.cpp` (`state_seq_load_incr` + helper `state_seq_load_incr_hybrid`), `src/llama-context.h` (declaration)

**Interfaces:**
- Consumes: Task 2/3 layout; `llama_kv_cache::state_read_append(io, seq_id)`; `llama_memory_recurrent::state_read(io, seq_id, flags = 0)`; `ggsd_rec_read_header`.
- Produces: `state_seq_load_incr` returns the restored token coverage (`rec n_tokens`) for hybrid models; server `autoload_ggsd` consumes it unchanged.

- [ ] **Step 1: Dispatch at the top of `state_seq_load_incr`**

```cpp
if (auto * mem = dynamic_cast<llama_memory_hybrid *>(memory.get())) {
    return state_seq_load_incr_hybrid(session_path, seq_id, prompt_tokens, n_prompt_tokens, min_prefix_tokens, *mem);
}
```

(Place before the standard-cast error return, same pattern as save.)

- [ ] **Step 2: Implement the hybrid load helper**

```cpp
size_t llama_context::state_seq_load_incr_hybrid(
        const char * session_path,
              llama_seq_id   seq_id,
        const llama_token * prompt_tokens,
              size_t   n_prompt_tokens,
              size_t   min_prefix_tokens,
              llama_memory_hybrid & mem) {
    if (n_prompt_tokens < GGSD_SEGMENT_TOKENS) {
        return 0;
    }

    auto * kv_attn  = mem.get_mem_attn();
    auto * mem_recr = mem.get_mem_recr();

    if (model.hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        return 0;
    }

    const std::string model_id  = llm_arch_name(model.arch);
    const std::string kv_params = ggsd_kv_params(model, kv_attn);

    const size_t n_seg = n_prompt_tokens / GGSD_SEGMENT_TOKENS;
    const std::vector<std::string> hashes = ggsd_hash_chain(model_id, kv_params, prompt_tokens, n_seg);

    // differential load is not supported for hybrid (rec state pins coverage);
    // always full replay. n_prefix_valid is ignored by design.
    size_t n_seg_ok = 0;
    for (; n_seg_ok < n_seg; ++n_seg_ok) {
        if (!std::filesystem::exists(ggsd_segment_path(session_path, hashes[n_seg_ok]))) {
            break;
        }
    }

    // best rec candidate: header chain_hash must equal the recomputed prefix
    // hash of the request, and segments + tail must exactly tile the coverage
    size_t best_tokens = 0;
    std::filesystem::path best_path;
    ggsd_rec_header best_h;
    for (const auto & entry : std::filesystem::directory_iterator(
            std::filesystem::path(session_path).parent_path())) {
        const std::string name = entry.path().filename().string();
        if (name.rfind("rec_", 0) != 0 || name.find(".bin") == std::string::npos) {
            continue;
        }
        try {
            llama_file f(entry.path().string().c_str(), "rb");
            ggsd_rec_header h;
            if (!ggsd_rec_read_header(f, h)) {
                continue;
            }
            if (h.model_id != model_id || h.kv_params != kv_params) {
                continue;
            }
            if (h.n_tokens < min_prefix_tokens || h.n_tokens > n_prompt_tokens) {
                continue;
            }
            if (h.n_tokens <= best_tokens) {
                continue;
            }
            if (h.chain_hash != ggsd_prefix_hash(model_id, kv_params, prompt_tokens, h.n_tokens)) {
                continue;
            }
            // the tail must start exactly where the on-disk segments end
            if (h.n_tail != h.n_tokens - n_seg_ok * GGSD_SEGMENT_TOKENS) {
                continue;
            }
            best_tokens = h.n_tokens;
            best_path   = entry.path();
            best_h      = std::move(h);
        } catch (...) {
            continue;
        }
    }

    if (best_tokens == 0) {
        return 0;
    }

    // full replay
    kv_attn->seq_rm(seq_id, -1, -1);
    mem_recr->seq_rm(seq_id, -1, -1);

    size_t n_loaded = 0;
    try {
        for (size_t k = 0; k < n_seg_ok; ++k) {
            // [segment open + header/model/hash verification + payload read:
            //  identical body to standard load lines 507-558, extracted into
            //  ggsd_read_segment(session_path, seq_id, k, hashes, model_id,
            //  kv_params, kv_attn) -> bool; do not duplicate]
            ...
            ++n_loaded;
        }

        llama_file file(best_path.string().c_str(), "rb");
        ggsd_rec_header h;
        if (!ggsd_rec_read_header(file, h)) {
            throw std::runtime_error("rec header changed on disk");
        }

        std::vector<uint8_t> payload(h.payload_size);
        file.read_raw(payload.data(), payload.size());

        uint8_t digest[32];
        {
            llama_sha256_t sha;
            llama_sha256_init(&sha);
            llama_sha256_update(&sha, payload.data(), payload.size());
            llama_sha256_final(&sha, digest);
        }
        if (memcmp(digest, h.payload_hash, GGSD_HASH_BYTES) != 0) {
            LLAMA_LOG_ERROR("%s: rec file %s is corrupt (payload hash mismatch)\n",
                    __func__, best_path.string().c_str());
            throw std::runtime_error("rec payload hash mismatch");
        }

        // one stream, same order as save: attn tail then recurrent state
        ggsd_io_read_host io(payload.data(), payload.size());
        if (h.n_tail > 0) {
            kv_attn->state_read_append(io, seq_id);
        }
        mem_recr->state_read(io, seq_id);
    } catch (...) {
        // undo the partial restore
        kv_attn->seq_rm(seq_id, -1, -1);
        mem_recr->seq_rm(seq_id, -1, -1);
        throw;
    }

    LLAMA_LOG_INFO("%s: restored %zu tokens (%zu segments + %u tail tokens + rec state)\n",
            __func__, best_tokens, n_loaded, best_h.n_tail);

    return best_tokens;
}
```

Implementation notes:
- Extract the standard segment-read body (lines 507-558) into `ggsd_read_segment(...)` shared by both load paths (same DRY rule as save).
- `best_h` tokens are not re-verified against the request beyond the chain hash (the hash IS the verification).
- The min-prefix check uses `min_prefix_tokens` (server passes `prompt_cache_ssd_min_prefix`).

- [ ] **Step 3: Build**

Run: `cmake --build build-linux --target test-save-load-state llama-server -j16 2>&1 | tail -3`
Expected: build succeeds.

- [ ] **Step 4: Regression**

Run: `cd build-linux && ./bin/test-save-load-state -m tinyllamas/stories15M-q4_0.gguf 2>&1 | grep -E "PASS|FAIL" | head -12`
Expected: 10 PASS.

- [ ] **Step 5: Commit (after user approval)**

```bash
git add src/llama-state-incr.cpp src/llama-context.h
git commit -m "ggsd : hybrid split-mode load (segment replay + rec/tail restore)

Assisted-by: <assistant name>"
```

---

### Task 5: Hybrid split-mode estimate

**Files:**
- Modify: `src/llama-state-incr.cpp` (`state_seq_load_incr_estimate`), `src/llama-context.h` (declaration of `state_seq_estimate_hybrid` or inline const helper)

**Interfaces:**
- Consumes: Task 2 helpers; `ggsd_kv_params` needs a `llama_kv_cache*` - in the const estimate path get it via `dynamic_cast<const llama_memory_hybrid *>(memory.get())->get_mem_attn()` (returns `llama_kv_cache *`, callable on const hybrid since the accessor is const).
- Produces: estimate returns best rec-file coverage for hybrid models; server arbitration (`autoload_ggsd`) works unchanged.

- [ ] **Step 1: Extend the estimate entry point**

```cpp
size_t llama_context::state_seq_load_incr_estimate(
        const char * session_path,
              llama_seq_id   seq_id,
        const llama_token * prompt_tokens,
              size_t   n_prompt_tokens,
              size_t   min_prefix_tokens) const {
    if (auto * mem = dynamic_cast<const llama_memory_hybrid *>(memory.get())) {
        return state_seq_estimate_hybrid(session_path, prompt_tokens, n_prompt_tokens, min_prefix_tokens, *mem);
    }

    auto * kv = dynamic_cast<const llama_kv_cache *>(memory.get());
    if (kv == nullptr || model.hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        return 0;
    }
    // [existing standard estimate body unchanged]
}
```

Note: the standard estimate keeps its early return `n_prompt_tokens < GGSD_SEGMENT_TOKENS` at the very top (applies to both paths).

- [ ] **Step 2: Implement the hybrid estimate (const, headers only)**

```cpp
size_t llama_context::state_seq_estimate_hybrid(
        const char * session_path,
        const llama_token * prompt_tokens,
              size_t   n_prompt_tokens,
              size_t   min_prefix_tokens,
              const llama_memory_hybrid & mem) const {
    if (n_prompt_tokens < GGSD_SEGMENT_TOKENS || model.hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        return 0;
    }

    const std::string model_id  = llm_arch_name(model.arch);
    const std::string kv_params = ggsd_kv_params(model, mem.get_mem_attn());

    size_t best_tokens = 0;
    for (const auto & entry : std::filesystem::directory_iterator(
            std::filesystem::path(session_path).parent_path())) {
        const std::string name = entry.path().filename().string();
        if (name.rfind("rec_", 0) != 0 || name.find(".bin") == std::string::npos) {
            continue;
        }
        try {
            llama_file f(entry.path().string().c_str(), "rb");
            ggsd_rec_header h;
            if (!ggsd_rec_read_header(f, h)) {
                continue;
            }
            if (h.model_id != model_id || h.kv_params != kv_params) {
                continue;
            }
            if (h.n_tokens < min_prefix_tokens || h.n_tokens > n_prompt_tokens || h.n_tokens <= best_tokens) {
                continue;
            }
            if (h.chain_hash != ggsd_prefix_hash(model_id, kv_params, prompt_tokens, h.n_tokens)) {
                continue;
            }
            best_tokens = h.n_tokens;
        } catch (...) {
            continue;
        }
    }

    return best_tokens;
}
```

(The tail/segment tiling check is deliberately omitted here - Task 4 re-verifies it before committing to a restore; estimate only prices candidates.)

- [ ] **Step 3: Build + regression**

Run: `cmake --build build-linux --target test-save-load-state llama-server -j16 2>&1 | tail -3 && cd build-linux && ./bin/test-save-load-state -m tinyllamas/stories15M-q4_0.gguf 2>&1 | grep -cE "=== Test"`
Expected: build ok, 10.

- [ ] **Step 4: Commit (after user approval)**

```bash
git add src/llama-state-incr.cpp src/llama-context.h
git commit -m "ggsd : hybrid split-mode restore estimate

Assisted-by: <assistant name>"
```

---

### Task 6: Documentation update

**Files:**
- Modify: `docs/ggsd-guide.md`, `docs/ggsd-autoload-guide.md`
- Reference: `docs/superpowers/specs/2026-07-31-ggsd-incremental-slot-storage-design.md` (the "only standard kv cache" wording lives wherever this limitation is stated - update it)

- [ ] **Step 1: Update the support matrix**

In both guides, replace the "standard kv cache only" limitation statement with the dispatch table: `llama_kv_cache` -> segment mode; `llama_memory_hybrid` (Qwen3.5 family) -> split mode (shared attn segments + per-conversation `rec_*.bin`); everything else still rejected. State the hybrid matching rule explicitly: restore coverage = the rec file's `n_tokens` (a request must extend a previously saved conversation exactly), fork conversations share the aligned attn segments on disk, per-conversation cost = rec state + non-aligned tail. Note the dominated-parent cleanup (each chain keeps only its latest rec file) and that `--swa-full` is not involved (SWA hybrids are rejected).

- [ ] **Step 2: Commit (after user approval)**

```bash
git add docs/ggsd-guide.md docs/ggsd-autoload-guide.md docs/superpowers/specs/2026-07-31-ggsd-incremental-slot-storage-design.md
git commit -m "docs : ggsd hybrid split mode

Assisted-by: <assistant name>"
```

---

### Task 7: Smoke test with Qwen3.5-9B (behavioral verification)

**Files:**
- None (verification only). Uses `start-server.sh`, model `/models/unslloth/Qwen3.5-9B-UD-Q4_K_XL.gguf`.

- [ ] **Step 1: Restart the GGSD-enabled server**

Run: `pkill -f llama-server; sleep 2; ./start-server.sh &` then poll `curl -s http://127.0.0.1:8080/health` until ok.
Expected: startup log shows `--prompt-cache-ssd` active and NO "incremental state save is only supported for the standard kv cache" warning.

- [ ] **Step 2: First completion triggers hybrid autosave**

```bash
curl -s http://127.0.0.1:8080/completion -d '{"prompt": "The little dog ran. ", "n_predict": 64}' | jq -r .content
ls -la saves/ | grep -E "seg_|rec_"
```
Expected: text output; at least one `seg_*.bin` (if the prompt reached 1024 tokens; with a short prompt, only `rec_*.bin` appears - both are valid outcomes) and one `rec_*.bin`.

- [ ] **Step 3: Continuation restores from SSD (zero prefill)**

```bash
curl -s http://127.0.0.1:8080/completion -d '{"prompt": "<previous prompt + generated text> and then", "n_predict": 32}' | jq -r .tokens_evaluated
grep "restored" server log
```
Expected: log line `ggsd hybrid ... restored N tokens (... + rec state)`; `tokens_evaluated` small (only the suffix). If the exact-continuation prompt is impractical to construct, verify via a second identical completion request instead: identical prompt+seed -> identical output and `restored N tokens` in the log.

- [ ] **Step 4: Restart persistence**

```bash
pkill -f llama-server; sleep 2; ./start-server.sh &
curl -s http://127.0.0.1:8080/completion -d '<same request as Step 3>'
```
Expected: after restart (RAM prompt cache empty), the log still shows the GGSD restore - this is the SSD closed loop.

- [ ] **Step 5: Fork sharing check**

```bash
curl -s http://127.0.0.1:8080/completion -d '<prefix-shared prompt with different suffix>'; ls saves/ | grep -c rec_
```
Expected: the second conversation's rec file appears while `seg_*` count does not grow for the shared prefix (fork shares segments).

- [ ] **Step 6: Full regression suite**

Run: `cd build-linux && ./bin/test-save-load-state -m tinyllamas/stories15M-q4_0.gguf 2>&1 | grep -E "PASS|FAIL" | head -12`
Expected: 10 PASS. No commits pending without approval.
