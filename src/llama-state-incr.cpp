#include "llama.h"

#include "llama-arch.h"
#include "llama-context.h"
#include "llama-impl.h"
#include "llama-io.h"
#include "llama-kv-cache.h"
#include "llama-mmap.h"
#include "llama-model.h"
#include "llama-sha256.h"

#include "ggml.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <vector>

//
// GGSD - incremental sequence state save/load
//
// see: docs/superpowers/specs/2026-07-31-ggsd-incremental-slot-storage-design.md
//

namespace {
    constexpr char GGSD_MAGIC[4] = { 'G', 'G', 'S', 'D' };

    constexpr uint32_t GGSD_VERSION                = 1;
    constexpr uint32_t GGSD_SEGMENT_TOKENS         = 1024;
    constexpr size_t   GGSD_HASH_BYTES             = 16; // truncated sha256
    constexpr size_t   GGSD_HASH_HEX_LEN           = 32;

    // segment header size before the payload hash field
    constexpr size_t GGSD_HEADER_PAYLOAD_HASH_OFFSET = 4 + 4 + 4 + 32 + 4 + 8;

    std::string ggsd_segment_path(const std::string & session_path, const std::string & hash) {
        std::filesystem::path p(session_path);
        return (p.parent_path() / ("seg_" + hash + ".bin")).string();
    }

    void ggsd_hash_hex_to_bytes(const std::string & hex, uint8_t * bytes) {
        for (size_t i = 0; i < GGSD_HASH_BYTES; ++i) {
            auto nibble = [](char c) -> uint8_t {
                if (c >= '0' && c <= '9') {
                    return c - '0';
                }
                return c - 'a' + 10;
            };
            bytes[i] = (nibble(hex[2*i]) << 4) | nibble(hex[2*i + 1]);
        }
    }

    std::string ggsd_segment_hash(
            const std::string & model_id,
            const std::string & kv_params,
            const uint8_t * prev_hash, // GGSD_HASH_BYTES, nullptr for chain head
            const llama_token * tokens) {
        llama_sha256_t sha;
        llama_sha256_init(&sha);

        const uint32_t len_model = (uint32_t) model_id.size();
        llama_sha256_update(&sha, (const uint8_t *) &len_model, sizeof(len_model));
        llama_sha256_update(&sha, (const uint8_t *) model_id.data(), len_model);

        const uint32_t len_params = (uint32_t) kv_params.size();
        llama_sha256_update(&sha, (const uint8_t *) &len_params, sizeof(len_params));
        llama_sha256_update(&sha, (const uint8_t *) kv_params.data(), len_params);

        if (prev_hash != nullptr) {
            llama_sha256_update(&sha, prev_hash, GGSD_HASH_BYTES);
        }

        llama_sha256_update(&sha, (const uint8_t *) tokens, GGSD_SEGMENT_TOKENS * sizeof(llama_token));

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

    void ggsd_write_str(llama_file & file, const std::string & str) {
        const uint32_t len = (uint32_t) str.size();
        file.write_raw(&len, sizeof(len));
        file.write_raw(str.data(), len);
    }

    std::string ggsd_read_str(llama_file & file) {
        uint32_t len;
        file.read_raw(&len, sizeof(len));
        std::string str(len, '\0');
        if (len > 0) {
            file.read_raw(&str[0], len);
        }
        return str;
    }

    // file IO that hashes the bytes as they are written
    class ggsd_io_write_file : public llama_io_write_i {
    public:
        ggsd_io_write_file(llama_file * f, llama_sha256_t * sha) : file(f), sha(sha) {}

        void write(const void * src, size_t size) override {
            file->write_raw(src, size);
            llama_sha256_update(sha, (const uint8_t *) src, size);
            size_written += size;
        }

        void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
            temp_buffer.resize(size);
            ggml_backend_tensor_get(tensor, temp_buffer.data(), offset, size);
            write(temp_buffer.data(), temp_buffer.size());
        }

        size_t n_bytes() override {
            return size_written;
        }

    private:
        llama_file * file;
        llama_sha256_t * sha;
        size_t size_written = 0;
        std::vector<uint8_t> temp_buffer;
    };

    // dummy IO for measuring the payload size
    class ggsd_io_write_dummy : public llama_io_write_i {
    public:
        void write(const void * /* src */, size_t size) override {
            size_written += size;
        }

        void write_tensor(ggml_tensor * /* tensor */, size_t /* offset */, size_t size) override {
            size_written += size;
        }

        size_t n_bytes() override {
            return size_written;
        }

    private:
        size_t size_written = 0;
    };

    // host memory IO for the payload, tensors are flushed on destruction
    class ggsd_io_read_host : public llama_io_read_i {
    public:
        ggsd_io_read_host(const uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

        ~ggsd_io_read_host() {
            for (const auto & rinfo : rinfos) {
                ggml_backend_tensor_set(rinfo.tensor, rinfo.ptr, rinfo.offset, rinfo.size);
            }
        }

        void read(void * dst, size_t size) override {
            if (size > buf_size) {
                throw std::runtime_error("unexpectedly reached end of buffer");
            }
            memcpy(dst, ptr, size);
            ptr += size;
            size_read += size;
            buf_size -= size;
        }

        void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
            if (size > buf_size) {
                throw std::runtime_error("unexpectedly reached end of buffer");
            }

            // save for later during destruction
            rinfos.push_back({tensor, ptr, size, offset});

            ptr += size;
            size_read += size;
            buf_size -= size;
        }

        size_t n_bytes() override {
            return size_read;
        }

    private:
        const uint8_t * ptr;
        size_t buf_size = 0;
        size_t size_read = 0;

        struct read_info {
            ggml_tensor * tensor;
            const uint8_t * ptr;
            size_t size;
            size_t offset;
        };
        std::vector<read_info> rinfos;
    };
} // namespace

size_t llama_context::state_seq_save_incr(
        const char * session_path,
              llama_seq_id   seq_id,
        const llama_token * tokens,
              size_t   n_token_count) {
    auto * kv = dynamic_cast<llama_kv_cache *>(memory.get());
    if (kv == nullptr) {
        LLAMA_LOG_ERROR("%s: incremental state save is only supported for the standard kv cache\n", __func__);
        return (size_t) -1;
    }

    const size_t n_seg = n_token_count / GGSD_SEGMENT_TOKENS;
    if (n_seg == 0) {
        return 0;
    }

    if (model.hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        LLAMA_LOG_ERROR("%s: incremental state save does not support SWA KV caches\n", __func__);
        return (size_t) -1;
    }

    const uint32_t n_pos_per_embd = model.hparams.n_pos_per_embd();

    if (n_pos_per_embd != 1) {
        LLAMA_LOG_ERROR("%s: incremental state save requires n_pos_per_embd == 1 (got %u)\n", __func__, n_pos_per_embd);
        return (size_t) -1;
    }

    // model identity: arch name + kv cache parameters, both part of the segment hash
    const std::string model_id = llm_arch_name(model.arch);
    const std::string kv_params = std::string(ggml_type_name(kv->type_k())) + "|" +
                                  std::string(ggml_type_name(kv->type_v())) + "|" +
                                  std::to_string(n_pos_per_embd);

    // read the session file (save-side hint only: last known chain length);
    // a corrupt or missing session simply starts a new chain. Restore does not
    // depend on this file - see the segment-pool semantics in the spec.
    uint32_t n_segments = 0;
    std::string tail_hash_read;
    if (std::filesystem::exists(session_path)) {
        bool ok = true;
        try {
            llama_file file(session_path, "rb");

            char magic[4];
            file.read_raw(magic, 4);

            const uint32_t version = file.read_u32();

            char tail_hash[GGSD_HASH_HEX_LEN];
            file.read_raw(tail_hash, sizeof(tail_hash));

            n_segments = file.read_u32();

            if (memcmp(magic, GGSD_MAGIC, 4) != 0 || version != GGSD_VERSION) {
                ok = false;
            } else {
                tail_hash_read = std::string(tail_hash, sizeof(tail_hash));
            }
        } catch (const std::exception & e) {
            ok = false;
        }

        if (!ok) {
            LLAMA_LOG_WARN("%s: invalid session file %s, starting a new chain\n", __func__, session_path);
            n_segments = 0;
        }
    }

    // hash chain of the prompt
    const std::vector<std::string> hashes = ggsd_hash_chain(model_id, kv_params, tokens, n_seg);

    // find the fork point: the first segment missing on disk. The KV cache is
    // deliberately NOT consulted here - already-persisted segments stay valid
    // and reachable even after the corresponding cells are evicted (R2).
    size_t k0 = 0;
    for (; k0 < std::min<size_t>(n_segments, n_seg); ++k0) {
        if (!std::filesystem::exists(ggsd_segment_path(session_path, hashes[k0]))) {
            break;
        }
    }

    // position guard: only when the head segment must be written. If all
    // segments are already on disk (k0 == n_seg), eviction or an empty cache
    // is irrelevant - the chain is preserved with zero IO (R2/R3). Correctness
    // of written segments relies on the caller passing tokens that correspond
    // to positions 0..n-1 of the sequence's cells.
    if (k0 < n_seg && kv->seq_pos_min(seq_id) != 0) {
        LLAMA_LOG_ERROR("%s: cannot write the head segment: sequence positions do not start at 0 (min pos = %d)\n",
                __func__, (int) kv->seq_pos_min(seq_id));
        return (size_t) -1;
    }

    // write new segments starting at the fork point
    size_t n_written = 0;
    for (size_t k = k0; k < n_seg; ++k) {
        const llama_pos pos_begin = (llama_pos) ( k       * GGSD_SEGMENT_TOKENS * n_pos_per_embd);
        const llama_pos pos_end   = (llama_pos) ((k + 1) * GGSD_SEGMENT_TOKENS * n_pos_per_embd);

        if (kv->count_cells_range(seq_id, pos_begin, pos_end) != GGSD_SEGMENT_TOKENS) {
            // partial KV cache: stop writing at the contiguous prefix
            break;
        }

        const std::string fname = ggsd_segment_path(session_path, hashes[k]);

        if (std::filesystem::exists(fname)) {
            // already persisted by a previous save (possibly from another
            // sequence computing the same chain); content addressing makes the
            // data equivalent - see the FP nondeterminism note in the spec
            continue;
        }

        // measure the payload size
        ggsd_io_write_dummy io_dummy;
        kv->state_write_range(io_dummy, seq_id, pos_begin, pos_end);
        const uint64_t payload_size = io_dummy.n_bytes();

        const std::string fname_tmp = fname + ".tmp";

        {
            llama_file file(fname_tmp.c_str(), "wb");

            file.write_raw(GGSD_MAGIC, sizeof(GGSD_MAGIC));
            file.write_u32(GGSD_VERSION);
            file.write_u32((uint32_t) k);

            char prev[GGSD_HASH_HEX_LEN];
            if (k == 0) {
                memset(prev, 0, sizeof(prev));
            } else {
                memcpy(prev, hashes[k - 1].c_str(), sizeof(prev));
            }
            file.write_raw(prev, sizeof(prev));

            file.write_u32(GGSD_SEGMENT_TOKENS);
            file.write_raw(&payload_size, sizeof(payload_size));

            uint8_t digest[32]; // full sha256; the header stores the first GGSD_HASH_BYTES
            memset(digest, 0, sizeof(digest));
            file.write_raw(digest, GGSD_HASH_BYTES); // patched below

            ggsd_write_str(file, model_id);
            ggsd_write_str(file, kv_params);
            file.write_raw(tokens + k * GGSD_SEGMENT_TOKENS, GGSD_SEGMENT_TOKENS * sizeof(llama_token));

            llama_sha256_t sha;
            llama_sha256_init(&sha);
            {
                ggsd_io_write_file io(&file, &sha);
                kv->state_write_range(io, seq_id, pos_begin, pos_end);
            }
            llama_sha256_final(&sha, digest);

            // patch the truncated payload hash into the header
            file.seek(GGSD_HEADER_PAYLOAD_HASH_OFFSET, SEEK_SET);
            file.write_raw(digest, GGSD_HASH_BYTES);
        }

        std::filesystem::rename(fname_tmp, fname);
        ++n_written;
    }

    // final chain length: the longest prefix of the hash chain present on disk.
    // This is the authoritative value - it counts persisted segments regardless
    // of whether their KV cells are still in the cache (R2), and it stops at a
    // missing file even when the write loop was cut short by evicted cells.
    size_t n_chain = 0;
    for (; n_chain < n_seg; ++n_chain) {
        if (!std::filesystem::exists(ggsd_segment_path(session_path, hashes[n_chain]))) {
            break;
        }
    }

    if (n_chain == 0) {
        LLAMA_LOG_ERROR("%s: KV cache is missing the head of the sequence (pos [0, %d)) and no head segment exists, nothing saved\n",
                __func__, (int) (GGSD_SEGMENT_TOKENS * n_pos_per_embd));
        return (size_t) -1;
    }

    LLAMA_LOG_INFO("%s: wrote %zu new segment(s), session chain covers %zu segment(s)\n",
            __func__, n_written, n_chain);

    // rewrite the session file when the chain changed (length or tail hash -
    // an equal-length re-fork keeps n_chain but changes the tail)
    if ((uint32_t) n_chain != n_segments || hashes[n_chain - 1] != tail_hash_read) {
        llama_file file(session_path, "wb");

        file.write_raw(GGSD_MAGIC, sizeof(GGSD_MAGIC));
        file.write_u32(GGSD_VERSION);
        file.write_raw(hashes[n_chain - 1].c_str(), hashes[n_chain - 1].size());
        file.write_u32((uint32_t) n_chain);
    }

    return n_chain;
}

size_t llama_context::state_seq_load_incr(
        const char * session_path,
              llama_seq_id   seq_id,
        const llama_token * prompt_tokens,
              size_t   n_prompt_tokens,
              size_t   min_prefix_tokens,
              size_t   n_prefix_valid) {
    if (n_prompt_tokens < GGSD_SEGMENT_TOKENS) {
        return 0;
    }

    auto * kv = dynamic_cast<llama_kv_cache *>(memory.get());
    if (kv == nullptr) {
        LLAMA_LOG_ERROR("%s: incremental state load is only supported for the standard kv cache\n", __func__);
        return 0;
    }

    if (model.hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        LLAMA_LOG_ERROR("%s: incremental state load does not support SWA KV caches\n", __func__);
        return 0;
    }

    if (model.hparams.n_pos_per_embd() != 1) {
        LLAMA_LOG_ERROR("%s: incremental state load requires n_pos_per_embd == 1 (got %u)\n",
                __func__, model.hparams.n_pos_per_embd());
        return 0;
    }

    // segment-pool semantics: the session file is not consulted. The prompt's
    // hash chain is matched directly against content-addressed segment files in
    // the directory of session_path, so prefixes persisted by any session (and
    // any sequence) are reusable (G2 / remediation R1).

    const uint32_t n_pos_per_embd = model.hparams.n_pos_per_embd();

    const std::string model_id = llm_arch_name(model.arch);
    const std::string kv_params = std::string(ggml_type_name(kv->type_k())) + "|" +
                                  std::string(ggml_type_name(kv->type_v())) + "|" +
                                  std::to_string(n_pos_per_embd);

    const size_t n_seg = n_prompt_tokens / GGSD_SEGMENT_TOKENS;

    // hash chain of the prompt prefix
    const std::vector<std::string> hashes = ggsd_hash_chain(model_id, kv_params, prompt_tokens, n_seg);

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

    // find the longest chain of segment files matching the prompt
    size_t n_seg_ok = k0;
    for (; n_seg_ok < n_seg; ++n_seg_ok) {
        if (!std::filesystem::exists(ggsd_segment_path(session_path, hashes[n_seg_ok]))) {
            break;
        }
    }

    const size_t n_restored = n_seg_ok * GGSD_SEGMENT_TOKENS;
    if (n_restored < min_prefix_tokens) {
        return 0;
    }

    // full replay clears the sequence itself; differential mode drops any
    // cells beyond the aligned boundary so the replayed segments do not
    // duplicate them (callers that already truncated are unaffected)
    if (k0 == 0) {
        kv->seq_rm(seq_id, -1, -1);
    } else {
        kv->seq_rm(seq_id, (llama_pos) (k0 * GGSD_SEGMENT_TOKENS), -1);
    }

    size_t n_loaded = 0;
    try {
        for (size_t k = k0; k < n_seg_ok; ++k) {
            const std::string fname = ggsd_segment_path(session_path, hashes[k]);

            llama_file file(fname.c_str(), "rb");

            char magic[4];
            file.read_raw(magic, 4);

            const uint32_t version   = file.read_u32();
            const uint32_t seg_index = file.read_u32();

            char prev[GGSD_HASH_HEX_LEN];
            file.read_raw(prev, sizeof(prev));

            const uint32_t n_tokens = file.read_u32();

            uint64_t payload_size;
            file.read_raw(&payload_size, sizeof(payload_size));

            uint8_t payload_hash[GGSD_HASH_BYTES];
            file.read_raw(payload_hash, sizeof(payload_hash));

            if (memcmp(magic, GGSD_MAGIC, 4) != 0 || version != GGSD_VERSION ||
                    seg_index != k || n_tokens != GGSD_SEGMENT_TOKENS) {
                LLAMA_LOG_ERROR("%s: invalid segment file %s (k = %zu)\n", __func__, fname.c_str(), k);
                break;
            }

            const std::string f_model_id = ggsd_read_str(file);
            const std::string f_kv_params = ggsd_read_str(file);
            if (f_model_id != model_id || f_kv_params != kv_params) {
                LLAMA_LOG_ERROR("%s: segment file %s was created with a different model\n", __func__, fname.c_str());
                break;
            }

            // skip the tokens; the file name is the content hash of the tokens
            file.seek(GGSD_SEGMENT_TOKENS * sizeof(llama_token), SEEK_CUR);

            std::vector<uint8_t> payload(payload_size);
            file.read_raw(payload.data(), payload.size());

            // verify the payload hash
            uint8_t digest[32];
            {
                llama_sha256_t sha;
                llama_sha256_init(&sha);
                llama_sha256_update(&sha, payload.data(), payload.size());
                llama_sha256_final(&sha, digest);
            }
            if (memcmp(digest, payload_hash, GGSD_HASH_BYTES) != 0) {
                LLAMA_LOG_ERROR("%s: segment file %s is corrupt (payload hash mismatch)\n", __func__, fname.c_str());
                break;
            }

            ggsd_io_read_host io(payload.data(), payload.size());
            kv->state_read_append(io, seq_id);

            ++n_loaded;
        }
    } catch (...) {
        // undo the partial restore
        kv->seq_rm(seq_id, -1, -1);
        throw;
    }

    // stop-at-first-corrupt semantics: segments verified and replayed before
    // the failure are kept - the return value always matches the cache state
    // (review M1). Only an exception (IO failure) unwinds via the catch above.

    return (n_loaded + k0) * GGSD_SEGMENT_TOKENS;
}

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

//
// C API
//

int32_t llama_state_seq_save_incr(
        llama_context * ctx,
        const char * session_path,
        llama_seq_id seq_id,
        const llama_token * tokens,
        size_t n_token_count) {
    ctx->synchronize();

    try {
        const size_t res = ctx->state_seq_save_incr(session_path, seq_id, tokens, n_token_count);
        if (res == (size_t) -1) {
            LLAMA_LOG_ERROR("%s: failed to save incremental sequence state\n", __func__);
            return -1;
        }
        return (int32_t) res;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving incremental sequence state: %s\n", __func__, err.what());
        return -1;
    }
}

size_t llama_state_seq_load_incr(
        llama_context * ctx,
        const char * session_path,
        llama_seq_id seq_id,
        const llama_token * prompt_tokens,
        size_t n_prompt_tokens,
        size_t min_prefix_tokens,
        size_t n_prefix_valid) {
    ctx->synchronize();

    try {
        return ctx->state_seq_load_incr(session_path, seq_id, prompt_tokens, n_prompt_tokens, min_prefix_tokens, n_prefix_valid);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading incremental sequence state: %s\n", __func__, err.what());
        return 0;
    }
}
