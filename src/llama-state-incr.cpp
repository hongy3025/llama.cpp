#include "llama-ggsd-cache.h"
#include "llama.h"

#include "llama-arch.h"
#include "llama-context.h"
#include "llama-impl.h"
#include "llama-io.h"
#include "llama-kv-cache.h"
#include "llama-memory-hybrid.h"
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

    // content-addressed pool layout: <pool>/seg/<hh>/<hash> and <pool>/rec/<hh>/<hash>,
    // where <hh> is the first two hex chars of the 32-char hash (256 shards)
    std::string ggsd_segment_path(const std::string & session_path, const std::string & hash) {
        std::filesystem::path p(session_path);
        return (p.parent_path() / "seg" / hash.substr(0, 2) / hash).string();
    }

    void ggsd_ensure_parent_dir(const std::string & fname) {
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(fname).parent_path(), ec);
    }

    // true for <hash> content files: exactly 32 lowercase hex chars
    bool ggsd_is_hash_name(const std::string & name) {
        if (name.size() != GGSD_HASH_HEX_LEN) {
            return false;
        }
        for (const char c : name) {
            if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) {
                return false;
            }
        }
        return true;
    }

    // list content files under pool_dir/<kind>/<shard>/, skipping junk
    std::vector<std::filesystem::path> ggsd_list_pool_files(const std::filesystem::path & pool_dir, const char * kind) {
        std::vector<std::filesystem::path> out;
        std::error_code ec;
        const std::filesystem::path root = pool_dir / kind;
        if (!std::filesystem::exists(root, ec)) {
            return out;
        }
        for (const auto & shard : std::filesystem::directory_iterator(root, ec)) {
            if (!shard.is_directory()) {
                continue;
            }
            for (const auto & entry : std::filesystem::directory_iterator(shard.path(), ec)) {
                if (entry.is_regular_file() && ggsd_is_hash_name(entry.path().filename().string())) {
                    out.push_back(entry.path());
                }
            }
        }
        return out;
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

    // identity string of the kv cache configuration, shared by all entry points
    std::string ggsd_kv_params(const llama_model & model, const llama_kv_cache * kv) {
        return std::string(ggml_type_name(kv->type_k())) + "|" +
               std::string(ggml_type_name(kv->type_v())) + "|" +
               std::to_string(model.hparams.n_pos_per_embd()) + "|" +
               std::to_string(model.hparams.n_layer());
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

    constexpr char GGSD_REC_MAGIC[4] = { 'G', 'G', 'S', 'R' };

    // rec header size before the payload hash field
    constexpr size_t GGSD_REC_HEADER_PAYLOAD_HASH_OFFSET = 4 + 4 + 4 + 4 + 32 + 8;

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
        return (p.parent_path() / "rec" / hash.substr(0, 2) / hash).string();
    }


    // reads magic, version, n_tokens, n_tail, chain_hash, payload_size, payload_hash,
    // model_id, kv_params; the token array is read only when read_tokens is set.
    bool ggsd_rec_read_header(llama_file & file, ggsd_rec_header & out, bool read_tokens = true) {
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

            // bound the header-derived allocation by the file size
            if (read_tokens) {
                if ((size_t) out.n_tokens * sizeof(llama_token) > file.size()) {
                    return false;
                }
                out.tokens.resize(out.n_tokens);
                if (out.n_tokens > 0) {
                    file.read_raw(out.tokens.data(), out.n_tokens * sizeof(llama_token));
                }
            }
        } catch (...) {
            return false;
        }
        return true;
    }

    // write one aligned segment file (.tmp + rename), shared by the standard
    // and hybrid save paths
    bool ggsd_write_segment(
            llama_kv_cache * kv,
            const std::string & session_path,
            llama_seq_id seq_id,
            size_t k,
            const std::vector<std::string> & hashes,
            const std::string & model_id,
            const std::string & kv_params,
            const llama_token * tokens,
            llama_pos pos_begin,
            llama_pos pos_end,
            size_t & n_written,
            llama_ggsd::cache * cache,
            llama_ggsd::cache_reservation * group_reservation = nullptr) {
        ggsd_io_write_dummy io_dummy;
        kv->state_write_range(io_dummy, seq_id, pos_begin, pos_end);
        const uint64_t payload_size = io_dummy.n_bytes();
        uint64_t file_size = 0;
        if (cache && !llama_ggsd::segment_serialized_size(model_id.size(), kv_params.size(), payload_size, file_size)) {
            return false;
        }
        const std::string fname = ggsd_segment_path(session_path, hashes[k]);
        const std::string fname_tmp = fname + ".tmp";
        std::optional<llama_ggsd::cache_reservation> reservation;
        if (group_reservation) {
            if (group_reservation->remaining() < file_size) return false;
        } else if (cache) {
            std::vector<llama_ggsd::object_id> protect;
            if (k > 0) protect.push_back({llama_ggsd::object_kind::segment, hashes[k - 1]});
            reservation = cache->reserve(file_size, protect);
            if (!reservation) return false;
        }
        try {
            ggsd_ensure_parent_dir(fname);
            {
                llama_file file(fname_tmp.c_str(), "wb");
                file.write_raw(GGSD_MAGIC, sizeof(GGSD_MAGIC));
                file.write_u32(GGSD_VERSION);
                file.write_u32((uint32_t) k);
                char prev[GGSD_HASH_HEX_LEN];
                if (k == 0) memset(prev, 0, sizeof(prev));
                else memcpy(prev, hashes[k - 1].c_str(), sizeof(prev));
                file.write_raw(prev, sizeof(prev));
                file.write_u32(GGSD_SEGMENT_TOKENS);
                file.write_raw(&payload_size, sizeof(payload_size));
                uint8_t digest[32] = {};
                file.write_raw(digest, GGSD_HASH_BYTES);
                ggsd_write_str(file, model_id);
                ggsd_write_str(file, kv_params);
                file.write_raw(tokens + k * GGSD_SEGMENT_TOKENS, GGSD_SEGMENT_TOKENS * sizeof(llama_token));
                llama_sha256_t sha; llama_sha256_init(&sha);
                { ggsd_io_write_file io(&file, &sha); kv->state_write_range(io, seq_id, pos_begin, pos_end); }
                llama_sha256_final(&sha, digest);
                file.seek(GGSD_HEADER_PAYLOAD_HASH_OFFSET, SEEK_SET);
                file.write_raw(digest, GGSD_HASH_BYTES);
            }
            if (group_reservation) {
                std::string error;
                if (!group_reservation->finalize_temporary(llama_ggsd::object_kind::segment, fname_tmp, fname, file_size, error)) {
                    group_reservation->abandon_temporary(fname_tmp, file_size);
                    return false;
                }
            } else if (reservation) {
                std::string error;
                if (!reservation->finalize_temporary(llama_ggsd::object_kind::segment, fname_tmp, fname, file_size, error)) {
                    reservation->abandon_temporary(fname_tmp, file_size);
                    return false;
                }
            } else {
                std::filesystem::rename(fname_tmp, fname);
            }
            ++n_written;
            return true;
        } catch (...) {
            if (group_reservation) group_reservation->abandon_temporary(fname_tmp, file_size);
            else if (reservation) reservation->abandon_temporary(fname_tmp, file_size);
            throw;
        }
    }

    // read, verify and replay one aligned segment file; returns false when the
    // file is invalid / mismatched / corrupt (caller stops), IO errors throw
    bool ggsd_read_segment(
            llama_kv_cache * kv,
            const std::string & session_path,
            size_t k,
            const std::vector<std::string> & hashes,
            const std::string & model_id,
            const std::string & kv_params,
            llama_seq_id seq_id) {
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
            return false;
        }

        const std::string f_model_id = ggsd_read_str(file);
        const std::string f_kv_params = ggsd_read_str(file);
        if (f_model_id != model_id || f_kv_params != kv_params) {
            LLAMA_LOG_ERROR("%s: segment file %s was created with a different model\n", __func__, fname.c_str());
            return false;
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
            return false;
        }

        ggsd_io_read_host io(payload.data(), payload.size());
        kv->state_read_append(io, seq_id);

        return true;
    }
} // namespace

size_t llama_context::state_seq_save_incr(
        const char * session_path,
              llama_seq_id   seq_id,
        const llama_token * tokens,
              size_t   n_token_count) {
    if (auto * mem = dynamic_cast<llama_memory_hybrid *>(memory.get())) {
        return state_seq_save_incr_hybrid(session_path, seq_id, tokens, n_token_count, *mem);
    }

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

    // model identity: arch name + kv cache parameters, both part of the segment hash
    const std::string model_id = llm_arch_name(model.arch);
    const std::string kv_params = ggsd_kv_params(model, kv);

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
        const llama_pos pos_begin = (llama_pos) ( k       * GGSD_SEGMENT_TOKENS);
        const llama_pos pos_end   = (llama_pos) ((k + 1) * GGSD_SEGMENT_TOKENS);

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

        if (!ggsd_write_segment(kv, session_path, seq_id, k, hashes, model_id, kv_params, tokens, pos_begin, pos_end, n_written,
                ggsd_cache_for_session(session_path))) break;
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
                __func__, (int) GGSD_SEGMENT_TOKENS);
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

    if (auto * cache = ggsd_cache_for_session(session_path)) {
        cache->touch({llama_ggsd::object_kind::segment, hashes[n_chain - 1]});
    }
    return n_chain;
}

size_t llama_context::state_seq_save_incr_hybrid(
        const char * session_path,
              llama_seq_id   seq_id,
        const llama_token * tokens,
              size_t   n_token_count,
              llama_memory_hybrid & mem) {
    auto * kv_attn  = mem.get_mem_attn();
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
    auto * cache = ggsd_cache_for_session(session_path);
    const bool target_rec_exists = std::filesystem::exists(
            ggsd_rec_path(session_path, ggsd_prefix_hash(model_id, kv_params, tokens, n_token_count)));
    std::optional<llama_ggsd::cache_reservation> group_reservation;
    if (cache) {
        size_t existing = 0;
        while (existing < n_seg && std::filesystem::exists(ggsd_segment_path(session_path, hashes[existing]))) {
            ++existing;
        }
        bool measurable = true;
        uint64_t group_size = 0;
        for (size_t k = existing; k < n_seg; ++k) {
            const llama_pos begin = (llama_pos) (k * GGSD_SEGMENT_TOKENS);
            const llama_pos end = (llama_pos) ((k + 1) * GGSD_SEGMENT_TOKENS);
            if (kv_attn->count_cells_range(seq_id, begin, end) != GGSD_SEGMENT_TOKENS) {
                measurable = false;
                break;
            }
            ggsd_io_write_dummy io;
            kv_attn->state_write_range(io, seq_id, begin, end);
            uint64_t size = 0;
            if (!llama_ggsd::segment_serialized_size(model_id.size(), kv_params.size(), io.n_bytes(), size) ||
                    group_size > UINT64_MAX - size) {
                measurable = false;
                break;
            }
            group_size += size;
        }
        if (measurable && !target_rec_exists) {
            const size_t tail = n_token_count - n_seg * GGSD_SEGMENT_TOKENS;
            ggsd_io_write_dummy io;
            if (tail > 0) {
                kv_attn->state_write_range(io, seq_id, (llama_pos) (n_seg * GGSD_SEGMENT_TOKENS), (llama_pos) n_token_count);
            }
            mem_recr->state_write(io, seq_id);
            uint64_t rec_size = 0;
            if (!llama_ggsd::rec_serialized_size(model_id.size(), kv_params.size(), n_token_count, io.n_bytes(), rec_size) ||
                    group_size > UINT64_MAX - rec_size) {
                measurable = false;
            } else {
                group_size += rec_size;
            }
        }
        if (measurable) {
            std::vector<llama_ggsd::object_id> protect;
            if (existing > 0) protect.push_back({llama_ggsd::object_kind::segment, hashes[existing - 1]});
            group_reservation = cache->reserve(group_size, protect);
        }
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

        if (!ggsd_write_segment(kv_attn, session_path, seq_id, k, hashes, model_id, kv_params, tokens, pos_begin, pos_end, n_written,
                cache, group_reservation ? &*group_reservation : nullptr)) break;
    }

    // longest on-disk chain prefix
    size_t n_chain = 0;
    for (; n_chain < n_seg; ++n_chain) {
        if (!std::filesystem::exists(ggsd_segment_path(session_path, hashes[n_chain]))) {
            break;
        }
    }

    const size_t n_tail = n_token_count - n_chain * GGSD_SEGMENT_TOKENS;

    // rec file: attn tail + recurrent snapshot, coverage = n_token_count.
    // the GGSD session file is deliberately NOT touched here (pool semantics).
    // no segment-alignment floor: the rec is self-contained, so short chains
    // are usable as long as the caller's min_prefix policy allows them.
    {
        const std::string chain_hash = ggsd_prefix_hash(model_id, kv_params, tokens, n_token_count);
        const std::string fname = ggsd_rec_path(session_path, chain_hash);
        const bool rec_exists = std::filesystem::exists(fname);

        if (!rec_exists) {
            const llama_pos tail_begin = (llama_pos) (n_chain * GGSD_SEGMENT_TOKENS);
            const llama_pos tail_end   = (llama_pos) n_token_count;

            if (kv_attn->count_cells_range(seq_id, tail_begin, tail_end) != (size_t) n_tail) {
                LLAMA_LOG_WARN("%s: attn cache missing tail cells [%d, %d), rec file skipped\n",
                        __func__, (int) tail_begin, (int) tail_end);
            } else {
                ggsd_io_write_dummy io_dummy;
                if (n_tail > 0) {
                    kv_attn->state_write_range(io_dummy, seq_id, tail_begin, tail_end);
                }
                mem_recr->state_write(io_dummy, seq_id);
                const uint64_t payload_size = io_dummy.n_bytes();
                uint64_t file_size = 0;
                if (cache && !llama_ggsd::rec_serialized_size(model_id.size(), kv_params.size(), n_token_count, payload_size, file_size)) {
                    return (size_t) -1;
                }
                std::optional<llama_ggsd::cache_reservation> reservation;
                if (group_reservation) {
                    if (group_reservation->remaining() < file_size) return n_chain;
                } else if (cache) {
                    reservation = cache->reserve(file_size, {});
                    if (!reservation) return n_chain;
                }
                const std::string fname_tmp = fname + ".tmp";
                try {
                    ggsd_ensure_parent_dir(fname);
                    {
                        llama_file file(fname_tmp.c_str(), "wb");
                        file.write_raw(GGSD_REC_MAGIC, sizeof(GGSD_REC_MAGIC));
                        file.write_u32(GGSD_VERSION);
                        file.write_u32((uint32_t) n_token_count);
                        file.write_u32((uint32_t) n_tail);
                        file.write_raw(chain_hash.c_str(), GGSD_HASH_HEX_LEN);
                        file.write_raw(&payload_size, sizeof(payload_size));
                        uint8_t digest[32] = {};
                        file.write_raw(digest, GGSD_HASH_BYTES);
                        ggsd_write_str(file, model_id);
                        ggsd_write_str(file, kv_params);
                        file.write_raw(tokens, n_token_count * sizeof(llama_token));
                        llama_sha256_t sha;
                        llama_sha256_init(&sha);
                        {
                            ggsd_io_write_file io(&file, &sha);
                            if (n_tail > 0) kv_attn->state_write_range(io, seq_id, tail_begin, tail_end);
                            mem_recr->state_write(io, seq_id);
                        }
                        llama_sha256_final(&sha, digest);
                        file.seek(GGSD_REC_HEADER_PAYLOAD_HASH_OFFSET, SEEK_SET);
                        file.write_raw(digest, GGSD_HASH_BYTES);
                    }
                    if (group_reservation) {
                        std::string error;
                        if (!group_reservation->finalize_temporary(llama_ggsd::object_kind::rec, fname_tmp, fname, file_size, error)) {
                            group_reservation->abandon_temporary(fname_tmp, file_size);
                            return (size_t) -1;
                        }
                    } else if (reservation) {
                        std::string error;
                        if (!reservation->finalize_temporary(llama_ggsd::object_kind::rec, fname_tmp, fname, file_size, error)) {
                            reservation->abandon_temporary(fname_tmp, file_size);
                            return (size_t) -1;
                        }
                    } else std::filesystem::rename(fname_tmp, fname);
                } catch (...) {
                    if (group_reservation) group_reservation->abandon_temporary(fname_tmp, file_size);
                    else if (reservation) reservation->abandon_temporary(fname_tmp, file_size);
                    throw;
                }
        }
    }

    }
    if (n_seg > 0 && n_chain == 0) {
        LLAMA_LOG_ERROR("%s: attn cache is missing the head of the sequence, nothing usable saved\n", __func__);
        return (size_t) -1;
    }

    LLAMA_LOG_INFO("%s: wrote %zu segment(s), rec coverage %zu tokens (chain %zu)\n",
            __func__, n_written, n_token_count, n_chain);

    if (auto * cache = ggsd_cache_for_session(session_path)) {
        cache->touch({llama_ggsd::object_kind::rec, ggsd_prefix_hash(model_id, kv_params, tokens, n_token_count)});
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
    if (auto * mem = dynamic_cast<llama_memory_hybrid *>(memory.get())) {
        return state_seq_load_incr_hybrid(session_path, seq_id, prompt_tokens, n_prompt_tokens, min_prefix_tokens, *mem);
    }

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

    // segment-pool semantics: the session file is not consulted. The prompt's
    // hash chain is matched directly against content-addressed segment files in
    // the directory of session_path, so prefixes persisted by any session (and
    // any sequence) are reusable (G2 / remediation R1).

    const std::string model_id = llm_arch_name(model.arch);
    const std::string kv_params = ggsd_kv_params(model, kv);

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
            if (!ggsd_read_segment(kv, session_path, k, hashes, model_id, kv_params, seq_id)) {
                break;
            }

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

    const size_t restored = (n_loaded + k0) * GGSD_SEGMENT_TOKENS;
    if (restored >= min_prefix_tokens && restored > 0) {
        if (auto * cache = ggsd_cache_for_session(session_path)) {
            cache->touch({llama_ggsd::object_kind::segment, hashes[(n_loaded + k0) - 1]});
        }
    }
    return restored;
}

size_t llama_context::state_seq_load_incr_hybrid(
        const char * session_path,
              llama_seq_id   seq_id,
        const llama_token * prompt_tokens,
              size_t   n_prompt_tokens,
              size_t   min_prefix_tokens,
              llama_memory_hybrid & mem) {
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
    // hash of the request, and the segments under its tail must exist on disk
    size_t best_tokens = 0;
    std::filesystem::path best_path;
    ggsd_rec_header best_h;
    size_t n_chain_best = 0;

    const std::filesystem::path pool_dir = std::filesystem::path(session_path).parent_path();
    for (const auto & path : ggsd_list_pool_files(pool_dir, "rec")) {
        try {
            llama_file f(path.string().c_str(), "rb");
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
            // the rec's own chain segments must exist for this request; the pool
            // may hold longer chains from other sessions (n_seg_ok > chain)
            const size_t n_chain_rec = (size_t) (h.n_tokens - h.n_tail) / GGSD_SEGMENT_TOKENS;
            if ((size_t) h.n_tokens - h.n_tail != n_chain_rec * GGSD_SEGMENT_TOKENS ||
                    n_seg_ok < n_chain_rec) {
                continue;
            }
            best_tokens   = h.n_tokens;
            best_path     = path;
            best_h        = std::move(h);
            n_chain_best  = n_chain_rec;
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
        // replay only the rec's own chain - the tail appends right after it
        for (size_t k = 0; k < n_chain_best; ++k) {
            if (!ggsd_read_segment(kv_attn, session_path, k, hashes, model_id, kv_params, seq_id)) {
                throw std::runtime_error("segment failed validation during replay");
            }
            ++n_loaded;
        }

        llama_file file(best_path.string().c_str(), "rb");
        ggsd_rec_header h;
        if (!ggsd_rec_read_header(file, h)) {
            throw std::runtime_error("rec header changed on disk");
        }

        // bound the payload allocation by the remaining file size
        if (h.payload_size > file.size() - file.tell()) {
            throw std::runtime_error("rec payload size out of range");
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

    if (auto * cache = ggsd_cache_for_session(session_path)) {
        cache->touch({llama_ggsd::object_kind::rec, best_path.filename().string()});
    }
    return best_tokens;
}

size_t llama_context::state_seq_load_incr_estimate(
        const char * session_path,
              llama_seq_id   seq_id,
        const llama_token * prompt_tokens,
              size_t   n_prompt_tokens,
              size_t   min_prefix_tokens) const {
    if (auto * mem = dynamic_cast<const llama_memory_hybrid *>(memory.get())) {
        return state_seq_estimate_hybrid(session_path, prompt_tokens, n_prompt_tokens, min_prefix_tokens, *mem);
    }

    if (n_prompt_tokens < GGSD_SEGMENT_TOKENS) {
        return 0;
    }

    auto * kv = dynamic_cast<const llama_kv_cache *>(memory.get());
    if (kv == nullptr || model.hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        return 0;
    }

    GGML_UNUSED(seq_id);

    const std::string model_id = llm_arch_name(model.arch);
    const std::string kv_params = ggsd_kv_params(model, kv);

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

size_t llama_context::state_seq_estimate_hybrid(
        const char * session_path,
        const llama_token * prompt_tokens,
              size_t   n_prompt_tokens,
              size_t   min_prefix_tokens,
              const llama_memory_hybrid & mem) const {
    if (model.hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        return 0;
    }

    const std::string model_id  = llm_arch_name(model.arch);
    const std::string kv_params = ggsd_kv_params(model, mem.get_mem_attn());

    // mirror the load selection: the aligned segment prefix must exist on disk,
    // so an estimate hit is never priced on segments that cannot be replayed
    const size_t n_seg = n_prompt_tokens / GGSD_SEGMENT_TOKENS;
    const std::vector<std::string> hashes = ggsd_hash_chain(model_id, kv_params, prompt_tokens, n_seg);

    size_t n_seg_ok = 0;
    for (; n_seg_ok < n_seg; ++n_seg_ok) {
        if (!std::filesystem::exists(ggsd_segment_path(session_path, hashes[n_seg_ok]))) {
            break;
        }
    }

    size_t best_tokens = 0;

    const std::filesystem::path pool_dir = std::filesystem::path(session_path).parent_path();
    for (const auto & path : ggsd_list_pool_files(pool_dir, "rec")) {
        try {
            llama_file f(path.string().c_str(), "rb");
            ggsd_rec_header h;
            if (!ggsd_rec_read_header(f, h, /* read_tokens = */ false)) {
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
            // same coverage rule as the load path
            const size_t n_chain_rec = (size_t) (h.n_tokens - h.n_tail) / GGSD_SEGMENT_TOKENS;
            if ((size_t) h.n_tokens - h.n_tail != n_chain_rec * GGSD_SEGMENT_TOKENS ||
                    n_seg_ok < n_chain_rec) {
                continue;
            }
            best_tokens = h.n_tokens;
        } catch (...) {
            continue;
        }
    }

    return best_tokens;
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

llama_ggsd_cache_params llama_ggsd_cache_default_params() {
    return { 0 };
}

bool llama_context::ggsd_cache_configure(const char * session_path, llama_ggsd_cache_params params) {
    if (!session_path) return false;
    std::error_code ec;
    const auto pool = std::filesystem::weakly_canonical(std::filesystem::path(session_path).parent_path(), ec);
    if (ec || !std::filesystem::is_directory(std::filesystem::symlink_status(pool, ec)) || ec) return false;
    const auto key = pool.string();
    if (params.max_bytes == 0) {
        ggsd_caches.erase(key);
        return true;
    }
    std::string error;
    auto state = llama_ggsd::cache::open(pool, params.max_bytes, llama_ggsd::cache_file_ops::system(), error);
    if (!state) {
        LLAMA_LOG_ERROR("%s: failed to configure GGSD cache: %s\n", __func__, error.c_str());
        return false;
    }
    ggsd_caches[key] = std::move(state);
    return true;
}

llama_ggsd::cache * llama_context::ggsd_cache_for_session(const char * session_path) {
    if (!session_path) return nullptr;
    std::error_code ec;
    const auto pool = std::filesystem::weakly_canonical(std::filesystem::path(session_path).parent_path(), ec);
    if (ec) return nullptr;
    const auto it = ggsd_caches.find(pool.string());
    return it == ggsd_caches.end() ? nullptr : it->second.get();
}

bool llama_context::ggsd_cache_get_stats(const char * session_path, llama_ggsd_cache_stats & out) const {
    auto * self = const_cast<llama_context *>(this);
    auto * c = self->ggsd_cache_for_session(session_path);
    if (!c) return false;
    const auto & s = c->stats();
    out = {s.bytes,s.limit_bytes,s.segments,s.rec_snapshots,s.gc_runs,s.gc_deleted_bytes,s.gc_failures,s.writes_rejected,s.touch_failures};
    return true;
}

bool llama_ggsd_cache_configure(llama_context * ctx, const char * path, llama_ggsd_cache_params params) {
    try { return ctx && ctx->ggsd_cache_configure(path, params); } catch (...) { return false; }
}

bool llama_ggsd_cache_get_stats(const llama_context * ctx, const char * path, llama_ggsd_cache_stats * stats) {
    try { return ctx && stats && ctx->ggsd_cache_get_stats(path, *stats); } catch (...) { return false; }
}
