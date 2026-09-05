#include "llama-ggsd.h"
#include "llama-mmap.h"
#include "llama-sha256.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace llama_ggsd {
namespace {
constexpr char SEG_MAGIC[4] = {'G','G','S','D'};
constexpr char REC_MAGIC[4] = {'G','G','S','R'};
constexpr uint64_t SEG_FIXED = 4 + 4 + 4 + HASH_HEX_LEN + 4 + 8 + HASH_BYTES;
constexpr uint64_t REC_FIXED = 4 + 4 + 4 + 4 + HASH_HEX_LEN + 8 + HASH_BYTES;

bool add(uint64_t a, uint64_t b, uint64_t & out) {
    if (b > UINT64_MAX - a) return false;
    out = a + b; return true;
}
bool mul(uint64_t a, uint64_t b, uint64_t & out) {
    if (a != 0 && b > UINT64_MAX / a) return false;
    out = a * b; return true;
}
std::string hex16(const uint8_t * digest) {
    std::string out(HASH_HEX_LEN, '0');
    for (size_t i = 0; i < HASH_BYTES; ++i) {
        out[2*i] = "0123456789abcdef"[digest[i] >> 4];
        out[2*i+1] = "0123456789abcdef"[digest[i] & 15];
    }
    return out;
}
void hash_update_string(llama_sha256_t & sha, const std::string & s) {
    if (s.size() > UINT32_MAX) throw std::overflow_error("GGSD identity is too large");
    const uint32_t n = (uint32_t) s.size();
    llama_sha256_update(&sha, (const uint8_t *) &n, sizeof(n));
    llama_sha256_update(&sha, (const uint8_t *) s.data(), s.size());
}
bool read_common(llama_file & f, size_t start, uint64_t payload, size_t & offset) {
    if (payload > SIZE_MAX || start > f.size() || payload > f.size() - start) return false;
    offset = start;
    return true;
}
}

bool object_id::operator<(const object_id & rhs) const { return kind != rhs.kind ? kind < rhs.kind : hash < rhs.hash; }
bool object_id::operator==(const object_id & rhs) const { return kind == rhs.kind && hash == rhs.hash; }

std::filesystem::path pool_dir(const char * session_path) { return std::filesystem::path(session_path).parent_path(); }
std::filesystem::path object_path(const std::filesystem::path & pool, const object_id & id) {
    const char * root = id.kind == object_kind::segment ? "seg" : "rec";
    return pool / root / id.hash.substr(0, 2) / id.hash;
}
std::filesystem::path segment_path(const char * p, const std::string & h) { return object_path(pool_dir(p), {object_kind::segment, h}); }
std::filesystem::path rec_path(const char * p, const std::string & h) { return object_path(pool_dir(p), {object_kind::rec, h}); }
bool is_hash_name(const std::string & n) {
    if (n.size() != HASH_HEX_LEN) return false;
    return std::all_of(n.begin(), n.end(), [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}
bool is_temporary_name(const std::string & n) { return n.size() == HASH_HEX_LEN + 4 && is_hash_name(n.substr(0, HASH_HEX_LEN)) && n.substr(HASH_HEX_LEN) == ".tmp"; }
std::vector<std::filesystem::path> list_pool_files(const std::filesystem::path & pool, object_kind kind, std::error_code & ec) {
    std::vector<std::filesystem::path> out; const auto root = pool / (kind == object_kind::segment ? "seg" : "rec");
    const auto status = std::filesystem::symlink_status(root, ec); if (ec || !std::filesystem::is_directory(status)) return out;
    for (const auto & shard : std::filesystem::directory_iterator(root, ec)) {
        if (ec) {
            break;
        }
        const auto ss = std::filesystem::symlink_status(shard.path(), ec);
        if (ec || !std::filesystem::is_directory(ss) || shard.path().filename().string().size() != 2) {
            continue;
        }
        for (const auto & e : std::filesystem::directory_iterator(shard.path(), ec)) {
            if (ec) {
                break;
            }
            const auto es = std::filesystem::symlink_status(e.path(), ec);
            if (!ec && std::filesystem::is_regular_file(es) && is_hash_name(e.path().filename().string()) && e.path().filename().string().substr(0,2) == shard.path().filename()) {
                out.push_back(e.path());
            }
        }
    }
    return out;
}

std::string segment_hash(const std::string & model, const std::string & kv, const std::string & prev, const llama_token * tokens) {
    if ((!prev.empty() && !is_hash_name(prev)) || (prev.empty() && tokens == nullptr)) throw std::invalid_argument("invalid GGSD predecessor");
    llama_sha256_t sha; llama_sha256_init(&sha); hash_update_string(sha, model); hash_update_string(sha, kv);
    if (!prev.empty()) { uint8_t b[HASH_BYTES]; for (size_t i=0;i<HASH_BYTES;++i) b[i] = (uint8_t)(std::stoi(prev.substr(i*2,2), nullptr, 16)); llama_sha256_update(&sha,b,sizeof(b)); }
    llama_sha256_update(&sha, (const uint8_t *) tokens, SEGMENT_TOKENS * sizeof(llama_token)); uint8_t d[32]; llama_sha256_final(&sha,d); return hex16(d);
}
std::vector<std::string> hash_chain(const std::string & m, const std::string & k, const llama_token * t, size_t n) {
    std::vector<std::string> out; out.reserve(n); for (size_t i=0;i<n;++i) out.push_back(segment_hash(m,k,i ? out.back() : std::string(), t+i*SEGMENT_TOKENS)); return out;
}
std::string prefix_hash(const std::string & m, const std::string & k, const llama_token * t, size_t n) {
    if (n > UINT32_MAX / sizeof(llama_token)) throw std::overflow_error("GGSD token count overflow");
    llama_sha256_t sha; llama_sha256_init(&sha); hash_update_string(sha,m); hash_update_string(sha,k); llama_sha256_update(&sha,(const uint8_t *)t,n*sizeof(llama_token)); uint8_t d[32]; llama_sha256_final(&sha,d); return hex16(d);
}
void write_string(llama_file & f, const std::string & s) { if (s.size()>UINT32_MAX) throw std::overflow_error("GGSD string overflow"); f.write_u32((uint32_t)s.size()); f.write_raw(s.data(),s.size()); }
std::string read_string(llama_file & f) { const uint32_t n=f.read_u32(); const size_t pos=f.tell(); if (n>MAX_IDENTITY_BYTES || pos>f.size() || (uint64_t)n>f.size()-pos) throw std::runtime_error("GGSD string exceeds file"); std::string s(n,'\0'); if(n) f.read_raw(s.data(),n); return s; }
bool read_segment_header(llama_file & f, segment_header & o, bool tokens) {
    try { char magic[4]; f.read_raw(magic,4); if(memcmp(magic,SEG_MAGIC,4)||f.read_u32()!=VERSION)return false; o.segment_index=f.read_u32(); char prev[HASH_HEX_LEN];f.read_raw(prev,sizeof(prev));o.prev_hash.assign(prev,sizeof(prev));o.n_tokens=f.read_u32();f.read_raw(&o.payload_size,8);f.read_raw(o.payload_hash.data(),HASH_BYTES);o.model_id=read_string(f);o.kv_params=read_string(f);uint64_t tb; const size_t pos=f.tell(); if(!mul(o.n_tokens,sizeof(llama_token),tb)||pos>f.size()||tb>f.size()-pos)return false;if(tokens){o.tokens.resize(o.n_tokens);if(tb)f.read_raw(o.tokens.data(),tb);}else f.seek(tb,SEEK_CUR);return read_common(f,f.tell(),o.payload_size,o.payload_offset); } catch (...) { return false; }
}
bool read_rec_header(llama_file & f, rec_header & o, bool tokens) {
    try { char magic[4];f.read_raw(magic,4);if(memcmp(magic,REC_MAGIC,4)||f.read_u32()!=VERSION)return false;o.n_tokens=f.read_u32();o.n_tail=f.read_u32();char chain[HASH_HEX_LEN];f.read_raw(chain,sizeof(chain));o.chain_hash.assign(chain,sizeof(chain));f.read_raw(&o.payload_size,8);f.read_raw(o.payload_hash.data(),HASH_BYTES);o.model_id=read_string(f);o.kv_params=read_string(f);if(o.n_tail>o.n_tokens)return false;uint64_t tb;if(!mul(o.n_tokens,sizeof(llama_token),tb)||tb>f.size()-f.tell())return false;if(tokens){o.tokens.resize(o.n_tokens);if(tb)f.read_raw(o.tokens.data(),tb);}else f.seek(tb,SEEK_CUR);return read_common(f,f.tell(),o.payload_size,o.payload_offset);}catch(...){return false;}
}
bool segment_serialized_size(size_t m,size_t k,uint64_t p,uint64_t & r){uint64_t x=SEG_FIXED;if(m>MAX_IDENTITY_BYTES||k>MAX_IDENTITY_BYTES||!add(x,4+m,x)||!add(x,4+k,x)||!add(x,p,x))return false;r=x;return true;}
bool rec_serialized_size(size_t m,size_t k,size_t n,uint64_t p,uint64_t & r){uint64_t x=REC_FIXED,t;if(m>MAX_IDENTITY_BYTES||k>MAX_IDENTITY_BYTES||!mul(n,sizeof(llama_token),t)||!add(x,4+m,x)||!add(x,4+k,x)||!add(x,t,x)||!add(x,p,x))return false;r=x;return true;}
} // namespace llama_ggsd
