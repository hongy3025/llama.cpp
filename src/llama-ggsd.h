#pragma once

#include "llama.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

class llama_file;

namespace llama_ggsd {

constexpr uint32_t VERSION = 2;
constexpr uint32_t SEGMENT_TOKENS = 256;
constexpr size_t HASH_BYTES = 16;
constexpr size_t HASH_HEX_LEN = 32;
constexpr uint32_t MAX_IDENTITY_BYTES = 1U << 20;

enum class object_kind : uint8_t { segment, rec };

struct object_id {
    object_kind kind;
    std::string hash;
    bool operator<(const object_id & rhs) const;
    bool operator==(const object_id & rhs) const;
};

struct segment_header {
    uint32_t segment_index = 0;
    std::string prev_hash;
    uint32_t n_tokens = 0;
    uint64_t payload_size = 0;
    std::array<uint8_t, HASH_BYTES> payload_hash = {};
    std::string model_id;
    std::string kv_params;
    std::vector<llama_token> tokens;
    size_t payload_offset = 0;
};

struct rec_header {
    uint32_t n_tokens = 0;
    uint32_t n_tail = 0;
    std::string chain_hash;
    uint64_t payload_size = 0;
    std::array<uint8_t, HASH_BYTES> payload_hash = {};
    std::string model_id;
    std::string kv_params;
    std::vector<llama_token> tokens;
    size_t payload_offset = 0;
};

std::filesystem::path pool_dir(const char * session_path);
std::filesystem::path object_path(const std::filesystem::path &, const object_id &);
std::filesystem::path segment_path(const char *, const std::string &);
std::filesystem::path rec_path(const char *, const std::string &);
bool is_hash_name(const std::string &);
bool is_temporary_name(const std::string &);
std::vector<std::filesystem::path> list_pool_files(const std::filesystem::path &, object_kind, std::error_code &);

std::string segment_hash(const std::string &, const std::string &, const std::string &, const llama_token *);
std::vector<std::string> hash_chain(const std::string &, const std::string &, const llama_token *, size_t);
std::string prefix_hash(const std::string &, const std::string &, const llama_token *, size_t);

void write_string(llama_file &, const std::string &);
std::string read_string(llama_file &);
bool read_segment_header(llama_file &, segment_header &, bool read_tokens = true);
bool read_rec_header(llama_file &, rec_header &, bool read_tokens = true);
bool segment_serialized_size(size_t, size_t, uint64_t, uint64_t &);
bool rec_serialized_size(size_t, size_t, size_t, uint64_t, uint64_t &);

} // namespace llama_ggsd
