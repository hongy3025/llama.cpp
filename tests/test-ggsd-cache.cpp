#include "llama-ggsd-cache.h"
#include "llama-mmap.h"

#include <cassert>
#include <chrono>
#include <filesystem>
#include <vector>

namespace {
std::filesystem::path object(const std::filesystem::path & pool, llama_ggsd::object_kind kind, const std::string & hash) {
    return llama_ggsd::object_path(pool, {kind, hash});
}

void write_segment(const std::filesystem::path & pool, const std::string & hash, const std::string & parent, uint32_t index) {
    const auto path = object(pool, llama_ggsd::object_kind::segment, hash);
    std::filesystem::create_directories(path.parent_path());
    llama_file file(path.string().c_str(), "wb");
    const char magic[] = {'G', 'G', 'S', 'D'};
    file.write_raw(magic, sizeof(magic));
    file.write_u32(llama_ggsd::VERSION);
    file.write_u32(index);
    char previous[llama_ggsd::HASH_HEX_LEN] = {};
    if (!parent.empty()) {
        std::copy(parent.begin(), parent.end(), previous);
    }
    file.write_raw(previous, sizeof(previous));
    file.write_u32(llama_ggsd::SEGMENT_TOKENS);
    const uint64_t payload = 0;
    file.write_raw(&payload, sizeof(payload));
    std::array<uint8_t, llama_ggsd::HASH_BYTES> payload_hash = {};
    file.write_raw(payload_hash.data(), payload_hash.size());
    llama_ggsd::write_string(file, "model");
    llama_ggsd::write_string(file, "kv");
    std::vector<llama_token> tokens(llama_ggsd::SEGMENT_TOKENS, 0);
    file.write_raw(tokens.data(), tokens.size() * sizeof(tokens[0]));
}

void test_empty_scan() {
    const auto pool = std::filesystem::temp_directory_path() / "llama-ggsd-cache-empty";
    std::error_code ec;
    std::filesystem::remove_all(pool, ec);
    std::filesystem::create_directories(pool, ec);
    const auto graph = llama_ggsd::scan_pool(pool, llama_ggsd::cache_file_ops::system());
    assert(graph.ok);
    assert(graph.managed_bytes == 0);
    std::filesystem::remove_all(pool, ec);
}

void test_graph_and_quota() {
    const auto pool = std::filesystem::temp_directory_path() / "llama-ggsd-cache-graph";
    std::error_code ec;
    std::filesystem::remove_all(pool, ec);
    std::filesystem::create_directories(pool, ec);
    std::vector<llama_token> tokens(llama_ggsd::SEGMENT_TOKENS, 0);
    const auto first = llama_ggsd::segment_hash("model", "kv", "", tokens.data());
    const auto second = llama_ggsd::segment_hash("model", "kv", first, tokens.data());
    write_segment(pool, first, "", 0);
    write_segment(pool, second, first, 1);
    const auto graph = llama_ggsd::scan_pool(pool, llama_ggsd::cache_file_ops::system());
    assert(graph.ok);
    assert(graph.segments.size() == 2);
    assert(graph.segments.at(first).child_count == 1);
    assert(graph.segments.at(second).parent_hash == first);
    assert(graph.invalid_objects == 0);

    std::string error;
    auto cache = llama_ggsd::cache::open(pool, graph.managed_bytes, llama_ggsd::cache_file_ops::system(), error);
    assert(cache != nullptr);
    assert(cache->stats().bytes == graph.managed_bytes);
    auto rejected = cache->reserve(graph.managed_bytes + 1, {});
    assert(!rejected.has_value());
    assert(cache->stats().writes_rejected == 1);
    std::filesystem::remove_all(pool, ec);
}
}

int main() {
    test_empty_scan();
    test_graph_and_quota();
    return 0;
}
