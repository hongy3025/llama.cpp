#pragma once
#include "llama-ggsd.h"
#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <optional>

namespace llama_ggsd {
struct cache_file_ops {
    std::function<bool(const std::filesystem::path &, std::error_code &)> remove;
    std::function<bool(const std::filesystem::path &, const std::filesystem::path &, std::error_code &)> rename;
    std::function<bool(const std::filesystem::path &, std::filesystem::file_time_type, std::error_code &)> set_mtime;
    std::function<std::chrono::steady_clock::time_point()> now_steady;
    static cache_file_ops system();
};
struct segment_node { object_id id; std::string parent_hash; uint32_t segment_index=0; uint64_t bytes=0; std::filesystem::file_time_type last_used{}; uint32_t child_count=0; uint32_t rec_ref_count=0; bool valid=true; bool protected_now=false; };
struct rec_node { object_id id; std::string required_tail_segment_hash; uint64_t bytes=0; std::filesystem::file_time_type last_used{}; bool valid=true; bool protected_now=false; };
struct cache_graph { bool ok=false; std::string error; uint64_t managed_bytes=0; uint64_t temporary_bytes=0; uint64_t enumerate_us=0; uint64_t validate_us=0; uint64_t estimated_graph_bytes=0; std::map<std::string,segment_node> segments; std::map<std::string,rec_node> recs; std::vector<std::filesystem::path> temporary_files; uint64_t invalid_objects=0; };
cache_graph scan_pool(const std::filesystem::path &, const cache_file_ops &);
struct collect_result { bool ok=false; uint64_t bytes_before=0,bytes_target=0,bytes_after=0,objects_deleted=0,bytes_deleted=0,invalid_found=0,protected_skipped=0,duration_us=0,candidate_queue_us=0,delete_us=0; std::string error; };
struct cache_stats { uint64_t bytes=0,limit_bytes=0,segments=0,rec_snapshots=0,gc_runs=0,gc_deleted_bytes=0,gc_failures=0,writes_rejected=0,touch_failures=0; };
class cache;
class cache_reservation { public: cache_reservation(cache_reservation &&) noexcept; cache_reservation & operator=(cache_reservation &&) noexcept; ~cache_reservation(); void commit(object_kind,uint64_t); void retain_temporary(uint64_t); bool finalize_temporary(object_kind,const std::filesystem::path &,const std::filesystem::path &,uint64_t,std::string &); void abandon_temporary(const std::filesystem::path &,uint64_t); uint64_t remaining() const; private: friend class cache; cache_reservation(cache &,uint64_t); cache * owner_=nullptr; uint64_t remaining_=0; };
class cache { public: static std::unique_ptr<cache> open(const std::filesystem::path &,uint64_t,cache_file_ops,std::string &); std::optional<cache_reservation> reserve(uint64_t,const std::vector<object_id> &); collect_result collect_to(uint64_t,const std::vector<object_id> &); void touch(const object_id &); const cache_stats & stats() const; private: friend class cache_reservation; cache(const std::filesystem::path &,uint64_t,cache_file_ops); void commit_reserved(object_kind,uint64_t); void release_reserved(uint64_t); std::filesystem::path pool_; uint64_t max_bytes_,reserved_=0; cache_file_ops ops_; cache_stats stats_; std::map<std::string,segment_node> segments_; std::map<std::string,rec_node> recs_; std::map<object_id,std::chrono::steady_clock::time_point> touched_; };
}
