#include "llama-ggsd-cache.h"
#include "llama-mmap.h"
#include <algorithm>
#include <chrono>
#include <fstream>

namespace llama_ggsd {
namespace {
uint64_t add_size(uint64_t a, uint64_t b, bool & ok) { if (b > UINT64_MAX-a) { ok=false; return 0; } return a+b; }
bool zero_hash(const std::string & s) { return s.size()==HASH_HEX_LEN && s.find_first_not_of('\0')==std::string::npos; }
void refresh(cache_graph & g, const std::filesystem::path & p, const cache_file_ops & ops) {
    std::error_code ec;
    for (auto kind : {object_kind::segment, object_kind::rec}) {
        for (const auto & path : list_pool_files(p, kind, ec)) {
            if (ec) { g.ok=false; g.error=ec.message(); return; }
            const auto sz=std::filesystem::file_size(path,ec); if(ec){g.ok=false;g.error=ec.message();return;}
            bool good=true; g.managed_bytes=add_size(g.managed_bytes,sz,good); if(!good){g.ok=false;g.error="managed byte overflow";return;}
            const auto mt=std::filesystem::last_write_time(path,ec); if(ec){g.ok=false;g.error=ec.message();return;}
            try {
                llama_file f(path.string().c_str(),"rb");
                if (kind==object_kind::segment) {
                    segment_header h; auto & n=g.segments[path.filename().string()]; n.id={kind,path.filename().string()}; n.bytes=sz;n.last_used=mt;
                    if(!read_segment_header(f,h,true)||h.n_tokens!=SEGMENT_TOKENS||(!zero_hash(h.prev_hash)&&!is_hash_name(h.prev_hash))||segment_hash(h.model_id,h.kv_params,zero_hash(h.prev_hash)?std::string():h.prev_hash,h.tokens.data())!=n.id.hash){n.valid=false;g.invalid_objects++;continue;} n.parent_hash=zero_hash(h.prev_hash)?std::string():h.prev_hash;n.segment_index=h.segment_index;
                } else {
                    rec_header h; auto & n=g.recs[path.filename().string()]; n.id={kind,path.filename().string()};n.bytes=sz;n.last_used=mt;
                    if(!read_rec_header(f,h,true)||!is_hash_name(h.chain_hash)||prefix_hash(h.model_id,h.kv_params,h.tokens.data(),h.n_tokens)!=h.chain_hash){n.valid=false;g.invalid_objects++;continue;}
                    const size_t aligned=h.n_tokens-h.n_tail; if(aligned%SEGMENT_TOKENS) {n.valid=false;g.invalid_objects++;continue;} if(aligned) n.required_tail_segment_hash=hash_chain(h.model_id,h.kv_params,h.tokens.data(),aligned/SEGMENT_TOKENS).back();
                }
            } catch (...) { if(kind==object_kind::segment)g.segments[path.filename().string()].valid=false; else g.recs[path.filename().string()].valid=false; g.invalid_objects++; }
        }
    }
    for (const char * root_name : {"seg", "rec"}) {
        const auto root = p / root_name;
        if (!std::filesystem::is_directory(std::filesystem::symlink_status(root, ec))) {
            ec.clear();
            continue;
        }
        for (const auto & shard : std::filesystem::directory_iterator(root, ec)) {
            if (ec) { g.ok = false; g.error = ec.message(); return; }
            if (!std::filesystem::is_directory(std::filesystem::symlink_status(shard.path(), ec))) {
                ec.clear();
                continue;
            }
            for (const auto & entry : std::filesystem::directory_iterator(shard.path(), ec)) {
                if (ec) { g.ok = false; g.error = ec.message(); return; }
                const auto name = entry.path().filename().string();
                if (!std::filesystem::is_regular_file(std::filesystem::symlink_status(entry.path(), ec)) ||
                        !is_temporary_name(name) ||
                        name.substr(0, 2) != shard.path().filename().string()) {
                    ec.clear();
                    continue;
                }
                const auto size = std::filesystem::file_size(entry.path(), ec);
                if (ec) { g.ok = false; g.error = ec.message(); return; }
                bool good = true;
                g.managed_bytes = add_size(g.managed_bytes, size, good);
                g.temporary_bytes = add_size(g.temporary_bytes, size, good);
                if (!good) { g.ok = false; g.error = "managed byte overflow"; return; }
                g.temporary_files.push_back(entry.path());
            }
        }
    }

    for (auto & [h,n] : g.segments) {
        if (!n.parent_hash.empty()) {
            auto it = g.segments.find(n.parent_hash);
            if (it == g.segments.end() || it->second.segment_index + 1 != n.segment_index) {
                n.valid = false;
                continue;
            }
            it->second.child_count++;
        }
    }
    for(auto & [h,n]:g.recs) if(!n.required_tail_segment_hash.empty()){auto it=g.segments.find(n.required_tail_segment_hash);if(it==g.segments.end()){n.valid=false;continue;}it->second.rec_ref_count++;}
    std::map<std::string,uint8_t> colors;
    std::function<bool(const std::string&)> visit=[&](const std::string & h){auto & c=colors[h];if(c==1){g.segments[h].valid=false;return false;}if(c==2)return g.segments[h].valid;c=1;auto & n=g.segments[h];bool ok=n.parent_hash.empty()|| (g.segments.count(n.parent_hash)&&visit(n.parent_hash));if(!ok)n.valid=false;c=2;return n.valid;};
    for(const auto & [h,n]:g.segments)visit(h);
    g.ok=true; g.estimated_graph_bytes=(g.segments.size()*sizeof(segment_node)+g.recs.size()*sizeof(rec_node));
}
}
cache_file_ops cache_file_ops::system(){return {[](const auto&p,std::error_code&e){return std::filesystem::remove(p,e);},[](const auto&a,const auto&b,std::error_code&e){std::filesystem::rename(a,b,e);return !e;},[](const auto&p,auto t,std::error_code&e){std::filesystem::last_write_time(p,t,e);return !e;},[](){return std::chrono::steady_clock::now();}};}
cache_graph scan_pool(const std::filesystem::path & p,const cache_file_ops & ops){cache_graph g;auto begin=std::chrono::steady_clock::now();std::error_code ec; if(!std::filesystem::is_directory(std::filesystem::symlink_status(p,ec))||ec){g.error=ec?ec.message():"pool is not a directory";return g;} refresh(g,p,ops);g.enumerate_us=(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-begin).count();return g;}
cache::cache(const std::filesystem::path&p,uint64_t m,cache_file_ops o):pool_(p),max_bytes_(m),ops_(std::move(o)){}
std::unique_ptr<cache> cache::open(const std::filesystem::path & p, uint64_t m, cache_file_ops o, std::string & e) {
    auto c = std::unique_ptr<cache>(new cache(p, m, std::move(o)));
    auto graph = scan_pool(p, c->ops_);
    if (!graph.ok) { e = graph.error; return nullptr; }
    for (const auto & temporary : graph.temporary_files) {
        std::error_code ec;
        c->ops_.remove(temporary, ec);
    }
    graph = scan_pool(p, c->ops_);
    if (!graph.ok) { e = graph.error; return nullptr; }
    c->segments_ = std::move(graph.segments);
    c->recs_ = std::move(graph.recs);
    c->stats_.bytes = graph.managed_bytes;
    c->stats_.limit_bytes = m;
    c->stats_.segments = c->segments_.size();
    c->stats_.rec_snapshots = c->recs_.size();
    if (m && c->stats_.bytes > m) {
        const auto collected = c->collect_to(m / 10 * 9, {});
        if (!collected.ok) { e = collected.error; return nullptr; }
    }
    return c;
}
std::optional<cache_reservation> cache::reserve(uint64_t bytes,const std::vector<object_id>& ids){if(bytes>max_bytes_||reserved_>max_bytes_-bytes){stats_.writes_rejected++;return std::nullopt;}if(stats_.bytes>max_bytes_-reserved_-bytes){auto r=collect_to(std::min(max_bytes_/10*9,max_bytes_-reserved_-bytes),ids);if(!r.ok){stats_.writes_rejected++;return std::nullopt;}}reserved_+=bytes;return cache_reservation(*this,bytes);}
collect_result cache::collect_to(uint64_t target, const std::vector<object_id> & ids) {
    collect_result result;
    result.bytes_before = stats_.bytes;
    result.bytes_target = target;
    const auto started = std::chrono::steady_clock::now();
    std::map<object_id, bool> protected_ids;
    for (const auto & id : ids) {
        protected_ids[id] = true;
    }

    auto reload = [&]() -> bool {
        auto graph = scan_pool(pool_, ops_);
        if (!graph.ok) {
            result.error = graph.error;
            return false;
        }
        segments_ = std::move(graph.segments);
        recs_ = std::move(graph.recs);
        stats_.bytes = graph.managed_bytes;
        stats_.segments = segments_.size();
        stats_.rec_snapshots = recs_.size();
        result.invalid_found = graph.invalid_objects;
        return true;
    };
    if (!reload()) {
        ++stats_.gc_failures;
        return result;
    }

    while (stats_.bytes > target) {
        std::map<object_id, bool> closure = protected_ids;
        for (const auto & [id, value] : protected_ids) {
            if (id.kind == object_kind::rec) {
                const auto rec = recs_.find(id.hash);
                if (rec != recs_.end() && !rec->second.required_tail_segment_hash.empty()) {
                    closure[{object_kind::segment, rec->second.required_tail_segment_hash}] = true;
                }
            }
        }
        for (auto it = closure.begin(); it != closure.end(); ++it) {
            if (it->first.kind != object_kind::segment) continue;
            auto segment = segments_.find(it->first.hash);
            while (segment != segments_.end() && !segment->second.parent_hash.empty()) {
                const object_id parent{object_kind::segment, segment->second.parent_hash};
                if (closure[parent]) break;
                closure[parent] = true;
                segment = segments_.find(segment->second.parent_hash);
            }
        }
        protected_ids = std::move(closure);
        std::vector<object_id> candidates;
        for (const auto & [hash, node] : recs_) {
            if (!protected_ids[node.id]) candidates.push_back(node.id);
        }
        for (const auto & [hash, node] : segments_) {
            if (!protected_ids[node.id] && node.child_count == 0 && node.rec_ref_count == 0) {
                candidates.push_back(node.id);
            }
        }
        std::sort(candidates.begin(), candidates.end(), [&](const object_id & lhs, const object_id & rhs) {
            const auto valid = [&](const object_id & id) {
                return id.kind == object_kind::segment ? segments_.at(id.hash).valid : recs_.at(id.hash).valid;
            };
            if (valid(lhs) != valid(rhs)) return !valid(lhs);
            const auto mtime = [&](const object_id & id) {
                return id.kind == object_kind::segment ? segments_.at(id.hash).last_used : recs_.at(id.hash).last_used;
            };
            if (mtime(lhs) != mtime(rhs)) return mtime(lhs) < mtime(rhs);
            const auto bytes = [&](const object_id & id) {
                return id.kind == object_kind::segment ? segments_.at(id.hash).bytes : recs_.at(id.hash).bytes;
            };
            if (bytes(lhs) != bytes(rhs)) return bytes(lhs) > bytes(rhs);
            return lhs.hash < rhs.hash;
        });
        if (candidates.empty()) {
            result.error = "collection target not reached";
            break;
        }
        bool deleted = false;
        for (const auto & id : candidates) {
            std::error_code ec;
            if (!ops_.remove(object_path(pool_, id), ec)) {
                continue;
            }
            ++result.objects_deleted;
            deleted = true;
            break;
        }
        if (!deleted) {
            result.error = "collection target not reached";
            break;
        }
        if (!reload()) {
            ++stats_.gc_failures;
            return result;
        }
    }
    result.bytes_after = stats_.bytes;
    result.bytes_deleted = result.bytes_before >= result.bytes_after ? result.bytes_before - result.bytes_after : 0;
    result.ok = stats_.bytes <= target;
    result.duration_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - started).count();
    ++stats_.gc_runs;
    stats_.gc_deleted_bytes += result.bytes_deleted;
    if (!result.ok) {
        ++stats_.gc_failures;
    }
    return result;
}
void cache::touch(const object_id&id){auto now=ops_.now_steady();auto it=touched_.find(id);if(it!=touched_.end()&&now-it->second<std::chrono::minutes(10))return;std::error_code ec;if(!ops_.set_mtime(object_path(pool_,id),std::filesystem::file_time_type::clock::now(),ec)){stats_.touch_failures++;return;}touched_[id]=now;}
const cache_stats& cache::stats()const{return stats_;}
cache_reservation::cache_reservation(cache&c,uint64_t b):owner_(&c),remaining_(b){}cache_reservation::cache_reservation(cache_reservation&&r)noexcept:owner_(r.owner_),remaining_(r.remaining_){r.owner_=nullptr;r.remaining_=0;}cache_reservation&cache_reservation::operator=(cache_reservation&&r)noexcept{if(this!=&r){if(owner_)owner_->release_reserved(remaining_);owner_=r.owner_;remaining_=r.remaining_;r.owner_=nullptr;r.remaining_=0;}return *this;}cache_reservation::~cache_reservation(){if(owner_)owner_->release_reserved(remaining_);}uint64_t cache_reservation::remaining()const{return remaining_;}void cache_reservation::commit(object_kind k,uint64_t b){if(!owner_||b>remaining_)return;remaining_-=b;owner_->commit_reserved(k,b);}void cache_reservation::retain_temporary(uint64_t b){if(b<=remaining_){} }bool cache_reservation::finalize_temporary(object_kind k,const std::filesystem::path&t,const std::filesystem::path&d,uint64_t expected,std::string&e){std::error_code ec;auto n=std::filesystem::file_size(t,ec);if(ec||n!=expected){e="temporary size mismatch";return false;}if(!owner_->ops_.rename(t,d,ec)){e=ec.message();return false;}commit(k,expected);return true;}void cache_reservation::abandon_temporary(const std::filesystem::path&t,uint64_t expected){std::error_code ec;owner_->ops_.remove(t,ec);if(!ec){owner_->reserved_=owner_->reserved_>expected?owner_->reserved_-expected:0;remaining_=remaining_>expected?remaining_-expected:0;}}void cache::commit_reserved(object_kind k,uint64_t b){reserved_=reserved_>b?reserved_-b:0;stats_.bytes+=b;if(k==object_kind::segment)stats_.segments++;else stats_.rec_snapshots++;}void cache::release_reserved(uint64_t b){reserved_=reserved_>b?reserved_-b:0;}
}
