# GGSD Cache Automatic GC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a filesystem-native, dependency-aware Leaf-LRU collector that keeps GGSD-managed bytes below a configured hard limit for both manual and automatic cache operations.

**Architecture:** Extract the GGSD disk-format primitives from `llama-state-incr.cpp`, then add a core `llama_ggsd::cache` object owned per canonical pool path by `llama_context`. The cache rebuilds its graph from `seg` and `rec` files, admits writes through byte reservations, touches only successfully used save/restore endpoints, and synchronously peels cold leaves when a write would cross the hard limit.

**Tech Stack:** C++17, `std::filesystem`, existing `llama_file`/SHA-256 utilities, CMake/CTest, server Python pytest suite, Prometheus text metrics.

**Spec:** `docs/superpowers/specs/2026-09-04-ggsd-cache-auto-gc-design.md`

## Global Constraints

- Preserve the current single-writer contract: one process/context writer per pool; add no cross-process lock.
- Add no database, journal, durable index, background collector thread, pin, TTL, or source priority.
- Keep files as the persistent source of truth; rebuild byte accounting and dependencies during configuration.
- The hard limit counts recognized `seg`, `rec`, and GGSD temporary files, including a temporary file while it is written; it excludes session files, traditional slot saves, directory allocation, and unrelated files.
- Use a fixed low watermark of exactly 90% of `max_bytes`.
- Manual `save_incr` and autosave use identical admission, last-use, and eviction policy.
- Preserve prefix closure: a retained segment has every ancestor, and a retained rec has its required aligned segment chain.
- Run GC synchronously in the existing serialized save path; a failed admission degrades caching only and never model inference.
- `max_bytes == 0` and an unconfigured pool retain existing unlimited behavior without a startup scan.
- Keep source compatibility for `llama_state_seq_save_incr` and `llama_state_seq_load_incr`; add configuration and stats APIs rather than replacing them.
- Keep the project at C++17 and add no third-party dependency.
- Use test-first changes and commit after every task.

---

## File Structure

**Create**

- `src/llama-ggsd.h` — private GGSD constants, paths, hashes, headers, and bounded header parsing shared by save/load and GC.
- `src/llama-ggsd.cpp` — implementations of the shared disk-format helpers.
- `src/llama-ggsd-cache.h` — private cache manager, object identifiers, reservation type, graph scan result, and injectable filesystem operations.
- `src/llama-ggsd-cache.cpp` — startup reconciliation, dependency validation, Leaf-LRU collection, reservations, touches, and statistics.
- `tests/test-ggsd-cache.cpp` — model-free tests for graph reconstruction, invalid objects, deterministic eviction, reservations, touch throttling, failures, and the synthetic benchmark mode.
- `tests/test-ggsd-hybrid-gc.cpp` — generated-model integration test for grouped hybrid admission and recovery.

**Modify**

- `src/CMakeLists.txt` — compile both new GGSD implementation units.
- `tests/CMakeLists.txt` — build/register the model-free test and generated hybrid-model test.
- `include/llama.h` — public cache parameters, stats structs, configure/default/stats functions.
- `src/llama-context.h` — per-canonical-pool cache ownership and internal lookup methods.
- `src/llama-context.cpp` — include the complete private cache type where the
  out-of-line `llama_context` destructor destroys cache instances.
- `src/llama-state-incr.cpp` — consume shared format helpers; route save/load through reservation and touch hooks.
- `common/common.h` — add `prompt_cache_ssd_max_mib`.
- `common/arg.cpp` — parse `--prompt-cache-ssd-max-mib` and its environment variable safely.
- `tests/test-arg-parser.cpp` — zero, positive, negative, malformed, and overflow parsing cases.
- `tools/server/server-context.cpp` — configure the core pool before serving and snapshot GGSD stats for metrics.
- `tools/server/server-common.h` — carry a GGSD metrics snapshot in `server_metrics`.
- `tools/server/server-task.cpp` — render GGSD Prometheus counters and gauges.
- `tools/server/tests/utils.py` — expose SSD-cache and max-MiB arguments to server tests.
- `tools/server/tests/unit/test_metrics.py` — assert GGSD metric names/types/values.
- `tools/server/tests/unit/test_slot_save.py` — cover manual GGSD under the configured pool and restart reconciliation.
- `tools/server/README.md` — document the server flag and metrics.
- `docs/ggsd-guide.md` — replace the no-GC statement with quota, policy, and failure semantics.
- `docs/ggsd-autoload-guide.md` — document automatic-cache behavior under pressure.

---

### Task 1: Extract Shared GGSD Disk-Format Primitives

**Files:**

- Create: `src/llama-ggsd.h`
- Create: `src/llama-ggsd.cpp`
- Modify: `src/llama-state-incr.cpp:26-367`
- Modify: `src/CMakeLists.txt:5-45`
- Test: `tests/test-save-load-state.cpp`

**Interfaces:**

- Consumes: existing `llama_file`, `llama_sha256_*`, `llama_token`, and the version-1 GGSD/GGSR layouts.
- Produces: the exact `llama_ggsd` constants, paths, hash functions, and header parsers shown below; Tasks 2-5 depend on these names.

- [ ] **Step 1: Record the pre-refactor GGSD regression baseline**

Run:

```bash
cmake --build build-linux --target test-save-load-state -j
ctest --test-dir build-linux -R '^test-save-load-state$' --output-on-failure
```

Expected: build succeeds and the existing GGSD roundtrip/append/fork/corruption/differential test passes.

- [ ] **Step 2: Declare the shared private format API**

Create `src/llama-ggsd.h` with these public-to-core names; keep serialization internals outside `include/llama.h`:

```cpp
#pragma once

#include "llama.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

class llama_file;

namespace llama_ggsd {

constexpr uint32_t VERSION        = 1;
constexpr uint32_t SEGMENT_TOKENS = 1024;
constexpr size_t HASH_BYTES       = 16;
constexpr size_t HASH_HEX_LEN     = 32;
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
std::filesystem::path object_path(const std::filesystem::path & pool, const object_id & id);
std::filesystem::path segment_path(const char * session_path, const std::string & hash);
std::filesystem::path rec_path(const char * session_path, const std::string & hash);
bool is_hash_name(const std::string & name);
bool is_temporary_name(const std::string & name);
std::vector<std::filesystem::path> list_pool_files(
        const std::filesystem::path & pool, object_kind kind, std::error_code & ec);

std::string segment_hash(
        const std::string & model_id,
        const std::string & kv_params,
        const std::string & prev_hash,
        const llama_token * tokens);
std::vector<std::string> hash_chain(
        const std::string & model_id,
        const std::string & kv_params,
        const llama_token * tokens,
        size_t n_segments);
std::string prefix_hash(
        const std::string & model_id,
        const std::string & kv_params,
        const llama_token * tokens,
        size_t n_tokens);

void write_string(llama_file & file, const std::string & value);
std::string read_string(llama_file & file);
bool read_segment_header(llama_file & file, segment_header & out, bool read_tokens);
bool read_rec_header(llama_file & file, rec_header & out, bool read_tokens);
bool segment_serialized_size(
        size_t model_id_bytes,
        size_t kv_params_bytes,
        uint64_t payload_bytes,
        uint64_t & result);
bool rec_serialized_size(
        size_t model_id_bytes,
        size_t kv_params_bytes,
        size_t token_count,
        uint64_t payload_bytes,
        uint64_t & result);

} // namespace llama_ggsd
```

The parsers must bound string lengths, token-array multiplication, payload size,
and `payload_offset + payload_size` against `file.size()` before allocating.
When `read_tokens == false`, seek over tokens and still leave the cursor at
`payload_offset`.

`read_string` rejects lengths above `MAX_IDENTITY_BYTES` or the remaining file
size before allocating. `write_string`, hash inputs, segment indices, and rec
token counts reject values that cannot be represented by their on-disk
`uint32_t` fields; do not preserve the current narrowing casts.

`segment_hash` accepts an empty `prev_hash` only for a chain head; otherwise it
requires 32 lowercase hexadecimal characters. `is_temporary_name` recognizes
only `<32-lowercase-hex>.tmp`, matching the writer's current naming convention.

- [ ] **Step 3: Move the existing implementations without changing bytes**

Create `src/llama-ggsd.cpp` by moving the constants and helpers currently named
`ggsd_segment_path`, `ggsd_rec_path`, `ggsd_is_hash_name`,
`ggsd_list_pool_files`, `ggsd_hash_chain`, `ggsd_prefix_hash`,
`ggsd_write_str`, `ggsd_read_str`, and `ggsd_rec_read_header` into the namespace.
Implement `read_segment_header` from the header parsing currently embedded in
`ggsd_read_segment`.

Use checked addition and multiplication for the fixed header, both
length-prefixed strings, token bytes, and payload before returning sizes:

```cpp
uint64_t file_size = 0;
if (!llama_ggsd::segment_serialized_size(
        model_id.size(), kv_params.size(), payload_size, file_size)) {
    throw std::overflow_error("GGSD segment serialized size overflow");
}
```

Reject a file rather than wrapping if any addition or multiplication exceeds
`uint64_t` or `size_t`.

- [ ] **Step 4: Rewire state save/load to the shared helpers**

Include `llama-ggsd.h` in `llama-state-incr.cpp`, remove the moved anonymous
namespace definitions, and use qualified names consistently:

```cpp
namespace ggsd = llama_ggsd;

const auto hashes = ggsd::hash_chain(model_id, kv_params, tokens, n_seg);
const auto fname  = ggsd::segment_path(session_path, hashes[k]).string();
```

Keep payload serialization, payload hashing, sequence mutation, and return
semantics unchanged. Add `llama-ggsd.cpp` to the `llama` target immediately
before `llama-state-incr.cpp` in `src/CMakeLists.txt`.

- [ ] **Step 5: Build and rerun the regression test**

Run:

```bash
cmake --build build-linux --target test-save-load-state -j
ctest --test-dir build-linux -R '^test-save-load-state$' --output-on-failure
```

Expected: PASS with the same GGSD test output and no disk-format migration.

- [ ] **Step 6: Commit the format extraction**

```bash
git add src/llama-ggsd.h src/llama-ggsd.cpp src/llama-state-incr.cpp src/CMakeLists.txt
git commit -m "refactor: share GGSD disk format helpers"
```

---

### Task 2: Build and Validate the Transient Dependency Graph

**Files:**

- Create: `src/llama-ggsd-cache.h`
- Create: `src/llama-ggsd-cache.cpp`
- Create: `tests/test-ggsd-cache.cpp`
- Modify: `src/CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**

- Consumes: `llama_ggsd::object_id`, `read_segment_header`, `read_rec_header`, `hash_chain`, and pool path helpers from Task 1.
- Produces: `llama_ggsd::cache_graph`, `scan_pool`, and `cache_file_ops`; Task 3 consumes them without reading model state.

- [ ] **Step 1: Register the model-free test target**

Add to `tests/CMakeLists.txt` near `test-save-load-state`:

```cmake
llama_build_and_test(test-ggsd-cache.cpp)
target_include_directories(test-ggsd-cache PRIVATE ${PROJECT_SOURCE_DIR}/src)
```

Add `llama-ggsd-cache.cpp` to the `llama` library sources in
`src/CMakeLists.txt`.

- [ ] **Step 2: Write failing graph reconstruction tests**

Create helpers in `tests/test-ggsd-cache.cpp` that write version-1 segment and
rec headers with zero-byte payloads using the Task 1 format functions. Add these
named cases:

```cpp
static void test_graph_shared_prefix() {
    fixture f;
    const auto a = f.segment(zero_hash(), 0, f.tokens(0));
    const auto b = f.segment(a,           1, f.tokens(1));
    const auto c = f.segment(b,           2, f.tokens(2));
    const auto d = f.segment(b,           2, f.tokens(3));

    auto scan = llama_ggsd::scan_pool(f.pool(), llama_ggsd::cache_file_ops::system());
    assert(scan.ok);
    assert(scan.segments.at(b).child_count == 2);
    assert(scan.segments.at(c).child_count == 0);
    assert(scan.segments.at(d).child_count == 0);
}

static void test_graph_rec_dependency() {
    fixture f;
    const auto chain = f.segment_chain(2);
    const auto r = f.rec(chain.tokens, 0);

    auto scan = llama_ggsd::scan_pool(f.pool(), llama_ggsd::cache_file_ops::system());
    assert(scan.ok);
    assert(scan.recs.at(r).required_tail_segment_hash == chain.hashes.back());
    assert(scan.segments.at(chain.hashes.back()).rec_ref_count == 1);
}
```

Also add cases for a head segment, a zero-aligned-segment rec, shard junk,
unrecognized root files, size totals, and multiple model/KV forests.

- [ ] **Step 3: Run the test to verify the scanner is absent**

Run:

```bash
cmake --build build-linux --target test-ggsd-cache -j
```

Expected: FAIL to compile because `llama_ggsd::scan_pool` and graph types are
not defined.

- [ ] **Step 4: Declare graph and filesystem-operation types**

Put these stable private interfaces in `src/llama-ggsd-cache.h`:

```cpp
namespace llama_ggsd {

struct cache_file_ops {
    std::function<bool(const std::filesystem::path &, std::error_code &)> remove;
    std::function<bool(
            const std::filesystem::path &,
            const std::filesystem::path &,
            std::error_code &)> rename;
    std::function<bool(
            const std::filesystem::path &,
            std::filesystem::file_time_type,
            std::error_code &)> set_mtime;
    std::function<std::chrono::steady_clock::time_point()> now_steady;

    static cache_file_ops system();
};

struct segment_node {
    object_id id;
    std::string parent_hash;
    uint32_t segment_index = 0;
    uint64_t bytes = 0;
    std::filesystem::file_time_type last_used;
    uint32_t child_count = 0;
    uint32_t rec_ref_count = 0;
    bool valid = true;
    bool protected_now = false;
};

struct rec_node {
    object_id id;
    std::string required_tail_segment_hash;
    uint64_t bytes = 0;
    std::filesystem::file_time_type last_used;
    bool valid = true;
    bool protected_now = false;
};

struct cache_graph {
    bool ok = false;
    std::string error;
    uint64_t managed_bytes = 0;
    uint64_t temporary_bytes = 0;
    uint64_t enumerate_us = 0;
    uint64_t validate_us = 0;
    uint64_t estimated_graph_bytes = 0;
    std::map<std::string, segment_node> segments;
    std::map<std::string, rec_node> recs;
    std::vector<std::filesystem::path> temporary_files;
    uint64_t invalid_objects = 0;
};

cache_graph scan_pool(
        const std::filesystem::path & pool,
        const cache_file_ops & ops);

} // namespace llama_ggsd
```

- [ ] **Step 5: Implement bounded scan and dependency validation**

In `src/llama-ggsd-cache.cpp`:

1. Enumerate all 256 possible shard directories through the shared listing
   helper; never recurse outside `seg` and `rec`.
2. Count every recognized content file before parsing it.
3. Use `symlink_status` for kind roots, shard directories, and files; ignore
   symlinks instead of following them outside the managed pool. Recognize
   content only in its canonical shard (`filename.substr(0, 2)`), and
   recognize temporary files only when `is_temporary_name` is true in that same
   shard. Count each recognized file's logical size with checked addition.
4. Parse segment headers, recompute their content hash with `segment_hash` from
   model/KV identity, predecessor hash, and tokens, and compare it to the file
   name.
5. Validate head/index rules and `parent.segment_index + 1 == child.segment_index`.
6. Parse rec tokens with allocation bounds, recompute its prefix hash, derive
   `n_chain = (n_tokens - n_tail) / 1024`, and attach its reference to the last
   segment when `n_chain > 0`.
7. Mark missing parents, missing rec tails, index mismatches, and every node in
   a detected DFS color-cycle invalid; propagate invalidity to descendants and
   dependent recs.
8. Return `ok == false` only for directory/accounting failures. Individual bad
   cache files produce invalid nodes rather than aborting the scan.

Use the standard tri-color transition exactly:

```cpp
enum class visit : uint8_t { white, gray, black };

bool visit_parent(const std::string & hash) {
    if (color[hash] == visit::gray) {
        mark_cycle_from(hash);
        return false;
    }
    if (color[hash] == visit::black) {
        return segments.at(hash).valid;
    }
    color[hash] = visit::gray;
    const bool ok = validate_parent_edge(hash);
    color[hash] = visit::black;
    return ok;
}
```

- [ ] **Step 6: Run graph tests**

Run:

```bash
cmake --build build-linux --target test-ggsd-cache -j
ctest --test-dir build-linux -R '^test-ggsd-cache$' --output-on-failure
```

Expected: all graph, dependency, accounting, and junk-file cases PASS.

- [ ] **Step 7: Commit the scanner**

```bash
git add src/llama-ggsd-cache.h src/llama-ggsd-cache.cpp src/CMakeLists.txt tests/test-ggsd-cache.cpp tests/CMakeLists.txt
git commit -m "feat: reconstruct GGSD cache dependency graph"
```

---

### Task 3: Implement Leaf-LRU Collection, Reservations, and Touches

**Files:**

- Modify: `src/llama-ggsd-cache.h`
- Modify: `src/llama-ggsd-cache.cpp`
- Modify: `tests/test-ggsd-cache.cpp`

**Interfaces:**

- Consumes: `cache_graph`, `scan_pool`, and `cache_file_ops` from Task 2.
- Produces: `llama_ggsd::cache`, movable `cache_reservation`, `collect_result`, `reserve`, `commit`, `retain_temporary`, and `touch`; Tasks 4-7 use these exact methods.

- [ ] **Step 1: Write failing leaf-order and closure tests**

Add named tests that set deterministic mtimes and call a direct collection
entry point:

```cpp
static void test_collect_peels_old_leaf_and_keeps_shared_prefix() {
    fixture f;
    auto fork = f.forked_chain(); // A <- B <- C and A <- B <- D
    f.mtime(fork.c, 10);
    f.mtime(fork.d, 20);
    f.mtime(fork.b, 30);

    auto cache = f.open_cache(f.managed_bytes());
    const auto result = cache.collect_to(f.bytes_without(fork.c), {});

    assert(result.ok);
    assert(!f.exists_segment(fork.c));
    assert(f.exists_segment(fork.d));
    assert(f.exists_segment(fork.b));
    assert(f.prefix_closed());
}

static void test_rec_release_makes_tail_collectible() {
    fixture f;
    auto chain = f.segment_chain(2);
    auto rec = f.rec_for(chain);
    f.mtime(rec, 10);
    f.mtime(chain.hashes.back(), 20);

    auto cache = f.open_cache(f.managed_bytes());
    const auto result = cache.collect_to(0, {});

    assert(result.ok);
    assert(!f.exists(rec));
    assert(!f.exists_segment(chain.hashes.back()));
    assert(!f.exists_segment(chain.hashes.front()));
}
```

Add separate cases for protected deepest nodes, invalid-subtree priority, cyclic
invalid components, unlink failure followed by another candidate, deterministic
mtime/size/hash ties, and stopping exactly once the requested target is reached.

- [ ] **Step 2: Run tests and verify the collector API is absent**

Run:

```bash
cmake --build build-linux --target test-ggsd-cache -j
```

Expected: FAIL to compile because `cache::collect_to` is missing.

- [ ] **Step 3: Declare the cache manager and reservation contract**

Extend `src/llama-ggsd-cache.h` with:

```cpp
struct collect_result {
    bool ok = false;
    uint64_t bytes_before = 0;
    uint64_t bytes_target = 0;
    uint64_t bytes_after = 0;
    uint64_t objects_deleted = 0;
    uint64_t bytes_deleted = 0;
    uint64_t invalid_found = 0;
    uint64_t protected_skipped = 0;
    uint64_t duration_us = 0;
    uint64_t candidate_queue_us = 0;
    uint64_t delete_us = 0;
    std::string error;
};

struct cache_stats {
    uint64_t bytes = 0;
    uint64_t limit_bytes = 0;
    uint64_t segments = 0;
    uint64_t rec_snapshots = 0;
    uint64_t gc_runs = 0;
    uint64_t gc_deleted_bytes = 0;
    uint64_t gc_failures = 0;
    uint64_t writes_rejected = 0;
    uint64_t touch_failures = 0;
};

class cache;

class cache_reservation {
public:
    cache_reservation(cache_reservation && rhs) noexcept;
    cache_reservation & operator=(cache_reservation && rhs) noexcept;
    ~cache_reservation();

    void commit(object_kind kind, uint64_t bytes);
    void retain_temporary(uint64_t bytes);
    bool finalize_temporary(
            object_kind kind,
            const std::filesystem::path & temporary,
            const std::filesystem::path & destination,
            uint64_t expected_bytes,
            std::string & error);
    void abandon_temporary(
            const std::filesystem::path & temporary,
            uint64_t expected_bytes);
    uint64_t remaining() const;

private:
    friend class cache;
    cache_reservation(cache & owner, uint64_t bytes);
    cache * owner_ = nullptr;
    uint64_t remaining_ = 0;
};

class cache {
public:
    static std::unique_ptr<cache> open(
            const std::filesystem::path & pool,
            uint64_t max_bytes,
            cache_file_ops ops,
            std::string & error);

    std::optional<cache_reservation> reserve(
            uint64_t bytes,
            const std::vector<object_id> & protected_ids);
    collect_result collect_to(
            uint64_t target_bytes,
            const std::vector<object_id> & protected_ids);
    void touch(const object_id & id);
    const cache_stats & stats() const;

private:
    friend class cache_reservation;
    void commit_reserved(object_kind kind, uint64_t bytes);
    void retain_reserved_temporary(uint64_t bytes);
    void release_reserved(uint64_t bytes);
};
```

Delete copy construction/assignment for both classes. `cache_reservation` must
release all unconsumed reservation in its destructor.

`finalize_temporary` verifies that the closed temp file's logical size equals
`expected_bytes`, renames through the injected operation, then converts exactly
that portion from reserved to committed. `abandon_temporary` removes through
the injected operation; if removal fails, it stats and retains the actual
logical temp bytes, bounded by the remaining reservation. These two methods
give the model-free suite deterministic rename/remove failure coverage. A stat
failure conservatively retains `expected_bytes` and makes the reservation fail
without weakening the hard bound.

`cache::open` must remove recognized stale temporary files, count any temporary
it cannot remove, initialize object counts from `scan_pool`, and immediately
collect an over-limit existing pool to the fixed 90% target. Return null with a
nonempty error if trustworthy accounting or the startup target cannot be
established.

- [ ] **Step 4: Implement deterministic Leaf-LRU**

Build candidates only from unprotected recs and true segment leaves. Apply
transitive protection by walking from each protected segment or rec tail to the
head. Use this comparator:

```cpp
if (lhs.valid != rhs.valid) return lhs.valid;            // invalid first
if (lhs.last_used != rhs.last_used) return lhs.last_used > rhs.last_used;
if (lhs.bytes != rhs.bytes) return lhs.bytes < rhs.bytes; // larger first on tie
return lhs.id.hash > rhs.id.hash;                         // lexical stability
```

After a successful rec unlink, decrement only its tail's `rec_ref_count`; parent
edges protect the remaining ancestry. After a segment unlink, decrement its
parent's `child_count`. Never decrement counters when unlink fails.

For an invalid cycle, first delete invalid recs and acyclic descendants. Once no
valid incoming dependency remains, unlink the cycle members in lexical hash
order and continue propagating to their parents.

- [ ] **Step 5: Write failing reservation and touch tests**

Add:

```cpp
static void test_reservation_never_crosses_limit() {
    fixture f;
    f.segment_chain(3);
    const uint64_t max = f.managed_bytes();
    auto cache = f.open_cache(max);

    auto denied = cache.reserve(max + 1, {});
    assert(!denied.has_value());
    assert(cache.stats().bytes <= max);
    assert(cache.stats().writes_rejected == 1);
}

static void test_touch_is_throttled() {
    fixture f;
    const auto leaf = f.segment_chain(1).hashes.back();
    uint64_t touches = 0;
    auto clock = f.fake_steady_clock();
    auto cache = f.open_cache_with_touch_counter(touches, clock);

    cache.touch({llama_ggsd::object_kind::segment, leaf});
    cache.touch({llama_ggsd::object_kind::segment, leaf});
    assert(touches == 1);
    clock.advance(std::chrono::minutes(10));
    cache.touch({llama_ggsd::object_kind::segment, leaf});
    assert(touches == 2);
}
```

Also cover exact-limit admission, one-byte-over collection, the
`min(90%, max - reservation)` target, reservation destructor release, partial
commit, retained undeletable temporary bytes, oversize rejection without a GC
run, startup cleanup of stale temporaries, startup collection of an over-limit
pool, startup failure when undeletable bytes prevent the target, and touch
failure statistics.

- [ ] **Step 6: Implement reservation accounting and ten-minute touch throttling**

Use checked arithmetic for every byte transition. `reserve` follows:

```cpp
if (bytes > max_bytes_ || reserved_bytes_ > max_bytes_ - bytes) {
    ++stats_.writes_rejected;
    return std::nullopt;
}
const uint64_t room_for_committed = max_bytes_ - reserved_bytes_ - bytes;
if (stats_.bytes > room_for_committed) {
    const uint64_t low = max_bytes_ / 10 * 9 + (max_bytes_ % 10) * 9 / 10;
    const uint64_t target = std::min(low, room_for_committed);
    if (!collect_to(target, protected_ids).ok) {
        ++stats_.writes_rejected;
        return std::nullopt;
    }
}
reserved_bytes_ += bytes;
return cache_reservation(*this, bytes);
```

At the beginning of every `collect_to`, call `scan_pool` and replace the
transient graph plus current byte/object gauges from that fresh result. This is
what makes newly committed objects visible to later GC runs while keeping the
filesystem authoritative. Preserve cumulative counters across this
reconciliation. If the scan cannot establish trustworthy accounting, fail the
collection without unlinking anything.

Before `reserved_bytes_ += bytes`, explicitly assert/check that it cannot
overflow. `commit`, `retain_temporary`, and destructor release must reject an
amount greater than `remaining_` rather than underflowing.

Throttle touches with `std::chrono::steady_clock` in memory while writing the
persistent ordering signal through `std::filesystem::last_write_time`. The
fixed interval is `std::chrono::minutes(10)`. A failed touch must not enter the
throttle map, so the next use can retry.

- [ ] **Step 7: Run all manager tests**

Run:

```bash
cmake --build build-linux --target test-ggsd-cache -j
ctest --test-dir build-linux -R '^test-ggsd-cache$' --output-on-failure
```

Expected: graph, collection, reservation, protection, error-injection, and
touch tests PASS.

- [ ] **Step 8: Commit the cache manager**

```bash
git add src/llama-ggsd-cache.h src/llama-ggsd-cache.cpp tests/test-ggsd-cache.cpp
git commit -m "feat: add dependency-aware GGSD Leaf-LRU"
```

---

### Task 4: Expose Core Policy API and Integrate Standard Segment Save/Load

**Files:**

- Modify: `include/llama.h:896-930`
- Modify: `src/llama-context.h:18-22,184-192,288-360`
- Modify: `src/llama-context.cpp:1-20,481-485`
- Modify: `src/llama-state-incr.cpp:369-647,784-877,1112-1160`
- Modify: `tests/test-save-load-state.cpp:29-145,483-930,1050-1100`

**Interfaces:**

- Consumes: `llama_ggsd::cache::open`, `reserve`, `cache_reservation::commit`, `touch`, and `stats` from Task 3.
- Produces: public `llama_ggsd_cache_params`, `llama_ggsd_cache_stats`, `llama_ggsd_cache_default_params`, `llama_ggsd_cache_configure`, and `llama_ggsd_cache_get_stats`; Task 6 configures them and Task 7 exports their stats.

- [ ] **Step 1: Write failing public API and standard-path tests**

Extend `tests/test-save-load-state.cpp` with these cases, using its existing
temporary GGSD directory and tiny model:

```cpp
static bool test_ggsd_gc_default_and_configure(llama_model * model, const common_params & params) {
    incr_test_cleanup();
    auto ctx = llama_context_ptr{llama_init_from_model(model, incr_context_params(params))};
    auto policy = llama_ggsd_cache_default_params();
    if (policy.max_bytes != 0) return false;

    policy.max_bytes = 1;
    if (!llama_ggsd_cache_configure(ctx.get(), incr_path("session_gc.bin").c_str(), policy)) return false;

    llama_ggsd_cache_stats stats = {};
    if (!llama_ggsd_cache_get_stats(ctx.get(), incr_path("session_gc.bin").c_str(), &stats)) return false;
    return stats.limit_bytes == 1 && stats.bytes == 0;
}
```

Add distinct tests that decode 2048+ tokens and verify:

- a one-byte limit rejects a new head and keeps managed bytes at zero;
- a measured exact-size limit admits one segment;
- a second cold branch causes leaf collection before its write;
- the configured stats byte/count totals equal a fresh filesystem traversal;
- a successful restore advances the deepest segment mtime;
- estimate and a mismatching restore do not change mtime;
- an unconfigured context retains the existing unlimited behavior.

- [ ] **Step 2: Run the test to verify the C API is absent**

Run:

```bash
cmake --build build-linux --target test-save-load-state -j
```

Expected: FAIL to compile on `llama_ggsd_cache_default_params`.

- [ ] **Step 3: Add public structs and functions**

Insert before the existing incremental state functions in `include/llama.h`:

```c
struct llama_ggsd_cache_params {
    uint64_t max_bytes;
};

struct llama_ggsd_cache_stats {
    uint64_t bytes;
    uint64_t limit_bytes;
    uint64_t segments;
    uint64_t rec_snapshots;
    uint64_t gc_runs;
    uint64_t gc_deleted_bytes;
    uint64_t gc_failures;
    uint64_t writes_rejected;
    uint64_t touch_failures;
};

LLAMA_API struct llama_ggsd_cache_params llama_ggsd_cache_default_params(void);
LLAMA_API bool llama_ggsd_cache_configure(
        struct llama_context * ctx,
        const char * session_path,
        struct llama_ggsd_cache_params params);
LLAMA_API bool llama_ggsd_cache_get_stats(
        const struct llama_context * ctx,
        const char * session_path,
        struct llama_ggsd_cache_stats * stats);
```

Document that configuration is pre-request, path-scoped, single-writer, and
that zero disables the limit.

- [ ] **Step 4: Add per-context cache ownership**

Forward-declare `llama_ggsd::cache` in `src/llama-context.h`, then add:

```cpp
bool ggsd_cache_configure(const char * session_path, llama_ggsd_cache_params params);
bool ggsd_cache_get_stats(const char * session_path, llama_ggsd_cache_stats & stats) const;
llama_ggsd::cache * ggsd_cache_for_session(const char * session_path);

std::map<std::string, std::unique_ptr<llama_ggsd::cache>> ggsd_caches;
```

Canonicalize with `std::filesystem::weakly_canonical` after verifying that the
pool directory exists. A zero policy erases the canonical map entry. A positive
policy calls `cache::open`, including startup reconciliation and over-limit
collection, before inserting the entry.

Implement the public wrappers in `llama-state-incr.cpp`; catch exceptions,
log with the existing LLAMA logger, and return `false` without installing
partial state. Include `llama-ggsd-cache.h` in `llama-context.cpp` so the
out-of-line destructor sees the complete cache type. Translate every field from
the private `llama_ggsd::cache_stats` to `llama_ggsd_cache_stats` explicitly in
`ggsd_cache_get_stats`; do not expose or alias the private type in `llama.h`.

- [ ] **Step 5: Integrate exact reservation into standard writes**

Split payload measurement from `ggsd_write_segment` so the caller supplies the
already measured `payload_size`. Compute the full serialized size from the
header fields before opening `.tmp`.

For every missing segment:

```cpp
auto * pool = ggsd_cache_for_session(session_path);
std::optional<llama_ggsd::cache_reservation> reservation;
if (pool != nullptr) {
    const std::vector<llama_ggsd::object_id> protect = k > 0
        ? std::vector<llama_ggsd::object_id>{{llama_ggsd::object_kind::segment, hashes[k - 1]}}
        : std::vector<llama_ggsd::object_id>{};
    reservation = pool->reserve(file_size, protect);
    if (!reservation) break;
}

const auto temporary = ggsd_write_segment_temporary(/* existing args */, payload_size);
if (reservation) {
    std::string error;
    if (!reservation->finalize_temporary(
            llama_ggsd::object_kind::segment, temporary, fname, file_size, error)) {
        reservation->abandon_temporary(temporary, file_size);
        throw std::runtime_error(error);
    }
} else {
    std::filesystem::rename(temporary, fname);
}
```

Have `ggsd_write_segment_temporary` close and return the temp path without
renaming. On an exception after the temp path is known, call
`reservation->abandon_temporary(temporary, file_size)`; the reservation
destructor releases the unused portion. Touch
`hashes[n_chain - 1]` after a successful save with nonzero coverage, including
the all-objects-already-existed path.

- [ ] **Step 6: Touch successful standard restores only**

After computing the existing return value, touch the deepest verified segment
only when the returned coverage meets `min_prefix_tokens`:

```cpp
const size_t restored_segments = n_loaded + k0;
if (restored_segments > 0) {
    if (auto * pool = ggsd_cache_for_session(session_path)) {
        pool->touch({llama_ggsd::object_kind::segment, hashes[restored_segments - 1]});
    }
}
return restored_segments * llama_ggsd::SEGMENT_TOKENS;
```

Do not add a touch to `state_seq_load_incr_estimate`.

- [ ] **Step 7: Run standard integration and manager tests**

Run:

```bash
cmake --build build-linux --target test-save-load-state test-ggsd-cache -j
ctest --test-dir build-linux -R '^(test-save-load-state|test-ggsd-cache)$' --output-on-failure
```

Expected: both PASS; exact-limit and one-byte-over assertions confirm the hard
bound, and existing unconfigured tests remain unchanged.

- [ ] **Step 8: Commit the core API and standard integration**

```bash
git add include/llama.h src/llama-context.h src/llama-context.cpp src/llama-state-incr.cpp tests/test-save-load-state.cpp
git commit -m "feat: enforce GGSD quota on segment saves"
```

---

### Task 5: Add Grouped Hybrid Admission and Rec Touches

**Files:**

- Create: `tests/test-ggsd-hybrid-gc.cpp`
- Modify: `tests/CMakeLists.txt`
- Modify: `src/llama-state-incr.cpp:649-781,879-1013`
- Modify: `tests/test-ggsd-cache.cpp`
- Modify: `tests/test-save-load-state.cpp`

**Interfaces:**

- Consumes: configured core cache API and movable reservation from Tasks 3-4.
- Produces: all-or-none quota admission for newly required hybrid segments plus rec, partial-commit accounting after I/O failure, and rec-only touch after successful hybrid save or restore.

- [ ] **Step 1: Register a generated hybrid-model test**

Add a focused executable and CTest entry after the generated Qwen3.5 fixture is
defined in `tests/CMakeLists.txt`:

```cmake
llama_build(test-ggsd-hybrid-gc.cpp)
llama_test(
    test-ggsd-hybrid-gc
    LABEL "model"
    ARGS -m "${MODEL_DIR}/qwen35-dense.gguf")
set_tests_properties(test-ggsd-hybrid-gc PROPERTIES FIXTURES_REQUIRED generate-models)
```

- [ ] **Step 2: Write failing hybrid group tests**

In `tests/test-ggsd-hybrid-gc.cpp`, load the supplied model, decode at least
`SEGMENT_TOKENS + 32` deterministic tokens into sequence zero, and first save to
an unlimited measurement pool. Record the exact total size of its one segment
plus rec. Then use fresh pools to assert:

```cpp
policy.max_bytes = measured_group_size - 1;
assert(llama_ggsd_cache_configure(ctx.get(), session.c_str(), policy));
assert(llama_state_seq_save_incr(ctx.get(), session.c_str(), 0, tokens.data(), tokens.size()) == 0);
assert(count_seg_files(pool) == 0);
assert(count_rec_files(pool) == 0);

policy.max_bytes = measured_group_size;
assert(llama_ggsd_cache_configure(ctx2.get(), session2.c_str(), policy));
assert(llama_state_seq_save_incr(ctx2.get(), session2.c_str(), 0, tokens.data(), tokens.size()) == 1);
assert(count_seg_files(pool2) == 1);
assert(count_rec_files(pool2) == 1);
```

Restore from the admitted pool in a new context and assert full rec coverage,
then set the rec mtime to an old value and assert successful load refreshes it
while segment mtimes remain unchanged.

Decode the same deterministic sequence independently into each fresh context;
never reuse KV state from the unlimited measurement context. Also repeat a save
when the target rec already exists and assert that only the rec mtime advances.

Add a model-free group-reservation case in `test-ggsd-cache.cpp`: finalize a
segment temp successfully, inject failure for the following rec rename, abandon
the rec temp, and destroy the reservation. Assert the committed segment bytes
and count remain, the unconsumed rec reservation is released, and the measured
managed total does not cross the limit.

- [ ] **Step 3: Run and verify grouped admission fails**

Run:

```bash
cmake --build build-linux --target test-ggsd-hybrid-gc test-ggsd-cache -j
ctest --test-dir build-linux -R '^(test-ggsd-hybrid-gc|test-ggsd-cache)$' --output-on-failure
```

Expected: the hybrid test fails because the current implementation writes the
segment before discovering that the rec does not fit.

- [ ] **Step 4: Premeasure and reserve the hybrid write group**

In `state_seq_save_incr_hybrid`:

1. Build descriptors for every missing, cache-covered segment with its measured
   payload and complete serialized size.
2. Measure the target rec payload and serialized size before opening any temp.
3. Sum all descriptor sizes with overflow checks.
4. Protect the deepest existing aligned segment and reserve the full sum once.
5. If reservation fails, write nothing and return the existing on-disk aligned
   coverage.
6. Write each segment temp and call the group reservation's
   `finalize_temporary`, committing that segment's portion.
7. Write the rec temp last and call `finalize_temporary` for its portion.
8. On failure, retain any undeletable temp bytes; the reservation destructor
   releases every unconsumed byte.

Represent pending work explicitly:

```cpp
struct pending_segment {
    size_t index;
    uint64_t payload_size;
    uint64_t file_size;
};

std::vector<pending_segment> pending;
uint64_t group_size = rec_file_size;
for (const auto & item : pending) {
    if (group_size > UINT64_MAX - item.file_size) {
        LLAMA_LOG_ERROR("%s: hybrid GGSD write group size overflow\n", __func__);
        return (size_t) -1;
    }
    group_size += item.file_size;
}
```

- [ ] **Step 5: Touch only a successfully used hybrid rec**

After the full segment-plus-rec restore succeeds, derive the rec ID from
`best_path.filename()` and call:

```cpp
if (auto * pool = ggsd_cache_for_session(session_path)) {
    pool->touch({llama_ggsd::object_kind::rec, best_path.filename().string()});
}
```

Do not touch rec candidates during `state_seq_estimate_hybrid` or while scanning
non-selected headers. After hybrid save, touch the target rec when it was newly
committed or already existed. Do not touch it when quota admission, tail-cell
validation, write, or rename fails. A quota rejection before the first segment
returns the currently available aligned coverage (`0` is not an inference
failure) and emits the GGSD rejection log required by the design.

- [ ] **Step 6: Run hybrid, standard, and manager tests**

Run:

```bash
cmake --build build-linux --target test-ggsd-hybrid-gc test-save-load-state test-ggsd-cache -j
ctest --test-dir build-linux -R '^(test-ggsd-hybrid-gc|test-save-load-state|test-ggsd-cache)$' --output-on-failure
```

Expected: all PASS; the rejected hybrid pool contains neither new segment nor
rec, and admitted restore refreshes only the rec mtime.

- [ ] **Step 7: Commit hybrid admission**

```bash
git add src/llama-state-incr.cpp tests/test-ggsd-hybrid-gc.cpp tests/test-ggsd-cache.cpp tests/CMakeLists.txt
git commit -m "feat: reserve GGSD hybrid snapshots as a group"
```

---

### Task 6: Add Server CLI Configuration and Startup Enforcement

**Files:**

- Modify: `common/common.h:676-678`
- Modify: `common/arg.cpp:3575-3610`
- Modify: `tests/test-arg-parser.cpp:115-225`
- Modify: `tools/server/server-context.cpp:1268-1327`
- Modify: `tools/server/tests/utils.py:62-124,215-235`
- Modify: `tools/server/tests/unit/test_slot_save.py`

**Interfaces:**

- Consumes: `llama_ggsd_cache_configure` from Task 4.
- Produces: `common_params::prompt_cache_ssd_max_mib`, CLI/env parsing, pre-request server configuration, and Python test harness fields; Task 7 assumes configuration is live before metrics scraping.

- [ ] **Step 1: Write failing argument parser cases**

Add a self-contained block to `tests/test-arg-parser.cpp`:

```cpp
{
    common_params p;
    argv = {"binary_name", "--prompt-cache-ssd-max-mib", "4096"};
    assert(common_params_parse(argv.size(), list_str_to_char(argv).data(), p, LLAMA_EXAMPLE_SERVER));
    assert(p.prompt_cache_ssd_max_mib == 4096);

    for (const char * bad : {"-1", "+1", " 1", "1x", "abc", "18446744073709551616"}) {
        common_params invalid;
        argv = {"binary_name", "--prompt-cache-ssd-max-mib", bad};
        assert(!common_params_parse(argv.size(), list_str_to_char(argv).data(), invalid, LLAMA_EXAMPLE_SERVER));
    }
}
```

Also assert that zero parses and that a MiB value larger than
`UINT64_MAX / (1024 * 1024)` is rejected. In the existing non-Windows
environment-variable block, set `LLAMA_ARG_PROMPT_CACHE_SSD_MAX_MIB=2048`,
parse an argument list containing only `binary_name`, and assert the field is
2048; then set it to `-1`, assert parsing fails, and reset it to `0` before the
rest of the process's parser cases.

- [ ] **Step 2: Run the parser test and verify failure**

Run:

```bash
cmake --build build-linux --target test-arg-parser -j
build-linux/bin/test-arg-parser
```

Expected: FAIL to compile because `prompt_cache_ssd_max_mib` is absent.

- [ ] **Step 3: Add overflow-safe CLI and environment parsing**

Add to `common_params`:

```cpp
uint64_t prompt_cache_ssd_max_mib = 0;
```

Register a string-valued option so parsing is not truncated through the
existing `int` handler; add `<charconv>` for strict, locale-independent parsing:

```cpp
add_opt(common_arg(
    {"--prompt-cache-ssd-max-mib"}, "N",
    "set the GGSD managed-cache hard limit in MiB (default: 0, unlimited)",
    [](common_params & params, const std::string & value) {
        if (value.empty() || !std::all_of(value.begin(), value.end(), [](unsigned char c) {
                return c >= '0' && c <= '9';
            })) {
            throw std::invalid_argument("GGSD cache limit must be decimal digits");
        }
        uint64_t mib = 0;
        const auto parsed = std::from_chars(value.data(), value.data() + value.size(), mib, 10);
        if (parsed.ec != std::errc() || parsed.ptr != value.data() + value.size() ||
                mib > UINT64_MAX / (1024ULL * 1024ULL)) {
            throw std::invalid_argument("GGSD cache limit is out of range");
        }
        params.prompt_cache_ssd_max_mib = mib;
    }
).set_examples({LLAMA_EXAMPLE_SERVER})
 .set_env("LLAMA_ARG_PROMPT_CACHE_SSD_MAX_MIB"));
```

- [ ] **Step 4: Configure the pool before `server_context_impl::init`**

After `ctx_tgt` exists and before `init()` wires request processing:

```cpp
if (params_base.prompt_cache_ssd_max_mib > 0) {
    if (params_base.slot_save_path.empty()) {
        SRV_ERR("%s", "--prompt-cache-ssd-max-mib requires --slot-save-path\n");
        return false;
    }
    llama_ggsd_cache_params gc = llama_ggsd_cache_default_params();
    gc.max_bytes = params_base.prompt_cache_ssd_max_mib * 1024ULL * 1024ULL;
    const std::string session = params_base.slot_save_path + "session___autosave__.bin";
    if (!llama_ggsd_cache_configure(ctx_tgt, session.c_str(), gc)) {
        SRV_ERR("%s", "failed to configure GGSD cache hard limit\n");
        return false;
    }
}
```

Do not require `--prompt-cache-ssd`; manual `save_incr` and `restore_incr` must
be quota-managed when only the max and slot path are configured.

- [ ] **Step 5: Extend the Python server harness and write startup/manual tests**

Add fields and argument emission in `tools/server/tests/utils.py`:

```python
prompt_cache_ssd: bool = False
prompt_cache_ssd_max_mib: int | None = None

if self.prompt_cache_ssd:
    server_args.append("--prompt-cache-ssd")
if self.prompt_cache_ssd_max_mib is not None:
    server_args.extend(["--prompt-cache-ssd-max-mib", self.prompt_cache_ssd_max_mib])
```

In `test_slot_save.py`, add a test that starts with a slot path and positive
limit but without autosave, creates a prompt longer than 1024 tokens, invokes
`action=save_incr`, verifies success, stops the server, restarts with the same
path/limit, and invokes `action=restore_incr`. Assert every recognized GGSD file
keeps the total at or below `limit_mib * 1024 * 1024`.

Use slot `0`, `n_slots = 1`, `n_ctx = 2048`, `n_predict = 1`, a deterministic
repeated-text prompt whose returned `prompt_n` is greater than 1024, and a
64-MiB test limit. Send `{"filename": "gc-manual"}` to `save_incr` and
`{"filename": "gc-manual", "prompt": prompt, "min_prefix": 1024}` to
`restore_incr`; assert `n_tokens_restored >= 1024`. Walk only `seg/*/*`,
`rec/*/*`, and matching `*.tmp` files when summing managed logical bytes.

- [ ] **Step 6: Run parser and focused server tests**

Run:

```bash
cmake --build build-linux --target test-arg-parser llama-server -j
build-linux/bin/test-arg-parser
LLAMA_SERVER_BIN_PATH="$PWD/build-linux/bin/llama-server" \
  ./tools/server/tests/tests.sh unit/test_slot_save.py -k ggsd
```

Expected: parser PASS; manual-only quota configuration survives restart and
the restored prefix is usable.

- [ ] **Step 7: Commit server configuration**

```bash
git add common/common.h common/arg.cpp tests/test-arg-parser.cpp tools/server/server-context.cpp tools/server/tests/utils.py tools/server/tests/unit/test_slot_save.py
git commit -m "feat: configure GGSD disk quota in server"
```

---

### Task 7: Export GGSD Logs and Prometheus Metrics

**Files:**

- Modify: `tools/server/server-common.h:431-492`
- Modify: `tools/server/server-context.cpp:826-832,2518-2538,4142-4245,5719-5737`
- Modify: `tools/server/server-task.cpp:1522-1603`
- Modify: `tools/server/tests/unit/test_metrics.py:38-85`
- Modify: `tests/test-ggsd-cache.cpp`

**Interfaces:**

- Consumes: `llama_ggsd_cache_get_stats` and configured session path from Tasks 4 and 6.
- Produces: a consistent metrics snapshot and the nine `llamacpp:ggsd_*` series required by the spec.

- [ ] **Step 1: Write failing metric-name/type assertions**

Extend `test_metrics_prometheus_format`:

```python
expected_counters += [
    "llamacpp:ggsd_gc_runs_total",
    "llamacpp:ggsd_gc_deleted_bytes_total",
    "llamacpp:ggsd_gc_failures_total",
    "llamacpp:ggsd_cache_writes_rejected_total",
    "llamacpp:ggsd_cache_touch_failures_total",
]
expected_gauges += [
    "llamacpp:ggsd_cache_bytes",
    "llamacpp:ggsd_cache_limit_bytes",
    "llamacpp:ggsd_cache_segments",
    "llamacpp:ggsd_cache_rec_snapshots",
]
```

Add a configured-GGSD test asserting `limit_bytes == max_mib * 1024 * 1024`,
cache bytes never exceed it, and counters remain cumulative across two metric
scrapes.

- [ ] **Step 2: Run and verify metrics are missing**

Run:

```bash
cmake --build build-linux --target llama-server -j
LLAMA_SERVER_BIN_PATH="$PWD/build-linux/bin/llama-server" \
  ./tools/server/tests/tests.sh unit/test_metrics.py::test_metrics_prometheus_format
```

Expected: FAIL because `llamacpp:ggsd_cache_bytes` has no TYPE entry.

- [ ] **Step 3: Snapshot core stats in `server_metrics`**

Add fields:

```cpp
bool ggsd_cache_configured = false;
llama_ggsd_cache_stats ggsd_cache = {};
```

Add `refresh_ggsd_metrics()` in `server_context_impl`; it zeroes the snapshot,
then queries the core only when `prompt_cache_ssd_max_mib > 0`. Call it before
copying `metrics` into a metrics task result and before `get_metrics()` returns
the cached sleep snapshot. Do not reset GGSD counters in `reset_bucket()`.

Make the private accessor non-const so the refresh is explicit:

```cpp
server_metrics get_metrics() {
    refresh_ggsd_metrics();
    return metrics;
}
```

In `SERVER_TASK_TYPE_METRICS`, call `refresh_ggsd_metrics()` immediately before
`res->metrics = metrics`. The queue remains the serialization boundary; add no
mutex or sampling thread.

- [ ] **Step 4: Render the exact counters and gauges**

Append these counter items in `server_task_result_metrics::to_metrics()`:

```cpp
{"ggsd_gc_runs_total", "Number of GGSD garbage collections", (double) metrics.ggsd_cache.gc_runs},
{"ggsd_gc_deleted_bytes_total", "GGSD bytes deleted by garbage collection", (double) metrics.ggsd_cache.gc_deleted_bytes},
{"ggsd_gc_failures_total", "GGSD garbage collection operation failures", (double) metrics.ggsd_cache.gc_failures},
{"ggsd_cache_writes_rejected_total", "GGSD cache writes rejected by quota", (double) metrics.ggsd_cache.writes_rejected},
{"ggsd_cache_touch_failures_total", "GGSD last-use metadata update failures", (double) metrics.ggsd_cache.touch_failures},
```

Append these gauges:

```cpp
{"ggsd_cache_bytes", "Current GGSD managed bytes", (double) metrics.ggsd_cache.bytes},
{"ggsd_cache_limit_bytes", "Configured GGSD managed-byte hard limit", (double) metrics.ggsd_cache.limit_bytes},
{"ggsd_cache_segments", "Current GGSD segment objects", (double) metrics.ggsd_cache.segments},
{"ggsd_cache_rec_snapshots", "Current GGSD recurrent snapshot objects", (double) metrics.ggsd_cache.rec_snapshots},
```

Expose zero-valued series when GC is unconfigured so the metrics schema is
stable.

- [ ] **Step 5: Emit one summary log per collection**

At the end of `collect_to`, emit one `LLAMA_LOG_INFO` or `LLAMA_LOG_WARN` record
with prefix `__GGSD__ gc:` and the fields `reason`, `bytes_before`,
`bytes_target`, `bytes_after`, `seg_scanned`, `rec_scanned`, `invalid_found`,
`objects_deleted`, `bytes_deleted`, `protected_skipped`, and `duration_ms`.
Pass `configure` or `reservation` as an enum/string argument from the caller;
do not emit per-file info logs.

Extend `test-ggsd-cache` with a temporary log callback. Trigger one successful
collection and one target-not-reached collection, asserting each produces
exactly one summary record, the success uses info severity, the failure uses
warning severity, and every required key occurs once in the record.

- [ ] **Step 6: Run metrics and core tests**

Run:

```bash
cmake --build build-linux --target llama-server test-ggsd-cache -j
ctest --test-dir build-linux -R '^test-ggsd-cache$' --output-on-failure
LLAMA_SERVER_BIN_PATH="$PWD/build-linux/bin/llama-server" \
  ./tools/server/tests/tests.sh unit/test_metrics.py
```

Expected: all tests PASS; GGSD series have correct types/help lines and remain
available while the server metrics response is cached for sleep.

- [ ] **Step 7: Commit observability**

```bash
git add tools/server/server-common.h tools/server/server-context.cpp tools/server/server-task.cpp tools/server/tests/unit/test_metrics.py src/llama-ggsd-cache.h src/llama-ggsd-cache.cpp
git commit -m "feat: expose GGSD cache GC metrics"
```

---

### Task 8: Document, Benchmark, and Verify the Complete Feature

**Files:**

- Modify: `tests/test-ggsd-cache.cpp`
- Modify: `tools/server/README.md:210-225,1116-1160`
- Modify: `docs/ggsd-guide.md:23-47,150-208`
- Modify: `docs/ggsd-autoload-guide.md:31-86,152-185`

**Interfaces:**

- Consumes: the completed API, CLI, collector, logs, and metrics from Tasks 1-7.
- Produces: reproducible synthetic scale output, user/operator documentation, and final verification evidence.

- [ ] **Step 1: Add a non-CTest benchmark mode to the model-free test binary**

Make `test-ggsd-cache` accept `--bench N`. The default no-argument path still
runs assertions only. Benchmark mode creates `N` total valid objects distributed
by hash across shards, with forked segments and a rec approximately every
hundredth object, opens the cache at its current managed size, then explicitly
collects to 90% of that size. Print exactly:

```text
objects=<N> enumerate_ms=<...> validate_ms=<...> candidate_ms=<...> delete_ms=<...> peak_graph_bytes=<...> deleted_bytes=<...>
```

Use `std::chrono::steady_clock` and report the graph's explicit capacity-based
memory estimate; do not claim process RSS. Assert after collection that a fresh
scan is prefix-closed and bytes are at or below the requested target.

Also make `test-save-load-state` consume and remove a test-only
`--ggsd-bench-iterations N` argument before calling `common_params_parse`.
Benchmark mode prepares one 2048-token standard GGSD chain, warms both paths,
then measures repeated all-content-already-exists saves and repeated successful
restores with `steady_clock`. It skips the ordinary test suite and prints:

```text
iterations=<N> steady_save_median_us=<...> steady_save_p95_us=<...> steady_load_median_us=<...> steady_load_p95_us=<...>
```

Keep this mode out of CTest so timing remains diagnostic, not a pass/fail budget.

- [ ] **Step 2: Run representative scale measurements**

Run:

```bash
cmake --build build-linux --target test-ggsd-cache -j
build-linux/bin/test-ggsd-cache --bench 1000
build-linux/bin/test-ggsd-cache --bench 10000
build-linux/bin/test-save-load-state \
  -m build-linux/tinyllamas/stories15M-q4_0.gguf \
  --ggsd-bench-iterations 10
```

Expected: all three exit zero, print the named values, and show increasing scan
and graph memory without a super-linear jump caused by repeated full-chain
searches. On big-endian s390x, use the CTest-configured
`stories15M-be.Q4_0.gguf` path for the model benchmark.

- [ ] **Step 3: Update server and GGSD documentation**

Document all of the following verbatim in the appropriate files:

- `--prompt-cache-ssd-max-mib N`, default `0`, positive hard limit, and the
  `LLAMA_ARG_PROMPT_CACHE_SSD_MAX_MIB` environment variable.
- The limit applies to recognized `seg`, `rec`, and temp bytes, not the entire
  save directory or filesystem allocation overhead.
- GC triggers synchronously before an over-limit write and collects to a fixed
  90% low watermark.
- Leaf-LRU preserves shared prefixes and treats manual/automatic saves equally.
- Successful standard restore touches the deepest segment; successful hybrid
  restore touches the selected rec; touch is throttled to ten minutes.
- Quota rejection loses cache coverage only; inference falls back to prefill.
- Positive max works for manual endpoints without enabling autosave/autoload.
- Single-writer and no-background-thread constraints remain.
- List the nine Prometheus metric names from Task 7.

Remove the old statement that GGSD has no automatic garbage collection, but
retain manual deletion guidance for unlimited mode.

- [ ] **Step 4: Run focused native tests**

Run:

```bash
cmake --build build-linux --target test-ggsd-cache test-save-load-state test-ggsd-hybrid-gc test-arg-parser -j
ctest --test-dir build-linux -R '^(test-ggsd-cache|test-save-load-state|test-ggsd-hybrid-gc|test-arg-parser)$' --output-on-failure
```

Expected: all four targets PASS.

- [ ] **Step 5: Run focused server tests**

Run:

```bash
cmake --build build-linux --target llama-server -j
LLAMA_SERVER_BIN_PATH="$PWD/build-linux/bin/llama-server" \
  ./tools/server/tests/tests.sh unit/test_metrics.py unit/test_slot_save.py
```

Expected: both Python modules PASS.

- [ ] **Step 6: Run broad regression and static checks**

Run:

```bash
ctest --test-dir build-linux --output-on-failure
git diff --check
cmake --build build-linux -j
```

Expected: CTest reports zero failures, `git diff --check` prints nothing, and
the complete configured build succeeds.

- [ ] **Step 7: Audit filesystem and quota invariants in the test suite**

Run the no-argument model-free binary once more. Every collection, startup, and
failure-injection case must finish with a fresh `scan_pool` assertion that
`managed_bytes <= limit_bytes`, no valid node has a missing/invalid ancestor,
no valid rec has a missing tail, and no removable temporary remains. Test
fixtures may delete only their own temporary directories; verification must
never clean a configured user save path.

- [ ] **Step 8: Commit docs and benchmark harness**

```bash
git add tests/test-ggsd-cache.cpp tests/test-save-load-state.cpp tools/server/README.md docs/ggsd-guide.md docs/ggsd-autoload-guide.md
git commit -m "docs: describe automatic GGSD cache GC"
```

- [ ] **Step 9: Review the final commit range**

Run:

```bash
git log --oneline --decorate -8
git diff --stat cdb0100ae..HEAD
git status --short
```

Expected: task-sized commits are present, the diff contains only the planned
GGSD GC/test/docs files, and the worktree is clean.
