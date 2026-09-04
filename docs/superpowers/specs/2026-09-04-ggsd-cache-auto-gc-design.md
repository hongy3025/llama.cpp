# GGSD Cache Automatic GC Design

Date: 2026-09-04

Status: Approved design, awaiting written-spec review

Related:

- `docs/superpowers/specs/2026-07-31-ggsd-incremental-slot-storage-design.md`
- `docs/superpowers/specs/2026-09-03-ggsd-autoload-design.md`
- `docs/superpowers/specs/2026-09-03-ggsd-hybrid-split-mode-design.md`

## 1. Problem Statement

GGSD stores immutable, content-addressed cache objects under one pool:

```text
<pool>/seg/<hh>/<hash>   aligned 1024-token attention KV segments
<pool>/rec/<hh>/<hash>   hybrid attention-tail plus recurrent snapshots
```

Forks retain old branches, autosave creates new restore points, and no existing
mechanism reclaims old objects. The pool therefore grows monotonically.

The cache cannot be collected as a set of independent files. Each `seg` names
its predecessor, so a retained segment is useful only when all of its ancestors
remain. A hybrid `rec` additionally depends on the complete aligned `seg` chain
under its tail. Deleting an interior segment while keeping its descendants
leaves files that consume space but can never be restored.

The new mechanism must enforce a configured hard byte limit without affecting
inference correctness. Manual `save_incr` and server autosave have identical
retention and eviction priority. There are no pinned cache entries.

## 2. Goals

- Enforce a hard upper bound on GGSD-managed disk bytes, including in-progress
  temporary object writes.
- Preserve the segment-prefix closure after every successful GC operation:
  every retained `seg` has every ancestor, and every retained `rec` has its
  required aligned segment chain.
- Use least-recently-used retention while respecting the dependency graph and
  shared prefixes.
- Keep GGSD files as the only persistent source of truth. Rebuild all GC state
  by scanning the pool; do not add a database, journal, or durable index.
- Keep GC in the existing single-writer execution domain. Do not introduce a
  background GC thread.
- Treat GC and quota failures as cache-admission failures, never as inference
  correctness failures.
- Keep unconfigured C API users and `--prompt-cache-ssd-max-mib 0` behavior
  compatible with the current unlimited pool.
- Provide enough logging and metrics to evaluate scan cost, reclamation,
  rejected writes, and whether a future persistent index is justified.

## 3. Non-Goals

- Multi-process or multi-writer access to one pool.
- Per-session, per-model, per-tenant, or manual-versus-automatic priorities.
- Pinned entries, TTL retention, LFU/ARC/TinyLFU, or benefit/cost scoring.
- A manual GC HTTP endpoint or runtime changes to the configured limit.
- A persistent root set, catalog, database, or access journal.
- Managing `session_*.bin`, traditional slot-state files, or arbitrary user
  files stored beside the GGSD pool.
- Changing GGSD payload formats solely for GC. Existing `rec` dependencies are
  reconstructed from their stored token arrays.

## 4. Chosen Policy: Dependency-Aware Leaf-LRU

The collector may delete:

1. Any unprotected `rec`, ordered by last use.
2. An unprotected `seg` only when it has no retained child segment and no
   retained `rec` dependency.

After a leaf segment is deleted, its parent may become a leaf and enter the
candidate set. After a `rec` is deleted, its required tail segment loses that
dependency and may become a candidate. This peels cold branches from their
leaves toward shared ancestors and never creates a new broken chain.

The scheme is preferred over a rooted mark-and-sweep collector because current
GGSD restore intentionally does not depend on session manifests. It is preferred
over a persistent catalog because pressure-triggered scans preserve the current
filesystem-as-truth architecture and have much simpler crash recovery.

## 5. Capacity Semantics

### 5.1 Managed bytes

The hard limit covers:

- regular content objects under `seg/<hh>/` and `rec/<hh>/` whose names are
  exactly 32 lowercase hexadecimal characters;
- GGSD temporary files created while writing those objects.

It excludes `session_*.bin`, traditional slot-state files, directories, and
unrecognized user files. Consequently the option bounds GGSD content bytes,
not the total apparent size of `--slot-save-path`.

File size means logical file size. GGSD object files are not sparse, so this is
also the useful cache-data measure; filesystem allocation overhead and directory
metadata are outside the promised bound.

### 5.2 Watermarks and reservation

`max_bytes` is the high watermark. The low watermark is fixed at 90% of
`max_bytes` in the first version.

Before creating a temporary object, the write path obtains its exact serialized
size using the existing dummy writer. Let:

```text
M = max_bytes
L = floor(0.90 * M)
U = current committed managed bytes
R = bytes reserved by the planned write or hybrid write group
```

If `U + R <= M`, the write may proceed without collection. Otherwise GC chooses
a pre-write target:

```text
target = min(L, M - R)
```

and deletes candidates until `U <= target`. If `R > M`, or collection cannot
reach the target, admission is rejected. A successful reservation is counted
before the `.tmp` file is created, so temporary writes cannot exceed the bound.
On rename, reserved bytes become committed bytes. A failed write releases the
reservation and removes its temporary file when possible.

The target leaves ordinary high/low-watermark headroom. An unusually large
object may finish above the low watermark but never above the high watermark.

### 5.3 Startup behavior

Configuration performs a synchronous initial scan before the server accepts
requests. It:

1. Canonicalizes the pool path.
2. Enumerates managed `seg`, `rec`, and temporary objects.
3. Removes stale GGSD temporary files. A temporary file that cannot be removed
   remains counted against the limit.
4. Builds and validates the dependency graph.
5. Reclaims invalid and cold objects if the existing pool exceeds the limit,
   stopping at the 90% low watermark.

If the scan cannot establish trustworthy accounting, or an over-limit pool
cannot be reduced to the target, explicit quota configuration fails. The server
must then fail startup rather than silently run without its requested limit.

## 6. Core API and Ownership

The core exposes pool policy while retaining the existing save/load entry
points:

```c
struct llama_ggsd_cache_params {
    uint64_t max_bytes; // 0 disables quota and GC
};

LLAMA_API struct llama_ggsd_cache_params
llama_ggsd_cache_default_params(void);

LLAMA_API bool llama_ggsd_cache_configure(
        struct llama_context * ctx,
        const char * session_path,
        struct llama_ggsd_cache_params params);
```

`session_path` retains its existing meaning: its normalized parent directory is
the pool identity. Configuration is stored in the `llama_context`, keyed by the
canonical pool path. Subsequent existing `llama_state_seq_save_incr` and
`llama_state_seq_load_incr` calls resolve their session path to that state and
automatically apply admission, GC, and access touches.

Callers that never configure a pool retain unlimited behavior and incur no pool
scan. Passing `max_bytes == 0` disables and removes quota state for that pool.
Configuration must occur before concurrent request processing and is not a
runtime reconfiguration API.

The core also exposes read-only statistics so server metrics do not duplicate
accounting:

```c
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

LLAMA_API bool llama_ggsd_cache_get_stats(
        const struct llama_context * ctx,
        const char * session_path,
        struct llama_ggsd_cache_stats * stats);
```

No manual `collect` API is introduced.

## 7. Server Configuration

The server adds one option:

```text
--prompt-cache-ssd-max-mib N
```

- `N > 0` enables quota enforcement for the pool selected by
  `--slot-save-path`.
- `N == 0` is the default and preserves unlimited behavior.
- Negative values and multiplication overflow are configuration errors.
- A positive value requires `--slot-save-path`. It applies to manual GGSD
  operations even when `--prompt-cache-ssd` (autosave/autoload) is disabled.

After creating the target context and before serving requests, the server calls
`llama_ggsd_cache_configure`. Manual `save_incr`, manual `restore_incr`,
autosave, and autoload subsequently use the same configured core pool and the
same policy. Their source does not affect last-use time or eviction priority.

## 8. Transient Graph Model

GC builds the following in-memory nodes:

```text
segment:
    hash
    parent_hash
    segment_index
    bytes
    last_used
    child_count
    rec_ref_count
    valid
    protected

rec snapshot:
    hash
    bytes
    last_used
    required_tail_segment_hash (optional when chain length is zero)
    valid
    protected
```

For segments, the path supplies the object hash and the existing header supplies
`prev_hash` and `segment_index`. For a hybrid rec, GC validates bounded header
fields, reads its stored token array, and uses the existing GGSD chain helper to
recompute the last aligned segment hash. Only that tail receives a rec reference;
its ancestry is protected transitively by segment parent edges.

Different model/KV identities naturally form separate trees because those
identities participate in object hashes. All trees compete in one LRU order and
one pool-wide quota.

### 8.1 Invalid graphs

A malformed rec is directly reclaimable. A malformed segment, a segment with a
missing parent, a parent/index mismatch, a dependency cycle, or a rec with a
missing required tail makes the dependent subtree unusable. GC detects cycles
during graph validation, marks the affected descendants and recs invalid, and
prioritizes them for deletion. A cyclic invalid component has no valid leaf, so
its members are unlinked in deterministic hash order after every incoming valid
dependency and dependent rec has been removed. Acyclic invalid subtrees still
delete children before parents. The end state restores prefix closure without
leaving newly stranded files.

Unrecognized files are ignored rather than guessed to be GGSD objects.

## 9. Last-Use Tracking

Filesystem access time is not used because `noatime` and `relatime` make it an
unreliable cache signal. The object mtime is explicitly updated:

- successful standard restore: touch the deepest restored segment;
- successful hybrid restore: touch the selected rec snapshot;
- new object: its creation mtime is its initial last-use time;
- a successful save whose content already exists: touch the deepest usable
  restore object for that save;
- estimate, miss, rejected restore, and corrupt candidate: do not touch.

Touching only the deepest standard segment is sufficient: its parent chain is
protected by dependency closure. Touching only a hybrid rec similarly protects
its segment chain.

Touches are rate-limited in memory to at most once per object per ten minutes.
A touch failure increments a counter and emits a warning but does not undo a
successful restore or save. Wall-clock jumps may make LRU ordering temporarily
imperfect; they cannot violate dependency or capacity invariants.

## 10. Collection Algorithm

After graph validation, initialize a deterministic oldest-first priority queue
with:

- every unprotected rec;
- every unprotected segment with `child_count == 0` and
  `rec_ref_count == 0`.

Candidate order is:

1. invalid before valid;
2. older mtime before newer mtime;
3. larger byte size first when mtimes are equal;
4. lexical hash order as the final deterministic tie-breaker.

For each selected candidate:

```text
if candidate is rec:
    unlink rec
    committed bytes -= rec bytes
    decrement required tail segment rec_ref_count
    enqueue the tail if it is now an unprotected leaf

if candidate is segment:
    assert child_count == 0 and rec_ref_count == 0
    unlink segment
    committed bytes -= segment bytes
    decrement parent child_count
    enqueue the parent if it is now an unprotected leaf
```

An unlink failure records a GC failure, leaves accounting unchanged, and moves
to the next candidate. Collection succeeds only if it reaches the requested
target. Otherwise the pending cache write is rejected.

### 10.1 Operation protection

When save triggers collection, the deepest existing object on the current target
chain is protected for that operation. Its ancestors are protected transitively.
Objects already staged or committed by the operation are also protected. This
prevents admission for a new tail from deleting the prefix that the same save
is extending.

If protected data leaves insufficient reclaimable space, the new object is not
admitted. GC never evicts the current chain in order to append to that chain.

Load does not trigger GC, and GC has no background thread, so no load-time file
lease is required under the single-writer contract.

## 11. Save and Load Data Flows

### 11.1 Standard save

1. Compute the prompt hash chain and discover the existing prefix as today.
2. For each new segment, determine exact serialized size before opening `.tmp`.
3. Protect the existing target prefix.
4. Reserve space, collecting when required.
5. If reservation fails, stop appending and return the actual existing chain
   coverage, matching the current best-effort partial-cache behavior.
6. Write `.tmp`, rename, and commit accounting.
7. Touch the deepest usable persisted segment for this save, subject to
   throttling.

If no head segment exists and the head cannot be admitted, the existing `-1`
failure behavior remains.

### 11.2 Hybrid save

Hybrid admission treats all newly required aligned segments plus the target rec
as one reservation group. Sizes are measured before writing, and the current
prefix is protected. If the group cannot fit, no new member of the group is
written.

After reservation, segment temporary files are written and renamed first; the
rec is renamed last. On a mid-commit I/O failure, already renamed immutable
segments remain valid and are committed to accounting, unfinished temporaries
are removed, and unused reservation is released. A later GC can reclaim any
unreferenced segment leaves.

If quota rejects the rec snapshot, the function returns the currently available
aligned segment coverage and logs that the target hybrid restore point was not
cached. This remains a cache-admission result, not an inference failure.

### 11.3 Load

Estimate remains read-only and never refreshes mtime. Actual restore follows
existing validation and replay rules. Only after the restore returns usable
coverage does core update mtime:

- deepest restored segment for standard KV;
- selected rec for hybrid.

An exception, corrupt object, or result below `min_prefix` does not refresh the
candidate. Failure to update mtime is non-fatal.

## 12. Error and Crash Semantics

| Condition | Behavior |
|---|---|
| Initial directory scan fails | Explicit quota configuration fails; server startup fails |
| Stale `.tmp` exists | Delete during configure; count it if deletion fails |
| Invalid object or broken dependency | Prioritize its dependent closure for leaf-first deletion |
| Candidate unlink fails | Log, count failure, try other candidates |
| Target cannot be reached | Reject the pending cache write |
| Single object exceeds limit | Reject that object without collection churn |
| Object write/rename fails | Remove temp when possible and release unused reservation |
| mtime touch fails | Keep successful cache operation; log and count |
| Process dies during GC | Deleted leaves remain deleted; startup rebuilds state |
| Process dies during write | Final immutable files remain truth; stale temp is handled at startup |
| Session hint names an evicted tail | Leave it stale; existing presence checks rediscover actual coverage |
| Underlying filesystem runs out of space first | Normal write failure and reservation cleanup |

Each unlink is independently safe. GC needs no transaction log or rollback.

## 13. Concurrency Contract

- Configuration occurs before request processing.
- Save, load, touch, and GC execute synchronously in the caller's existing
  serialized task path.
- There is no collector thread and no concurrent unlink/read race inside a
  supported server instance.
- Sharing one pool between processes or independent contexts remains unsupported,
  as it already is for GGSD writes. External mutation may make in-memory byte
  accounting stale and is outside the contract; restart/configure rescans it.

## 14. Observability

Every collection emits one summary record with a consistent `__GGSD__ gc`
prefix and at least:

```text
reason=configure|reservation
bytes_before
bytes_target
bytes_after
seg_scanned
rec_scanned
invalid_found
objects_deleted
bytes_deleted
protected_skipped
duration_ms
```

Server metrics expose:

```text
ggsd_cache_bytes
ggsd_cache_limit_bytes
ggsd_cache_segments
ggsd_cache_rec_snapshots
ggsd_gc_runs_total
ggsd_gc_deleted_bytes_total
ggsd_gc_failures_total
ggsd_cache_writes_rejected_total
ggsd_cache_touch_failures_total
```

Autosave logs quota rejection and continues serving. Manual save retains its
existing response semantics: standard mode reports actual aligned coverage;
hybrid mode continues to report aligned segment coverage, while a rejected rec
snapshot is explicit in the GGSD log. Both use exactly the same admission and
eviction policy.

## 15. Testing

### 15.1 Graph algorithm tests

- A single chain is evicted from tail to head.
- Forks retain shared ancestors until the final child disappears.
- Removing a child promotes its parent to leaf candidacy.
- A rec protects its tail and all ancestors; deleting it releases that chain.
- A hybrid rec with zero aligned segments is independently evictable.
- Protected nodes never become candidates.
- Equal-time candidates use size and hash tie-breakers deterministically.
- Collection stops at its target.
- Randomly generated trees retain the prefix-closure invariant after every
  deletion.

### 15.2 Filesystem and reconstruction tests

- Sharded enumeration counts managed bytes and ignores unrelated files.
- Startup removes stale temporaries and accounts for undeletable ones.
- Current rec files reconstruct their segment dependency from bounded token
  arrays.
- Invalid headers, missing parents, index mismatches, dependency cycles, and
  missing rec tails are detected and reclaimed under the invalid-graph rules.
- Injected unlink, write, rename, and touch errors follow the specified behavior.
- Restart after partial GC or write rebuilds the same committed accounting.
- In-memory accounting matches a fresh filesystem rescan after every scenario.

### 15.3 Reservation tests

- A write that exactly reaches `max_bytes` succeeds.
- Exceeding by one byte triggers collection.
- Collection reaches the computed target before admission.
- A single oversize object is rejected.
- Protected data can force rejection without being evicted.
- Temporary-file bytes remain within the hard bound.
- Failed writes release reservations.
- Hybrid group rejection writes no new group members.
- An over-limit startup pool is reduced before requests are accepted.

### 15.4 Restore and server tests

- A retained standard branch restores fully after GC.
- An evicted branch restores only through its remaining shared prefix and then
  prefills normally with unchanged inference correctness.
- A retained hybrid rec and its segment dependencies restore fully.
- Removing the last rec dependency makes its cold exclusive segments eligible.
- Manual and automatic saves enter the same LRU order.
- Successful load touches; estimate, miss, rejection, and corruption do not.
- The ten-minute touch window suppresses repeat metadata writes.
- CLI zero keeps unlimited behavior, positive values enable quota, and negative
  or overflowing values fail validation.
- Configuration failure prevents server startup.
- Logs and exported metrics match observed files and operations.
- Existing GGSD roundtrip, append, fork, corrupt, mismatch, and differential
  tests continue to pass.

### 15.5 Scale measurements

Synthetic pools with increasing segment and rec counts record startup scan time,
graph construction time, candidate-queue time, deletion time, peak memory, and
steady-state save/load latency. Acceptance is structural rather than tied to an
unmeasured millisecond number:

- a save that does not require collection never scans the entire pool;
- estimate and miss add no persistent writes;
- one hot object is touched at most once per throttle window;
- scan time and memory grow linearly with object and edge count;
- measurements are reproducible through benchmark output and GC logs.

## 16. Acceptance Criteria

1. After successful configuration, managed bytes never exceed the hard limit at
   any point in an object write.
2. Every GC deletion preserves or restores segment-prefix closure.
3. GC can reduce reuse coverage but cannot change inference correctness.
4. Manual and automatic save sources receive identical policy.
5. A restart reconstructs accounting and dependencies from files alone.
6. Unconfigured and zero-limit operation retains existing unlimited behavior.
7. No database, durable index, background GC thread, or new multi-writer promise
   is introduced.

## 17. Anticipated Implementation Areas

- `include/llama.h`: cache policy and statistics C API.
- `src/llama-context.h`: per-context pool-state ownership.
- `src/llama-state-incr.cpp`: scanning, graph construction, touch, reservation,
  leaf collection, and save/load integration.
- `common/common.h` and `common/arg.cpp`: server parameter and CLI parsing.
- `tools/server/server-context.cpp`: startup configuration, logs, and use of the
  shared policy by manual and automatic paths.
- Server metrics plumbing: GGSD cache gauges and counters.
- `tests/test-save-load-state.cpp` plus focused server/argument tests: graph,
  filesystem, quota, error injection, restart, and regression coverage.
