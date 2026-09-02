# GGSD Autoload - Design

Date: 2026-09-03

Status: Draft, awaiting review

Related: docs/superpowers/specs/2026-07-31-ggsd-incremental-slot-storage-design.md
(the base GGSD spec; this document extends it)

## Problem Statement

The server currently has two KV-reuse mechanisms that never cooperate:

1. **RAM prompt cache** (`--cache-ram`): automatic, per-request, LRU, volatile.
2. **GGSD segment pool** (save_incr / restore_incr): explicit, persistent,
   content-addressed.

A request whose prompt shares a long prefix with data saved on disk gains
nothing unless the caller manually calls restore_incr. The goal is to make
the segment pool participate in normal request processing: when slot and RAM
cache cannot reuse enough of the prompt, the server automatically restores
the matching prefix from the segment pool.

## Goals

- G1. Three-source arbitration at slot selection: reuse the best of
  (slot KV, RAM cache candidate, GGSD prefix estimate). Exactly one path
  executes; sources are never combined.
- G2. Zero-IO arbitration: deciding whether GGSD can help costs only hash
  computation plus one `stat` per segment - no payload reads on the miss
  path.
- G3. Differential replay: when the slot already holds valid KV for a
  request prefix, GGSD replays only the segments after the aligned boundary;
  skipped segments are not even opened.
- G4. Off by default; when disabled, behavior is byte-identical to today.

## Non-Goals

- No automatic saving (the pool only grows via explicit save_incr or future
  policy; not part of this design).
- No changes to the manual restore_incr endpoint semantics from the
  caller's perspective.
- No eviction or GC of segment files.
- No sub-segment (partial segment) replay; the segment stays the minimum
  unit of IO and the slot is truncated to the segment-aligned boundary
  before replay.

## Architecture Overview

The decision lives in `get_available_slot`, inside the existing
`update_cache` block (`server-context.cpp`). Three reusable-prefix lengths
are compared; the winner executes:

```
n_slot  = LCP(slot.prompt.tokens, task.tokens)              (existing logic)
n_cache = prompt_cache->peek(task.tokens)                   (new method)
n_ggsd  = state_seq_load_incr_estimate(session, tokens)     (new, zero-IO)

best = max(n_slot, n_cache, n_ggsd)
if best < GGSD_AUTOLOAD_MIN_PREFIX: existing behavior (consume cache
    candidate if any, else prompt_clear; full prefill)
else: execute the winning path only
```

Trigger conditions for entering arbitration are unchanged (LRU-selected slot
or f_keep < 0.5 on the similarity path, prompt cache enabled, completion
task). `prompt_save` rescue of the slot's current context is unchanged and
runs before arbitration, so the truncated tail is already preserved in the
RAM cache when GGSD wins.

The three sources are alternatives, not layers: the GGSD hash chain is
self-authenticating from position 0, so a segment can only extend a chain
that matches from the head. Combining a RAM-cache restore with a GGSD
mid-chain append is therefore both semantically impossible and redundant.

## New Configuration

```
--slot-incr-autoload      (flag, default: off)
```

Precondition: `--slot-save-path` must be set; otherwise the server logs a
warning at startup and treats the flag as off.

Constants (fixed, not configurable):

```
GGSD_AUTOLOAD_MIN_PREFIX = 1024   # at least one full segment must be reusable
GGSD_AUTOLOAD_MARGIN     = 256    # GGSD must beat the runner-up by this much
```

The margin prevents churn: a GGSD restore with ~65 ms IO per segment is not
worth it when it saves only a few tokens of prefill over the runner-up.

## C API Change

`llama_state_seq_load_incr` gains one parameter (all callers - llama.cpp
binding, tests - are migrated; no backward-compat shim):

```c
size_t llama_state_seq_load_incr(
        struct llama_context * ctx,
        const char * session_path,
        llama_seq_id   seq_id,
        const llama_token * prompt_tokens,
        size_t   n_prompt_tokens,
        size_t   min_prefix_tokens,
        size_t   n_prefix_valid);    // NEW
```

- `n_prefix_valid == 0`: current semantics - the sequence is treated as
  empty; replay starts at the chain head. Used by the manual endpoint.
- If `m` is not a multiple of 1024, the implementation floors it internally
  (`k0' = m / 1024`); the caller must truncate the sequence to the same
  boundary `floor(m / 1024) * 1024` before the call, otherwise the replayed
  segments overlap cells the slot still holds.
   Truncation is the caller's responsibility: the C API never mutates the
   sequence beyond appending cells. The server calls
   `llama_memory_seq_rm(mem, seq_id, m_aligned, -1)` before the load.
  1. Guards: `seq_pos_min(seq_id) == 0` (positions must not be shifted),
     `m >= GGSD_SEGMENT_TOKENS` (less than a segment has nothing to skip),
     SWA / `n_pos_per_embd != 1` rejected as today.
  2. Hash chain computation is unchanged. `k0' = m / GGSD_SEGMENT_TOKENS`.
  3. Segments `[0, k0')` are skipped entirely - no stat, no read.
  4. Segments `k >= k0'` follow the current logic (file existence, payload
     hash verification, `state_read_append` replay). The return value is the
     longest chain prefix present on disk times 1024 (M1 semantics: the
     return always matches the restored coverage).
- If `m` is not a multiple of 1024, the implementation floors it internally
  (`k0' = m / 1024`); the caller is expected to have truncated the sequence
  to the same aligned boundary.

Truncation is the caller's responsibility: the C API never mutates the
sequence beyond appending cells. The server calls
`llama_memory_seq_rm(mem, seq_id, m_aligned, -1)` before the load.

`llama_context::state_seq_load_incr` (member, `llama-context.h`) gains the
same parameter.

## New Zero-IO Estimate

Internal member function (not exported in llama.h):

```cpp
// Returns the number of tokens a load_incr call would restore for this
// prompt (floor-aligned to 1024), assuming an empty sequence.
// Costs: hash chain computation + one stat per segment. No payload IO.
size_t llama_context::state_seq_load_incr_estimate(
        const char * session_path,
        llama_seq_id   seq_id,
        const llama_token * prompt_tokens,
        size_t   n_prompt_tokens,
        size_t   min_prefix_tokens) const;
```

- Reuses the (now shared) hash-chain helper that save and load both use;
  the duplicated inline chain code in save/load is consolidated into one
  anonymous-namespace function as part of this work.
- SWA / `n_pos_per_embd != 1` return 0, aligned with load's rejection
  semantics.
- Thread safety: reads only `model.hparams` and the filesystem; callable
  from the main thread where `get_available_slot` runs. The single-writer
  assumption on the pool directory is inherited from the base GGSD spec.

## Server-Side Flow (get_available_slot)

Inside the existing `update_cache` block, when `--slot-incr-autoload` is on
and the task is text-only:

1. `prompt_save` rescue runs first, unchanged (LRU / f_keep < 0.5 paths).
2. Arbitration:
   - `n_cache` comes from a new `server_prompt_cache::peek(tokens)` that
     finds the best entry by the existing f_keep/f_sim criteria and returns
     its LCP with the task tokens without consuming it. `load` is
     refactored into `peek` + `consume` (the erase moves into consume), so
     the consuming semantics are unchanged when GGSD does not win.
   - `n_ggsd` from the estimate function.
   - `n_slot` from the existing LCP.
3. Winner execution:
   - `slot` wins: nothing to do; the existing KV is continued (this is the
     current similarity path).
   - `cache` wins: `consume` the peeked entry and `set_data` into the slot
     (exactly today's load, split in two steps).
   - `ggsd` wins (and beats the runner-up by >= GGSD_AUTOLOAD_MARGIN):
     ```
     m_aligned = floor(min(n_slot, n_ggsd) / 1024) * 1024
     if n_slot >= 1024 and LCP is complete (slot's tokens are a prefix of
             the task's tokens):
         llama_memory_seq_rm(mem, seq_id, m_aligned, -1)
         n_restored = llama_state_seq_load_incr(..., n_prefix_valid = m_aligned)
     else:
         n_restored = llama_state_seq_load_incr(..., n_prefix_valid = 0)
     slot->prompt.tokens = server_tokens(first n_restored task tokens,
                                         has_mtmd = false)
     slot->prompt.checkpoints.clear()
     ```
     The LCP-completeness check guards differential replay: the slot's
     existing tokens must be exactly a prefix of the request's tokens, or
     the skipped segments would not correspond to the slot's KV.

Multimodal tasks skip the GGSD branch entirely (the hash chain cannot
represent `LLAMA_TOKEN_NULL` placeholders).

## Edge Cases and Error Handling

| Case | Handling |
|---|---|
| GGSD estimate below MIN_PREFIX | GGSD not considered; existing behavior |
| GGSD wins but a segment goes missing/corrupt between estimate and load | load stops at the verified prefix (M1); server proceeds with the restored prefix; full prefill covers the rest |
| Slot tokens are not a prefix of the task tokens | `n_prefix_valid = 0` (full replay); slot's KV was already rescued to the RAM cache in step 1 |
| Positions shifted (context shift) | `seq_pos_min != 0` rejects differential mode; full replay only if the head can be rebuilt, else falls through to full prefill |
| `set_data` of a consumed cache entry fails | Existing path: `prompt_clear`, full prefill |
| `--slot-incr-autoload` without `--slot-save-path` | Startup warning, flag treated as off |
| Disabled flag | Behavior byte-identical to today (arbitration block not entered) |

## Testing

C level (test-save-load-state.cpp):

1. **Deleted-head strong validation**: decode 2500, save (2 segments),
   restore with `n_prefix_valid = 1024` returns 2048; delete segment 0's
   file, restore again with `n_prefix_valid = 1024` still returns 2048
   (proves skipped segments are never read); with `n_prefix_valid = 0` the
   same deletion limits the restore to 0.
2. **Differential correctness**: decode 1500, save (1 segment), decode to
   2600, save (2 segments); fresh context decodes 1500 tokens, restore with
   `n_prefix_valid = 1024` returns 2048; generation matches the reference.
3. **Guards**: `seq_pos_min != 0` rejects differential mode; flooring of a
   non-aligned `n_prefix_valid`.

Server level (test-server.py style):

4. With `--slot-incr-autoload`: second request with a prompt matching saved
   segments restores automatically (n_prompt_tokens_cache reflects the
   reuse; server logs show the ggsd branch).
5. Without the flag: behavior identical to today (regression).

## Risks and Accepted Trade-offs

- **Trust in slot KV identity**: the slot's existing KV is assumed to match
  `prompt_tokens[0..m)` without byte-level verification (the payload is not
  part of the hash; base spec section on FP nondeterminism). The manual
  endpoint never faces this (it always sees an empty slot); the autoload
  path accepts the same equivalence assumption as the prompt cache.
- **Truncation discards up to 1023 valid tail tokens** before replay; those
  tokens were rescued to the RAM cache by step 1 when the LRU/f_keep
  trigger fired, and were part of the "difference" the request must
  re-process anyway.
- **Cross-process pool mutation** between estimate and load: handled by
  per-segment verification inside load; the return value is authoritative.
