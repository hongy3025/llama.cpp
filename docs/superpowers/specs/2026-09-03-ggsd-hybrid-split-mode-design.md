# GGSD Split Mode for Hybrid Caches (Qwen3.5 and friends)

Date: 2026-09-03
Status: approved design, pending implementation plan
Depends on: `docs/superpowers/specs/2026-07-31-ggsd-incremental-slot-storage-design.md` (segment mode, autoload, autosave)

## Problem

`--prompt-cache-ssd` (GGSD) currently rejects every model whose KV memory is
not a plain `llama_kv_cache`:

```
incremental state save is only supported for the standard kv cache
```

Qwen3.5 (LLM_ARCH_QWEN35) is a hybrid architecture: interleaved
gated-delta-net (linear attention) layers and full-attention layers
(`src/models/qwen35.cpp`: 1 attention layer per 4). Its memory is
`llama_memory_hybrid` (attn: filtered `llama_kv_cache`, rec:
`llama_memory_recurrent`), so the `dynamic_cast<llama_kv_cache*>` guard fails.

The whole-sequence state (`--cache-ram`) works because `llama_memory_hybrid`
implements `state_write/state_read` on the generic `llama_memory_i` interface.
A pure whole-blob SSD mode would work for continuation but duplicates the
shared attention prefix on disk for every forked conversation.

## Information-theoretic boundary

- Attention layers (8/32 for Qwen3.5-9B) are cell-based and position-sliceable.
  Their KV for a token prefix is byte-identical across conversations sharing
  that prefix. Fully segment-shareable, exactly like standard GGSD.
- Recurrent layers (24/32) hold a running summary `S_t = A(token_t)*S_{t-1} + b`.
  It is not position-sliceable, past-boundary snapshots cannot be reconstructed
  after the fact (transition inputs require hidden states), and the delta of
  two dense states is dense. The only recoverable snapshot is the current one,
  taken at autosave time, valid for exactly the token prefix it was taken at.

Consequence: the finest correct granularity for hybrid models is
"attention segments (shared) + one recurrent snapshot per conversation state".

## Design

Dispatch by concrete memory class inside GGSD:

| memory class | mode |
|---|---|
| `llama_kv_cache` | existing segment mode (unchanged) |
| `llama_memory_hybrid` | new split mode (this spec) |
| everything else (`llama_kv_cache_iswa` standalone, `llama_memory_recurrent`, ...) | still rejected with the existing error (non-goal) |

### Files

Both file kinds live in the existing `--slot-save-path` pool:

1. `seg_*.bin` - unchanged segment format, written from the **attn sub-cache**
   (`get_mem_attn()`), i.e. payload covers only attention layers. Shared by all
   conversations (forks included) that hash to the same chain.
2. `rec_<chain_hash>.bin` - one per saved conversation state:
   - header: magic, version, u32 `n_tokens` (full sequence length), u32
     `n_tail` (= `n_tokens - 1024*(n_tokens/1024)`), u8[32] `prev_chain_hash`,
     u8[32] `chain_hash`, kv_params/model identity, tokens array
     (`n_tokens` x i32), u32 `codec = 0` (reserved for future compression);
   - payload: attn KV for the non-aligned tail `[1024*k, n_tokens)` serialized
     with the same range format as segments (`state_write_range` on the attn
     sub-cache), followed by the full recurrent state (`state_write` on the
     rec sub-cache).

### Save (autosave at completion end)

1. Hash chain over the slot tokens as in segment mode; write any newly
   completed 1024-token attention segments from `get_mem_attn()`.
2. Write `rec_<chain_hash>.bin` with the non-aligned attn tail + rec snapshot.
   Skip the write if a file with the same `chain_hash` already exists
   (identical sequence re-saved).
3. Dominated-parent cleanup: delete any `rec_*.bin` whose `chain_hash` equals
   the new file's `prev_chain_hash` (same model identity). The new state
   strictly dominates the old one for extension-only matching, so this is loss
   free. Forked chains never collide (different chain hashes).

### Load (autoload arbitration + restore)

- Estimate: scan `rec_*.bin` headers; a candidate matches when
  `chain_hash(tokens[0..n_tokens))` of the request equals the file's
  `chain_hash` and `n_tokens >= min_prefix`. The matched coverage is
  `n_tokens` (never the segment-aligned prefix: the rec state pins coverage).
  Segments are byte sources only; they never extend coverage for hybrid.
- Restore: verify the chain, replay attention segments `[0, 1024*k)` into the
  attn sub-cache via the existing range-read path, then read the rec file:
  attn tail via range-read (continuing positions) + rec state via
  `get_mem_recr()->state_read`. Save order and read order must match.
- Mismatch of any kind (short payload, hash mismatch, rec read failure) aborts
  the whole restore; no partial fallback.

### IMROPE / cell_ext restore prerequisite

Qwen3.5 attention cells carry 4-component rope positions
(`n_pos_per_embd() == 4`). The whole-cache restore branch of
`llama_kv_cache::state_read_meta` already restores `cell_ext`, but the
ubatch (append) branch does not - upstream TODO at `llama-kv-cache.cpp`
("we cannot yet restore llama_kv_cell_ext as apply_ubatch() does not support
it yet"). This work must close that gap: after `apply_ubatch` in the append
path, set the restored `cell_ext` on the touched cells (`cells.ext_set`),
mirroring the whole-cache branch. Without it, restored attention positions are
wrong for every mrope-style model.

With ext restore in place, the GGSD guard `n_pos_per_embd != 1` is relaxed to
accept the append path for any `n_pos_per_embd` (the primary position
identity `pos_get(i) == i` still holds for text tokens; sections live in ext).

### Server integration

- Autosave: existing hook fires for hybrid models too; dispatches to split
  save. `n_tokens >= min threshold` rule unchanged.
- Autoload: unchanged interface. Estimate walks the chain and scans `rec_*`
  headers (a few KB per header); `min_prefix` / margin arbitration unchanged.
- `--prompt-cache-ssd` for hybrid no longer logs the rejection; `--cache-ram`
  and GGSD may coexist exactly as for standard models.

## Error handling

- Corrupt/short `rec_*.bin`: skip candidate during estimate; error out if hit
  during an accepted restore.
- Attn segment missing while its range is required by the chain: restore
  fails (segments are shared, so a pruned segment only affects chains that
  reference it - same policy as standard mode).
- Save failure: log, keep slot serving (autosave is best-effort, as today).

## Testing

- C-level (`tests/test-save-load-state.cpp`): no hybrid model fixture exists
  in the test suite, so the hybrid dispatch stays uncovered there; all
  existing tests must keep passing (standard segment path unchanged).
- The cell_ext append-branch fix is exercised by the existing roundtrip tests
  only for `n_pos_per_embd == 1`; mrope correctness is verified via server
  smoke test with Qwen3.5-9B (available locally, 5.6 GB): completion ->
  autosave -> restart server -> autoload must restore with zero prefill and
  produce output identical to the non-restarted run.
- Smoke matrix: continue same conversation (extension match), fork from a
  shared prefix (shared segments on disk, distinct rec files), reload after
  restart.

## Non-goals

- `llama_kv_cache_iswa` (non-hybrid SWA) and pure recurrent models.
- Hybrid-iswa variants (SWA attention sub-cache).
- LRU/GC for the segment pool (unchanged policy: manual prune only).
- Compression (codec field reserved), delta/diff storage (evaluated,
  rejected: rec delta is dense; attn sharing already achieved by segments).
- Changes to `--cache-ram` semantics or `server_prompt_cache`.

## Risks

- Upstream may close the `cell_ext` TODO differently; our fix should stay
  minimal and be easy to rebase.
- `rec_*.bin` size is dominated by the attn tail (up to 1023 tokens x
  ~32 KB/token for Qwen3.5-9B ~ 32 MB max per conversation) plus the rec
  snapshot (fixed, a few MB). Acceptable; only the latest per chain is kept.
- The `ext_set` addition must not perturb the non-mrope path (guarded by
  `n_pos_per_embd > 1`).

## Amendment (2026-09-04, post-review)

Deviations from the design text above, settled during implementation review:

- The `rec_*.bin` header ships **without** `prev_chain_hash` and without the
  reserved `codec` field. Fields: magic, version, u32 `n_tokens`, u32
  `n_tail`, u8[32] `chain_hash`, u64 `payload_size`, u8[16] truncated
  `payload_hash`, model_id, kv_params, token array.
- Dominated-parent cleanup deletes **all** strictly dominated rec files
  (a candidate whose stored token array is a strict prefix of the freshly
  saved sequence, same model identity - compared via the token array, not
  via `prev_chain_hash`), not only the direct parent. This achieves
  one-rec-per-chain even when intermediate states were saved at
  non-contiguous lengths. The cleanup runs only when the child rec file
  exists (written now or already on disk), so a skipped child write never
  deletes the surviving parent.
- Estimate verifies segment availability with the same tiling rule as load:
  the longest existing aligned segment prefix is computed first and a
  candidate is only priced when `n_tail == n_tokens - 1024 * n_seg_ok`, so
  an estimate hit cannot be priced on missing segments. Estimate reads
  headers only (no token array).
