# Incremental Slot Storage (GGSD) - Design

Date: 2026-07-31 (amended 2026-09-02, see docs/plans/2026-09-02-ggsd-remediation.md)

Status: Approved, amended after design review
Amendment (2026-09-06): GGSD segment granularity is reduced to 256 tokens and the on-disk version is 2. References to 1024-token segment boundaries below are historical values from version 1; the implementation and current user guides use 256.


Amendment (2026-09-03): the "standard kv cache only" scope below is superseded
by `docs/superpowers/specs/2026-09-03-ggsd-hybrid-split-mode-design.md`:
`llama_memory_hybrid` models (Qwen3.5 family) are now supported via split mode
(shared attention segments + one `rec_<chain_hash>.bin` per saved conversation
state); all other cache classes stay rejected. The `n_pos_per_embd != 1`
rejection is lifted (append-path restore now rebuilds `cell_ext`), and the
`kv_params` identity additionally includes `n_layer`, which invalidates
segment pools written before the change. The rest of this document is kept as
a historical record.

## Problem Statement

The current slot save/restore mechanism (`/slots/{id}/save`, `/slots/{id}/restore`)
writes a full snapshot of the slot state (all prompt tokens + all KV cache cells) to
a single GGSQ v2 file on every save. This has two problems:

1. **Slow consecutive saves**: For long conversations the KV cache is large
   (hundreds of MB), so every save rewrites the whole state even when only a few
   tokens were generated since the last save.
2. **No cross-slot prefix reuse**: When a new prompt shares a common prefix with a
   previously saved prompt, the shared prefix is fully re-prefilled instead of
   being restored from the save file.

## Goals

- G1. Fast consecutive saves: a save only writes data for tokens that were not
  saved before. High-frequency save calls (e.g. after every generation step) must
  have near-zero cost when no new 1024-token segment boundary was crossed.
- G2. Cross-slot prefix reuse: on restore with a prompt, the prompt's hash chain
  is matched directly against the content-addressed segment pool (all
  `seg_<hash>.bin` files in the save directory, regardless of which session or
  sequence wrote them). The KV for the aligned prefix is restored and only the
  remainder is prefilled. The session file plays no role in restore.
- G3. Full independence from the existing GGSQ v2 path: new file format, new C API,
  new server endpoints. The existing save/restore code is untouched.

## Non-Goals

- No automatic garbage collection of orphaned segment files (forked chains leave
  old segments behind; cleanup is manual).
- No incremental savings for the last partial segment (< 1024 tokens); partial
  segments are never written to disk and are re-prefilled after restore.
- No changes to the existing GGSQ v2 format or its code paths.
- No rollback-by-truncation: a rollback/fork writes new segments; old segments are
  left in place (this is what enables shared prefixes across forks).
- SWA (sliding window attention) caches are unsupported: evicted cells make
  per-segment completeness impossible; both save and load reject them explicitly.
- Multimodal (mtmd) sequences are unsupported: the hash chain covers text token
  ids only; the server rejects `save_incr` on sequences containing media.
- Single-writer: one process per save directory. Server-internal calls are
  serialized by the task queue; the C API itself does not guard concurrent
  writers to the same session/segments.

## Architecture Overview

Two artifacts:

- **Segment files** `seg_<hash>.bin` in the slot save directory: one file per
  1024-token chunk of KV state, content-addressed by a hash chain. The directory
  as a whole is the **segment pool**: restore matches the prompt's hash chain
  against any segment file in it, regardless of which session or sequence wrote
  it (G2).
- **Session files** `session_<name>.bin`: a small manifest recording the chain
  tail (last segment hash) of one saved conversation. Save-side hint only -
  restore never reads it.

A conversation chain is a linked list of segments:

```
seg_A <- seg_B <- seg_C        (chain head -> tail)
tokens [0,1024) [1024,2048) [2048,3072)
```

Forking (rollback or a new conversation sharing a prefix) writes new segments
after the divergence point; the hash chain makes the new segments reference the
old prefix segments, so shared prefixes are physically shared with zero copying.
Because identity is content-based, a restore does not need the session file of
the conversation that produced a segment - only the directory.

## File Formats

### Segment file

Name: `seg_<hash>.bin`, where

```
hash_0 = sha256(model_id || kv_params || ""       || tokens_0)   (chain head)
hash_k = sha256(model_id || kv_params || hash_{k-1} || tokens_k)
```

- `model_id`: the model arch name (`llm_arch_name(model.arch)`).
- `kv_params`: serialized string of KV-affecting parameters: `type_k`, `type_v`,
  `n_pos_per_embd` (values joined, e.g. "f16|f16|1").
- `tokens_k`: the 1024 token ids of segment k (raw bytes).
- The hash is truncated to 16 bytes (32 hex chars) for the file name.
- Chain head uses 16 zero bytes for the previous hash.

Content layout (little-endian):

```
u32  magic "GGSD"
u32  version = 1
u32  seg_index                 # position in chain, 0-based
[32] prev_hash                 # previous segment hash, 32 hex chars (all '0' for head)
u32  n_tokens = 1024           # constant; field present for future extension
u64  payload_size              # size of the KV payload in bytes
[16] payload_hash              # truncated sha256 of the payload, hex-verified on read
str  model_id                  # string with length prefix
str  kv_params                 # string with length prefix
i32  tokens[1024]              # the segment tokens
     payload                   # KV cells for this token range (see below)
```

`payload_size` is measured with a dummy write pass before the real write; the
payload hash is computed while streaming the payload and patched into the
header afterwards. Restore verifies it and stops at the first mismatching
segment (corruption guard).

The payload reuses the existing KV cell serialization produced by
`llama_kv_cache::state_write` (same stream/cell metadata + K/V tensor data layout
as GGSQ v2), but the write is restricted to the cell range covering this segment's
tokens via a position-range filter (see Implementation Notes).

### Session file (save-side hint only)

Name: `session_<name>.bin` (name user-supplied).

```
u32  magic "GGSD"
u32  version = 1
[32] tail_hash                # last segment hash, 32 hex chars
u32  n_segments               # number of segments in the chain
```

`n_segments * 1024` is the number of tokens covered by the chain. The session file
is rewritten (small, fixed size) whenever the chain length changes.

**The session file is only consulted by save** (as the last known chain length
and to skip redundant rewrites). Restore uses segment-pool semantics and never
reads it: a corrupt, stale, or missing session file degrades nothing.

## C API

New functions in `llama.h`, implemented in `src/llama-state-incr.cpp` (member
functions on `llama_context`, declared in `llama-context.h`), independent of
the GGSQ v2 functions:

```c
// Save the KV state of a sequence as a chain of 1024-token segments,
// content-addressed by sha256(model_id || kv_params || prev_hash || tokens).
// The full current token list of the sequence is passed; the function finds
// the fork point against already-persisted segment files (by file existence
// only - KV eviction does not invalidate persisted segments) and appends new
// segments while the KV cache covers them.
// Returns the number of segments the session chain covers after the save
// (not the number newly written), or -1 on error.
// Rejected with -1: SWA caches, n_pos_per_embd != 1, sequence positions not
// starting at 0 (checked only when the head segment must be written;
// already-persisted chains are unaffected by eviction or an empty cache),
// KV head neither on disk nor writable.
int32_t llama_state_seq_save_incr(
        struct llama_context * ctx,
        const char * session_path,
        llama_seq_id   seq_id,
        const llama_token * tokens,
        size_t   n_token_count);

// Restore the aligned prefix of the prompt's hash chain by matching
// content-addressed segment files in the directory of session_path
// (segment-pool semantics; the session file is not read). The hash chain is
// computed from the prompt tokens (zero disk reads for the match itself);
// each matched segment's payload is hash-verified and replayed into the KV
// cache, floor-aligned to 1024. If the matched prefix is shorter than
// min_prefix_tokens, nothing is restored (returns 0).
// Returns the number of tokens restored, or 0 if no match / error.
// Rejected with 0: SWA caches, n_pos_per_embd != 1.
size_t llama_state_seq_load_incr(
        struct llama_context * ctx,
        const char * session_path,
        llama_seq_id   seq_id,
        const llama_token * prompt_tokens,
        size_t   n_prompt_tokens,
        size_t   min_prefix_tokens);
```

Notes:

- `session_path` locates the save directory (its parent); segment files live
  next to it and form the segment pool (`llama_file` helpers / `fs_*` utilities
  used by the server). The session file itself is a save-side hint only.
- Save semantics with a partial KV cache: already-persisted segments stay valid
  and the chain is preserved regardless of eviction; new segments are written
  only while the KV cache covers them contiguously. Save fails only when the
  head segment can neither be loaded from disk nor written from the cache.
- Both functions call `ctx->synchronize()`; they share the caller's
  synchronization requirements of the existing `llama_state_seq_*` API. They
  are NOT safe against multiple concurrent writers to the same directory
  (single-writer assumption).

## Server Endpoints

New endpoints with dedicated handlers (`handle_slots_save_incr`,
`handle_slots_restore_incr`), implemented as new task types
(`SERVER_TASK_TYPE_SLOT_SAVE_INCR`, `SERVER_TASK_TYPE_SLOT_RESTORE_INCR`). The
existing `SERVER_TASK_TYPE_SLOT_SAVE` / `RESTORE` handlers are not modified.
```
POST /slots/{id}/save_incr
    body: { "filename": "session_name" }
    -> { "n_segments": 1, "n_tokens": 2048 }
    # n_segments / n_tokens = the session chain coverage after the save
    # (not the amount newly written this call)

POST /slots/{id}/restore_incr
    body: { "filename": "session_name", "prompt": "...", "min_prefix": 64? }
    -> { "n_tokens_restored": 1024 }
```

Restore flow (segment pool):

1. Tokenize `prompt`, call `llama_state_seq_load_incr` with the slot's context.
   The hash chain of the prompt is walked against `seg_<hash>.bin` files in the
   directory - segments saved by any session or sequence match.
2. On success, set `slot.prompt.tokens` to the restored prefix and position the
   slot so prefill starts at the restored token count (the server's normal prompt
   processing continues from `n_past = n_tokens_restored`).
3. If nothing was restored (below `min_prefix` or no matching chain), the prompt
   is processed normally (full prefill).

The `min_prefix` request field overrides the default threshold (64 tokens).
Session file names are validated with the existing `fs_validate_filename`.

## Edge Cases and Error Handling

| Case | Handling |
|---|---|
| Save, all segments already on disk | Zero IO, chain unchanged |
| Save, boundary crossed, KV covers it | Append the new segment, update session file |
| Save, prompt diverges from chain (rollback/fork) | Write new fork segments after the divergence point (file-existence based), update session file; old segments kept and still restorable (pool) |
| KV cache partial / cells evicted | Chain preserved (files stay valid); only segments the KV still covers can be (re)written; no error, no shortening (R2) |
| KV head missing and no head file on disk | Save fails with error (single head-missing rule, R3) |
| Restore, matched prefix < min_prefix | No-op, return 0, full prefill |
| Restore, segment file missing in matched chain | Stop at first missing segment, restore shorter prefix |
| Restore, segment file corrupt (payload hash mismatch) | Stop at corrupt segment; the verified prefix stays restored and is returned; log warning |
| Session file corrupt / absent / stale | Save: treated as a new chain; restore: unaffected (never read) |
| Cross-model / cross-config restore | Hash chain mismatch (model_id, kv_params in hash) -> natural no-op |
| Sequence positions shifted (context shift) | Save rejected only when the head segment must be written; if all segments are already on disk, save is a no-op (R4, scoped) |

## Implementation Notes

### Position-range cell slicing

`llama_kv_cache::state_write` currently iterates all cells of a sequence
(llama-kv-cache.cpp:1953). Slicing adds a `(pos_begin, pos_end)` filter:

- Token range to pos range: `pos_begin = t0 * n_pos_per_embd`,
  `pos_end = t1 * n_pos_per_embd`. This identity is only verified for
  `n_pos_per_embd == 1`; other values are rejected explicitly (R4).
- In the cell loop, add `add_cell = add_cell && (pos >= pos_begin && pos < pos_end)`
  (mirrors the existing SWA-mask filter using `cells.pos_get(i)`).
- The K/V tensor data is written per cell row via `io.write_tensor` with row
  offsets; no data layout change is needed.

### KV completeness check

Before writing a segment covering tokens `[t0, t1)`, verify the sequence's cells
are contiguous in that range (count cells matching the pos range; compare to the
expected count). Used only to decide whether new segments can be written; it
does NOT gate fork detection (file-existence based, R2) and does not shorten
an existing chain.

### Content identity and FP nondeterminism (R6)

Segment identity covers (model arch, kv params, chain prefix, tokens) - the KV
payload is NOT part of the hash. The same token chain computed twice may yield
bitwise-different KV data (batch composition, thread count, backend). When a
save finds an existing segment file it skips writing; restore then continues
generation from KV computed under a possibly different configuration. This is
the same equivalence assumption the cross-slot prompt cache already makes, and
is accepted explicitly. If stricter determinism is ever required, the payload
hash would have to enter the content address.

### Reusable pieces

- `llama_io_write_file` / `llama_file` for file IO.
- `fs_*` utilities and `fs_validate_filename` from common.
- Hash: SHA-256, vendored as `src/llama-sha256.{h,c}` (Igor Pavlov, public
  domain). NOTE: the original plan suggested `ggml_sha256`, but no such public
  symbol exists in ggml (only private copies inside opencl/hexagon backends);
  vendoring is required.
- Segment directory: reuse `slot_save_path` from server params.

## Testing

### C-level tests (extend `tests/test-save-load-state.cpp`)

1. Save/load roundtrip: generate 2500 tokens, save, clear context, restore with
   matching prompt, verify KV-equivalent generation (logits match reference).
2. Incremental save: save at 1500, then at 2600; verify only one new segment file
   was created and the chain covers 2048 tokens.
3. Fork: save chain A (3000 tokens), diverge at 1500, save again; verify prefix
   segment files are shared (same file names) and restore works.
4. Partial KV: evict tail cells, save; verify the chain is preserved and
   nothing new is written; restore returns the full on-disk chain; saving a
   fresh session with the head evicted fails.
5. Corrupt segment: flip a byte in a segment file; verify restore stops at the
   previous segment with a warning.
6. Cross-model: hash with different model_id does not match on-disk files.

### Server API tests

- `tests/test-server.py` style: save_incr / restore_incr roundtrip through the
  HTTP API with a small model; verify `n_tokens_restored` and correct completion
  continuation. Cross-session prefix reuse: save session A, restore with a
  different session name but a prompt sharing a prefix -> tokens restored.
