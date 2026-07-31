# Incremental Slot Storage (GGSD) - Design

Date: 2026-07-31

Status: Approved

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
- G2. Cross-slot prefix reuse: on restore with a prompt, automatically compute the
  longest common prefix (LCP) between the saved token chain and the prompt, restore
  the KV for the aligned prefix, and prefill only the remainder.
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

## Architecture Overview

Two artifacts:

- **Segment files** `seg_<hash>.bin` in the slot save directory: one file per
  1024-token chunk of KV state, content-addressed by a hash chain.
- **Session files** `session_<name>.bin`: a small manifest recording the chain
  tail (last segment hash) of one saved conversation.

A conversation chain is a linked list of segments:

```
seg_A <- seg_B <- seg_C        (chain head -> tail)
tokens [0,1024) [1024,2048) [2048,3072)
```

Forking (rollback or a new conversation sharing a prefix) writes new segments
after the divergence point; the hash chain makes the new segments reference the
old prefix segments, so shared prefixes are physically shared with zero copying.

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
- The hash is truncated to a fixed length for the file name (e.g. 16 hex bytes).
- Chain head uses an empty string for the previous hash.

Content layout (little-endian):

```
u32  magic "GGSD"
u32  version = 1
u32  seg_index                 # position in chain, 0-based
[32] prev_hash                 # previous segment hash (all zeros for head)
u32  n_tokens = 1024           # constant; field present for future extension
str  model_id                  # string with length prefix
str  kv_params                 # string with length prefix
i32  tokens[1024]              # the segment tokens
     payload                   # KV cells for this token range (see below)
```

The payload reuses the existing KV cell serialization produced by
`llama_kv_cache::state_write` (same stream/cell metadata + K/V tensor data layout
as GGSQ v2), but the write is restricted to the cell range covering this segment's
tokens via a position-range filter (see Implementation Notes).

### Session file

Name: `session_<name>.bin` (name user-supplied).

```
u32  magic "GGSD"
u32  version = 1
[32] tail_hash                # hash of the last (tail) segment of the chain
u32  n_segments               # number of segments in the chain
```

`n_segments * 1024` is the number of tokens covered by the chain. The session file
is rewritten (small, fixed size) whenever the chain tail changes.

## C API

New functions in `llama.h` / `llama-context.cpp`, independent of the GGSQ v2
functions:

```c
// Save an incremental segment to the content-addressed pool.
// The full current token list of the sequence is passed; the function reads the
// session file (if any), computes the chain LCP with the given tokens, and:
//   - writes nothing if no new full 1024 segment boundary was crossed
//     (returns 0);
//   - appends a new segment (or a fork segment after a divergence point)
//     otherwise, updating the session file.
// Returns the number of segments written, or -1 on error.
int32_t llama_state_seq_save_incr(llama_context * ctx,
                                  const char * session_path,
                                  llama_seq_id seq_id,
                                  const llama_token * tokens,
                                  size_t n_token_count);

// Restore the aligned prefix of the chain matching the prompt.
// Computes the hash chain for the prompt tokens (zero disk reads), matches it
// against on-disk segment files, and restores KV + tokens for the matched prefix
// (floor-aligned to 1024). If the matched prefix is shorter than
// min_prefix_tokens, nothing is restored (returns 0).
// Returns the number of tokens restored, or 0 if no match / error.
size_t llama_state_seq_load_incr(llama_context * ctx,
                                 const char * session_path,
                                 llama_seq_id seq_id,
                                 const llama_token * prompt_tokens,
                                 size_t n_prompt_tokens,
                                 size_t min_prefix_tokens);
```

Notes:

- `session_path` points at the session file; segment files live in the same
  directory (`llama_file` helpers / `fs_*` utilities used by the server).
- Save semantics with a partial KV cache (context shift evicted early cells):
  the saved chain covers only the contiguous KV prefix starting at token 0,
  aligned down to 1024. If even the first cell range is missing (head not
  contiguous), save fails with an error.
- Both functions are thread-safe with respect to the caller's synchronization
  requirements of the existing `llama_state_seq_*` API (they call
  `ctx->synchronize()`).

## Server Endpoints

New endpoints with dedicated handlers (`handle_slots_save_incr`,
`handle_slots_restore_incr`), implemented as new task types
(`SERVER_TASK_TYPE_SLOT_SAVE_INCR`, `SERVER_TASK_TYPE_SLOT_RESTORE_INCR`). The
existing `SERVER_TASK_TYPE_SLOT_SAVE` / `RESTORE` handlers are not modified.

```
POST /slots/{id}/save_incr
    body: { "filename": "session_name" }
    -> { "n_segments": 1, "n_tokens": 2048 }

POST /slots/{id}/restore_incr
    body: { "filename": "session_name", "prompt": "...", "min_prefix": 64? }
    -> { "n_tokens_restored": 1024 }
```

Restore flow:

1. Tokenize `prompt`, call `llama_state_seq_load_incr` with the slot's context.
2. On success, set `slot.prompt.tokens` to the prompt tokens and position the
   slot so prefill starts at the restored token count (the server's normal prompt
   processing continues from `n_past = n_tokens_restored`).
3. If nothing was restored, the prompt is processed normally (full prefill).

The `min_prefix` request field overrides the default threshold (64 tokens).
Session file names are validated with the existing `fs_validate_filename`.

## Edge Cases and Error Handling

| Case | Handling |
|---|---|
| Save, chain tail matches, no boundary crossed | Zero IO, return 0 |
| Save, chain tail matches, boundary crossed | Append one new segment, update session file |
| Save, prompt diverges from chain (rollback/fork) | Write new fork segment after LCP-aligned boundary, update session file; old segments kept |
| KV cache partial (early cells evicted) | Save only contiguous prefix from token 0, aligned down to 1024; session chain shortened |
| KV head missing (not contiguous from token 0) | Save fails with error |
| Restore, LCP < min_prefix | No-op, return 0, full prefill |
| Restore, segment file missing in matched chain | Stop at first missing segment, restore shorter prefix |
| Segment file corrupt (hash mismatch) | Stop at corrupt segment, restore shorter prefix; log warning |
| Session file corrupt / absent | Error (save: treat as new chain; restore: return 0) |
| Cross-model restore | Hash chain mismatch (model_id in hash) -> natural no-op |
| Save with non-contiguous KV (middle gap) | Treat as head missing -> error |

## Implementation Notes

### Position-range cell slicing

`llama_kv_cache::state_write` currently iterates all cells of a sequence
(llama-kv-cache.cpp:1953). Slicing adds a `(pos_begin, pos_end)` filter:

- Token range to pos range: `pos_begin = t0 * n_pos_per_embd`,
  `pos_end = t1 * n_pos_per_embd` (identity when `n_pos_per_embd == 1`).
- In the cell loop, add `add_cell = add_cell && (pos >= pos_begin && pos < pos_end)`
  (mirrors the existing SWA-mask filter using `cells.pos_get(i)`).
- The K/V tensor data is written per cell row via `io.write_tensor` with row
  offsets; no data layout change is needed.

### KV completeness check

Before writing a segment covering tokens `[t0, t1)`, verify the sequence's cells
are contiguous in that range (count cells matching the pos range; compare to the
expected count). Used for the partial-KV degradation and head-missing error.

### Reusable pieces

- `llama_io_write_file` / `llama_file` for file IO.
- `fs_*` utilities and `fs_validate_filename` from common.
- Hash: use the existing SHA-256 implementation in ggml (`ggml_sha256`).
- Segment directory: reuse `slot_save_path` from server params.

## Testing

### C-level tests (extend `tests/test-save-load-state.cpp`)

1. Save/load roundtrip: generate 2500 tokens, save, clear context, restore with
   matching prompt, verify KV-equivalent generation (logits match reference).
2. Incremental save: save at 1500, then at 2600; verify only one new segment file
   was created and the chain covers 2048 tokens.
3. Fork: save chain A (3000 tokens), diverge at 1500, save again; verify prefix
   segment files are shared (same file names) and restore works.
4. Partial KV: evict early cells (context shift), save; verify chain shrinks to
   the contiguous prefix; restore + prefill remainder.
5. Corrupt segment: flip a byte in a segment file; verify restore stops at the
   previous segment with a warning.
6. Cross-model: hash with different model_id does not match on-disk files.

### Server API tests

- `tests/test-server.py` style: save_incr / restore_incr roundtrip through the
  HTTP API with a small model; verify `n_tokens_restored` and correct completion
  continuation.
