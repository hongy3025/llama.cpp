# GGSD: Incremental Slot Storage - User Guide and Technical Reference

GGSD (incremental slot storage) adds content-addressed, incremental KV-cache
save/restore to llama.cpp. Long conversations can be saved repeatedly at
near-zero cost, and a restored prompt can reuse saved KV state from *any*
previous session, as long as the token prefix matches.

Design spec: `docs/superpowers/specs/2026-07-31-ggsd-incremental-slot-storage-design.md`

---

## Part 1: User Perspective

### 1.1 What problem does it solve

The existing `/slots/{id}/save` / `restore` endpoints (GGSQ v2) write a full
snapshot of the slot state on every call: for a long conversation that is
hundreds of MB per save, even when only a few tokens changed.

GGSD instead splits the KV state into fixed 1024-token segments. Each segment
is written once and never rewritten:

- **Consecutive saves are cheap.** Saving after generating a few tokens costs
  microseconds when no 1024-token boundary was crossed.
- **Prefix reuse across sessions and slots.** On restore, the prompt's saved
  prefix is replayed from disk and only the remainder is prefilled. The
  session that produced the prefix does not matter - any session saved to the
  same directory can be reused.

### 1.2 New configuration and parameters

**No new server flags.** GGSD reuses the existing `--slot-save-path PATH`
directory (the same directory used by the classic slot save). It must exist
and is disabled unless provided:

```
llama-server -m model.gguf --slot-save-path saves
```

All GGSD artifacts live in this single directory:

```
saves/
  seg_<hash>.bin          # one KV segment per 1024 tokens
  session_<name>.bin      # tiny save-side hint (chain tail, 44 bytes)
```

### 1.3 HTTP API

Two new actions on the existing slots route. Both are synchronous.

#### Save

```
POST /slots/{id_slot}?action=save_incr
{ "filename": "my-session" }
```

Response:

```json
{ "id_slot": 0, "filename": "my-session",
  "n_segments": 3, "n_tokens": 3072, "t_ms": 12.4 }
```

- `n_segments` / `n_tokens` describe the **whole chain on disk after the
  save**, not the amount written by this call. A repeated save with no new
  full segment returns the same numbers with `t_ms` near zero.
- `filename` is validated with the server's filename rules; it maps to
  `session_<filename>.bin`.

#### Restore

```
POST /slots/{id_slot}?action=restore_incr
{ "filename": "my-session", "prompt": "...", "min_prefix": 64 }
```

Response:

```json
{ "id_slot": 0, "filename": "my-session",
  "n_tokens_restored": 1024, "t_ms": 65.5 }
```

- `prompt` is tokenized text-only. Its hash chain is matched against
  `seg_<hash>.bin` files in the save directory. `filename` is only used to
  locate the directory - the session file itself is never read.
- Restored tokens are floor-aligned to 1024. The slot's prompt is set to the
  restored prefix and the next `/completion` continues from it; only the
  tokens after the prefix are prefilled.
- `min_prefix` (optional, default 64) is the minimum restored length. If the
  match is shorter, nothing is restored and the request returns
  `n_tokens_restored: 0`; the prompt is then processed by a full prefill.
  Setting `min_prefix` above the expected match is how you express "restore
  only if it is worth it".

#### Typical workflow

```sh
# 1. run a long completion
curl http://localhost:8080/completion -d '{"prompt": "<long text>", "n_predict": 512}'

# 2. checkpoint the slot (call as often as you like)
curl http://localhost:8080/slots/0?action=save_incr -d '{"filename": "story"}'

# 3. later / in another process: restore the prefix and continue
curl http://localhost:8080/slots/0?action=restore_incr \
     -d '{"filename": "story", "prompt": "<same long text>", "min_prefix": 1024}'
```

Cross-session reuse: save session `A`, then call `restore_incr` with
`filename: "B"` (never saved) and A's prompt - the prefix still restores,
because the segment pool matches by content, not by session name.

### 1.4 C API

For applications embedding llama.cpp (`llama_state_seq_save_incr` /
`llama_state_seq_load_incr` in `llama.h`, implementation in
`src/llama-state-incr.cpp`):

```c
// Returns the chain length after the save (segments, not newly written),
// or -1 on error.
int32_t llama_state_seq_save_incr(
        struct llama_context * ctx,
        const char * session_path,   // path of the session file; its parent
                                     // directory is the segment pool
        llama_seq_id   seq_id,
        const llama_token * tokens,  // full token list of the sequence,
        size_t   n_token_count);     // positions must start at 0

// Returns the number of tokens restored (floor-aligned to 1024),
// or 0 if no match / error.
size_t llama_state_seq_load_incr(
        struct llama_context * ctx,
        const char * session_path,
        llama_seq_id   seq_id,
        const llama_token * prompt_tokens,
        size_t   n_prompt_tokens,
        size_t   min_prefix_tokens);
```

Both share the caller's synchronization requirements of the existing
`llama_state_seq_*` API (call `llama_synchronize()` / decode on the same
thread as usual).

### 1.5 Disk usage

Per segment file (1024 tokens):

header      ~110 bytes   magic, version, seg_index, prev_hash, payload
                         size, payload hash, model_id, kv_params
tokens      4 KiB        the 1024 token ids (i32)
payload     depends      the KV cells for those 1024 tokens
```

The payload dominates. Per segment it is approximately:

```
n_layer x n_embd_kv x 2 (K and V) x sizeof(type_k/v) x n_streams
```

Examples (K/V in f16, single stream):

| Model                     | layers x n_embd_kv | payload/segment |
|---------------------------|--------------------|-----------------|
| stories15M (6 x 288)      | 6 x 288 x 2 x 2 B  | ~6.8 KiB        |
| 7B-class (32 x 1024)      | 32 x 1024 x 2 x 2 B| ~128 KiB        |
| 70B-class GQA (80 x 1024) | 80 x 1024 x 2 x 2 B| ~320 KiB        |

Session files are 44 bytes each.

**Growth and cleanup.** Segments are content-addressed and shared: forks and
rewrites (e.g. a context rollback followed by new generation) create *new*
segments and leave the old ones in place, so the directory grows monotonically.
There is **no automatic garbage collection** - orphaned segments (chains no
session points to) must be removed manually. A safe rule: stop the server,
then delete `seg_*.bin` files you no longer want to restore; every session
degrades gracefully (restore falls back to a shorter prefix, or a full
prefill) if its segments are missing.

### 1.6 Semantics, limits and error behavior

| Situation | Behavior |
|---|---|
| Save, everything already on disk | Zero IO, chain unchanged |
| Save, < 1024 new tokens since last save | Nothing written (partial segments are never persisted) |
| Save, prompt diverges from a saved chain | New fork segments written from the divergence point; old segments stay restorable |
| Save, KV cache partially evicted | Chain on disk is preserved; only segments the cache still covers contiguously can be written; no error |
| Save, head segment missing in cache AND on disk | Error (`-1` / HTTP 500) |
| Restore, matched prefix < min_prefix | No-op, 0 tokens restored |
| Restore, segment missing or corrupt mid-chain | Restore stops there; the verified prefix stays restored |
| Restore across models / KV configs | Natural no-op (identity includes arch + KV types) |

Explicit rejections:

- **SWA (sliding window) caches** - save and load both reject with an error;
  evicted SWA cells make per-segment completeness impossible.
- **`n_pos_per_embd != 1`** (e.g. M-RoPE) - rejected; the hash chain is
  defined over text token positions.
- **Multimodal (mtmd) sequences** - the server rejects `save_incr` on slots
  containing media (the chain covers text token ids only). `restore_incr`
  always tokenizes plain text, so it simply does not match media sequences.
- **Shifted positions** (context shift) - save is rejected only when the head
  segment would have to be written; already-persisted chains are unaffected.

Concurrency: one writer per save directory. Inside the server the task queue
serializes calls; the C API itself does not guard against two processes
saving to the same directory.

---

## Part 2: Technical Principles

### 2.1 Content-addressed segment chain

A conversation is a linked list of 1024-token segments. Each segment's name
is derived from a truncated SHA-256 hash over its identity:

```
hash_0 = sha256(model_id || kv_params || 16 zero bytes || tokens_0)   # chain head
hash_k = sha256(model_id || kv_params || hash_{k-1}     || tokens_k)

model_id  = llm_arch_name(model.arch)
kv_params = "<type_k>|<type_v>|<n_pos_per_embd>"      # e.g. "f16|f16|1"
```

Key properties:

- **Immutable.** The KV payload is *not* part of the hash. A segment file is
  written once and never modified, which makes file existence a reliable
  fork-detection signal.
- **Self-authenticating.** Given a token list, anyone can recompute the whole
  hash chain with zero disk reads. Restore needs no index, no manifest, and
  no session file: it hashes the prompt and checks which `seg_<hash>.bin`
  files exist. This is what makes cross-session (G2) reuse free.
- **Chain-linked.** Each hash binds to its predecessor, so a segment is only
  valid at its position in one specific token chain - you cannot splice
  segments from different conversations.
- **Config-bound.** `model_id` and `kv_params` are inside the hash, so a
  restore with a different model or cache type mismatches silently and
  degrades to a normal full prefill.

### 2.2 Segment file format (little-endian)

```
u32  magic "GGSD"
u32  version = 1
u32  seg_index
[32] prev_hash                 # hex ascii, all '0' for the head
u32  n_tokens = 1024
u64  payload_size
[16] payload_hash              # truncated sha256 of the payload
str  model_id
str  kv_params
i32  tokens[1024]
     payload
```

The payload is the standard `llama_kv_cache::state_write` cell serialization
(same format as GGSQ v2), restricted to the segment's position range. The
payload hash is computed while streaming and patched into the header; restore
verifies it and stops at the first mismatch (corruption guard, R: corrupted
data can only shorten a restore, never poison it).

The session file is a 44-byte hint (magic, version, tail hash, chain length)
used only by save, to skip redundant rewrites and to know the previous chain
length. Restore never reads it.

### 2.3 Save flow

1. Guard: standard KV cache only, non-SWA, `n_pos_per_embd == 1`.
2. Read the session file (optional; corrupt/missing simply starts fresh).
3. Compute the hash chain for `tokens[0 .. n/1024)`.
4. **Fork detection is file-existence based** (R2): walk the chain and find
   the first hash whose `seg_<hash>.bin` is missing. The KV cache is
   deliberately not consulted here - persisted segments remain valid even
   after their cells are evicted.
5. From that fork point, write segments while `count_cells_range` shows the
   cache still covers the range contiguously; stop at the first gap
   (partial-KV best effort, never shortens the chain). Existing files are
   skipped (content addressing makes a rewrite redundant; see 2.5).
6. The chain length is re-derived from disk (longest existing prefix). If it
   is 0 - head neither on disk nor writable - the save fails (R3, the single
   head-missing rule). A position guard (`seq_pos_min == 0`) applies only
   when the head must be written (M2).
7. Rewrite the 44-byte session file when length or tail hash changed.

Consequence (G1): a save with no new 1024-token boundary touches nothing on
disk - measured at ~0.07 ms vs ~67 ms for a real segment write.

### 2.4 Restore flow

1. Guard: non-SWA, `n_pos_per_embd == 1`.
2. Compute the prompt's hash chain (pure arithmetic, no disk reads).
3. For each hash, open `seg_<hash>.bin` from the directory of `session_path`
   (segment-pool semantics, R1). Stop at the first missing file.
4. For each matched segment: verify header (magic, version, model_id,
   kv_params, prev_hash matches the chain), verify the payload hash while
   reading, and replay the payload into the KV cache via
   `llama_kv_cache::state_read_append` (appends without clearing the
   sequence, so multiple segments restore back-to-back).
5. Floor-align to 1024, apply `min_prefix`; below it, report 0 and leave the
   cache untouched. On mid-chain corruption, the already-verified prefix
   stays restored and the true length is returned (M1: return value always
   matches the cache state).
6. The server sets the slot's prompt to the restored prefix; normal prompt
   processing continues from `n_past = n_tokens_restored`.

### 2.5 Accepted nondeterminism

Segment identity does not include the KV payload. The same token chain
prefilled twice can produce bitwise-different KV data (batch composition,
thread count, backend). When a save finds an existing segment file it skips
writing, so a later restore may replay KV computed under a different
configuration. This is the same equivalence assumption the cross-slot prompt
cache already makes, and is accepted explicitly. If bit-exact determinism
were ever required, the payload hash would have to enter the content
address.

### 2.6 Implementation footprint

- `src/llama-sha256.{h,c}`: vendored public-domain SHA-256 (ggml has no
  public one).
- `src/llama-state-incr.cpp`: the two entry points, hashing, file IO
  (reuses `llama_file`, `llama_io_write_i`/`llama_io_read_i`).
- `src/llama-kv-cache.cpp`: `state_write_range` (position-range cell
  slicing), `state_read_append` (append-without-clear read),
  `count_cells_range`. The existing `state_write`/`state_read` now delegate
  to these; their external behavior is unchanged.
- `tools/server`: two task types and two slot actions; no routing changes
  (`POST /slots/{id}?action=save_incr|restore_incr`).

The GGSQ v2 format and its code paths are untouched; both mechanisms can
coexist in the same server (they share only the save directory).
