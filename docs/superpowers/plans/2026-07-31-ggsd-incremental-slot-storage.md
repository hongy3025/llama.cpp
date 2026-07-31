# GGSD Incremental Slot Storage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement content-addressed incremental sequence-state save/load ("GGSD") in llama.cpp: 1024-token KV segments shared across saves and slots, plus server endpoints `save_incr` / `restore_incr`.

**Architecture:** Segment files `seg_<hash>.bin` are content-addressed by a SHA-256 hash chain over (model_id, kv_params, previous hash, tokens); a small session file `session_<name>.bin` records the chain tail. Save appends (or forks) segments only past the chain LCP; restore recomputes the hash chain from prompt tokens (zero disk reads), opens only matched segment files, and replays their KV payloads. The KV payload per segment is the existing `llama_kv_cache::state_write` cell serialization restricted to a position range. Standard GGSQ v2 paths are untouched.

**Tech Stack:** C++17 (project standard), CMake, SHA-256 (vendored public-domain implementation), llama.cpp C API, llama.cpp server task queue.

## Global Constraints

- Follow llama.cpp style: 4-space indent, no comments unless they explain a non-obvious invariant, ASCII only (no `-` emdash, no `->` arrow), concise.
- Do NOT modify any existing GGSQ v2 code path or its file formats (backward compatibility).
- Do NOT modify `server.cpp` routing; new endpoints are `action` values on the existing `POST /slots/{id_slot}` route.
- Do NOT modify `llama_context::state_seq_save_file` / `state_seq_load_file` or `llama_kv_cache::state_write` / `state_read` signatures; extend with new methods instead.
- Session file name validation reuses `fs_validate_filename` (server-side only).
- No garbage collection of orphaned segment files (explicitly out of scope).
- Every task ends with a build + test verification and a commit. Never commit on the user's behalf without approval; `Assisted-by:` is used only for user-approved commits.
- All new public C API functions must be declared in `include/llama.h` with `LLAMA_API` and defined in a file compiled into the `llama` target.
- New source files must be registered in `src/CMakeLists.txt` (the library lists sources explicitly).
- The environment is Windows + PowerShell. Do not use `wc`, `sed`, `awk`, `head`, `tail`; use PowerShell equivalents.

---

## Spec Corrections (decided during plan writing; applied in Task 7)

1. The spec says "Hash: use the existing SHA-256 implementation in ggml (`ggml_sha256`)" - **no such function exists** in ggml. We vendor the public-domain SHA-256 from `examples/gguf-hash/deps/sha256/` (Igor Pavlov) into `src/llama-sha256.{h,c}`.
2. The spec's segment layout has no payload hash, but its own corruption test (flip a byte in a segment file -> restore stops) requires detecting payload corruption. The segment header gains a `u8[16] payload_hash` field (sha256 of the payload, truncated to 16 bytes).
3. `llama_state_seq_save_incr` returns the **chain length after the save** (number of segments in the session), not the number of segments written. `0` = no change / empty chain, `-1` = error. This lets the server report `n_segments`/`n_tokens` of the whole chain.
4. Endpoints are `POST /slots/{id_slot}?action=save_incr` and `POST /slots/{id_slot}?action=restore_incr` (the server routes all slot actions through `post_slots`; there are no per-action paths in `server.cpp`).
5. `kv_params` serialization: `"<type_k>|<type_v>|<n_pos_per_embd>"` (e.g. `"f16|f16|1"`).

---

### Task 1: Vendor SHA-256

**Files:**
- Create: `src/llama-sha256.h`
- Create: `src/llama-sha256.c`
- Modify: `src/CMakeLists.txt`

**Interfaces:**
- Produces: `void llama_sha256_init(llama_sha256_t * p);`, `void llama_sha256_update(llama_sha256_t * p, const unsigned char * data, size_t size);`, `void llama_sha256_final(llama_sha256_t * p, unsigned char * digest);` - used by Tasks 4. The `llama_sha256_t` struct holds streaming state.

- [ ] **Step 1: Create `src/llama-sha256.h`**

The implementation is a verbatim rename of the public-domain code from `examples/gguf-hash/deps/sha256/sha256.c` (Igor Pavlov, "This code is based on public domain code from Wei Dai's Crypto++ library"), with the rotate macros inlined (from `examples/gguf-hash/deps/rotate-bits/rotate-bits.h`) and symbols renamed to avoid any collision.

```c
/* llama-sha256.h - SHA-256 Hash
   based on: examples/gguf-hash/deps/sha256/sha256.c
   Igor Pavlov : Public domain */

#ifndef LLAMA_SHA256_H
#define LLAMA_SHA256_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LLAMA_SHA256_DIGEST_SIZE 32

typedef struct llama_sha256_t {
  uint32_t state[8];
  uint64_t count;
  unsigned char buffer[64];
} llama_sha256_t;

void llama_sha256_init(llama_sha256_t * p);
void llama_sha256_update(llama_sha256_t * p, const unsigned char * data, size_t size);
void llama_sha256_final(llama_sha256_t * p, unsigned char * digest);

#ifdef __cplusplus
}
#endif

#endif
```

- [ ] **Step 2: Create `src/llama-sha256.c`**

The full transform/compress implementation. The rotate macros from `rotate-bits.h` are inlined (only `ROTR32` is used by SHA-256).

```c
/* llama-sha256.c - SHA-256 Hash
   based on: examples/gguf-hash/deps/sha256/sha256.c
   Igor Pavlov : Public domain
   This code is based on public domain code from Wei Dai's Crypto++ library. */

#include "llama-sha256.h"

/* rotate bits macros, inlined from examples/gguf-hash/deps/rotate-bits/rotate-bits.h */

#ifdef _MSC_VER

#include <stdlib.h>

#define ROTL32(v, n) _rotl((v), (n))

#else

#include <stdint.h>

#define U32V(v) ((uint32_t)(v) & 0xFFFFFFFFU)

#define ROTL32(v, n) \
  (U32V((uint32_t)(v) << (n)) | ((uint32_t)(v) >> (32 - (n))))

#endif

#define ROTR32(v, n) ROTL32(v, 32 - (n))

/* define it for speed optimization */
#define _SHA256_UNROLL
#define _SHA256_UNROLL2

void
llama_sha256_init(llama_sha256_t *p)
{
  p->state[0] = 0x6a09e667;
  p->state[1] = 0xbb67ae85;
  p->state[2] = 0x3c6ef372;
  p->state[3] = 0xa54ff53a;
  p->state[4] = 0x510e527f;
  p->state[5] = 0x9b05688c;
  p->state[6] = 0x1f83d9ab;
  p->state[7] = 0x5be0cd19;
  p->count = 0;
}

#define S0(x) (ROTR32(x, 2) ^ ROTR32(x,13) ^ ROTR32(x, 22))
#define S1(x) (ROTR32(x, 6) ^ ROTR32(x,11) ^ ROTR32(x, 25))
#define s0(x) (ROTR32(x, 7) ^ ROTR32(x,18) ^ (x >> 3))
#define s1(x) (ROTR32(x,17) ^ ROTR32(x,19) ^ (x >> 10))

#define blk0(i) (W[i] = data[i])
#define blk2(i) (W[i&15] += s1(W[(i-2)&15]) + W[(i-7)&15] + s0(W[(i-15)&15]))

#define Ch(x,y,z) (z^(x&(y^z)))
#define Maj(x,y,z) ((x&y)|(z&(x|y)))

#define a(i) T[(0-(i))&7]
#define b(i) T[(1-(i))&7]
#define c(i) T[(2-(i))&7]
#define d(i) T[(3-(i))&7]
#define e(i) T[(4-(i))&7]
#define f(i) T[(5-(i))&7]
#define g(i) T[(6-(i))&7]
#define h(i) T[(7-(i))&7]

#ifdef _SHA256_UNROLL2

#define R(a,b,c,d,e,f,g,h, i) h += S1(e) + Ch(e,f,g) + K[i+j] + (j?blk2(i):blk0(i));\
  d += h; h += S0(a) + Maj(a, b, c)

#define RX_8(i) \
  R(a,b,c,d,e,f,g,h, i); \
  R(h,a,b,c,d,e,f,g, (i+1)); \
  R(g,h,a,b,c,d,e,f, (i+2)); \
  R(f,g,h,a,b,c,d,e, (i+3)); \
  R(e,f,g,h,a,b,c,d, (i+4)); \
  R(d,e,f,g,h,a,b,c, (i+5)); \
  R(c,d,e,f,g,h,a,b, (i+6)); \
  R(b,c,d,e,f,g,h,a, (i+7))

#else

#define R(i) h(i) += S1(e(i)) + Ch(e(i),f(i),g(i)) + K[i+j] + (j?blk2(i):blk0(i));\
  d(i) += h(i); h(i) += S0(a(i)) + Maj(a(i), b(i), c(i))

#ifdef _SHA256_UNROLL

#define RX_8(i) R(i+0); R(i+1); R(i+2); R(i+3); R(i+4); R(i+5); R(i+6); R(i+7);

#endif

#endif

static const uint32_t K[64] = {
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
  0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
  0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
  0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
  0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
  0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
  0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
  0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
  0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

static void
llama_sha256_transform(uint32_t state[8], const uint32_t data[16])
{
  uint32_t W[16];
  unsigned j;
  #ifdef _SHA256_UNROLL2
  uint32_t a,b,c,d,e,f,g,h;
  a = state[0];
  b = state[1];
  c = state[2];
  d = state[3];
  e = state[4];
  f = state[5];
  g = state[6];
  h = state[7];
  #else
  uint32_t T[8];
  for (j = 0; j < 8; j++)
    T[j] = state[j];
  #endif

  for (j = 0; j < 64; j += 16)
  {
    #if defined(_SHA256_UNROLL) || defined(_SHA256_UNROLL2)
    RX_8(0); RX_8(8);
    #else
    unsigned i;
    for (i = 0; i < 16; i++) { R(i); }
    #endif
  }

  #ifdef _SHA256_UNROLL2
  state[0] += a;
  state[1] += b;
  state[2] += c;
  state[3] += d;
  state[4] += e;
  state[5] += f;
  state[6] += g;
  state[7] += h;
  #else
  for (j = 0; j < 8; j++)
    state[j] += T[j];
  #endif

  /* Wipe variables */
  /* memset(W, 0, sizeof(W)); */
  /* memset(T, 0, sizeof(T)); */
}

#undef S0
#undef S1
#undef s0
#undef s1

static void
llama_sha256_write_byte_block(llama_sha256_t *p)
{
  uint32_t data32[16];
  unsigned i;
  for (i = 0; i < 16; i++)
    data32[i] =
      ((uint32_t)(p->buffer[i * 4    ]) << 24) +
      ((uint32_t)(p->buffer[i * 4 + 1]) << 16) +
      ((uint32_t)(p->buffer[i * 4 + 2]) <<  8) +
      ((uint32_t)(p->buffer[i * 4 + 3]));
  llama_sha256_transform(p->state, data32);
}

void
llama_sha256_final(llama_sha256_t *p, unsigned char *digest)
{
  uint64_t curBufferPos = (uint32_t)p->count & 0x3F;
  unsigned i;
  p->buffer[curBufferPos++] = 0x80;
  while (curBufferPos != (64 - 8))
  {
    p->buffer[curBufferPos++] = 0;
    if (curBufferPos == 64)
    {
      llama_sha256_write_byte_block(p);
      curBufferPos = 0;
    }
  }
  for (i = 0; i < 8; i++)
  {
    p->buffer[64 - 8 + i] = (unsigned char)(p->count >> (56 - 8 * i));
  }
  llama_sha256_write_byte_block(p);
  for (i = 0; i < 8; i++)
  {
    digest[i * 4] = (unsigned char)(p->state[i] >> 24);
    digest[i * 4 + 1] = (unsigned char)(p->state[i] >> 16);
    digest[i * 4 + 2] = (unsigned char)(p->state[i] >> 8);
    digest[i * 4 + 3] = (unsigned char)(p->state[i]);
  }
}

void
llama_sha256_update(llama_sha256_t *p, const unsigned char *data, size_t size)
{
  uint32_t curBufferPos = (uint32_t)p->count & 0x3F;
  while (size > 0)
  {
    p->buffer[curBufferPos++] = *data++;
    p->count++;
    size--;
    if (curBufferPos == 64)
    {
      llama_sha256_write_byte_block(p);
      curBufferPos = 0;
    }
  }
}
```

- [ ] **Step 3: Register the sources in `src/CMakeLists.txt`**

In the `add_library(llama ...)` list (after `llama-sampler.cpp`):

```cmake
            llama-sampler.cpp
            llama-sha256.c
            llama-state-incr.cpp
            llama-vocab.cpp
```

`llama-state-incr.cpp` does not exist yet (created in Task 4) - registering it now keeps the file list complete at the end of Task 4; CMake errors until then, so only add the `llama-sha256.c` line in this step:

```cmake
            llama-sampler.cpp
            llama-sha256.c
            llama-vocab.cpp
```

- [ ] **Step 4: Verify the build**

Run (PowerShell):

```powershell
cmake --build build --config Release --target llama
```

Expected: builds without errors.

- [ ] **Step 5: Commit**

```bash
git add src/llama-sha256.h src/llama-sha256.c src/CMakeLists.txt
git commit -m "llama : add vendored sha256 for GGSD incremental state"
```

---

### Task 2: KV cache position-range write and append read

**Files:**
- Modify: `src/llama-kv-cache.h`
- Modify: `src/llama-kv-cache.cpp`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces (used by Task 4):
  - `void llama_kv_cache::state_write_range(llama_io_write_i & io, llama_seq_id seq_id, llama_pos pos_begin, llama_pos pos_end) const;` - like `state_write`, but only cells whose position is in `[pos_begin, pos_end)`.
  - `size_t llama_kv_cache::count_cells_range(llama_seq_id seq_id, llama_pos pos_begin, llama_pos pos_end) const;` - number of cells of `seq_id` (unmasked by SWA) with position in the range.
  - `void llama_kv_cache::state_read_append(llama_io_read_i & io, llama_seq_id seq_id);` - like `state_read` with a specific `seq_id`, but does NOT clear the sequence first (appends to existing cells; used to restore multiple segments).
  - Private helper `bool state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, slot_info & sinfo, llama_seq_id dest_seq_id = -1, bool append = false);`

- [ ] **Step 1: Modify `src/llama-kv-cache.h`**

In the public "state write/load" section (line 147-150):

```cpp
    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    // state write/load for a subset of positions [pos_begin, pos_end)
    // (used for incremental state save/load)
    void state_write_range(llama_io_write_i & io, llama_seq_id seq_id, llama_pos pos_begin, llama_pos pos_end) const;
    void state_read_append (llama_io_read_i  & io, llama_seq_id seq_id);

    // number of cells of a sequence within [pos_begin, pos_end)
    size_t count_cells_range(llama_seq_id seq_id, llama_pos pos_begin, llama_pos pos_end) const;
```

In the private section (line 316):

```cpp
    bool state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count,       slot_info & sinfo, llama_seq_id dest_seq_id = -1, bool append = false);
```

- [ ] **Step 2: Modify `src/llama-kv-cache.cpp` - `state_write` -> `state_write_range`**

Replace the body of `llama_kv_cache::state_write` (line 1953) and add the range variant. The `flags` argument is now unused in the no-range version:

```cpp
void llama_kv_cache::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    GGML_UNUSED(flags);

    state_write_range(io, seq_id, 0, LLAMA_POS_MAX);
}

void llama_kv_cache::state_write_range(llama_io_write_i & io, llama_seq_id seq_id, llama_pos pos_begin, llama_pos pos_end) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    io.write(&n_stream, sizeof(n_stream));

    for (uint32_t s = 0; s < n_stream; ++s) {
        cell_ranges_t cr { s, {} };

        uint32_t cell_count = 0;

        const auto & cells = v_cells[s];

        // Count the number of cells with the specified seq_id
        // Find all the ranges of cells with this seq id (or all, when -1)
        uint32_t cell_range_begin = cells.size();

        for (uint32_t i = 0; i < cells.size(); ++i) {
            bool add_cell = true;

            add_cell = add_cell && !cells.is_empty(i);
            add_cell = add_cell && (seq_id == -1 || cells.seq_has(i, seq_id));

            // check the cell is not SWA-masked
            if (add_cell && seq_id != -1) {
                const bool is_masked = llama_hparams::is_masked_swa(n_swa, swa_type, cells.pos_get(i), cells.seq_pos_max(seq_id));

                add_cell = !is_masked;
            }

            // check the cell is in the requested position range
            if (add_cell) {
                const llama_pos pos = cells.pos_get(i);

                add_cell = pos >= pos_begin && pos < pos_end;
            }

            if (add_cell) {
                ++cell_count;
                if (cell_range_begin == cells.size()) {
                    cell_range_begin = i;
                }
            } else {
                if (cell_range_begin != cells.size()) {
                    cr.data.emplace_back(cell_range_begin, i);
                    cell_range_begin = cells.size();
                }
            }
        }

        if (cell_range_begin != cells.size()) {
            cr.data.emplace_back(cell_range_begin, cells.size());
        }

        // DEBUG CHECK: Sum of cell counts in ranges should equal the total cell count
        uint32_t cell_count_check = 0;
        for (const auto & range : cr.data) {
            cell_count_check += range.second - range.first;
        }
        GGML_ASSERT(cell_count == cell_count_check);

        io.write(&cell_count, sizeof(cell_count));

        // skip empty streams
        if (cell_count == 0) {
            continue;
        }

        state_write_meta(io, cr, seq_id);
        state_write_data(io, cr);
    }
}
```

Note: `LLAMA_POS_MAX` comes from `llama.h` (already included transitively by `llama-kv-cache.h`).

- [ ] **Step 3: Modify `src/llama-kv-cache.cpp` - `state_read` -> shared impl + `state_read_append`**

Replace the body of `llama_kv_cache::state_read` (line 2023):

```cpp
void llama_kv_cache::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(flags);

    state_read_impl(io, seq_id, false);
}

void llama_kv_cache::state_read_append(llama_io_read_i & io, llama_seq_id seq_id) {
    state_read_impl(io, seq_id, true);
}

void llama_kv_cache::state_read_impl(llama_io_read_i & io, llama_seq_id seq_id, bool append) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    uint32_t n_stream_cur;
    io.read(&n_stream_cur, sizeof(n_stream_cur));
    if (n_stream_cur != n_stream) {
        throw std::runtime_error("n_stream mismatch");
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
        uint32_t cell_count;
        io.read(&cell_count, sizeof(cell_count));

        if (cell_count == 0) {
            continue;
        }

        const uint32_t strm = seq_id == -1 ? s : seq_to_stream[seq_id];

        slot_info sinfo;

        bool res = true;
        res = res && state_read_meta(io, strm, cell_count, sinfo, seq_id, append);
        res = res && state_read_data(io, strm, cell_count, sinfo);

        if (!res) {
            if (seq_id == -1) {
                clear(true);
            } else {
                seq_rm(seq_id, -1, -1);
            }
            throw std::runtime_error("failed to restore kv cache");
        }
    }
}
```

- [ ] **Step 4: Modify `src/llama-kv-cache.cpp` - `state_read_meta` append flag**

Update the signature and the seq-clearing branch (line 2198-2204):

```cpp
bool llama_kv_cache::state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, slot_info & sinfo, llama_seq_id dest_seq_id, bool append) {
    auto & cells = v_cells[strm];
    auto & head  = v_heads[strm];

    if (dest_seq_id != -1) {
        // single sequence
        if (!append) {
            seq_rm(dest_seq_id, -1, -1);
        }
```

The rest of the function is unchanged.

- [ ] **Step 5: Add `count_cells_range`**

Add after the `state_read_impl` function (same file, before `state_write_meta`):

```cpp
size_t llama_kv_cache::count_cells_range(llama_seq_id seq_id, llama_pos pos_begin, llama_pos pos_end) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return 0;
    }

    size_t count = 0;

    for (uint32_t s = 0; s < n_stream; ++s) {
        const auto & cells = v_cells[s];

        for (uint32_t i = 0; i < cells.size(); ++i) {
            bool add_cell = true;

            add_cell = add_cell && !cells.is_empty(i);
            add_cell = add_cell && cells.seq_has(i, seq_id);

            // check the cell is not SWA-masked
            if (add_cell) {
                const bool is_masked = llama_hparams::is_masked_swa(n_swa, swa_type, cells.pos_get(i), cells.seq_pos_max(seq_id));

                add_cell = !is_masked;
            }

            // check the cell is in the requested position range
            if (add_cell) {
                const llama_pos pos = cells.pos_get(i);

                add_cell = pos >= pos_begin && pos < pos_end;
            }

            if (add_cell) {
                ++count;
            }
        }
    }

    return count;
}
```

- [ ] **Step 6: Build and run the existing state tests**

```powershell
cmake --build build --config Release --target test-save-load-state
```

Expected: builds clean. The GGSD tests are not yet present; the existing tests do not exercise the new methods, so only the build is verified here.

- [ ] **Step 7: Commit**

```bash
git add src/llama-kv-cache.h src/llama-kv-cache.cpp
git commit -m "kv-cache : add position-range state write and append read"
```

---

### Task 3: Write the failing GGSD C tests

**Files:**
- Modify: `tests/test-save-load-state.cpp`

**Interfaces:**
- Consumes: nothing (tests call the not-yet-existing `llama_state_seq_save_incr` / `llama_state_seq_load_incr` from `llama.h`; the compile failure is the "red" state).
- Produces: the acceptance tests for Task 4. Test helper functions: `make_random_tokens(model, n, seed)`, `decode_tokens(ctx, tokens, i_begin, i_end, seq_id)`, `count_segment_files()`, `segment_file_by_index(idx)`.

- [ ] **Step 1: Add includes and test helpers**

At the top of `tests/test-save-load-state.cpp`, add `#include <filesystem>` and `#include <fstream>` after the existing includes:

```cpp
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama-cpp.h"

#include <clocale>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>
```

Add the GGSD test helpers after the `generate_tokens` function (after line 48):

```cpp
// GGSD incremental state tests (see docs/superpowers/specs/2026-07-31-ggsd-incremental-slot-storage-design.md)

static const char * const k_incr_dir = "incr_test";

static std::string session_path(const std::string & name) {
    return std::string(k_incr_dir) + "/" + name;
}

static void incr_test_cleanup() {
    std::filesystem::remove_all(k_incr_dir);
    std::filesystem::create_directories(k_incr_dir);
}

static llama_tokens make_random_tokens(struct llama_model * model, size_t n, uint32_t seed) {
    const auto * vocab = llama_model_get_vocab(model);
    const auto n_vocab = llama_vocab_n_tokens(vocab);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<llama_token> dist(0, n_vocab - 1);

    llama_tokens tokens;
    tokens.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        tokens.push_back(dist(rng));
    }

    return tokens;
}

static bool decode_tokens(llama_context * ctx, const llama_tokens & tokens, size_t i_begin, size_t i_end, llama_seq_id seq_id) {
    llama_batch_ptr batch(512, 0, 1);

    for (size_t i = i_begin; i < i_end; ) {
        const size_t n = std::min<size_t>(512, i_end - i);

        common_batch_clear(batch.get());
        for (size_t j = 0; j < n; ++j) {
            common_batch_add(batch.get(), tokens[i + j], (llama_pos) (i + j), {seq_id}, true);
        }

        if (llama_decode(ctx, batch.get())) {
            LOG_ERR("%s: failed to decode tokens [%zu, %zu)\n", __func__, i, i + n);
            return false;
        }

        i += n;
    }

    return true;
}

static size_t count_segment_files() {
    size_t n = 0;
    for (const auto & entry : std::filesystem::directory_iterator(k_incr_dir)) {
        const std::string name = entry.path().filename().string();
        if (name.rfind("seg_", 0) == 0) {
            ++n;
        }
    }
    return n;
}

static std::string segment_file_by_index(uint32_t idx) {
    for (const auto & entry : std::filesystem::directory_iterator(k_incr_dir)) {
        const std::string name = entry.path().filename().string();
        if (name.rfind("seg_", 0) != 0) {
            continue;
        }

        std::ifstream f(entry.path(), std::ios::binary);
        char     magic[4];
        uint32_t version;
        uint32_t seg_index;
        f.read(magic, 4);
        f.read((char *) &version, sizeof(version));
        f.read((char *) &seg_index, sizeof(seg_index));
        if (seg_index == idx) {
            return entry.path().string();
        }
    }
    return "";
}

static bool compare_generation(
        struct llama_model * model,
        const struct common_params & params,
        llama_context * ctx,
        int n_past,
        const llama_tokens & expected_result) {
    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    auto result = generate_tokens(ctx, smpl.get(), n_past, params.n_predict, 0);
    if (result.empty()) {
        return false;
    }

    if (result != expected_result) {
        LOG_ERR("\n%s: error: generation differs from expected\n", __func__);
        return false;
    }

    LOG("\nPASS\n");
    return true;
}
```

- [ ] **Step 2: Add test 5 - save/load roundtrip with generation equivalence**

Add after `test_seq_cp_device` (after line 270):

```cpp
// Test 5: GGSD save/load roundtrip
// - decode 2500 tokens, save (2 segments)
// - restore 2048 tokens in a fresh context, decode the remainder, generate
// - compare the generation against the reference (same KV -> same logits)
static bool test_incr_roundtrip(struct llama_model * model, const struct common_params & params) {
    incr_test_cleanup();

    const std::string session = session_path("session_roundtrip.bin");
    const llama_tokens tokens = make_random_tokens(model, 2500, 1234);

    auto ctx = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};
    if (!decode_tokens(ctx.get(), tokens, 0, tokens.size(), 0)) {
        return false;
    }

    const int32_t n_segments = llama_state_seq_save_incr(ctx.get(), session.c_str(), 0, tokens.data(), tokens.size());
    if (n_segments != 2) {
        LOG_ERR("\n%s: error: expected 2 segments after first save, got %d\n", __func__, n_segments);
        return false;
    }

    // reference generation
    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    const llama_tokens expected_result = generate_tokens(ctx.get(), smpl.get(), (int) tokens.size(), params.n_predict, 0);
    if (expected_result.empty()) {
        return false;
    }

    // fresh context: restore the aligned prefix, decode the remainder
    auto ctx2 = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};

    const size_t n_restored = llama_state_seq_load_incr(ctx2.get(), session.c_str(), 0, tokens.data(), tokens.size(), 64);
    if (n_restored != 2048) {
        LOG_ERR("\n%s: error: expected 2048 tokens restored, got %zu\n", __func__, n_restored);
        return false;
    }

    if (!decode_tokens(ctx2.get(), tokens, n_restored, tokens.size(), 0)) {
        return false;
    }

    return compare_generation(model, params, ctx2.get(), (int) tokens.size(), expected_result);
}
```

- [ ] **Step 3: Add test 6 - incremental append**

```cpp
// Test 6: GGSD incremental save
// - decode 1500 tokens, save -> 1 segment
// - decode 1100 more, save -> 2 segments, only 1 new file
// - restore -> 2048 tokens
static bool test_incr_append(struct llama_model * model, const struct common_params & params) {
    incr_test_cleanup();

    const std::string session = session_path("session_append.bin");
    const llama_tokens tokens = make_random_tokens(model, 2600, 4321);

    auto ctx = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};
    if (!decode_tokens(ctx.get(), tokens, 0, 1500, 0)) {
        return false;
    }

    int32_t n_segments = llama_state_seq_save_incr(ctx.get(), session.c_str(), 0, tokens.data(), 1500);
    if (n_segments != 1) {
        LOG_ERR("\n%s: error: expected 1 segment after first save, got %d\n", __func__, n_segments);
        return false;
    }
    if (count_segment_files() != 1) {
        LOG_ERR("\n%s: error: expected 1 segment file, got %zu\n", __func__, count_segment_files());
        return false;
    }

    if (!decode_tokens(ctx.get(), tokens, 1500, 2600, 0)) {
        return false;
    }

    n_segments = llama_state_seq_save_incr(ctx.get(), session.c_str(), 0, tokens.data(), 2600);
    if (n_segments != 2) {
        LOG_ERR("\n%s: error: expected 2 segments after append save, got %d\n", __func__, n_segments);
        return false;
    }
    if (count_segment_files() != 2) {
        LOG_ERR("\n%s: error: expected 2 segment files after append, got %zu\n", __func__, count_segment_files());
        return false;
    }

    auto ctx2 = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};

    const size_t n_restored = llama_state_seq_load_incr(ctx2.get(), session.c_str(), 0, tokens.data(), tokens.size(), 64);
    if (n_restored != 2048) {
        LOG_ERR("\n%s: error: expected 2048 tokens restored, got %zu\n", __func__, n_restored);
        return false;
    }

    LOG("\nPASS\n");
    return true;
}
```

- [ ] **Step 4: Add test 7 - fork shares prefix segments**

```cpp
// Test 7: GGSD fork
// - chain A: 3000 tokens, save -> 2 segments
// - chain B: A[0..1500) + 1500 new tokens, save -> 2 segments, 1 shared file
// - restore with B -> 2048 tokens (matched chain)
// - restore with A -> 2048 tokens (orphaned chain still on disk)
static bool test_incr_fork(struct llama_model * model, const struct common_params & params) {
    incr_test_cleanup();

    const std::string session = session_path("session_fork.bin");
    const llama_tokens tokens_a = make_random_tokens(model, 3000, 111);
    const llama_tokens tokens_b_tail = make_random_tokens(model, 1500, 222);

    llama_tokens tokens_b = tokens_a;
    tokens_b.resize(1500);
    tokens_b.insert(tokens_b.end(), tokens_b_tail.begin(), tokens_b_tail.end());

    auto ctx_a = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};
    if (!decode_tokens(ctx_a.get(), tokens_a, 0, tokens_a.size(), 0)) {
        return false;
    }
    const int32_t n_segments_a = llama_state_seq_save_incr(ctx_a.get(), session.c_str(), 0, tokens_a.data(), tokens_a.size());
    if (n_segments_a != 2 || count_segment_files() != 2) {
        LOG_ERR("\n%s: error: chain A expected 2 segments/2 files, got %d/%zu\n", __func__, n_segments_a, count_segment_files());
        return false;
    }

    auto ctx_b = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};
    if (!decode_tokens(ctx_b.get(), tokens_b, 0, tokens_b.size(), 0)) {
        return false;
    }
    const int32_t n_segments_b = llama_state_seq_save_incr(ctx_b.get(), session.c_str(), 0, tokens_b.data(), tokens_b.size());
    if (n_segments_b != 2) {
        LOG_ERR("\n%s: error: chain B expected 2 segments, got %d\n", __func__, n_segments_b);
        return false;
    }
    if (count_segment_files() != 3) {
        LOG_ERR("\n%s: error: expected 3 segment files after fork (1 shared + 2 unique), got %zu\n", __func__, count_segment_files());
        return false;
    }

    // restore chain B (the session tail)
    auto ctx_rb = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};
    const size_t n_restored_b = llama_state_seq_load_incr(ctx_rb.get(), session.c_str(), 0, tokens_b.data(), tokens_b.size(), 64);
    if (n_restored_b != 2048) {
        LOG_ERR("\n%s: error: chain B restore expected 2048 tokens, got %zu\n", __func__, n_restored_b);
        return false;
    }

    // restore chain A (orphaned but still on disk)
    auto ctx_ra = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};
    const size_t n_restored_a = llama_state_seq_load_incr(ctx_ra.get(), session.c_str(), 0, tokens_a.data(), tokens_a.size(), 64);
    if (n_restored_a != 2048) {
        LOG_ERR("\n%s: error: chain A restore expected 2048 tokens, got %zu\n", __func__, n_restored_a);
        return false;
    }

    LOG("\nPASS\n");
    return true;
}
```

- [ ] **Step 5: Add test 8 - partial KV degradation and head-missing error**

```cpp
// Test 8: GGSD partial KV
// - decode 3000 tokens, save -> 2 segments
// - evict cells [1024, 3000), save -> chain shrinks to 1 segment
// - restore with the full prompt -> 1024 tokens
// - evict the head cells [0, 1024), save -> error (-1)
static bool test_incr_partial(struct llama_model * model, const struct common_params & params) {
    incr_test_cleanup();

    const std::string session = session_path("session_partial.bin");
    const llama_tokens tokens = make_random_tokens(model, 3000, 555);

    auto ctx = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};
    if (!decode_tokens(ctx.get(), tokens, 0, tokens.size(), 0)) {
        return false;
    }

    int32_t n_segments = llama_state_seq_save_incr(ctx.get(), session.c_str(), 0, tokens.data(), tokens.size());
    if (n_segments != 2) {
        LOG_ERR("\n%s: error: expected 2 segments, got %d\n", __func__, n_segments);
        return false;
    }

    // evict the tail of the KV cache
    common_context_seq_rm(ctx.get(), 0, 1024, 3000);

    n_segments = llama_state_seq_save_incr(ctx.get(), session.c_str(), 0, tokens.data(), tokens.size());
    if (n_segments != 1) {
        LOG_ERR("\n%s: error: expected chain to shrink to 1 segment, got %d\n", __func__, n_segments);
        return false;
    }

    auto ctx2 = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};
    const size_t n_restored = llama_state_seq_load_incr(ctx2.get(), session.c_str(), 0, tokens.data(), tokens.size(), 64);
    if (n_restored != 1024) {
        LOG_ERR("\n%s: error: expected 1024 tokens restored, got %zu\n", __func__, n_restored);
        return false;
    }

    // evict the head: save must fail
    common_context_seq_rm(ctx.get(), 0, 0, 1024);

    const int32_t n_segments_err = llama_state_seq_save_incr(ctx.get(), session.c_str(), 0, tokens.data(), tokens.size());
    if (n_segments_err != -1) {
        LOG_ERR("\n%s: error: expected save to fail with head missing, got %d\n", __func__, n_segments_err);
        return false;
    }

    LOG("\nPASS\n");
    return true;
}
```

- [ ] **Step 6: Add test 9 - corrupt segment**

```cpp
// Test 9: GGSD corrupt segment
// - decode 2500 tokens, save -> 2 segments
// - flip a byte in the payload of segment 1
// - restore -> stops at the corrupt segment (1024 tokens) with a warning
// - corrupt session file -> restore returns 0
static bool test_incr_corrupt(struct llama_model * model, const struct common_params & params) {
    incr_test_cleanup();

    const std::string session = session_path("session_corrupt.bin");
    const llama_tokens tokens = make_random_tokens(model, 2500, 777);

    auto ctx = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};
    if (!decode_tokens(ctx.get(), tokens, 0, tokens.size(), 0)) {
        return false;
    }
    if (llama_state_seq_save_incr(ctx.get(), session.c_str(), 0, tokens.data(), tokens.size()) != 2) {
        return false;
    }

    // flip the last byte of segment 1 (inside the payload)
    {
        const std::string fname = segment_file_by_index(1);
        if (fname.empty()) {
            LOG_ERR("\n%s: error: segment 1 file not found\n", __func__);
            return false;
        }

        std::fstream f(fname, std::ios::binary | std::ios::in | std::ios::out);
        f.seekg(-1, std::ios::end);
        char c;
        f.read(&c, 1);
        c ^= 0xFF;
        f.seekp(-1, std::ios::end);
        f.write(&c, 1);
    }

    auto ctx2 = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};
    const size_t n_restored = llama_state_seq_load_incr(ctx2.get(), session.c_str(), 0, tokens.data(), tokens.size(), 64);
    if (n_restored != 1024) {
        LOG_ERR("\n%s: error: expected 1024 tokens restored (stop at corrupt segment), got %zu\n", __func__, n_restored);
        return false;
    }

    // corrupt session file: restore is a no-op
    {
        std::ofstream f(session, std::ios::binary);
        f << "garbage";
    }

    const size_t n_restored2 = llama_state_seq_load_incr(ctx2.get(), session.c_str(), 0, tokens.data(), tokens.size(), 64);
    if (n_restored2 != 0) {
        LOG_ERR("\n%s: error: expected 0 tokens restored from corrupt session, got %zu\n", __func__, n_restored2);
        return false;
    }

    LOG("\nPASS\n");
    return true;
}
```

- [ ] **Step 7: Add test 10 - kv params mismatch and min_prefix**

```cpp
// Test 10: GGSD kv params mismatch and min_prefix
// - save with an F32 K cache -> hashes differ from the default F16 cache
// - restore from the default cache -> 0 tokens (cross-config mismatch)
// - min_prefix: restore with a shorter prompt, threshold above the match -> 0
static bool test_incr_mismatch(struct llama_model * model, const struct common_params & params) {
    incr_test_cleanup();

    const std::string session = session_path("session_kv.bin");
    const llama_tokens tokens = make_random_tokens(model, 2048, 999);

    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.type_k = GGML_TYPE_F32;

    auto ctx_f32 = llama_context_ptr{llama_init_from_model(model, params_ctx)};
    if (!decode_tokens(ctx_f32.get(), tokens, 0, tokens.size(), 0)) {
        return false;
    }
    if (llama_state_seq_save_incr(ctx_f32.get(), session.c_str(), 0, tokens.data(), tokens.size()) != 2) {
        return false;
    }

    // restore with the default F16 cache: kv_params is part of the hash, no match
    auto ctx = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};
    const size_t n_restored = llama_state_seq_load_incr(ctx.get(), session.c_str(), 0, tokens.data(), tokens.size(), 64);
    if (n_restored != 0) {
        LOG_ERR("\n%s: error: expected 0 tokens restored (kv params mismatch), got %zu\n", __func__, n_restored);
        return false;
    }

    // restore with the F32 cache and a shorter prompt: match is floor-aligned to 1024
    const size_t n_restored2 = llama_state_seq_load_incr(ctx_f32.get(), session.c_str(), 0, tokens.data(), 1500, 1024);
    if (n_restored2 != 1024) {
        LOG_ERR("\n%s: error: expected 1024 tokens restored, got %zu\n", __func__, n_restored2);
        return false;
    }

    // min_prefix above the match -> no-op
    const size_t n_restored3 = llama_state_seq_load_incr(ctx_f32.get(), session.c_str(), 0, tokens.data(), 1500, 2000);
    if (n_restored3 != 0) {
        LOG_ERR("\n%s: error: expected 0 tokens restored (min_prefix), got %zu\n", __func__, n_restored3);
        return false;
    }

    // nonexistent session -> no-op
    const size_t n_restored4 = llama_state_seq_load_incr(ctx.get(), session_path("session_none.bin").c_str(), 0, tokens.data(), tokens.size(), 64);
    if (n_restored4 != 0) {
        LOG_ERR("\n%s: error: expected 0 tokens restored (no session), got %zu\n", __func__, n_restored4);
        return false;
    }

    LOG("\nPASS\n");
    return true;
}
```

- [ ] **Step 8: Call the new tests from `main`**

After the existing test 4 call (line 353), add:

```cpp
    // Test 5: GGSD save/load roundtrip
    if (!test_incr_roundtrip(model, params)) {
        return 1;
    }

    // Test 6: GGSD incremental save
    if (!test_incr_append(model, params)) {
        return 1;
    }

    // Test 7: GGSD fork
    if (!test_incr_fork(model, params)) {
        return 1;
    }

    // Test 8: GGSD partial KV
    if (!test_incr_partial(model, params)) {
        return 1;
    }

    // Test 9: GGSD corrupt segment
    if (!test_incr_corrupt(model, params)) {
        return 1;
    }

    // Test 10: GGSD kv params mismatch
    if (!test_incr_mismatch(model, params)) {
        return 1;
    }
```

- [ ] **Step 9: Run the build to verify the tests fail**

```powershell
cmake --build build --config Release --target test-save-load-state
```

Expected: FAIL with compile errors `llama_state_seq_save_incr` / `llama_state_seq_load_incr` not declared (the API does not exist yet). This is the required red state.

- [ ] **Step 10: Commit (only the failing tests)**

```bash
git add tests/test-save-load-state.cpp
git commit -m "tests : add failing GGSD incremental state save/load tests"
```

---

### Task 4: GGSD C API and implementation

**Files:**
- Modify: `include/llama.h`
- Modify: `src/llama-context.h`
- Create: `src/llama-state-incr.cpp`
- Modify: `src/CMakeLists.txt` (add `llama-state-incr.cpp`; `llama-sha256.c` was added in Task 1)

**Interfaces:**
- Consumes: `llama_sha256_*` (Task 1), `llama_kv_cache::state_write_range` / `state_read_append` / `count_cells_range` (Task 2), failing tests (Task 3).
- Produces (used by Tasks 5-6):
  - `LLAMA_API int32_t llama_state_seq_save_incr(struct llama_context * ctx, const char * session_path, llama_seq_id seq_id, const llama_token * tokens, size_t n_token_count);` - chain length after save, 0 = no change, -1 = error.
  - `LLAMA_API size_t llama_state_seq_load_incr(struct llama_context * ctx, const char * session_path, llama_seq_id seq_id, const llama_token * prompt_tokens, size_t n_prompt_tokens, size_t min_prefix_tokens);` - tokens restored, 0 = no match/error.
  - `size_t llama_context::state_seq_save_incr(...)` / `llama_context::state_seq_load_incr(...)` (member functions, declared in `llama-context.h`).

- [ ] **Step 1: Declare the C API in `include/llama.h`**

Insert after `llama_state_seq_load_file` (after line 872):

```c
    // GGSD - incremental sequence state save/load
    //
    // Save the KV state of a sequence as a chain of 1024-token segments,
    // content-addressed by a hash of (model, kv params, previous hash, tokens).
    // Segments shared with previous saves are reused (append / fork semantics).
    // Returns the number of segments in the session chain after the save,
    // 0 if nothing changed, or -1 on error.
    LLAMA_API int32_t llama_state_seq_save_incr(
            struct llama_context * ctx,
                      const char * session_path,
                    llama_seq_id   seq_id,
               const llama_token * tokens,
                          size_t   n_token_count);

    // Restore the aligned prefix of the session chain matching the prompt.
    // Returns the number of tokens restored, or 0 if no match / error.
    LLAMA_API size_t llama_state_seq_load_incr(
            struct llama_context * ctx,
                      const char * session_path,
                    llama_seq_id   seq_id,
               const llama_token * prompt_tokens,
                          size_t   n_prompt_tokens,
                          size_t   min_prefix_tokens);
```

- [ ] **Step 2: Declare the member functions in `src/llama-context.h`**

After the existing `state_seq_save_file` declaration (line 173):

```cpp
    size_t state_seq_save_incr(const char * session_path, llama_seq_id seq_id, const llama_token * tokens, size_t n_token_count);
    size_t state_seq_load_incr(const char * session_path, llama_seq_id seq_id, const llama_token * prompt_tokens, size_t n_prompt_tokens, size_t min_prefix_tokens);
```

- [ ] **Step 3: Create `src/llama-state-incr.cpp`**

The full implementation. File layout matches the spec:

```
segment file (little-endian):
  u32  magic "GGSD"
  u32  version = 1
  u32  seg_index
  [32] prev_hash (hex ascii, zeros for head)
  u32  n_tokens = 1024
  u64  payload_size
  [16] payload_hash (sha256 of payload, truncated)
  str  model_id
  str  kv_params
  i32  tokens[1024]
       payload (llama_kv_cache::state_write output for the pos range)

session file:
  u32  magic "GGSD"
  u32  version = 1
  [32] tail_hash (hex ascii)
  u32  n_segments
```

```cpp
#include "llama.h"

#include "llama-arch.h"
#include "llama-context.h"
#include "llama-io.h"
#include "llama-kv-cache.h"
#include "llama-sha256.h"

#include "ggml.h"

#include <cstring>
#include <filesystem>
#include <vector>

//
// GGSD - incremental sequence state save/load
//
// see: docs/superpowers/specs/2026-07-31-ggsd-incremental-slot-storage-design.md
//

namespace {
    constexpr char GGSD_MAGIC[4] = { 'G', 'G', 'S', 'D' };

    constexpr uint32_t GGSD_VERSION                = 1;
    constexpr uint32_t GGSD_SEGMENT_TOKENS         = 1024;
    constexpr size_t   GGSD_HASH_BYTES             = 16; // truncated sha256
    constexpr size_t   GGSD_HASH_HEX_LEN           = 32;

    // segment header size before the payload hash field
    constexpr size_t GGSD_HEADER_PAYLOAD_HASH_OFFSET = 4 + 4 + 4 + 32 + 4 + 8;

    std::string ggsd_segment_path(const std::string & session_path, const std::string & hash) {
        std::filesystem::path p(session_path);
        return (p.parent_path() / ("seg_" + hash + ".bin")).string();
    }

    void ggsd_hash_hex_to_bytes(const std::string & hex, uint8_t * bytes) {
        for (size_t i = 0; i < GGSD_HASH_BYTES; ++i) {
            auto nibble = [](char c) -> uint8_t {
                if (c >= '0' && c <= '9') {
                    return c - '0';
                }
                return c - 'a' + 10;
            };
            bytes[i] = (nibble(hex[2*i]) << 4) | nibble(hex[2*i + 1]);
        }
    }

    std::string ggsd_segment_hash(
            const std::string & model_id,
            const std::string & kv_params,
            const uint8_t * prev_hash, // GGSD_HASH_BYTES, nullptr for chain head
            const llama_token * tokens) {
        llama_sha256_t sha;
        llama_sha256_init(&sha);

        const uint32_t len_model = (uint32_t) model_id.size();
        llama_sha256_update(&sha, (const uint8_t *) &len_model, sizeof(len_model));
        llama_sha256_update(&sha, (const uint8_t *) model_id.data(), len_model);

        const uint32_t len_params = (uint32_t) kv_params.size();
        llama_sha256_update(&sha, (const uint8_t *) &len_params, sizeof(len_params));
        llama_sha256_update(&sha, (const uint8_t *) kv_params.data(), len_params);

        if (prev_hash != nullptr) {
            llama_sha256_update(&sha, prev_hash, GGSD_HASH_BYTES);
        }

        llama_sha256_update(&sha, (const uint8_t *) tokens, GGSD_SEGMENT_TOKENS * sizeof(llama_token));

        uint8_t digest[32];
        llama_sha256_final(&sha, digest);

        std::string hex(GGSD_HASH_HEX_LEN, '0');
        for (size_t i = 0; i < GGSD_HASH_BYTES; ++i) {
            const uint8_t hi = digest[i] >> 4;
            const uint8_t lo = digest[i] & 0x0F;
            hex[2*i + 0] = hi < 10 ? '0' + hi : 'a' + hi - 10;
            hex[2*i + 1] = lo < 10 ? '0' + lo : 'a' + lo - 10;
        }

        return hex;
    }

    void ggsd_write_str(llama_file & file, const std::string & str) {
        const uint32_t len = (uint32_t) str.size();
        file.write_raw(&len, sizeof(len));
        file.write_raw(str.data(), len);
    }

    std::string ggsd_read_str(llama_file & file) {
        uint32_t len;
        file.read_raw(&len, sizeof(len));
        std::string str(len, '\0');
        if (len > 0) {
            file.read_raw(&str[0], len);
        }
        return str;
    }

    // file IO that hashes the bytes as they are written
    class ggsd_io_write_file : public llama_io_write_i {
    public:
        ggsd_io_write_file(llama_file * f, llama_sha256_t * sha) : file(f), sha(sha) {}

        void write(const void * src, size_t size) override {
            file->write_raw(src, size);
            llama_sha256_update(sha, (const uint8_t *) src, size);
            size_written += size;
        }

        void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
            temp_buffer.resize(size);
            ggml_backend_tensor_get(tensor, temp_buffer.data(), offset, size);
            write(temp_buffer.data(), temp_buffer.size());
        }

        size_t n_bytes() override {
            return size_written;
        }

    private:
        llama_file * file;
        llama_sha256_t * sha;
        size_t size_written = 0;
        std::vector<uint8_t> temp_buffer;
    };

    // dummy IO for measuring the payload size
    class ggsd_io_write_dummy : public llama_io_write_i {
    public:
        void write(const void * /* src */, size_t size) override {
            size_written += size;
        }

        void write_tensor(ggml_tensor * /* tensor */, size_t /* offset */, size_t size) override {
            size_written += size;
        }

        size_t n_bytes() override {
            return size_written;
        }

    private:
        size_t size_written = 0;
    };

    // host memory IO for the payload, tensors are flushed on destruction
    class ggsd_io_read_host : public llama_io_read_i {
    public:
        ggsd_io_read_host(const uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

        ~ggsd_io_read_host() {
            for (const auto & rinfo : rinfos) {
                ggml_backend_tensor_set(rinfo.tensor, rinfo.ptr, rinfo.offset, rinfo.size);
            }
        }

        void read(void * dst, size_t size) override {
            if (size > buf_size) {
                throw std::runtime_error("unexpectedly reached end of buffer");
            }
            memcpy(dst, ptr, size);
            ptr += size;
            size_read += size;
            buf_size -= size;
        }

        void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
            if (size > buf_size) {
                throw std::runtime_error("unexpectedly reached end of buffer");
            }

            // save for later during destruction
            rinfos.push_back({tensor, ptr, size, offset});

            ptr += size;
            size_read += size;
            buf_size -= size;
        }

        size_t n_bytes() override {
            return size_read;
        }

    private:
        const uint8_t * ptr;
        size_t buf_size = 0;
        size_t size_read = 0;

        struct read_info {
            ggml_tensor * tensor;
            const uint8_t * ptr;
            size_t size;
            size_t offset;
        };
        std::vector<read_info> rinfos;
    };
} // namespace

size_t llama_context::state_seq_save_incr(
        const char * session_path,
              llama_seq_id   seq_id,
        const llama_token * tokens,
              size_t   n_token_count) {
    auto * kv = dynamic_cast<llama_kv_cache *>(memory.get());
    if (kv == nullptr) {
        LLAMA_LOG_ERROR("%s: incremental state save is only supported for the standard kv cache\n", __func__);
        return (size_t) -1;
    }

    const size_t n_seg = n_token_count / GGSD_SEGMENT_TOKENS;
    if (n_seg == 0) {
        return 0;
    }

    const uint32_t n_pos_per_embd = model.hparams.n_pos_per_embd();

    // model identity: arch name + kv cache parameters, both part of the segment hash
    const std::string model_id = llm_arch_name(model.arch);
    const std::string kv_params = std::string(ggml_type_name(kv->type_k())) + "|" +
                                  std::string(ggml_type_name(kv->type_v())) + "|" +
                                  std::to_string(n_pos_per_embd);

    // read the session file; a corrupt or missing session starts a new chain
    uint32_t n_segments = 0;
    std::string tail_hash_read;
    if (std::filesystem::exists(session_path)) {
        bool ok = true;
        try {
            llama_file file(session_path, "rb");

            char magic[4];
            file.read_raw(magic, 4);

            const uint32_t version = file.read_u32();

            char tail_hash[GGSD_HASH_HEX_LEN];
            file.read_raw(tail_hash, sizeof(tail_hash));

            n_segments = file.read_u32();

            if (memcmp(magic, GGSD_MAGIC, 4) != 0 || version != GGSD_VERSION) {
                ok = false;
            } else {
                tail_hash_read = std::string(tail_hash, sizeof(tail_hash));
            }
        } catch (const std::exception & e) {
            ok = false;
        }

        if (!ok) {
            LLAMA_LOG_WARN("%s: invalid session file %s, starting a new chain\n", __func__, session_path);
            n_segments = 0;
        }
    }

    // hash chain of the prompt
    std::vector<std::string> hashes(n_seg);
    for (size_t k = 0; k < n_seg; ++k) {
        uint8_t prev[GGSD_HASH_BYTES];
        if (k == 0) {
            memset(prev, 0, sizeof(prev));
        } else {
            ggsd_hash_hex_to_bytes(hashes[k - 1], prev);
        }

        hashes[k] = ggsd_segment_hash(model_id, kv_params, prev, tokens + k * GGSD_SEGMENT_TOKENS);
    }

    // find the fork point: the first segment that is not already saved,
    // i.e. missing on disk or missing from the KV cache (eviction case)
    size_t k0 = 0;
    for (; k0 < std::min<size_t>(n_segments, n_seg); ++k0) {
        const llama_pos pos_begin = (llama_pos) ( k0       * GGSD_SEGMENT_TOKENS * n_pos_per_embd);
        const llama_pos pos_end   = (llama_pos) ((k0 + 1) * GGSD_SEGMENT_TOKENS * n_pos_per_embd);

        if (!std::filesystem::exists(ggsd_segment_path(session_path, hashes[k0])) ||
                kv->count_cells_range(seq_id, pos_begin, pos_end) != GGSD_SEGMENT_TOKENS) {
            break;
        }
    }

    // write new segments starting at the fork point
    size_t n_written = 0;
    for (size_t k = k0; k < n_seg; ++k) {
        const llama_pos pos_begin = (llama_pos) ( k       * GGSD_SEGMENT_TOKENS * n_pos_per_embd);
        const llama_pos pos_end   = (llama_pos) ((k + 1) * GGSD_SEGMENT_TOKENS * n_pos_per_embd);

        if (kv->count_cells_range(seq_id, pos_begin, pos_end) != GGSD_SEGMENT_TOKENS) {
            // partial KV cache: stop at the contiguous prefix
            break;
        }

        // measure the payload size
        ggsd_io_write_dummy io_dummy;
        kv->state_write_range(io_dummy, seq_id, pos_begin, pos_end);
        const uint64_t payload_size = io_dummy.n_bytes();

        const std::string fname     = ggsd_segment_path(session_path, hashes[k]);
        const std::string fname_tmp = fname + ".tmp";

        {
            llama_file file(fname_tmp.c_str(), "wb");

            file.write_raw(GGSD_MAGIC, sizeof(GGSD_MAGIC));
            file.write_u32(GGSD_VERSION);
            file.write_u32((uint32_t) k);

            char prev[GGSD_HASH_HEX_LEN];
            if (k == 0) {
                memset(prev, 0, sizeof(prev));
            } else {
                memcpy(prev, hashes[k - 1].c_str(), sizeof(prev));
            }
            file.write_raw(prev, sizeof(prev));

            file.write_u32(GGSD_SEGMENT_TOKENS);
            file.write_raw(&payload_size, sizeof(payload_size));

            uint8_t payload_hash[GGSD_HASH_BYTES];
            memset(payload_hash, 0, sizeof(payload_hash));
            file.write_raw(payload_hash, sizeof(payload_hash)); // patched below

            ggsd_write_str(file, model_id);
            ggsd_write_str(file, kv_params);
            file.write_raw(tokens + k * GGSD_SEGMENT_TOKENS, GGSD_SEGMENT_TOKENS * sizeof(llama_token));

            llama_sha256_t sha;
            llama_sha256_init(&sha);
            {
                ggsd_io_write_file io(&file, &sha);
                kv->state_write_range(io, seq_id, pos_begin, pos_end);
            }
            llama_sha256_final(&sha, payload_hash);

            // patch the payload hash into the header
            file.seek(GGSD_HEADER_PAYLOAD_HASH_OFFSET, SEEK_SET);
            file.write_raw(payload_hash, sizeof(payload_hash));
        }

        std::filesystem::rename(fname_tmp, fname);
        ++n_written;
    }

    const uint32_t n_chain = (uint32_t) (k0 + n_written);
    if (n_chain == 0) {
        LLAMA_LOG_ERROR("%s: KV cache is missing the head of the sequence (pos [0, %d)), nothing saved\n",
                __func__, (int) (GGSD_SEGMENT_TOKENS * n_pos_per_embd));
        return (size_t) -1;
    }

    // rewrite the session file when the chain changed
    const std::string tail_hash = hashes[n_chain - 1];
    if (n_chain != n_segments || tail_hash != tail_hash_read) {
        llama_file file(session_path, "wb");

        file.write_raw(GGSD_MAGIC, sizeof(GGSD_MAGIC));
        file.write_u32(GGSD_VERSION);
        file.write_raw(tail_hash.c_str(), tail_hash.size());
        file.write_u32(n_chain);
    }

    return n_chain;
}

size_t llama_context::state_seq_load_incr(
        const char * session_path,
              llama_seq_id   seq_id,
        const llama_token * prompt_tokens,
              size_t   n_prompt_tokens,
              size_t   min_prefix_tokens) {
    if (n_prompt_tokens < GGSD_SEGMENT_TOKENS) {
        return 0;
    }
    if (!std::filesystem::exists(session_path)) {
        return 0;
    }

    auto * kv = dynamic_cast<llama_kv_cache *>(memory.get());
    if (kv == nullptr) {
        LLAMA_LOG_ERROR("%s: incremental state load is only supported for the standard kv cache\n", __func__);
        return 0;
    }

    // read the session file
    uint32_t n_segments = 0;
    {
        llama_file file(session_path, "rb");

        char magic[4];
        file.read_raw(magic, 4);

        const uint32_t version = file.read_u32();

        char tail_hash[GGSD_HASH_HEX_LEN];
        file.read_raw(tail_hash, sizeof(tail_hash));

        n_segments = file.read_u32();

        if (memcmp(magic, GGSD_MAGIC, 4) != 0 || version != GGSD_VERSION) {
            LLAMA_LOG_ERROR("%s: invalid session file %s\n", __func__, session_path);
            return 0;
        }
    }

    const uint32_t n_pos_per_embd = model.hparams.n_pos_per_embd();

    const std::string model_id = llm_arch_name(model.arch);
    const std::string kv_params = std::string(ggml_type_name(kv->type_k())) + "|" +
                                  std::string(ggml_type_name(kv->type_v())) + "|" +
                                  std::to_string(n_pos_per_embd);

    const size_t n_seg = std::min<size_t>(n_segments, n_prompt_tokens / GGSD_SEGMENT_TOKENS);

    // hash chain of the prompt prefix
    std::vector<std::string> hashes(n_seg);
    for (size_t k = 0; k < n_seg; ++k) {
        uint8_t prev[GGSD_HASH_BYTES];
        if (k == 0) {
            memset(prev, 0, sizeof(prev));
        } else {
            ggsd_hash_hex_to_bytes(hashes[k - 1], prev);
        }

        hashes[k] = ggsd_segment_hash(model_id, kv_params, prev, prompt_tokens + k * GGSD_SEGMENT_TOKENS);
    }

    // find the longest chain of segment files matching the prompt
    size_t n_seg_ok = 0;
    for (; n_seg_ok < n_seg; ++n_seg_ok) {
        if (!std::filesystem::exists(ggsd_segment_path(session_path, hashes[n_seg_ok]))) {
            break;
        }
    }

    const size_t n_restored = n_seg_ok * GGSD_SEGMENT_TOKENS;
    if (n_restored < min_prefix_tokens) {
        return 0;
    }

    // clear the sequence and restore the matched segments
    kv->seq_rm(seq_id, -1, -1);

    size_t n_loaded = 0;
    try {
        for (size_t k = 0; k < n_seg_ok; ++k) {
            const std::string fname = ggsd_segment_path(session_path, hashes[k]);

            llama_file file(fname.c_str(), "rb");

            char magic[4];
            file.read_raw(magic, 4);

            const uint32_t version   = file.read_u32();
            const uint32_t seg_index = file.read_u32();

            char prev[GGSD_HASH_HEX_LEN];
            file.read_raw(prev, sizeof(prev));

            const uint32_t n_tokens = file.read_u32();

            uint64_t payload_size;
            file.read_raw(&payload_size, sizeof(payload_size));

            uint8_t payload_hash[GGSD_HASH_BYTES];
            file.read_raw(payload_hash, sizeof(payload_hash));

            if (memcmp(magic, GGSD_MAGIC, 4) != 0 || version != GGSD_VERSION ||
                    seg_index != k || n_tokens != GGSD_SEGMENT_TOKENS) {
                LLAMA_LOG_ERROR("%s: invalid segment file %s (k = %zu)\n", __func__, fname.c_str(), k);
                break;
            }

            const std::string f_model_id = ggsd_read_str(file);
            const std::string f_kv_params = ggsd_read_str(file);
            if (f_model_id != model_id || f_kv_params != kv_params) {
                LLAMA_LOG_ERROR("%s: segment file %s was created with a different model\n", __func__, fname.c_str());
                break;
            }

            // skip the tokens; the file name is the content hash of the tokens
            file.seek(GGSD_SEGMENT_TOKENS * sizeof(llama_token), SEEK_CUR);

            std::vector<uint8_t> payload(payload_size);
            file.read_raw(payload.data(), payload.size());

            // verify the payload hash
            uint8_t digest[32];
            {
                llama_sha256_t sha;
                llama_sha256_init(&sha);
                llama_sha256_update(&sha, payload.data(), payload.size());
                llama_sha256_final(&sha, digest);
            }
            if (memcmp(digest, payload_hash, GGSD_HASH_BYTES) != 0) {
                LLAMA_LOG_ERROR("%s: segment file %s is corrupt (payload hash mismatch)\n", __func__, fname.c_str());
                break;
            }

            ggsd_io_read_host io(payload.data(), payload.size());
            kv->state_read_append(io, seq_id);

            ++n_loaded;
        }
    } catch (...) {
        // undo the partial restore
        kv->seq_rm(seq_id, -1, -1);
        throw;
    }

    if (n_loaded != n_seg_ok) {
        // undo the partial restore
        kv->seq_rm(seq_id, -1, -1);
    }

    return n_loaded * GGSD_SEGMENT_TOKENS;
}

//
// C API
//

int32_t llama_state_seq_save_incr(
        llama_context * ctx,
        const char * session_path,
        llama_seq_id seq_id,
        const llama_token * tokens,
        size_t n_token_count) {
    ctx->synchronize();

    try {
        const size_t res = ctx->state_seq_save_incr(session_path, seq_id, tokens, n_token_count);
        if (res == (size_t) -1) {
            LLAMA_LOG_ERROR("%s: failed to save incremental sequence state\n", __func__);
            return -1;
        }
        return (int32_t) res;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving incremental sequence state: %s\n", __func__, err.what());
        return -1;
    }
}

size_t llama_state_seq_load_incr(
        llama_context * ctx,
        const char * session_path,
        llama_seq_id seq_id,
        const llama_token * prompt_tokens,
        size_t n_prompt_tokens,
        size_t min_prefix_tokens) {
    ctx->synchronize();

    try {
        return ctx->state_seq_load_incr(session_path, seq_id, prompt_tokens, n_prompt_tokens, min_prefix_tokens);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading incremental sequence state: %s\n", __func__, err.what());
        return 0;
    }
}
```

Note: `LLAMA_LOG_ERROR` / `LLAMA_LOG_WARN` and `LLAMA_POS_MAX` are defined in `llama.h` (included first). `ggml_backend_tensor_get` / `ggml_backend_tensor_set` come from `ggml.h`.

- [ ] **Step 4: Register `llama-state-incr.cpp` in `src/CMakeLists.txt`**

In the `add_library(llama ...)` list:

```cmake
            llama-sampler.cpp
            llama-sha256.c
            llama-state-incr.cpp
            llama-vocab.cpp
```

- [ ] **Step 5: Build and run the tests**

```powershell
cmake --build build --config Release --target test-save-load-state
```

Then run (requires any small GGUF model; `MODEL` is the path, e.g. a `stories260K.gguf`):

```powershell
build/bin/Release/test-save-load-state.exe -m <MODEL>
```

Expected: all tests pass, including the new tests 5-10. The GGSD tests write to `incr_test/` in the working directory and clean it up per test.

If `n_ctx` of the model's default is below 4096 (the test decodes up to 3000 tokens), pass `-c 4096`.

- [ ] **Step 6: Commit**

```bash
git add include/llama.h src/llama-context.h src/llama-state-incr.cpp src/CMakeLists.txt
git commit -m "llama : add GGSD incremental sequence state save/load"
```

---

### Task 5: Server task plumbing

**Files:**
- Modify: `tools/server/server-task.h`
- Modify: `tools/server/server-task.cpp`

**Interfaces:**
- Consumes: `llama_state_seq_save_incr` / `llama_state_seq_load_incr` (Task 4).
- Produces (used by Task 6):
  - enum values `SERVER_TASK_TYPE_SLOT_SAVE_INCR`, `SERVER_TASK_TYPE_SLOT_RESTORE_INCR`
  - `server_task::slot_action` gains `llama_tokens prompt_tokens;` and `size_t min_prefix = 64;`
  - `server_task_result_slot_incr` with fields `filename`, `is_save`, `n_segments`, `n_tokens`, `t_ms`, and `to_json()`.

- [ ] **Step 1: Modify `tools/server/server-task.h` - task types**

In the `server_task_type` enum (line 16-30), after `SERVER_TASK_TYPE_SLOT_ERASE`:

```cpp
    SERVER_TASK_TYPE_SLOT_SAVE_INCR,
    SERVER_TASK_TYPE_SLOT_RESTORE_INCR,
```

- [ ] **Step 2: Modify `tools/server/server-task.h` - slot_action**

In the `slot_action` struct (line 162-166):

```cpp
    // used by SERVER_TASK_TYPE_SLOT_SAVE, SERVER_TASK_TYPE_SLOT_RESTORE, SERVER_TASK_TYPE_SLOT_ERASE
    struct slot_action {
        int id_slot;
        std::string filename;
        std::string filepath;

        // used by SERVER_TASK_TYPE_SLOT_RESTORE_INCR
        llama_tokens prompt_tokens;
        size_t min_prefix = 64;
    };
```

- [ ] **Step 3: Modify `tools/server/server-task.h` - result struct**

After `server_task_result_slot_save_load` (line 551):

```cpp
struct server_task_result_slot_incr : server_task_result {
    std::string filename;
    bool is_save; // true = save, false = load

    size_t n_segments; // save: number of segments in the session chain
    size_t n_tokens;   // save: tokens covered by the chain; load: tokens restored
    double t_ms;

    virtual json to_json() override;
};
```

- [ ] **Step 4: Implement `to_json` in `tools/server/server-task.cpp`**

After the `server_task_result_slot_save_load::to_json` implementation (ends near line 1935):

```cpp
json server_task_result_slot_incr::to_json() {
    if (is_save) {
        return json {
            { "id_slot",    id_slot },
            { "filename",   filename },
            { "n_segments", n_segments },
            { "n_tokens",   n_tokens },
            { "t_ms",       t_ms },
        };
    }

    return json {
        { "id_slot",           id_slot },
        { "filename",          filename },
        { "n_tokens_restored", n_tokens },
        { "t_ms",              t_ms },
    };
}
```

- [ ] **Step 5: Build**

```powershell
cmake --build build --config Release --target llama-server
```

Expected: builds clean (the new task types are unused yet, so no behavior change).

- [ ] **Step 6: Commit**

```bash
git add tools/server/server-task.h tools/server/server-task.cpp
git commit -m "server : add GGSD slot task types and results"
```

---

### Task 6: Server handlers and dispatch

**Files:**
- Modify: `tools/server/server-context.h`
- Modify: `tools/server/server-context.cpp`

**Interfaces:**
- Consumes: task types and result structs (Task 5), C API (Task 4), `common_tokenize` (from `common.h`, already used by server), `json_value` helper, `fs_validate_filename` (from `common.h`), `params.slot_save_path`.
- Produces: handlers `handle_slots_save_incr`, `handle_slots_restore_incr`; dispatch entries `action=save_incr`, `action=restore_incr` in `post_slots`; `process_single_task` cases.

- [ ] **Step 1: Modify `tools/server/server-context.h` - handler declarations**

After `handle_slots_erase` (line 143):

```cpp
    std::unique_ptr<server_res_generator> handle_slots_save_incr(const server_http_req & req, int id_slot);
    std::unique_ptr<server_res_generator> handle_slots_restore_incr(const server_http_req & req, int id_slot);
```

- [ ] **Step 2: Modify `tools/server/server-context.cpp` - post_slots dispatch**

In the `post_slots` lambda (line 4170-4178), after the `erase` branch:

```cpp
        if (action == "save_incr") {
            return handle_slots_save_incr(req, id_slot);
        }
        if (action == "restore_incr") {
            return handle_slots_restore_incr(req, id_slot);
        }
```

- [ ] **Step 3: Modify `tools/server/server-context.cpp` - handlers**

After `handle_slots_restore` (ends line 4833):

```cpp
std::unique_ptr<server_res_generator> server_routes::handle_slots_save_incr(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + "session_" + filename + ".bin";

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_SAVE_INCR);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_restore_incr(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    const std::string prompt = request_data.at("prompt").get<std::string>();
    const size_t min_prefix = json_value(request_data, "min_prefix", 64);
    std::string filepath = params.slot_save_path + "session_" + filename + ".bin";

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_RESTORE_INCR);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        task.slot_action.prompt_tokens = common_tokenize(ctx_server.vocab, prompt, true, true);
        task.slot_action.min_prefix = min_prefix;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_incr*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}
```

- [ ] **Step 4: Modify `tools/server/server-context.cpp` - process_single_task cases**

After the `SERVER_TASK_TYPE_SLOT_ERASE` case (ends line 2346):

```cpp
            case SERVER_TASK_TYPE_SLOT_SAVE_INCR:
                {
                    if (!check_no_mtmd(task.id)) {
                        break;
                    }
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    const llama_tokens & tokens = slot->prompt.tokens.get_tokens();
                    const int32_t n_segments = llama_state_seq_save_incr(ctx_tgt, filepath.c_str(), slot->id, tokens.data(), tokens.size());

                    const int64_t t_end = ggml_time_us();
                    const double t_save_ms = (t_end - t_start) / 1000.0;

                    if (n_segments < 0) {
                        send_error(task, "Failed to save slot state (GGSD): KV cache is missing the head of the sequence", ERROR_TYPE_SERVER);
                        break;
                    }

                    auto res = std::make_unique<server_task_result_slot_incr>();
                    res->id         = task.id;
                    res->id_slot    = id_slot;
                    res->filename   = filename;
                    res->is_save    = true;
                    res->n_segments = n_segments;
                    res->n_tokens   = (size_t) n_segments * 1024;
                    res->t_ms       = t_save_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_RESTORE_INCR:
                {
                    if (!check_no_mtmd(task.id)) {
                        break;
                    }
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    const llama_tokens & tokens = task.slot_action.prompt_tokens;
                    const size_t n_restored = llama_state_seq_load_incr(ctx_tgt, filepath.c_str(), slot->id, tokens.data(), tokens.size(), task.slot_action.min_prefix);

                    const int64_t t_end = ggml_time_us();
                    const double t_restore_ms = (t_end - t_start) / 1000.0;

                    // keep only the restored prefix in the prompt cache;
                    // the remainder is re-processed by the next completion
                    slot->prompt.tokens.clear();
                    if (n_restored > 0) {
                        slot->prompt.tokens.insert(llama_tokens(tokens.begin(), tokens.begin() + n_restored));
                    }

                    auto res = std::make_unique<server_task_result_slot_incr>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = false;
                    res->n_tokens = n_restored;
                    res->t_ms     = t_restore_ms;
                    queue_results.send(std::move(res));
                } break;
```

- [ ] **Step 5: Build**

```powershell
cmake --build build --config Release --target llama-server
```

Expected: builds clean.

- [ ] **Step 6: Manual smoke test**

Start the server with a small model and `--slot-save-path saves` (create the dir first), then exercise the endpoints:

```powershell
mkdir saves
build/bin/Release/llama-server.exe -m <MODEL> --slot-save-path saves -c 4096
```

```powershell
$prompt = "The quick brown fox jumps over the lazy dog. " * 100
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8080/completion" -ContentType "application/json" -Body (@{ prompt = $prompt; id_slot = 0; cache_prompt = $true; n_predict = 1 } | ConvertTo-Json) | ConvertTo-Json -Depth 5
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8080/slots/0?action=save_incr" -ContentType "application/json" -Body (@{ filename = "test1" } | ConvertTo-Json) | ConvertTo-Json -Depth 5
```

Expected: the save returns `n_segments >= 1` and `n_tokens == n_segments * 1024`; the `saves/` directory contains `session_test1.bin` and `seg_*.bin` files.

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8080/slots/0?action=restore_incr" -ContentType "application/json" -Body (@{ filename = "test1"; prompt = $prompt } | ConvertTo-Json) | ConvertTo-Json -Depth 5
```

Expected: `n_tokens_restored == n_segments * 1024` from the save.

- [ ] **Step 7: Commit**

```bash
git add tools/server/server-context.h tools/server/server-context.cpp
git commit -m "server : add save_incr and restore_incr slot actions"
```

---

### Task 7: Server tests, README, and spec corrections

**Files:**
- Create: `tools/server/tests/unit/test_slot_save_incr.py`
- Modify: `tools/server/README.md`
- Modify: `docs/superpowers/specs/2026-07-31-ggsd-incremental-slot-storage-design.md` (apply the Spec Corrections from the top of this plan)

**Interfaces:**
- Consumes: the endpoints from Task 6, `ServerPreset.tinyllama2()` from `tools/server/tests/utils.py`.

- [ ] **Step 1: Create `tools/server/tests/unit/test_slot_save_incr.py`**

Mirrors `test_slot_save.py`. Uses a prompt long enough for 3 segments (over 3072 tokens) and verifies the roundtrip and the `n_past` behavior after restore.

```python
import pytest
from utils import *

server = ServerPreset.tinyllama2()

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.slot_save_path = "./tmp"
    server.temperature = 0.0
    server.n_ctx = 4096
    server.n_batch = 128


def test_slot_save_incr_restore():
    global server
    server.start()

    # long prompt spanning 3 GGSD segments (>= 3072 tokens)
    prompt = "What is the capital of France? " * 400

    # fully process the prompt in slot 1
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Whiskers|Flana)+", res.body["content"])
    n_prompt = res.body["timings"]["prompt_n"]
    assert n_prompt >= 3072

    # incremental save
    res = server.make_request("POST", "/slots/1?action=save_incr", data={
        "filename": "slot1",
    })
    assert res.status_code == 200
    assert res.body["n_segments"] >= 3
    assert res.body["n_tokens"] == res.body["n_segments"] * 1024

    # process the same prompt in slot 0 without a cache
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] == n_prompt

    # restore the cache into slot 0
    res = server.make_request("POST", "/slots/0?action=restore_incr", data={
        "filename": "slot1",
        "prompt": prompt,
    })
    assert res.status_code == 200
    assert res.body["n_tokens_restored"] == res.body["n_tokens"]

    # re-run the same prompt, only the tail should be processed
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Whiskers|Flana)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] < 64
```

- [ ] **Step 2: Run the server test**

Requires the server binary (built in Task 6) and the test model cache:

```powershell
python -m pytest tools/server/tests/unit/test_slot_save_incr.py -v
```

Expected: the test passes (it downloads the stories260K test model on first run). If a server from a previous test is still running on port 8080, stop it first.

- [ ] **Step 3: Document the endpoints in `tools/server/README.md`**

After the existing `action=erase` section (line 1111-1117):

```markdown
### POST `/slots/{id_slot}?action=save_incr`: Incrementally save the prompt cache of the specified slot.

Saves the KV cache as a chain of content-addressed 1024-token segments
(`seg_<hash>.bin` files) plus a small session manifest
(`session_<filename>.bin`) in the slot save directory. Segments shared with
previous saves are reused, so consecutive saves only write the newly crossed
segment boundaries. Standard `action=save` (GGSQ v2) is unaffected.

Request body:

- `filename`: (Required) The session name to save to.

Response body:

```json
{
  "id_slot": 0,
  "filename": "session_name",
  "n_segments": 2,
  "n_tokens": 2048,
  "t_ms": 7
}
```

### POST `/slots/{id_slot}?action=restore_incr`: Restore the prompt cache of the specified slot from an incremental session.

Matches the aligned prefix of the session chain against the given `prompt` and
restores the KV cache for the matched prefix. The remainder of the prompt is
processed normally by the next completion.

Request body:

- `filename`: (Required) The session name to restore from.
- `prompt`: (Required) The prompt to match against the saved chain.
- `min_prefix`: (Optional) Minimum number of matched tokens for the restore to
  be applied. Default: `64`.

Response body:

```json
{
  "id_slot": 0,
  "filename": "session_name",
  "n_tokens_restored": 2048,
  "t_ms": 5
}
```
```

Note: use the markdown code fences exactly as in the existing README (the section above shows them nested for readability here; in the file use plain triple backticks).

- [ ] **Step 4: Update the spec document with the corrections**

In `docs/superpowers/specs/2026-07-31-ggsd-incremental-slot-storage-design.md`:

1. In the segment file layout (line 84-91), add the payload hash after `n_tokens`:

```
u32  magic "GGSD"
u32  version = 1
u32  seg_index                 # position in chain, 0-based
[32] prev_hash                 # previous segment hash (all zeros for head)
u32  n_tokens = 1024           # constant; field present for future extension
[16] payload_hash              # sha256 of the payload, truncated to 16 bytes
str  model_id                  # string with length prefix
str  kv_params                 # string with length prefix
i32  tokens[1024]              # the segment tokens
     payload                   # KV cells for this token range (see below)
```

2. Replace the `llama_state_seq_save_incr` doc comment (line 125-130) to say it returns the number of segments in the session chain after the save (0 = no change, -1 = error).

3. Replace the "Reusable pieces" Hash bullet (line 226):

```
- Hash: vendored SHA-256 in `src/llama-sha256.{h,c}` (public domain, based on
  the implementation used by `examples/gguf-hash`).
```

4. Replace the endpoint syntax (line 165-173) with the action-param form:

```
POST /slots/{id_slot}?action=save_incr
    body: { "filename": "session_name" }
    -> { "n_segments": 2, "n_tokens": 2048 }

POST /slots/{id_slot}?action=restore_incr
    body: { "filename": "session_name", "prompt": "...", "min_prefix": 64? }
    -> { "n_tokens_restored": 2048 }
```

- [ ] **Step 5: Final verification**

```powershell
cmake --build build --config Release --target llama llama-server test-save-load-state
```

```powershell
build/bin/Release/test-save-load-state.exe -m <MODEL>
```

```powershell
python -m pytest tools/server/tests/unit/test_slot_save_incr.py -v
```

All expected to pass.

- [ ] **Step 6: Commit**

```bash
git add tools/server/tests/unit/test_slot_save_incr.py tools/server/README.md docs/superpowers/specs/2026-07-31-ggsd-incremental-slot-storage-design.md
git commit -m "server : add GGSD slot tests and docs"
```

---

## Self-Review Notes

- **Spec coverage:** every edge case in the spec's table (line 190-200) maps to a test in Task 3 or a code path in Task 4: append (test 6), fork (test 7), partial KV (test 8), head missing (test 8), LCP < min_prefix (test 10), corrupt segment (test 9), corrupt/absent session (tests 9/10), cross-config mismatch (test 10). Cross-MODEL mismatch is not unit-tested (requires two models); it is structurally identical to the kv_params path since `model_id` and `kv_params` are both hash inputs.
- **Placeholder scan:** no TBD/TODO items; all code is complete. The only behavioral constants are `GGSD_SEGMENT_TOKENS = 1024` and server `min_prefix` default `64`, both from the spec.
- **Type consistency:** `llama_state_seq_save_incr` returns `int32_t` (chain length, `-1` on error); `llama_state_seq_load_incr` returns `size_t` (tokens restored). Server response keys: save `n_segments`/`n_tokens`, restore `n_tokens_restored`, both including `id_slot`/`filename`/`t_ms`.
- **Self-review trace (2026-07-31):** every test in Task 3 was traced through the Task 4 code; this caught three bugs that are fixed in the code above:
  1. The fork-point scan must also check KV cell presence (`count_cells_range`), not just file existence - otherwise tail eviction (test 8) would not shrink the session chain.
  2. The session rewrite condition compares the chain **tail hash**, not just the segment count - otherwise a fork with an equal segment count (test 7) would leave the session pointing at the old chain.
  3. The restore loop is wrapped in try/catch: an IO exception mid-restore (e.g. truncated segment file) now undoes the partial restore before propagating.
- **Verified APIs:** `llama_file::seek(size_t, int)`, `read_u32`, `write_u32` exist (llama-mmap.h:26-34); `llama_io_write_i`/`llama_io_read_i` interfaces match the IO classes (llama-io.h:9-32); the deferred-tensor-set pattern of `ggsd_io_read_host` mirrors `llama_io_read_host` (llama-context.cpp:2464); `llama_context::memory` exists (llama-context.h:277); `server_tokens::insert(const llama_tokens&)`/`clear()` exist (server-common.h:192,206); `llama_cparams` uses `type_k`/`type_v` (not `cache_type_k`); the `llama` target compiles with `cxx_std_17` (src/CMakeLists.txt:53), so `std::filesystem` is available.
