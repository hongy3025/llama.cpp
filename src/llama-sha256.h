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
