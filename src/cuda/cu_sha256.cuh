
#pragma once

#include <stdint.h>

__device__ uint64_t cu_sha256d_to_u64(uint8_t* indata, uint32_t inlen );

__global__ void test_for_sha256d_to_u64(uint8_t * in, uint32_t inlen, uint64_t *out);