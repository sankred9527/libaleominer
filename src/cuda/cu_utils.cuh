#pragma once

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>
#include "bls12_377.cuh"
#include "cu_bigint.cuh"
#include "ptx.cuh"
#include "cu_common.h"



static __device__ __forceinline__  void multiply_u64(uint64_t a, uint64_t b, aleo_u128_t result) 
{    
    result[1] = __umul64hi(a,b);
    result[0] = b*a;
}


__device__ void cu_strncat(char *dest, const char *src, size_t n);
__device__ size_t cu_strlen(const char *src);
__device__ size_t cu_ultoa(char *s, unsigned long int n);
__device__ size_t cu_sltoa(char *s, long int n);
__device__  uint32_t checked_next_power_of_n(uint32_t base, uint32_t n);



#ifdef ENABLE_TEST
__global__ void test_fp256_to_bigint(void);
#endif 