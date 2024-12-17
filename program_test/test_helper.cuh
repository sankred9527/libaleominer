#pragma once

#include "cu_macro.h"
#include <cstdio>
#include <cuda_runtime.h>
#include "cuda/cu_aleo_globals.cuh"
#include "const_leaf_u16.h"
#include "const_inverse_u16.h"

constexpr uint64_t skip_table_size = 6ULL*1024ULL*1024ULL;

#define MKCODE_DEVICE(name) static __global__ void  test_kernel_##name(void *ctx)

#define MKCODE_HOST(name)\
extern "C" void testlib_##name(void)\
{\
    CUDA_SAFE_CALL(cudaSetDevice(0));\
    CUDA_SAFE_CALL(cudaDeviceReset());\
\
    uint32_t grid_size = 92 , block_size = 256;\
    uint32_t max_thread = grid_size*block_size;\
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_max_thread, (void*)&max_thread, sizeof(max_thread)));\
\
    cudaFuncSetAttribute(test_kernel_##name,\
        cudaFuncAttributeMaxDynamicSharedMemorySize, 99*1024);\
    void *dev_mem = nullptr;\
    size_t mem_size = grid_size*block_size*200ULL*1024ULL + skip_table_size; \
    CUDA_SAFE_CALL(cudaMalloc((void **)&dev_mem, mem_size));\
    CUDA_SAFE_CALL(cudaMemcpy(dev_mem, const_u16_leaf_hash, sizeof(const_u16_leaf_hash), cudaMemcpyHostToDevice));\
    CUDA_SAFE_CALL(cudaMemcpy( \
                static_cast<uint8_t*>(dev_mem) + 2*1024*1024, \
                const_u16_inverse_hash, sizeof(const_u16_inverse_hash), cudaMemcpyHostToDevice));    \
    test_kernel_##name<<<grid_size,block_size, 99*1024>>>(static_cast<uint8_t*>(dev_mem) + skip_table_size);\
    cudaDeviceSynchronize();\
}

