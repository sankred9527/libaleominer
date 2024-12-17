#include <cuda_runtime.h>
#include <functional>
#include <sstream>  
//#include "cu_merkle_tree.cuh"
#include "cu_aleo_interface.h"
#include "CudaAleoAlgo.cuh"
#include "cu_macro.h"
#include "cu_synthesis.cuh"



void cu_aleo_search(int grid_size, int block_size, cudaStream_t stream,
            uint8_t *leaves_memory,             
            uint64_t start_counter, volatile Search_results *result)
{
    //only support CUDA ARCH=86,89
    cudaFuncSetAttribute(cu_aleo_kernel_search,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 99*1024);

    //skip the u16 leaf hash and  u16 inverse table 
    leaves_memory += 6*1024*1024;

    cu_aleo_kernel_search<<<grid_size,block_size, 99*1024, stream>>>(leaves_memory, start_counter, result);    
    
}

