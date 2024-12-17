#define DUMP_LEAF 1
#include "cu_macro.h"
#include <cstdio>
#include <cuda_runtime.h>
#include "cuda/circuit/field/circuit_field.cuh"
#include "cuda/cu_aleo_globals.cuh"

static ALEO_ADI void debug_show_mtree_leaves(SVM_PARAM_DEF, uint32_t &soa_leaves_index)
{
    uint32_t max_full_leaves;
    uint32_t len_in_smem;
    uint32_t len_in_global_mem;
    RoundZeroMem::get_leaf_len_in_share_global_memory(max_full_leaves, len_in_smem, len_in_global_mem);

    uint32_t total_len = len_in_smem + len_in_global_mem;
    for (int offset = 0; offset < total_len; offset++)
    {
        uint32_t v32_bit_array;
        if ( offset < len_in_smem ) 
        {                
            v32_bit_array = gl_static_shared_memory[ (SMEM_CACHE_SIZE_IN_U32 - 1 - offset) * MY_BLOCK_SIZE + (idx & MY_BLOCK_MASK) ];
        } else {
            v32_bit_array = svm_get_full_leaves_by_offset(SVM_PARAM, offset - len_in_smem);
        }
                
        for ( int i = 0; i < 16; i++, v32_bit_array = v32_bit_array>>2)
        {
            int v = v32_bit_array & 0b11;
            if ( v <= 1 ) 
            {
                hash_bhp256_debug_dump_leaf((bool)v);
            } else {
                bigint_u256_t tmp;
                for (int k = 0; k<8; k++)
                   tmp.uint32[k] = smem_get_soa_leaves_by_offset(soa_leaves_index, k);
                hash_bhp256_debug_dump_leaf(tmp);
                soa_leaves_index++;
            }
        }
    }
}

static ALEO_ADI void debug_cache_leaves(SVM_PARAM_DEF, uint32_t &soa_leaves_index)
{    
    for ( int n = 0; n < r0_bits_cache_len; n++)
    {        
        uint32_t leaf = ( (r0_bits_cache>>(2*n)) & 0b11);
        if ( leaf <= 1 )
            hash_bhp256_debug_dump_leaf(leaf);
        else {
            //full leaf            
            if ( soa_leaves_index == UINT32_MAX )
                continue;                        
            bigint_u256_t tmp;            
            for (int i = 0; i < 8; i++)
                tmp.uint32[i] = smem_get_soa_leaves_by_offset(soa_leaves_index, i);  
            hash_bhp256_debug_dump_leaf(tmp);
            soa_leaves_index++;
        }
    }
    r0_bits_cache_len = 0;
    r0_bits_cache = 0;
        
}

static void __global__ mykernel_test_field_to_bigint_with_witeness(void *ctx)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t r0_bits_cache = 0;
    uint32_t r0_bits_cache_len = 0;
    bool is_const = false;
    if ( idx != 0 ) 
        return;
    

    svm_set_program_index(SVM_PARAM, 0);    
    svm_set_reg_index(SVM_PARAM, 0);
    svm_set_reg_stack_index(SVM_PARAM, 0);
    svm_set_hash_psd2_count(SVM_PARAM, 0);
    svm_set_soa_leaves_length(SVM_PARAM, 0);
    
    smem_set_soa_meta(0);    
    smem_set_circuit_meta2(0);    
    smem_set_circuit_meta(0);    
    smem_set_circuit_is_u8_leaf_hash(0);      

    // bigint_u256_t tmp = { 
    //     .uint64 = { 1,2,3,4}
    // };    

    uint64_t tmp[4] = { 
         1,2,3,4
    };    
    
    //CirCuit_Fields::field_to_bigint_with_witness(SVM_PARAM, 0, &tmp, &tmp);        

    //CirCuit_Fields::field_to_bigint_with_witness(SVM_PARAM, 0, tmp.uint64, tmp.uint64);

    nv_uint128 dest;
    CirCuit_Fields::operator_cast_lossy(SVM_PARAM_C, tmp, dest);    

    dest.stdint += 0x1234;
    dest.stdint *= 0x4567;

    printf("dest=%lx\n", (uint64_t)dest.stdint);
    
    uint32_t soa_leaves_index = 0;
    uint32_t read_cursor, soa_full_leaves_total_len;
    smem_get_soa_leaves_len(read_cursor , soa_full_leaves_total_len);    
    // debug_show_mtree_leaves(SVM_PARAM, soa_leaves_index);
    // debug_cache_leaves(SVM_PARAM, soa_leaves_index);   
}

extern "C" void testlib_field_to_bigint_with_witeness()
{
    CUDA_SAFE_CALL(cudaSetDevice(0));
    CUDA_SAFE_CALL(cudaDeviceReset());       

    uint32_t grid_size = 92 , block_size = 256;
    uint32_t max_thread = grid_size*block_size;
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_max_thread, (void*)&max_thread, sizeof(max_thread)));
    
    cudaFuncSetAttribute(mykernel_test_field_to_bigint_with_witeness,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 99*1024);

    void *dev_mem = nullptr;
    size_t mem_size = grid_size*block_size*200ULL*1024ULL; // more memory isn't bad    
    CUDA_SAFE_CALL(cudaMalloc((void **)&dev_mem, mem_size));
    mykernel_test_field_to_bigint_with_witeness<<<grid_size,block_size, 99*1024>>>(dev_mem);
    cudaDeviceSynchronize();
}