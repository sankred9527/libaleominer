
#define DUMP_LEAF 1
#include "test_helper.cuh"

#include <cstdio>
#include <cuda_runtime.h>
#include "cu_macro.h"
#include "cuda/cu_aleo_globals.cuh"
#include "cuda/cu_synthesis.cuh"
#include "cuda/cu_smem.cuh"
#include "cuda/cu_bigint.cuh"
#include "cuda/circuit/hash/ped64.cuh"




MKCODE_DEVICE(hash_ped64)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t r0_bits_cache = 0;
    uint32_t r0_bits_cache_len = 0;
    uint32_t gl_bits = 8;
    bool is_signed = 1;
    
    nv_uint128 src;


    if ( idx != 0 ) return;

    svm_set_program_index(SVM_PARAM, 0);    
    svm_set_reg_index(SVM_PARAM, 0);
    svm_set_reg_stack_index(SVM_PARAM, 0);
    svm_set_hash_psd2_count(SVM_PARAM, 0);
    svm_set_soa_leaves_length(SVM_PARAM, 0);
            
    smem_set_soa_meta(0);    
    smem_set_circuit_meta2(0);
    smem_set_circuit_meta(0);
    smem_set_circuit_is_u8_leaf_hash(0); 

    /*
    hash.ped64 r6 into r14 as i32;
    */
    src.v64[0] = 0x51dc9f59;
    src.v64[1] = 0;

    
    HashPed64::prepare(SVM_PARAM, src, gl_bits, is_signed);

    uint32_t soa_leaves_index;
    bool should_finish;
    do {
        RoundZeroMem::set_max_allow_full_leaves_in_smem(7);
        should_finish = HashPed64::iter(SVM_PARAM);
        smem_set_circuit_is_u8_leaf_hash(0);
        smem_set_soa_leaves_len(0, 0);
    } while(!should_finish);

    printf("done\n");
}


MKCODE_HOST(hash_ped64)
