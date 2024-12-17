#include <cstdio>
#include <cuda_runtime.h>
#include "cu_macro.h"
#include "cuda/cu_aleo_globals.cuh"
#include "cuda/cu_synthesis.cuh"
#include "cuda/cu_smem.cuh"
#include "cuda/cu_bigint.cuh"
#include "cuda/cu_keccak_v2.cuh"
#include "cuda/cu_mtree_define.cuh"

static __global__ void mykernel()
{
    Sha3StateV2 st[25];
    uint8_t last_bit = 0;
    memset(st, 0, sizeof(Sha3StateV2)*25);

    bigint_u256_t leaf = {
        .uint64 = { 0xcd82ec9c72b5b8ca, 0x6f2ab0fc9e582271, 0x1d23e09abe0841bc, 0xbb6158afb80d3039 }
    };

    #define KUP(i) KeccakV2::fast_leaf_hash_update<i>(st, last_bit, leaf.uint32[i])
    ALEO_ITER_8_CODE(KUP)

    #undef KUP

    KeccakV2::fast_leaf_hash_final_state(st);

    bigint_u256_t result;
    #define MAKE_CODE_mtree_convert_with_leaf_hash(i)\
        result.uint32[i] = KeccakV2::fast_path_hash_final_by_u32<i>(st)
    
    ALEO_ITER_8_CODE(MAKE_CODE_mtree_convert_with_leaf_hash)

    #undef MAKE_CODE_mtree_convert_with_leaf_hash

    bigint256_dump_with_prefix("leaf hash=","", &result);
}

static __global__ void mykernel_test_path_hash()
{
    Sha3StateV2 st[25];
    uint8_t last_bit = 1;
    memset(st, 0, sizeof(Sha3StateV2)*25);

    bigint_u256_t leaf = {
        .uint64 = { 0xcd82ec9c72b5b8ca, 0x6f2ab0fc9e582271, 0x1d23e09abe0841bc, 0xbb6158afb80d3039 }
    };
    

    #define KUP(r, i) path_hash_update_state_with_u32<r,i>(st, last_bit, leaf.uint32[i])

    ALEO_ITER_8_CODE_WITH_2_PARAMS(KUP, 0)
    ALEO_ITER_8_CODE_WITH_2_PARAMS(KUP, 1)
    ALEO_ITER_8_CODE_WITH_2_PARAMS(KUP, 2)
    ALEO_ITER_8_CODE_WITH_2_PARAMS(KUP, 3)
    ALEO_ITER_8_CODE_WITH_2_PARAMS(KUP, 4)
    ALEO_ITER_8_CODE_WITH_2_PARAMS(KUP, 5)
    ALEO_ITER_8_CODE_WITH_2_PARAMS(KUP, 6)
    ALEO_ITER_8_CODE_WITH_2_PARAMS(KUP, 7)

    #undef KUP

    KeccakV2::fast_path_hash_final_state(st, last_bit);

    bigint_u256_t result;
    #define MAKE_CODE_mtree_convert_with_leaf_hash(i)\
        result.uint32[i] = KeccakV2::fast_path_hash_final_by_u32<i>(st)
    
    ALEO_ITER_8_CODE(MAKE_CODE_mtree_convert_with_leaf_hash)

    #undef MAKE_CODE_mtree_convert_with_leaf_hash

    bigint256_dump_with_prefix("path hash=","", &result);
}

static __global__ void generate_leaf_hash_for_u8()
{
    bigint_u256_t leaf = { .uint8 = {0} };

    for(int n = 0; n <= (UINT16_MAX); n++)
    {
        Sha3StateV2 st[25];
        uint8_t last_bit = 0;
        memset(st, 0, sizeof(Sha3StateV2)*25);

        leaf.uint16[0] = n;

        #define KUP(i) KeccakV2::fast_leaf_hash_update<i>(st, last_bit, leaf.uint32[i])
        ALEO_ITER_8_CODE(KUP)

        #undef KUP

        KeccakV2::fast_leaf_hash_final_state(st);

        bigint_u256_t result;
        #define MAKE_CODE_mtree_convert_with_leaf_hash(i)\
            result.uint32[i] = KeccakV2::fast_path_hash_final_by_u32<i>(st)
        
        ALEO_ITER_8_CODE(MAKE_CODE_mtree_convert_with_leaf_hash)

        #undef MAKE_CODE_mtree_convert_with_leaf_hash

        //bigint256_dump(&result);

        for ( int i = 0; i < 8; i++ )
        {
            printf("UINT32_C(0x%08x), ", result.uint32[i]);
        }
        printf("\n");

        bigint_u256_t load;
        for (int i = 0; i < 8; i++)
        {
            load.uint32[i] = CONST_leaf_hash_for_u8[n*8 + i];
        }

        // int c = bigint_compare(load, result);
        // if ( c != 0 )
        //     printf("wrong\n");
    }

}


extern "C" void test_generate_leaf_hash_for_u8()
{
    generate_leaf_hash_for_u8<<<1,1>>>();
    cudaDeviceSynchronize();
}


extern "C" void test_keccak_entry_1()
{
    mykernel<<<1,1>>>();
    cudaDeviceSynchronize();
}


extern "C" void test_keccak_path_hash()
{
    mykernel_test_path_hash<<<1,1>>>();
    cudaDeviceSynchronize();
}

