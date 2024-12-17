

#include <cstdio>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include "cu_macro.h"
#include "cuda/cu_bigint.cuh"
#include "cuda/circuit/integer/circuit_integer.cuh"
#include "cuda/cu_utils.cuh"

using namespace std;

#define LOOP_CNT 1

static __device__ void generate_leaf_hash(bigint_u256_t leaf)
{        
    Sha3StateV2 st[25];
    uint8_t last_bit = 0;
    memset(st, 0, sizeof(Sha3StateV2)*25);

    #define KUP(i) KeccakV2::fast_leaf_hash_update<i>(st, last_bit, leaf.uint32[i])
    ALEO_ITER_8_CODE(KUP)

    #undef KUP

    KeccakV2::fast_leaf_hash_final_state(st);

    bigint_u256_t result;
    #define MAKE_CODE_mtree_convert_with_leaf_hash(i)\
        result.uint32[i] = KeccakV2::fast_path_hash_final_by_u32<i>(st)
    
    ALEO_ITER_8_CODE(MAKE_CODE_mtree_convert_with_leaf_hash)

    #undef MAKE_CODE_mtree_convert_with_leaf_hash    

    for ( int i = 0; i < 8; i++ )
    {
        printf("UINT32_C(0x%08x), ", result.uint32[i]);
    }
    printf("\n");
}

static __global__ void gen_inverse()
{
    uint64_t field_self[QBigInt::BIGINT_LEN] = {0};
    
    nv_uint128 reg0;
    
    int loop_value = 1;

    /*
    1-65535 映射到 表里的 0-65534 item 
    -1 to -65535 映射到 65535 - 131069
    总共： 131072 个 leaf hash, 合计  131072*8 = 1048576 = 1M 个 uint32
    */

    //for ( loop_value = 1; loop_value <= 65535; loop_value++ )
    for ( loop_value = -1; loop_value >= -65535; loop_value-- )
    {
        reg0.stdint = loop_value;        
        
        QBigInt::merge_from_u128(field_self, reg0.stdint, 0);
        if (  loop_value < 0  )
        {
            field_self[2] = UINT64_MAX;
            field_self[3] = UINT64_MAX;
            QBigInt::add_with_modulus(field_self);
        }        
        
        QBigInt::bigint_to_field(field_self);        

        if ( bigint_inverse_for_quick_bigint(field_self)  )
        {
            QBigInt::bigint_from_field(field_self);
            bigint_u256_t leaf ;
            memcpy(leaf.uint64, field_self, sizeof(bigint_u256_t));

            //generate_leaf_hash(leaf);
            for ( int i = 0; i < 8; i++ )
            {
                printf("UINT32_C(0x%08x), ", leaf.uint32[i]);
            }
            printf("\n");
        }

    }
    
}

static __global__ void kernel_curve_old()
{
    nv_uint128 src = { .stdint = 0x1234567887654321 };
    bigint_u256_t dest;
    bigint_u256_t tmp;

    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    src.stdint += idx;

    Circuit_Integer::to_field(src, tmp);

    for ( int n = 0; n < LOOP_CNT; n++)    
    {
        dest = tmp;
        bigint_inverse_old(dest);
        //bigint_inverse_old(dest);
    }
    // if ( idx == 0 )
    //     bigint256_dump(&dest);   
}

static void __device__ print_str(bigint_u256_t data)
{
    printf("data= 0x");
    printf("%016lx",data.uint64[3]);
    printf("%016lx",data.uint64[2]);
    printf("%016lx",data.uint64[1]);
    printf("%016lx",data.uint64[0]);
    printf("\n");
}

static __global__ void kernel_curve_new(uint32_t seed)
{
    nv_uint128 src = { .stdint = 0x1234567887654321 };
    bigint_u256_t tmp;    

    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    src.stdint = 6;

    nv_uint128 dest;
    dest.stdint = 2;

    bigint256_dump_with_prefix("moduls=","", &BLS12_377_FR_PARAMETER_MODULUS);
    
    if (1)
    {
        bigint_u256_t p1 = { .uint128 = { 12, 0 }};
        bigint_u256_t p2 = { .uint128 = { 5, 0 }};
        bigint_to_field(p1);
        bigint_to_field(p2);

        CirCuit_Fields_Base::field_neg(p2);
        CirCuit_Fields_Base::field_add_assign(p1, p2);
        

        bigint_inverse(p1);
        bigint256_dump(&p1);        
        

        // bigint_mont_mul_assign(t3, p1, t3);
        // bigint256_dump_with_prefix("t3=","", &t3);
    }

    if (1)
    {
        bigint_u256_t p1 = { .uint128 = { 12, 0 }};
        bigint_u256_t p2 = { .uint128 = { 5, 0 }};

        bool less_than;
        less_than = ( p1.uint128[0] < p2.uint128[0] );

        p1.uint128[0] -= p2.uint128[0];        
        if ( less_than ) {            
            p1.uint128[1] = (__uint128_t)-1;
            bigint_add_with_modulus(p1, p1);
        }
        
        bigint_to_field(p1);
        bigint_inverse(p1);
        bigint256_dump(&p1);                

        // bigint_mont_mul_assign(t3, p1, t3);
        // bigint256_dump_with_prefix("t3=","", &t3);
    }

   
    
}

extern "C" void test_inverse()
{
    gen_inverse<<<1,1>>>();
    cudaDeviceSynchronize();
}

extern "C" void test_curve()
{
    std::chrono::_V2::system_clock::time_point start;
    std::chrono::_V2::system_clock::time_point end;
    std::chrono::microseconds duration;
    
    uint32_t grid_size = 46*2;
#if 0
    for ( int n = 0; n < 100; n++)
    {
        start = std::chrono::high_resolution_clock::now();
        kernel_curve_old<<<grid_size,256>>>();
        //kernel_curve_old<<<1,1>>>();
        cudaDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Execution time: " << duration.count() / 1000.0 << " microseconds" << std::endl;    

    }

#else
    for ( int n = 0; n < 1; n++)
    {
        start = std::chrono::high_resolution_clock::now();
        //kernel_curve_new<<<grid_size,256>>>(n*100000);
        kernel_curve_new<<<1,1>>>(0);
        cudaDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Execution time: " << duration.count() / 1000.0 << " microseconds" << std::endl;    

    }
#endif 

}