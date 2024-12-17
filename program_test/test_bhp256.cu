
#include "test_helper.cuh"

#include "cuda/circuit/hash/bhp256.cuh"


MKCODE_DEVICE(hash_bhp256)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t r0_bits_cache = 0;
    uint32_t r0_bits_cache_len = 0;    

    uint32_t gl_bits = 64;
    bool gl_is_signed = 1;
        
    if ( idx != 0 ) return;

    nv_uint128 reg0 = { .stdint = 7714464406177627373ULL };        


    HashBhp256::exec_hash(SVM_PARAM, reg0, gl_bits, gl_is_signed);
    
    while ( 1  )
    {
        RoundZeroMem::set_max_allow_full_leaves_in_smem(7);

        bool should_fold = HashBhp256::exec_hash_iter(SVM_PARAM);
        if ( should_fold ) 
        {
            if ( !HashBhp256::exec_hash_fold(SVM_PARAM) )
            {
                break;
            }

        } 
    }
    bigint_u256_t digest_x;
    smem_load_circuit_params(0, digest_x);
    bigint256_dump_with_prefix("digest_x = ", "", &digest_x);
        
    HashBhp256::exec_hash_cast_lossy(SVM_PARAM);

    // uint32_t meta;
    // meta = smem_get_circuit_meta();            
    // uint32_t preimage_bits_len = (meta >> 16)  & 0x7FFF;    
    // bool is_signed = (meta>>31) & 1;
    // uint32_t bits = preimage_bits_len - 64 - 26; // hack

    // nv_uint128 dest;
    // {
    //     bigint_u256_t tmp;
    //     smem_load_circuit_params(0, tmp);
    //     dest.stdint = tmp.uint128[0];
    // }    
    

    // uint64_t tmp[QBigInt::BIGINT_LEN];
    // smem_load_circuit_params(0, tmp);
    // QBigInt::dump("s0=", tmp);
    // smem_load_circuit_params(1, tmp);
    // QBigInt::dump("s1=", tmp);
    // smem_load_circuit_params(2, tmp);
    // QBigInt::dump("s2=", tmp);
}


MKCODE_HOST(hash_bhp256)