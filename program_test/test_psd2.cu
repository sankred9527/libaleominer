
#include "test_helper.cuh"

#include "cuda/circuit/hash/poseidon.cuh"

MKCODE_DEVICE(hash_psd2)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t r0_bits_cache = 0;
    uint32_t r0_bits_cache_len = 0;    
        
    if ( idx != 0 ) return;

    uint64_t src[QBigInt::BIGINT_LEN] = { 
        0x2034174940775aadULL,
        0x1b36b501c56d3a85ULL,
        0xeec40964cfcef5a0ULL,
        0x304843389eb35830ULL
    };
    
    // uint64_t input1_reg[QBigInt::BIGINT_LEN] = { 0 };
    // uint64_t input2_reg[QBigInt::BIGINT_LEN] = { 0 };
    // circuit_plaintext_to_field(SVM_PARAM, VARIANT_Field, src, input1_reg, input2_reg);

    // QBigInt::dump("input1=", input1_reg);
    // QBigInt::dump("input2=", input2_reg);

    HashPoseidon::prepare_context(SVM_PARAM, src);

    uint64_t tmp[QBigInt::BIGINT_LEN];
    smem_load_circuit_params(0, tmp);
    QBigInt::dump("s0=", tmp);
    smem_load_circuit_params(1, tmp);
    QBigInt::dump("s1=", tmp);
    smem_load_circuit_params(2, tmp);
    QBigInt::dump("s2=", tmp);
}


MKCODE_HOST(hash_psd2)