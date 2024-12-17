
#include "test_helper.cuh"
#include "cuda/circuit/integer/circuit_integer.cuh"

MKCODE_DEVICE(poww)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t r0_bits_cache = 0;
    uint32_t r0_bits_cache_len = 0;    

    uint32_t gl_bits = 16;
    bool gl_is_signed = 0;

    uint8_t is_const = 0;
        
    if ( idx != 0 ) return;    
    

    nv_uint128 reg0;
    //reg0.stdint = 0x1234567887654321;
    reg0.stdint = UINT16_MAX-1234;

    nv_uint128 reg1;
    reg1.stdint = 0x5a5a5a5a;
    uint32_t exp_bits = 32;


    Circuit_Integer::operator_pow_wrap_prepare(SVM_PARAM_C, gl_bits, gl_is_signed, reg0, reg1.stdint, exp_bits);

    bool ret = false;
    int cnt = 0;
    while ( !ret )
    {
        printf("run cnt=%d\n", cnt++);
        RoundZeroMem::set_max_allow_full_leaves_in_smem(6);
        nv_uint128 dest; 
        ret = Circuit_Integer::operator_pow_wrap_run(SVM_PARAM);
    }

    printf("done\n");
}

MKCODE_HOST(poww)

