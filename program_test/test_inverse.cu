//#define DUMP_LEAF 1

#include "test_helper.cuh"
#include "cuda/circuit/integer/circuit_integer.cuh"


static ALEO_ADI void __svm_push_reg_as_variant_int(SVM_PARAM_DEF, nv_uint128 nvint , uint8_t bits, bool is_signed)
{
    switch (bits>>3)
    {
        case 1:
            if ( is_signed ) {
                char1 dest;
                dest.x = nvint.stdint & 0xff;
                svm_push_reg_implement_new(SVM_PARAM, dest);
            } else {
                uchar1 dest;
                dest.x = nvint.stdint & 0xff;
                svm_push_reg_implement_new(SVM_PARAM, dest);
            }
            break;
        case 2:
            if ( is_signed ) {
                short1 dest;
                dest.x = nvint.stdint & UINT16_MAX;
                svm_push_reg_implement_new(SVM_PARAM, dest);
            } else {
                ushort1 dest;
                dest.x = nvint.stdint & UINT16_MAX;
                svm_push_reg_implement_new(SVM_PARAM, dest);
            }
            break;
        case 4:
            if ( is_signed ) {
                int1 dest;
                dest.x = nvint.nvint.x;
                svm_push_reg_implement_new(SVM_PARAM, dest);
            } else {
                uint1 dest;
                dest.x = nvint.nvint.x;
                svm_push_reg_implement_new(SVM_PARAM, dest);
            }
            break;
        case 8:
            if ( is_signed ) {
                int2 dest;
                dest.x = nvint.nvint.x;
                dest.y = nvint.nvint.y;
                svm_push_reg_implement_new(SVM_PARAM, dest);
            } else {
                uint2 dest;
                dest.x = nvint.nvint.x;
                dest.y = nvint.nvint.y;
                svm_push_reg_implement_new(SVM_PARAM, dest);
            }
            break;
        case 16:
            if ( is_signed ) {
                svm_push_reg_implement_new(SVM_PARAM, nvint.nvint_signed);
            } else {                
                svm_push_reg_implement_new(SVM_PARAM, nvint.nvint);
            }
            break;
        default:
            break;
    }
}

static __device__ void old_inverse(void *ctx)
{    
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t r0_bits_cache = 0;
    uint32_t r0_bits_cache_len = 0;    
    uint8_t is_const = 0;
    uint32_t gl_bits = 16;
    bool gl_is_signed = 1;

    nv_uint128 reg0;
    nv_uint128 reg1;

    short1 v1, v2;
    v1.x = INT16_MAX;  //0x7fff
    v2.x = INT16_MAX+100;  //0x8000

    svm_push_reg_implement_new(SVM_PARAM, v1);
    svm_push_reg_implement_new(SVM_PARAM, v2);
    
    CircuitTypes ct; 
    uint32_t raw_value;
    svm_load_register_raw(SVM_PARAM, 0, ct, raw_value);
    reg0 = svm_load_reg_implement_for_int(SVM_PARAM, raw_value, 8);

    svm_load_register_raw(SVM_PARAM, 1, ct, raw_value);
    reg1 = svm_load_reg_implement_for_int(SVM_PARAM, raw_value, 8);    

    bool ret;
    Circuit_Integer::operator_is_not_eq(CINT_PARAM_C, reg0, reg1, ret );

    printf("old ret=%u\n", ret);
}

static __device__ void dump_v128(nv_uint128 nv)
{
    printf("dump v128, low=%016lx high=%016lx\n", (uint64_t)nv.stdint, (uint64_t)(nv.stdint>>64) );    
}

static __device__ void new_inverse(void *ctx)
{    
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t r0_bits_cache = 0;
    uint32_t r0_bits_cache_len = 0;    
    uint8_t is_const = 0;
    uint32_t gl_bits = 16;
    bool gl_is_signed = 1;

    nv_uint128 reg0;
    nv_uint128 reg1;

    short1 v1, v2;
    v1.x = INT16_MAX;  //0x7fff
    v2.x = INT16_MAX+100;  //0x8000

    svm_push_reg_implement_new(SVM_PARAM, v1);
    svm_push_reg_implement_new(SVM_PARAM, v2);
    
    CircuitTypes ct; 
    uint32_t raw_value;
    svm_load_register_raw(SVM_PARAM, 0, ct, raw_value);
    reg0 = svm_load_reg_implement_for_int(SVM_PARAM, raw_value, 8);

    svm_load_register_raw(SVM_PARAM, 1, ct, raw_value);
    reg1 = svm_load_reg_implement_for_int(SVM_PARAM, raw_value, 8);
        
    // short1 v3; v3.x = v1.x - v2.x;
    // printf("t=%d\n", v3.x);    

    bool ret;
    Circuit_Integer::operator_is_not_eq_v2(CINT_PARAM_C, reg0, reg1, ret );

    printf("new ret=%u\n", ret);
}

MKCODE_DEVICE(inverse_int)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;    

    if ( idx != 0 ) return ;

    new_inverse(ctx);
    old_inverse(ctx);
}




MKCODE_HOST(inverse_int)

