#pragma once
#include "cu_synthesis.cuh"
#include "cuda/cu_smem.cuh"
#include "circuit/circuit_type.cuh"
#include <type_traits>

static ALEO_ADI void svm_push_register_stack(SVM_PARAM_DEF, uint32_t value)
{
    uint32_t rindex = svm_get_reg_stack_index(SVM_PARAM);
    svm_set_reg_stack_by_offset(SVM_PARAM, rindex, value);
    svm_set_reg_stack_index(SVM_PARAM, rindex + 1);
}

static ALEO_ADI void svm_push_register_simple(SVM_PARAM_DEF, CircuitTypes ct, uint32_t value)
{
    uint32_t rindex = svm_get_reg_index(SVM_PARAM);
    svm_set_regs_by_offset(SVM_PARAM, rindex, ((uint32_t)ct) | (value<<16));
    svm_set_reg_index(SVM_PARAM, rindex + 1);
}


template <typename T>
static ALEO_ADI void svm_push_reg_implement_new(SVM_PARAM_DEF, T raw);

template<>
ALEO_ADI void svm_push_reg_implement_new<bool>(SVM_PARAM_DEF, bool v) {
    uint32_t raw = v;
    svm_push_register_simple(SVM_PARAM, CT_BOOLEAN, raw);
}

template<>
ALEO_ADI void svm_push_reg_implement_new<uint64_t *>(SVM_PARAM_DEF, uint64_t *v) {
    uint32_t current_stack_index =  svm_get_reg_stack_index(SVM_PARAM);
    svm_push_register_simple(SVM_PARAM, CT_FIELD, current_stack_index);
    svm_push_register_stack(SVM_PARAM, v[0]);
    svm_push_register_stack(SVM_PARAM, v[0]>>32);
    svm_push_register_stack(SVM_PARAM, v[1]);
    svm_push_register_stack(SVM_PARAM, v[1]>>32);
    
    svm_push_register_stack(SVM_PARAM, v[2]);
    svm_push_register_stack(SVM_PARAM, v[2]>>32);
    svm_push_register_stack(SVM_PARAM, v[3]);
    svm_push_register_stack(SVM_PARAM, v[3]>>32);
}


#define svm_push_reg_implement_new_for_qbigint(v) do {\
    uint32_t current_stack_index =  svm_get_reg_stack_index(SVM_PARAM);\
    svm_push_register_simple(SVM_PARAM, CT_FIELD, current_stack_index);\
    svm_set_reg_stack_by_offset(SVM_PARAM, current_stack_index++, v[0]);\
    svm_set_reg_stack_by_offset(SVM_PARAM, current_stack_index++, v[0]>>32);\
    svm_set_reg_stack_by_offset(SVM_PARAM, current_stack_index++, v[1]);\
    svm_set_reg_stack_by_offset(SVM_PARAM, current_stack_index++, v[1]>>32);\
    svm_set_reg_stack_by_offset(SVM_PARAM, current_stack_index++, v[2]);\
    svm_set_reg_stack_by_offset(SVM_PARAM, current_stack_index++, v[2]>>32);\
    svm_set_reg_stack_by_offset(SVM_PARAM, current_stack_index++, v[3]);\
    svm_set_reg_stack_by_offset(SVM_PARAM, current_stack_index++, v[3]>>32);\
    svm_set_reg_stack_index(SVM_PARAM, current_stack_index);\
} while(0)

template<>
ALEO_ADI void svm_push_reg_implement_new<char1>(SVM_PARAM_DEF, char1 v) {
    uint32_t t = 0; 
    t =  (uint8_t)(v.x) ;
    svm_push_register_simple(SVM_PARAM, CT_I8, t);
}

template<>
ALEO_ADI void svm_push_reg_implement_new<uchar1>(SVM_PARAM_DEF, uchar1 v) {    
    svm_push_register_simple(SVM_PARAM, CT_U8, v.x);
}

template<>
ALEO_ADI void svm_push_reg_implement_new<short1>(SVM_PARAM_DEF, short1 v) {
    uint32_t t = 0; 
    t =  (uint16_t)(v.x) ;
    svm_push_register_simple(SVM_PARAM, CT_I16, t);
}

template<>
ALEO_ADI void svm_push_reg_implement_new<ushort1>(SVM_PARAM_DEF, ushort1 v) {    
    svm_push_register_simple(SVM_PARAM, CT_U16, v.x);
}

template<>
ALEO_ADI void svm_push_reg_implement_new<uint1>(SVM_PARAM_DEF,uint1 v) {
    uint32_t current_stack_index = svm_get_reg_stack_index(SVM_PARAM);
    svm_push_register_simple(SVM_PARAM, CT_U32, current_stack_index);
    svm_push_register_stack(SVM_PARAM, v.x);    
}
template<>
ALEO_ADI void svm_push_reg_implement_new<int1>(SVM_PARAM_DEF,int1 v) {
    uint32_t current_stack_index = svm_get_reg_stack_index(SVM_PARAM);
    svm_push_register_simple(SVM_PARAM, CT_I32, current_stack_index);
    svm_push_register_stack(SVM_PARAM, v.x);
}

template<>
ALEO_ADI void svm_push_reg_implement_new<uint2>(SVM_PARAM_DEF,uint2 v) {    
    uint32_t current_stack_index = svm_get_reg_stack_index(SVM_PARAM);
    svm_push_register_simple(SVM_PARAM, CT_U64, current_stack_index);
    svm_push_register_stack(SVM_PARAM, v.x);
    svm_push_register_stack(SVM_PARAM, v.y);
}
template<>
ALEO_ADI void svm_push_reg_implement_new<int2>(SVM_PARAM_DEF,int2 v) {
    uint32_t current_stack_index = svm_get_reg_stack_index(SVM_PARAM);
    svm_push_register_simple(SVM_PARAM, CT_I64, current_stack_index);
    svm_push_register_stack(SVM_PARAM, v.x);
    svm_push_register_stack(SVM_PARAM, v.y);
}
template<>
ALEO_ADI void svm_push_reg_implement_new<int4>(SVM_PARAM_DEF,int4 v) {
    uint32_t current_stack_index = svm_get_reg_stack_index(SVM_PARAM);
    svm_push_register_simple(SVM_PARAM, CT_I128, current_stack_index);
    svm_push_register_stack(SVM_PARAM, v.x);
    svm_push_register_stack(SVM_PARAM, v.y);
    svm_push_register_stack(SVM_PARAM, v.z);
    svm_push_register_stack(SVM_PARAM, v.w);
}
template<>
ALEO_ADI void svm_push_reg_implement_new<uint4>(SVM_PARAM_DEF, uint4 v) {
    uint32_t current_stack_index = svm_get_reg_stack_index(SVM_PARAM);
    svm_push_register_simple(SVM_PARAM, CT_U128, current_stack_index);
    svm_push_register_stack(SVM_PARAM, v.x);
    svm_push_register_stack(SVM_PARAM, v.y);
    svm_push_register_stack(SVM_PARAM, v.z);
    svm_push_register_stack(SVM_PARAM, v.w);
}

template<>
ALEO_ADI void svm_push_reg_implement_new<bigint_u256_t>(SVM_PARAM_DEF, bigint_u256_t v) {
    uint32_t current_stack_index =  svm_get_reg_stack_index(SVM_PARAM);
    svm_push_register_simple(SVM_PARAM, CT_FIELD, current_stack_index);
    svm_push_register_stack(SVM_PARAM, v.uint4s[0].x);
    svm_push_register_stack(SVM_PARAM, v.uint4s[0].y);
    svm_push_register_stack(SVM_PARAM, v.uint4s[0].z);
    svm_push_register_stack(SVM_PARAM, v.uint4s[0].w);
    
    svm_push_register_stack(SVM_PARAM, v.uint4s[1].x);
    svm_push_register_stack(SVM_PARAM, v.uint4s[1].y);
    svm_push_register_stack(SVM_PARAM, v.uint4s[1].z);
    svm_push_register_stack(SVM_PARAM, v.uint4s[1].w);
    
}



static ALEO_ADI void svm_load_register_raw(SVM_PARAM_DEF, uint32_t reg_index, CircuitTypes &ct, uint32_t &value)
{
    uint32_t raw_value = svm_get_regs_by_offset(SVM_PARAM, reg_index);
    ct = (CircuitTypes)(raw_value & 0xFFFF);
    value = (uint32_t)(raw_value >> 16);
}



static ALEO_ADI bool svm_load_reg_implement_for_bool(SVM_PARAM_DEF,  uint32_t reg_raw_value)
{
    bool ret;
    ret = (bool)reg_raw_value;
    return ret;
}

static ALEO_ADI nv_uint128 svm_load_reg_implement_for_int(SVM_PARAM_DEF,  uint32_t reg_raw_value, uint32_t bits)
{
    nv_uint128 ret;
    ret.stdint = 0;
    if ( bits <= 16 ) {
        ret.nvint.x = reg_raw_value;
        ret.nvint.y = 0;
        ret.nvint.z = 0;
        ret.nvint.w = 0;
    }
    else if ( bits == 32 )
    {
        ret.nvint.x = svm_get_reg_stack_by_offset(SVM_PARAM, reg_raw_value);
        ret.nvint.y = 0;
        ret.nvint.z = 0;
        ret.nvint.w = 0;
    } else if ( bits == 64 )
    {
        ret.nvint.x = svm_get_reg_stack_by_offset(SVM_PARAM, reg_raw_value);
        ret.nvint.y = svm_get_reg_stack_by_offset(SVM_PARAM, reg_raw_value+1);
        ret.nvint.z = 0;
        ret.nvint.w = 0;
    }
    else if ( bits == 128 )
    {
        ret.nvint.x = svm_get_reg_stack_by_offset(SVM_PARAM, reg_raw_value);
        ret.nvint.y = svm_get_reg_stack_by_offset(SVM_PARAM, reg_raw_value+1);
        ret.nvint.z = svm_get_reg_stack_by_offset(SVM_PARAM, reg_raw_value+2);
        ret.nvint.w = svm_get_reg_stack_by_offset(SVM_PARAM, reg_raw_value+3);
    }
    return ret;
}


static ALEO_ADI bigint_u256_t svm_load_reg_implement_for_bigint256 (SVM_PARAM_DEF,  uint32_t reg_raw_value)
{
    bigint_u256_t ret;
    #pragma unroll
    for (int n = 0;n<8;n++)
    {
        BIGINT256_GET_U32(&ret, n) = svm_get_reg_stack_by_offset(SVM_PARAM, reg_raw_value + n);
    }
    return ret;
}

static ALEO_ADI void svm_load_reg_implement_for_qbigint (SVM_PARAM_DEF, uint64_t dest[QBigInt::BIGINT_LEN],  uint32_t reg_raw_value)
{
    
    #pragma unroll
    for (int n = 0;n<8;n++)
    {
        uint32_t tmp = svm_get_reg_stack_by_offset(SVM_PARAM, reg_raw_value + n);    
        QBigInt::set_by_u32(dest, tmp, n);
    }
    
}


template <typename T>
static ALEO_ADI T svm_load_reg_implement(SVM_PARAM_DEF,  uint32_t reg_raw_value, uint32_t bits = 0)
{
     if constexpr (std::is_same_v<T, bool> ) {
        return svm_load_reg_implement_for_bool(SVM_PARAM, reg_raw_value);
     }

     if constexpr  (std::is_same_v<T, bigint_u256_t> ) {
        return svm_load_reg_implement_for_bigint256(SVM_PARAM, reg_raw_value);
     }

     if constexpr  (std::is_same_v<T, nv_uint128> ) {
        return svm_load_reg_implement_for_int(SVM_PARAM, reg_raw_value, bits);
     }
}