#pragma once
#include <cstdio>
#include <cstdint>
#include "cu_common.h"
#include "cu_aleo_globals.cuh"



#define MAX_LEAVES_COMPRESS (3072)

/* 
    目前的epoch program 似乎51200个 leaf足够了, ，每个leaf 2bit

    需要多少个 uint32_t 才能存放全部的 leaves 代号
*/
#define MAX_FULL_LEAVES (204800) 
//#define MAX_FULL_LEAVES_IN_U32 ((MAX_FULL_LEAVES*2/8)/4)
static constexpr unsigned int MAX_FULL_LEAVES_IN_U32 = (MAX_FULL_LEAVES*2/8)/4;  //3600 

#define MAX_REG_STACK_SIZE (512)
#define MAX_REG_SIZE (150)

typedef union {
    uint8_t bytes[4];
    uint1 nv_int;
} union_nv_int_u8;

enum SOA_LEAF_SAVE_TYPE {
    LEAF_ZERO = 0,
    LEAF_ONE = 1,
    LEAF_HASH_PSD2 = 2, // 0b10
    LEAF_FULL = 3,     // 0b11 

    LEAF_UNKNOWN,
};

#define SOA_LEAF_TYPE_MASK (0b11)

typedef struct { 
    uint32_t soa_leavs_stack[MAX_LEAVES_COMPRESS][8];
    uint32_t soa_leaves_length;

    /*
     参考： SOA_LEAF_SAVE_TYPE
     full_leaves： 2bit 存放 vector的值： 00 是全零的 vector , 01 是最低位为1的vector , 0b11代表需要全量存储的 vector
                    0b10 代表 hash_psd2 的leaf
    */  
    uint32_t full_leaves[MAX_FULL_LEAVES_IN_U32];

    uint32_t hash_psd2_count;
    uint32_t reg_stack[MAX_REG_STACK_SIZE];
    uint32_t reg_stack_index;

    /*
     低16bit 存放 寄存器类型 
     高16bit 存放 寄存器的"值"(如果 <=16bit 直接存放，否则存放 reg stack 里的 index)
     对于 boolean , u8, u16 类型来说 16bit 足够存放
    */
    uint32_t regs[MAX_REG_SIZE];
    uint32_t reg_index;
    uint32_t program_index;
    
} gl_thread_data_wrap;



typedef union {
    uint8_t stdint;
    uchar1 nvint;
    char1 nvint_signed;
} nv_uint8;


typedef union {
    __uint128_t stdint;
    uint4 nvint;
    int4 nvint_signed;
    uint64_t v64[2];
    uint32_t v32[4];
    uint8_t v8[16];
} nv_uint128;


#define SVM_PARAM_DEF void *ctx,  unsigned int idx, uint32_t &r0_bits_cache, uint32_t &r0_bits_cache_len
                           
//是否有const 常量
#define SVM_PARAM_CDEF  SVM_PARAM_DEF,  uint8_t is_const

#define SVM_PARAM  ctx, idx, r0_bits_cache, r0_bits_cache_len
#define SVM_PARAM_C  SVM_PARAM, is_const

#define SVM_PARAM_NOT_NULL  (ctx != nullptr)


#define SVM_MAKE_CODE_BY_NAME(name)\
static ALEO_ADI uint32_t svm_get_##name(SVM_PARAM_DEF)\
{\
    constexpr gl_thread_data_wrap *ptdata = NULL; \
    constexpr uint32_t offset_in_u32 = ((uint64_t)( &( ptdata->MAKE_CODE_CONTEXT_CONCAT(name,) ) )) / sizeof(uint32_t);\
    uint32_t offset_of_name;  \
    offset_of_name = offset_in_u32*d_max_thread;\
    uint32_t *p = ((uint32_t*)(ctx)) + offset_of_name;\
    return p[ idx ];\
}\
static ALEO_ADI void svm_set_##name(SVM_PARAM_DEF,uint32_t data)\
{\
    constexpr gl_thread_data_wrap *ptdata = NULL; \
    constexpr uint32_t offset_in_u32 = ((uint64_t)( &( ptdata->MAKE_CODE_CONTEXT_CONCAT(name,) ) )) / sizeof(uint32_t);\
    uint32_t offset_of_name;    \
    offset_of_name = offset_in_u32*d_max_thread;\
    uint32_t *p = ((uint32_t*)(ctx)) + offset_of_name;\
    p[ idx ] = data;\
}

SVM_MAKE_CODE_BY_NAME(program_index)
SVM_MAKE_CODE_BY_NAME(reg_index)
SVM_MAKE_CODE_BY_NAME(reg_stack_index)
SVM_MAKE_CODE_BY_NAME(hash_psd2_count)
SVM_MAKE_CODE_BY_NAME(soa_leaves_length)


static ALEO_ADI uint32_t svm_get_full_leaves_by_offset(SVM_PARAM_DEF, uint32_t offset)
{
    constexpr gl_thread_data_wrap *ptdata = NULL; 
    uint32_t offset_in_u32 = ((uint64_t)( &(ptdata->full_leaves[offset]) )) / sizeof(uint32_t);
    uint32_t offset_of_name;  
    offset_of_name = offset_in_u32*d_max_thread;
    uint32_t *p = ((uint32_t*)(ctx)) + offset_of_name;
    return p[ idx ];
}

static ALEO_ADI void svm_set_full_leaves_by_offset(SVM_PARAM_DEF, uint32_t offset, uint32_t value)
{
    constexpr gl_thread_data_wrap *ptdata = NULL; 
    uint32_t offset_in_u32 = ((uint64_t)( &(ptdata->full_leaves[offset]) )) / sizeof(uint32_t);
    uint32_t offset_of_name;  
    offset_of_name = offset_in_u32*d_max_thread;
    uint32_t *p = ((uint32_t*)(ctx)) + offset_of_name;
    p[ idx ] = value;
}

static ALEO_ADI uint32_t svm_get_regs_by_offset(SVM_PARAM_DEF, uint32_t offset)
{
    constexpr gl_thread_data_wrap *ptdata = NULL; 
    uint32_t offset_in_u32 = ((uint64_t)( &(ptdata->regs[offset]) )) / sizeof(uint32_t);
    uint32_t offset_of_name;  
    offset_of_name = offset_in_u32*d_max_thread;
    uint32_t *p = ((uint32_t*)(ctx)) + offset_of_name;
    return p[ idx ];
}

static ALEO_ADI void svm_set_regs_by_offset(SVM_PARAM_DEF, uint32_t offset, uint32_t value)
{
    constexpr gl_thread_data_wrap *ptdata = NULL; 
    uint32_t offset_in_u32 = ((uint64_t)( &(ptdata->regs[offset]) )) / sizeof(uint32_t);
    uint32_t offset_of_name;  
    offset_of_name = offset_in_u32*d_max_thread;
    uint32_t *p = ((uint32_t*)(ctx)) + offset_of_name;
    p[ idx ] = value;
}


static ALEO_ADI uint32_t svm_get_reg_stack_by_offset(SVM_PARAM_DEF, uint32_t offset)
{
    constexpr gl_thread_data_wrap *ptdata = NULL; 
    uint32_t offset_in_u32 = ((uint64_t)( &(ptdata->reg_stack[offset]) )) / sizeof(uint32_t);
    uint32_t offset_of_name;  
    offset_of_name = offset_in_u32*d_max_thread;
    uint32_t *p = ((uint32_t*)(ctx)) + offset_of_name;
    return p[ idx ];
}

static ALEO_ADI void svm_set_reg_stack_by_offset(SVM_PARAM_DEF, uint32_t offset, uint32_t value)
{
    constexpr gl_thread_data_wrap *ptdata = NULL; 
    uint32_t offset_in_u32 = ((uint64_t)( &(ptdata->reg_stack[offset]) )) / sizeof(uint32_t);
    uint32_t offset_of_name;  
    offset_of_name = offset_in_u32*d_max_thread;
    uint32_t *p = ((uint32_t*)(ctx)) + offset_of_name;
    p[ idx ] = value;
}

