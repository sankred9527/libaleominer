#pragma once
#include <cstdint>
#include "cu_bigint_define.cuh"
#include "cu_common.h"
#include <cstdio>
#include "cu_synthesis.cuh"
#include "cu_qbigint.cuh"


#ifdef ALEO_DEBUG
#define cudbg(...) do { printf(__VA_ARGS__); } while(0)
#define cuerr(...) do { printf(__VA_ARGS__); } while(0)
#else
#define cudbg(...) do {}while(0)
#define cuerr(...) do {}while(0)
#endif

#define MY_BLOCK_SIZE (256)
#define MY_BLOCK_MASK (0xFF)


#define SMEM_SOA_LEAVES_LEN (7)

#define SMEM_CACHE_SIZE_IN_U32 (64)

/*
ADA 和 Ampere 架构:
warp_meta_t 的 size 尽量控制在 640 字节以内,  每个sm 共 4个block， 每个block 32 thread ， 每个block允许的share memory： 24576 以内

 mtree_round_0_cache 的开销等价于每个线程 4k/32 = 128 字节 ， 640+128=768
 768*32*4 = 98304 ，刚好是一个sm 能使用的最大 share memory ( 100k - 4k)

Turing 架构:

每个 sm 64个线程 ， 2个block

*/
typedef struct {    
    uint32_t soa_leaves[SMEM_CACHE_SIZE_IN_U32]; // SMEM_SOA_LEAVES_LEN 个 bigint_u256
    uint32_t soa_meta;  // 低16bit ： 存 soa_leaves in share mem 长度 。  高 16bit: 保留
    uint32_t circuit_params[3][8];
    uint32_t circuit_meta;

    /* 
    [0-8) bit : 是否是 uint8 的 simple leaf, bool 类型
    [8-24) bit: 存 soa_cache 里  bits array 的数量
    */
    uint32_t circuit_meta2;
} warp_meta_t;


//仅仅存放 前 128个 bits array的 path hash
// extern __device__ __shared__ uint32_t mtree_round_0_cache[128*8];
// extern __device__ __shared__ uint32_t gl_static_shared_memory[ sizeof(warp_meta_t) * MY_BLOCK_SIZE / sizeof(uint32_t) ];
extern __device__ __shared__ uint32_t __gl_static_shared_memory[];

#define mtree_round_0_cache (&__gl_static_shared_memory[0])
#define gl_static_shared_memory (&__gl_static_shared_memory[8*256])


#define MAKE_CODE_SMEM_GET_OFFSET(name)\
static ALEO_ADI uint32_t *smem_get_offset_of_##name()\
{\
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x; \
    constexpr warp_meta_t *pmeta = NULL; \
    uint32_t offset_of_name;\
    offset_of_name = ((uint64_t)( &( pmeta->MAKE_CODE_CONTEXT_CONCAT(name,) ) )) / sizeof(uint32_t);\
    offset_of_name = offset_of_name*MY_BLOCK_SIZE;\
    uint32_t *p = &gl_static_shared_memory[ offset_of_name ];\
    return p;\
}

#define MAKE_CODE_SMEM_CONTEXT(name) \
MAKE_CODE_SMEM_GET_OFFSET(name)\
static ALEO_ADI uint32_t smem_get_##name(void)\
{\
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;    \
    constexpr warp_meta_t *pmeta = NULL;    \
    uint32_t *p = smem_get_offset_of_##name() ;\
    /*if (1)printf("sem get " #name  " 1, idx=%u addr=%p offset=%u v=%u\n", idx, p, offset_of_name, p[ idx & MY_BLOCK_MASK]); */\
    return p[ idx & MY_BLOCK_MASK] ; \
}\
static ALEO_ADI void smem_set_##name(uint32_t value)\
{\
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;    \
    constexpr warp_meta_t *pmeta = NULL;    \
    uint32_t *p = smem_get_offset_of_##name() ;\
    /*if ( 1 ) printf("sem set " #name  " 1, idx=%u addr=%p offset=%u value=%u\n", idx, p, offset_of_name, value); */\
    p[ idx & MY_BLOCK_MASK] = value ; \
}

//MAKE_CODE_SMEM_GET_OFFSET(r0_bis_array)

MAKE_CODE_SMEM_CONTEXT(circuit_meta)
MAKE_CODE_SMEM_CONTEXT(soa_meta)
MAKE_CODE_SMEM_CONTEXT(circuit_meta2)


/*
round0 的 内存管理模块

meta2 :
[0, 4) 共 4个bit  为 是否为 uint8 的简单 full leaf , bool 值
[4, 8) 共 4个bit ，表示 soa_leaves 里可以容纳的最大 full leaf 个数
[8, 20) 12 bit 表示 round 0 的 leaf 在 share mem 里的 长度 ， 每个 leaf 由2个 bit 表达
[20, 32) 12 bit 表示 round 0 的 leaf 在 global mem 里的 长度 ， 每个 leaf 由2个 bit 表达

12 bit的int可以表示 4096个 leaf 长度，够用了
*/
class RoundZeroMem
{
public:

    static ALEO_ADI uint32_t get_max_allow_full_leaves_in_smem()
    {
        uint32_t meta2 = smem_get_circuit_meta2();
        return (meta2 >> 4) & 0xf;
    }

    static ALEO_ADI void set_max_allow_full_leaves_in_smem(uint32_t value)
    {
        /*
        value 的取值范围： [0, 8） 暂时不做检测 
        */
        uint32_t meta2 = smem_get_circuit_meta2();
        meta2 = meta2 & 0xFFFFFF0F;
        meta2 = meta2 | (value << 4 );

        smem_set_circuit_meta2(meta2);
    }

    static ALEO_ADI void get_leaf_len_in_share_global_memory(
                uint32_t &max_full_leaves,
                uint32_t &len_in_smem, 
                uint32_t &len_in_global_mem                
    )
    {
        uint32_t meta2 = smem_get_circuit_meta2();
        meta2 = meta2 >> 4;
        max_full_leaves = meta2 & 0xf;
        meta2 = meta2 >> 4;
        len_in_smem = meta2 & 0xfff;
        meta2 = meta2 >> 12;
        len_in_global_mem = meta2;
    }

    static ALEO_ADI void set_leaf_len_in_share_global_memory(uint32_t len_in_smem, uint32_t len_in_global_mem)
    {
        uint32_t meta2 = smem_get_circuit_meta2();
        meta2 = meta2 & 0xff;
        
        meta2 = ( len_in_smem << 8 ) | meta2 ;
        meta2 = ( len_in_global_mem << 20 ) | meta2;

        smem_set_circuit_meta2(meta2);
    }

    static ALEO_ADI void save_r0_bits_cache(SVM_PARAM_DEF)
    {        
        uint32_t max_full_leaves; // 最多 7 个 bigint_u256t 
        uint32_t len_in_smem; // 对应的是 leaf 占据的 u32 个数
        uint32_t len_in_global_mem;

        get_leaf_len_in_share_global_memory(max_full_leaves, len_in_smem, len_in_global_mem);
        
        uint32_t len_in_smem_max_in_u32 = SMEM_CACHE_SIZE_IN_U32 - max_full_leaves*8;

        if ( len_in_smem >= len_in_smem_max_in_u32) {
            //save in global memory
            svm_set_full_leaves_by_offset(SVM_PARAM, len_in_global_mem, r0_bits_cache);
            len_in_global_mem++;
        } else {
            //save to share memory            
            //desc 逆序存储
            uint32_t save_offset = SMEM_CACHE_SIZE_IN_U32 - 1 - len_in_smem;
            gl_static_shared_memory[ save_offset * MY_BLOCK_SIZE + (idx & MY_BLOCK_MASK) ] = r0_bits_cache;
            len_in_smem++;
        }
        set_leaf_len_in_share_global_memory(len_in_smem, len_in_global_mem);
    }
};


/*
1 : 代表 是  pow.w 的  32,64,128, bit
2 : 代表 是  pow.w 的  16 bit
3 : 代表 是  pow.w 的  8 bit

*/
static ALEO_ADI void smem_set_circuit_is_u8_leaf_hash(uint32_t v)
{
    uint32_t meta2 = smem_get_circuit_meta2();
    meta2 = ( meta2 & 0xfffffff0 ) | v;
    smem_set_circuit_meta2(meta2);
}

static ALEO_ADI uint32_t smem_get_circuit_is_u8_leaf_hash(void)
{
    uint32_t meta2 = smem_get_circuit_meta2();
    return meta2 & 0xf;
}


static ALEO_ADI void smem_get_soa_leaves_len(uint32_t &read_index, uint32_t &len)
{
    /*
        低 16 bit 存 full leaves 的总个数
        高 16bit 存 read cursor        
    */
    uint32_t meta = smem_get_soa_meta();
    len = meta & 0xffff;
    read_index = meta >> 16;
}


static ALEO_ADI void smem_set_soa_leaves_len(uint32_t read_index, uint32_t len)
{
    uint32_t meta = smem_get_soa_meta();
    meta = (read_index << 16) | ( len & 0xffff);
    smem_set_soa_meta(meta);
}




static ALEO_ADI uint32_t smem_get_soa_leaves_by_offset(uint32_t leaf_offset, uint32_t leaf_u32_offset )
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if ( leaf_offset >= SMEM_SOA_LEAVES_LEN || leaf_u32_offset >= 8) {
        //printf("bug leaf offset\n");
        return 0;
    }
    // 一个leaf 是 8 个 u32 ， 所以 and 0b111
    uint32_t foo = (leaf_offset * 8 + (leaf_u32_offset & 0b111) )*MY_BLOCK_SIZE  ;
    return gl_static_shared_memory[ foo + ( idx & MY_BLOCK_MASK )];
}

static ALEO_ADI void smem_set_soa_leaves_by_offset(uint32_t leaf_offset, uint32_t leaf_u32_offset, uint32_t value )
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if ( leaf_offset >= SMEM_SOA_LEAVES_LEN || leaf_u32_offset >= 8) {
        //printf("bug leaf offset\n");
        return;
    }
    // 一个leaf 是 8 个 u32 ， 所以 and 0b111
    uint32_t foo = (leaf_offset * 8 + (leaf_u32_offset & 0b111) )*MY_BLOCK_SIZE  ;
    gl_static_shared_memory[ foo + (idx & MY_BLOCK_MASK) ] = value;
}

static ALEO_ADI uint32_t smem_soa_leaves_alloc_free_index(void)
{
    uint32_t read_cursor, total_len;
    smem_get_soa_leaves_len(read_cursor , total_len);    

    //printf("smem_soa_leaves_alloc_free_index, step1, head=%u len=%u meta=%x\n", head, len, meta);

    if ( total_len >= SMEM_SOA_LEAVES_LEN )
    {
        cudbg("soa leaves is full 1, read_index=%u, total_len=%u\n", read_cursor, total_len);
        return UINT32_MAX;
    }
        
    smem_set_soa_leaves_len(read_cursor, total_len+1);

    {
        // uint32_t v1 ,v2;
        // smem_get_soa_leaves_len(v1, v2);
        // printf("smem_soa_leaves_alloc_free_index, read_cursor=%u, total_len=%u\n", v1, v2);
    }
    return total_len;
}

static ALEO_ADI uint32_t smem_soa_leaves_peek_oldest_index(void)
{    
    uint32_t read_cursor, total_len;
    smem_get_soa_leaves_len(read_cursor , total_len);    
    
    if ( total_len == 0 ) //empty
        return UINT32_MAX;

    return read_cursor;
}

static ALEO_ADI uint32_t smem_soa_leaves_pop_by_index(void)
{
    uint32_t read_cursor, total_len;
    smem_get_soa_leaves_len(read_cursor , total_len);

    if ( read_cursor >= total_len ) //empty 
    {
        cudbg("soa leaves is empty\n");
        return UINT32_MAX;
    }
        
    smem_set_soa_leaves_len(read_cursor+1, total_len);
    //printf("smem_soa_leaves_pop_by_index, head=%u meta=%x\n", head, meta);
    return read_cursor;

}


static ALEO_ADI void __r0_bits_cache_meta_unpack(SVM_PARAM_DEF, 
        uint8_t &real_cache_len, uint8_t &total_pad_len, uint8_t &already_pad_len, uint16_t &pad_bit_array)
{
    real_cache_len = r0_bits_cache_len & 0xFF;    
    total_pad_len = (r0_bits_cache_len >> 8) & 0xF;
    already_pad_len = (r0_bits_cache_len >> 12) & 0xF;
    pad_bit_array = r0_bits_cache_len >> 16;
}

static ALEO_ADI void __r0_bits_cache_meta_pack(SVM_PARAM_DEF, 
        uint8_t real_cache_len, uint8_t total_pad_len, uint8_t already_pad_len, uint16_t pad_bit_array)
{
    r0_bits_cache_len = pad_bit_array;
    r0_bits_cache_len = (r0_bits_cache_len<<4) | already_pad_len;    
    r0_bits_cache_len = (r0_bits_cache_len<<4) | total_pad_len;
    r0_bits_cache_len = (r0_bits_cache_len<<8) | real_cache_len;    

}


/* index 取值范围: 0 - 2 */
static ALEO_ADI void smem_save_circuit_params(uint32_t index, bigint_u256_t data)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    constexpr warp_meta_t *pmeta = NULL;
    for ( int m = 0; m < 8; m++)
    {
        uint32_t offset_of_state = ((uint64_t)( &( pmeta->circuit_params[index][m] ) )) / sizeof(uint32_t);
        offset_of_state = offset_of_state*MY_BLOCK_SIZE + (idx & MY_BLOCK_MASK);
        gl_static_shared_memory[ offset_of_state ] = data.uint32[m];
    }
}

/* index 取值范围: 0 - 2 */
static ALEO_ADI void smem_save_circuit_params(uint32_t index, uint64_t data[QBigInt::BIGINT_LEN])
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    constexpr warp_meta_t *pmeta = NULL;
    for ( int n = 0; n < 4; n++)
    {
        for ( int k = 0; k < 2; k++)
        {
            int m = n*2 + k;
            uint32_t offset_of_state = ((uint64_t)( &( pmeta->circuit_params[index][m] ) )) / sizeof(uint32_t);
            offset_of_state = offset_of_state*MY_BLOCK_SIZE + (idx & MY_BLOCK_MASK);
            gl_static_shared_memory[ offset_of_state ] = data[n] >> (k*32);
        }
        
    }
}

/* index 取值范围: 0 - 2 */
static ALEO_ADI void smem_load_circuit_params(uint32_t index, uint64_t data[QBigInt::BIGINT_LEN])
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    constexpr warp_meta_t *pmeta = NULL;
    for ( int n = 0; n < 4; n++)
    {
        uint32_t tmp[2];
        for ( int k = 0; k < 2; k++)
        {
            int m = n*2 + k;
            uint32_t offset_of_state = ((uint64_t)( &( pmeta->circuit_params[index][m] ) )) / sizeof(uint32_t);
            offset_of_state = offset_of_state*MY_BLOCK_SIZE + (idx & MY_BLOCK_MASK);
            tmp[k] = gl_static_shared_memory[ offset_of_state ];
        }
        data[n] = tmp[1];
        data[n] = (data[n] << 32) | tmp[0];            
    }
}

/* index 取值范围: 0 - 2 */
static ALEO_ADI void smem_load_circuit_params(uint32_t index, bigint_u256_t &data)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    constexpr warp_meta_t *pmeta = NULL;
    for ( int m = 0; m < 8; m++)
    {
        uint32_t offset_of_state = ((uint64_t)( &( pmeta->circuit_params[index][m] ) )) / sizeof(uint32_t);
        offset_of_state = offset_of_state*MY_BLOCK_SIZE + (idx & MY_BLOCK_MASK);
        data.uint32[m] = gl_static_shared_memory[ offset_of_state ];
    }
}

static ALEO_ADI void smem_save_circuit_params_by_u32(uint32_t index, uint32_t offset, uint32_t data)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    constexpr warp_meta_t *pmeta = NULL;
    
    uint32_t offset_of_data = ((uint64_t)( &( pmeta->circuit_params[index][offset] ) )) / sizeof(uint32_t);
    offset_of_data = offset_of_data*MY_BLOCK_SIZE + (idx & MY_BLOCK_MASK);
    gl_static_shared_memory[ offset_of_data ] = data;    
}

static ALEO_ADI uint32_t smem_load_circuit_params_by_u32(uint32_t index, uint32_t offset)
{    
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    constexpr warp_meta_t *pmeta = NULL;
    
    uint32_t offset_of_data = ((uint64_t)( &( pmeta->circuit_params[index][offset] ) )) / sizeof(uint32_t);
    offset_of_data = offset_of_data*MY_BLOCK_SIZE + (idx & MY_BLOCK_MASK);
    return gl_static_shared_memory[ offset_of_data ];
}

#undef cudbg
#undef cuerr        

#undef MAKE_CODE_SMEM_CONTEXT_CONCAT
#undef MAKE_CODE_SMEM_CONTEXT
#undef MAKE_CODE_SMEM_GET_OFFSET



