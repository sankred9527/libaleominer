#pragma once
#include "cu_synthesis.cuh"
#include "cu_smem.cuh"
#include "cu_aleo_globals.cuh"
#include "cu_mtree_define.cuh"
#include "cu_qbigint.cuh"

#define MTREE_PARAM_DEF Sha3StateV2 gl_r0_st[25], uint8_t &gl_r0_last_bit, uint32_t &gl_r0_round, \
                        Sha3StateV2 gl_r1_st[25], uint8_t &gl_r1_last_bit, uint32_t &gl_r1_round                        

#define MTREE_PARAM gl_r0_st, gl_r0_last_bit, gl_r0_round, \
                    gl_r1_st, gl_r1_last_bit, gl_r1_round  



//#define DUMP_LEAF 1

#ifdef DUMP_LEAF

#define tr_dbg(...) do { unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;   printf(__VA_ARGS__); } while(0)
static ALEO_ADI void hash_bhp256_debug_dump_leaf(bool leaf)
{    
    tr_dbg("leaf_hash=[%u, ",leaf);
    for(int n = 0; n < 31; n++) {
        tr_dbg("0, ");
    }
    tr_dbg("]\n");
}

static ALEO_ADI void hash_bhp256_debug_dump_leaf(uint64_t leaf[QBigInt::BIGINT_LEN])
{
    tr_dbg("leaf_hash=[");
    for(int n = 0; n < 32; n++) {        
        if ( n == 31)
        {
            tr_dbg("%u", QBigInt::get_u8_by_offset(leaf, n) );
        } else {
            tr_dbg("%u, ", QBigInt::get_u8_by_offset(leaf, n));
        }
        
    }
    tr_dbg("]\n");
}


static ALEO_ADI void hash_bhp256_debug_dump_leaf(uint32_t leaf[8])
{
    tr_dbg("leaf_hash=[");
    for(int n = 0; n < 32; n++) {        
        uint32_t v = leaf[n / 4];
        uint8_t v8 = v >> ((n%4)*8);
        if ( n == 31)
        {
            tr_dbg("%u", v8);
        } else {
            tr_dbg("%u, ", v8);
        }
        
    }
    tr_dbg("]\n");
}

static ALEO_ADI void hash_bhp256_debug_dump_leaf(bigint_u256_t leaf)
{
    tr_dbg("leaf_hash=[");
    for(int n = 0; n < 32; n++) {        
        uint8_t v8 = leaf.uint8[n];
        if ( n == 31)
        {
            tr_dbg("%u", v8);
        } else {
            tr_dbg("%u, ", v8);
        }
        
    }
    tr_dbg("]\n");
}

static ALEO_ADI void debug_show_bits_array(SVM_PARAM_DEF, uint16_t bits)
{
    
    for (int n =0; n < 8; n++, bits = bits>>2)
    {
        uint32_t v = bits & 0b11;
        if (  v <= 1 )
            hash_bhp256_debug_dump_leaf(v);
    }    
}

#else 
#define debug_show_bits_array(...) do {} while(0)
#define hash_bhp256_debug_dump_leaf(...) do{} while(0)
#define tr_dbg(...) do { } while(0)
#endif 


static ALEO_ADI uint32_t svm_mtree_pop_soa_leaf_u32_order_asc(SVM_PARAM_DEF, uint32_t counter, uint32_t vector_offset)
{
    // gl_thread_data_t *gctx = (gl_thread_data_t*)(ctx);                
    // return ((union_soa_leaf_t*)(&gctx->soa_leaves_stack[counter]))->v[vector_offset][idx].nv_int.x;
    constexpr gl_thread_data_wrap *ptdata = NULL; 
    uint32_t offset_in_u32 = ((uint64_t)( &(ptdata->soa_leavs_stack[counter][vector_offset]) )) / sizeof(uint32_t);
    uint32_t offset_of_name;  
    offset_of_name = offset_in_u32*d_max_thread;
    uint32_t *p = ((uint32_t*)(ctx)) + offset_of_name;
    return p[ idx ];
}

static ALEO_ADI void svm_mtree_push_soa_leaf_u32_order_asc(SVM_PARAM_DEF, uint32_t counter, uint32_t vector_offset, uint32_t data)
{    
    constexpr gl_thread_data_wrap *ptdata = NULL; 
    uint32_t offset_in_u32 = ((uint64_t)( &(ptdata->soa_leavs_stack[counter][vector_offset]) )) / sizeof(uint32_t);
    uint32_t offset_of_name;  
    offset_of_name = offset_in_u32*d_max_thread;
    uint32_t *p = ((uint32_t*)(ctx)) + offset_of_name;
    p[ idx ] = data;
}

static ALEO_ADI void svm_mtree_push_soa_leaf_u32_order_desc(SVM_PARAM_DEF, uint32_t counter, uint32_t vector_offset, uint32_t data)
{
    constexpr gl_thread_data_wrap *ptdata = NULL; 
    uint32_t offset_in_u32 = ((uint64_t)( &(ptdata->soa_leavs_stack[MAX_LEAVES_COMPRESS -1 - counter][vector_offset]) )) / sizeof(uint32_t);
    uint32_t offset_of_name;  
    offset_of_name = offset_in_u32*d_max_thread;
    uint32_t *p = ((uint32_t*)(ctx)) + offset_of_name;
    p[ idx ] = data;
}

static ALEO_ADI uint32_t svm_mtree_pop_soa_leaf_u32_order_desc(SVM_PARAM_DEF, uint32_t counter, uint32_t vector_offset)
{
    constexpr gl_thread_data_wrap *ptdata = NULL; 
    uint32_t offset_in_u32 = ((uint64_t)( &(ptdata->soa_leavs_stack[MAX_LEAVES_COMPRESS -1 - counter][vector_offset]) )) / sizeof(uint32_t);
    uint32_t offset_of_name;  
    offset_of_name = offset_in_u32*d_max_thread;
    uint32_t *p = ((uint32_t*)(ctx)) + offset_of_name;
    return p[ idx ];
    
}

static ALEO_ADI void svm_mtree_round1_try_final_action(SVM_PARAM_DEF, MTREE_PARAM_DEF)
{
    if ( gl_r1_round == 8 )
    {
        KeccakV2::fast_path_hash_final_state(gl_r1_st, gl_r1_last_bit);

        uint32_t tmp[8];
        #define MAKE_CODE_svm_mtree_round1_try_final_action(i) do{ \
            tmp[i] = KeccakV2::fast_path_hash_final_by_u32<i>(gl_r1_st);\
        } while(0)

        ALEO_ITER_8_CODE(MAKE_CODE_svm_mtree_round1_try_final_action)
        #undef MAKE_CODE_svm_mtree_round1_try_final_action

        #ifdef DUMP_LEAF
        {
            bigint_u256_t tmpint;
            for(int n = 0; n<8;n++)
                tmpint.uint32[n] = tmp[n];
            bigint256_dump_with_prefix("round 1 save=","" , &tmpint);
        }
        #endif                

        {            
            uint32_t len = svm_get_soa_leaves_length(SVM_PARAM);
            for (int n = 0; n < 8; n++)
                svm_mtree_push_soa_leaf_u32_order_asc(SVM_PARAM, len, n, tmp[n]);            
            svm_set_soa_leaves_length(SVM_PARAM, len+1);
        }

        memset(gl_r1_st, 0, sizeof(Sha3StateV2)*25);
        gl_r1_round = 0;
        gl_r1_last_bit = 1;
    }
}

static ALEO_ADI void svm_mtree_round0_try_final_action(SVM_PARAM_DEF, MTREE_PARAM_DEF)
{
    if ( gl_r0_round == 8)
    {        
        KeccakV2::fast_path_hash_final_state(gl_r0_st, gl_r0_last_bit);

        bigint_u256_t tmp;
        #define MKCODE_mtree_do_sha3(i)\
            tmp.uint32[i] = KeccakV2::fast_path_hash_final_by_u32<i>(gl_r0_st)
        
        ALEO_ITER_8_CODE(MKCODE_mtree_do_sha3)

        #undef MKCODE_mtree_do_sha3

        memset(gl_r0_st, 0, sizeof(Sha3StateV2)*25);
        gl_r0_round = 0;
        gl_r0_last_bit = 1;

        SHA3_update_state_with_uin32x8_dynamic(gl_r1_round, gl_r1_st, gl_r1_last_bit, tmp.uint32);
        gl_r1_round++;

        //bigint256_dump_with_prefix("r0 round save=","", &tmp);

        svm_mtree_round1_try_final_action(SVM_PARAM, MTREE_PARAM);        
    }
}

static ALEO_ADI void svm_mtree_round0_final_action_with_8bits_array(SVM_PARAM_DEF, MTREE_PARAM_DEF, uint8_t r0_bits_array)
{
#if 1
    uint32_t const_data[8];
    #pragma unroll
    for(int n = 0; n< 8;n++)
    {
        const_data[n] = mtree_round_0_cache[n*256 + r0_bits_array ];
    }
     
    SHA3_update_state_with_uin32x8_dynamic(gl_r1_round, gl_r1_st, gl_r1_last_bit, const_data);
    gl_r1_round++;    

    svm_mtree_round1_try_final_action(SVM_PARAM, MTREE_PARAM);
    
#endif    
}



static ALEO_ADI void svm_push_leaf_one_bit_bool(SVM_PARAM_DEF, bool leaf)
{     
    uint8_t real_cache_len;
    uint8_t total_pad_len; 
    uint8_t already_pad_len;
    uint16_t pad_bit_array;
    
    //hash_bhp256_debug_dump_leaf(leaf);

    uint32_t v = (uint32_t)leaf;
    __r0_bits_cache_meta_unpack(SVM_PARAM,
        real_cache_len, total_pad_len, already_pad_len, pad_bit_array);
    if ( already_pad_len < total_pad_len )
    {              
        pad_bit_array = (v<<(already_pad_len*2)) | pad_bit_array ;
        already_pad_len++;
        //printf("svm_push_leaf_one_bit_bool  pad2, already_pad_len=%u v=%u pad_bit_array=%x\n", already_pad_len, v, pad_bit_array);
    } else {
        r0_bits_cache = (v << (real_cache_len*2) ) | r0_bits_cache;
        real_cache_len++;
        if ( real_cache_len == 16 )
        {            
            RoundZeroMem::save_r0_bits_cache(SVM_PARAM);
            r0_bits_cache = 0;
            real_cache_len = 0;
        }         
    }   
    __r0_bits_cache_meta_pack(SVM_PARAM, real_cache_len, total_pad_len, already_pad_len, pad_bit_array); 
}


template <typename T>
static ALEO_ADI void svm_push_leaf_full(SVM_PARAM_DEF, T &leaf)
{     
    uint8_t real_cache_len;
    uint8_t total_pad_len; 
    uint8_t already_pad_len;
    uint16_t pad_bit_array;
    
    //hash_bhp256_debug_dump_leaf(leaf);

    __r0_bits_cache_meta_unpack(SVM_PARAM,
        real_cache_len, total_pad_len, already_pad_len, pad_bit_array);

    uint32_t v = (uint32_t)LEAF_FULL;    

    uint32_t leaves_len = smem_soa_leaves_alloc_free_index();    
    if ( leaves_len == UINT32_MAX )
        return;

    if constexpr (std::is_same_v<T, __uint128_t> ) {    
        #pragma unroll
        for(int n = 0; n < 4; n++)
            smem_set_soa_leaves_by_offset(leaves_len, n, leaf>>(n*32) );        

        #pragma unroll
        for(int n = 4; n < 8; n++)
            smem_set_soa_leaves_by_offset(leaves_len, n, 0 );
    }
    else if constexpr (std::is_array_v<T> &&     
                std::is_same_v<std::remove_extent_t<T>, uint64_t> && 
                std::extent_v<T> == 4) 
    {
        for(int n = 0; n < 4; n++)
        {
            for ( int i = 0; i < 2; i++ )
            {
                smem_set_soa_leaves_by_offset(leaves_len, n*2+i, leaf[n]>>(i*32) );
            }
        }
    } else if constexpr (std::is_same_v<T, bigint_u256_t> ) {
        for(int n = 0; n < 8; n++)
            smem_set_soa_leaves_by_offset(leaves_len, n, leaf.uint32[n]);  
    } else {
        static_assert(!std::is_same_v<T, T>, "Unsupported type for svm_push_leaf_full");
    }

    

    if ( already_pad_len < total_pad_len )
    {
        pad_bit_array = (v<<(already_pad_len*2)) | pad_bit_array ;
        already_pad_len++;
    } else {
        r0_bits_cache = (v << (real_cache_len*2) ) | r0_bits_cache;
        real_cache_len++;
        if ( real_cache_len == 16 )
        {
            RoundZeroMem::save_r0_bits_cache(SVM_PARAM);
            r0_bits_cache = 0;
            real_cache_len = 0;
        }
    }
    __r0_bits_cache_meta_pack(SVM_PARAM, real_cache_len, total_pad_len, already_pad_len, pad_bit_array);
}


// static ALEO_ADI void svm_push_leaf_full_with_convert(SVM_PARAM_DEF, bigint_u256_t leaf)
// {
//     bigint_u256_t tmp_field = leaf;
//     bigint_from_field(&tmp_field);
//     svm_push_leaf_full(SVM_PARAM, tmp_field);
// }

// static ALEO_ADI void svm_push_leaf_full_with_convert(SVM_PARAM_DEF, uint64_t leaf[4])
// {
//     QBigInt::bigint_from_field(leaf);    
//     svm_push_leaf_full(SVM_PARAM, leaf);
// }

template <typename T>
static ALEO_ADI void svm_push_leaf_full_with_convert(SVM_PARAM_DEF, T &leaf)
{
    if constexpr (( (std::is_array_v<T> &&     
                std::is_same_v<std::remove_extent_t<T>, uint64_t> && 
                std::extent_v<T> == QBigInt::BIGINT_LEN))
        ||  (std::is_same_v<T, uint64_t*> ) )
    {
        uint64_t tmp[QBigInt::BIGINT_LEN];
        QBigInt::copy(tmp, leaf);
        QBigInt::bigint_from_field(tmp);
        svm_push_leaf_full(SVM_PARAM, tmp);

    } else if constexpr (std::is_same_v<T, bigint_u256_t> ) {
        bigint_u256_t tmp_field = leaf;
        bigint_from_field(&tmp_field);
        svm_push_leaf_full(SVM_PARAM, tmp_field);
    } else {
        static_assert(!std::is_same_v<T, T>, "Unsupported type for svm_push_leaf_full_with_convert");
    }    
}


static ALEO_ADI void svm_push_leaf_full_with_convert_quick_bigint(SVM_PARAM_DEF, uint64_t leaf[QBigInt::BIGINT_LEN])
{
    uint64_t tmp[QBigInt::BIGINT_LEN];
    QBigInt::copy(tmp, leaf);
    QBigInt::bigint_from_field(tmp);
    svm_push_leaf_full(SVM_PARAM, tmp);   
}

template <typename T>
static ALEO_ADI void svm_push_leaf_bit_with_uint_type(SVM_PARAM_DEF, T value, uint32_t max_bits_len = 0)
{
#if 1
    uint32_t remain_bits = sizeof(T)*8;
    if ( max_bits_len > 0 && max_bits_len < remain_bits )
        remain_bits = max_bits_len;

    uint8_t real_cache_len;
    uint8_t total_pad_len; 
    uint8_t already_pad_len;
    uint16_t pad_bit_array;

    while (remain_bits > 0)
    {
        __r0_bits_cache_meta_unpack(SVM_PARAM,
            real_cache_len, total_pad_len, already_pad_len, pad_bit_array);
        
        if ( (real_cache_len > 0 && real_cache_len < 16) ||
            ( already_pad_len < total_pad_len )
         )
        {            
            uint32_t leaf = value & 1;
            value = value >> 1 ;
            svm_push_leaf_one_bit_bool(SVM_PARAM, leaf);
            remain_bits--;
        } else {            
            break;
        }
    }
    
    //if (sizeof(T)==2) printf("step4 remain_bits=%u  real_cache_len=%u r0_bits_cache=%08x\n", remain_bits, real_cache_len, r0_bits_cache);
    if ( remain_bits == 0 || real_cache_len != 0 )
        return;

    uint32_t loop_cnt;
    for (loop_cnt = 0; loop_cnt < remain_bits/16; loop_cnt++, value = value >> 16, r0_bits_cache = 0 )
    {
        uint16_t v16 = (uint16_t)value;

        for ( int n = 0; n < 16;n++, v16=v16>>1)
        {
            bool leaf = (v16 & 1);            
            r0_bits_cache = r0_bits_cache | ( leaf << (n*2) );
        }

        RoundZeroMem::save_r0_bits_cache(SVM_PARAM);        
    }
        
    remain_bits -= (loop_cnt*16);

    if ( remain_bits == 0 )
        return;

    for (loop_cnt = 0; loop_cnt < remain_bits; loop_cnt++, value=value>>1)
    {        
        bool leaf = (value & 1);
        r0_bits_cache = r0_bits_cache | ( leaf << (loop_cnt*2) );        
        real_cache_len += 1;        
    }

    __r0_bits_cache_meta_pack(SVM_PARAM, real_cache_len, total_pad_len, already_pad_len, pad_bit_array);

#endif     
}


static ALEO_ADI void svm_push_leaf_bit_with_u8(SVM_PARAM_DEF, __uint8_t value, uint32_t max_bits_len = 0)
{    
    svm_push_leaf_bit_with_uint_type(SVM_PARAM, value, max_bits_len );
    //svm_push_leaf_bit_as_int128(SVM_PARAM, value, 8);
}

static ALEO_ADI  void svm_push_leaf_bit_with_u16(SVM_PARAM_DEF, __uint16_t value, uint32_t max_bits_len = 0)
{
    svm_push_leaf_bit_with_uint_type(SVM_PARAM, value, max_bits_len );
    //svm_push_leaf_bit_as_int128(SVM_PARAM, value, 16);
}

static ALEO_ADI  void svm_push_leaf_bit_with_u32(SVM_PARAM_DEF, __uint32_t value, uint32_t max_bits_len = 0)
{
    svm_push_leaf_bit_with_uint_type(SVM_PARAM, value, max_bits_len );
    //svm_push_leaf_bit_as_int128(SVM_PARAM, value, 32);
}

static ALEO_ADI  void svm_push_leaf_bit_with_u64(SVM_PARAM_DEF, __uint64_t value, uint32_t max_bits_len = 0)
{
    svm_push_leaf_bit_with_uint_type(SVM_PARAM, value, max_bits_len );
    //svm_push_leaf_bit_as_int128(SVM_PARAM, value, 64);
}

static ALEO_ADI  void svm_push_leaf_bit_with_u128(SVM_PARAM_DEF, __uint128_t value, uint32_t max_bits_len = 0)
{
    svm_push_leaf_bit_with_uint_type(SVM_PARAM, value, max_bits_len );
    //svm_push_leaf_bit_as_int128(SVM_PARAM, value, 128);
}

static ALEO_ADI  void svm_push_leaf_bit_with_bls12_377_bigint(SVM_PARAM_DEF, uint64_t value[QBigInt::BIGINT_LEN])
{
    svm_push_leaf_bit_with_u64(SVM_PARAM, value[0]);
    svm_push_leaf_bit_with_u64(SVM_PARAM, value[1]);
    svm_push_leaf_bit_with_u64(SVM_PARAM, value[2]);
    svm_push_leaf_bit_with_u64(SVM_PARAM, value[3], 61);
}

static ALEO_ADI  void svm_push_leaf_bit_with_bls12_377_bigint(SVM_PARAM_DEF, bigint_u256_t value)
{
    svm_push_leaf_bit_with_u128(SVM_PARAM, value.uint128[0] );
    svm_push_leaf_bit_with_u128(SVM_PARAM, value.uint128[1], 125 );
    // svm_push_leaf_bit_with_u32(SVM_PARAM, value.uint4s[1].x);
    // svm_push_leaf_bit_with_u32(SVM_PARAM, value.uint4s[1].y);
    // svm_push_leaf_bit_with_u32(SVM_PARAM, value.uint4s[1].z);
    
    // constexpr uint8_t bits = BLS12_377_FR_PARAMETER_MODULUS_BITS - 224;
    // uint32_t v = value.uint4s[1].w;
    // #pragma unroll
    // for ( int n = 0; n < bits; n++ )
    // {
    //     bool to_set = ( v >> n ) & 1;
    //     svm_push_leaf_one_bit_bool(SVM_PARAM, to_set);
    // }
}

static ALEO_ADI void mtree_convert_with_leaf_hash(uint32_t input[8])
{
    Sha3StateV2 st[25];
    memset(st, 0 , sizeof(st));
    uint8_t last_bit = 0;
    KeccakV2::fast_leaf_hash_update<0>(st, last_bit, input[0]);
    KeccakV2::fast_leaf_hash_update<1>(st, last_bit, input[1]);
    KeccakV2::fast_leaf_hash_update<2>(st, last_bit, input[2]);
    KeccakV2::fast_leaf_hash_update<3>(st, last_bit, input[3]);
    KeccakV2::fast_leaf_hash_update<4>(st, last_bit, input[4]);
    KeccakV2::fast_leaf_hash_update<5>(st, last_bit, input[5]);
    KeccakV2::fast_leaf_hash_update<6>(st, last_bit, input[6]);
    KeccakV2::fast_leaf_hash_update<7>(st, last_bit, input[7]);
    
    KeccakV2::fast_leaf_hash_final_state(st);

    #define MAKE_CODE_mtree_convert_with_leaf_hash(i)\
        input[i] = KeccakV2::fast_path_hash_final_by_u32<i>(st)
    
    ALEO_ITER_8_CODE(MAKE_CODE_mtree_convert_with_leaf_hash)

    #undef MAKE_CODE_mtree_convert_with_leaf_hash
    
}

static ALEO_ADI void svm_push_leaf_bit_with_var_int(SVM_PARAM_DEF, nv_uint128 value, uint32_t bits)
{    
    switch (bits)
    {
    case 8:
        svm_push_leaf_bit_with_u8(SVM_PARAM, value.nvint.x);
        break;
    case 16:
        svm_push_leaf_bit_with_u16(SVM_PARAM, value.nvint.x);
        break;    
    case 32:
        svm_push_leaf_bit_with_u32(SVM_PARAM, value.nvint.x);
        break;
    case 64:
        svm_push_leaf_bit_with_u64(SVM_PARAM, value.stdint);
        break;
    case 128:
        svm_push_leaf_bit_with_u128(SVM_PARAM, value.stdint);
        break;        
    default:
        break;
    }
}


static ALEO_ADI uint8_t svm_mtree_round0_bits_array_to_uint8(uint16_t r0_bits_array, uint32_t round)
{
    uint8_t bits_array = 0;
    
    for(int n = 0; n < round; n++)
    {
        bits_array |= ( (r0_bits_array >> (n*2)) & 1 ) << n;
    }
    return bits_array;
}

static ALEO_ADI void __mtree_deal_uint16_bits_array_partitial(SVM_PARAM_DEF, MTREE_PARAM_DEF, uint16_t v16, uint8_t start_index, uint8_t end_index)
{    

    if ( start_index == 0 ) {
        gl_r0_round = 0;
        gl_r0_last_bit = 1;
        memset(gl_r0_st, 0, sizeof(Sha3StateV2)*25);
    }

    uint32_t leaf_is_simple_u8 = smem_get_circuit_is_u8_leaf_hash();

    int r0_current_round = start_index;
    for ( ; r0_current_round < end_index; r0_current_round++)
    {
        uint32_t offset_round = r0_current_round - start_index;
        
        uint32_t leaf = ( v16 >> (offset_round*2) ) & 0b11;
        if ( leaf == 0 || leaf == 1 )
        {
            //tr_dbg("r0 hash simple leaf=%u\n", leaf);
            hash_bhp256_debug_dump_leaf(leaf);
            uint32_t *const_data = CONST_LEAF_HASH_ZERO_ONE_U32X8[leaf];
            SHA3_update_state_with_uin32x8_dynamic(r0_current_round, gl_r0_st, gl_r0_last_bit, const_data);
        } else {
            // full leaf
            uint32_t soa_leaves_index = smem_soa_leaves_pop_by_index();
            //tr_dbg("soa leaves pop index=%u\n", soa_leaves_index);
            if ( soa_leaves_index == UINT32_MAX )
                continue;
            uint32_t tmp[8];
            for (int i = 0; i < 8; i++)
            {
                tmp[i] = smem_get_soa_leaves_by_offset(soa_leaves_index, i); 
            }
            #ifdef DUMP_LEAF
                //bigint256_dump_with_prefix("r0 hash full leaf=", "", &tmp);
                hash_bhp256_debug_dump_leaf(tmp);
            #endif 
            //if ( leaf_is_simple_u8 )
            // bool cond = (tmp[7] == 0 && tmp[6] == 0 && tmp[5] == 0 && tmp[4] == 0 && tmp[3] == 0 && tmp[2] == 0 && tmp[1] == 0) ;
            // bool cond_for_u16 = cond && ( (tmp[0] >> 16) == 0 );
            // bool cond_for_u8 = cond && ( (tmp[0] >> 8) == 0 );
            
            if ( leaf_is_simple_u8 == 2 )
            {
                uint16_t u16_value = tmp[0];
                constexpr uint32_t u16_leaf_hash_table_offset = 6*1024*1024;
                uint32_t *p = (uint32_t*)( ((uint8_t*)ctx) - u16_leaf_hash_table_offset );

                #pragma unroll
                for ( int n = 0; n < 8; n++)
                {
                    //tmp[n] = svm_mtree_pop_soa_leaf_u32_order_asc(SVM_PARAM, u8_value, n);
                    tmp[n] = p[u16_value*8  + n ];
                }
            }
            else
            if ( leaf_is_simple_u8 == 3 )
            {
                //load from constant table
                uint8_t u8_value = tmp[0];
                #pragma unroll
                for ( int n = 0; n < 8; n++)
                {
                    //tmp[n] = svm_mtree_pop_soa_leaf_u32_order_asc(SVM_PARAM, u8_value, n);
                    tmp[n] = CONST_leaf_hash_for_u8[ u8_value*8 + n];
                }
            }    
            else if ( leaf_is_simple_u8 == 1 )                     
            {
                //hack for square in pow.w 
                if ( soa_leaves_index != 2 )
                {
                    mtree_convert_with_leaf_hash(tmp);
                    if ( soa_leaves_index ==  1)
                    {                            
                        for (int i = 0; i < 8; i++)
                        {
                            smem_set_soa_leaves_by_offset(soa_leaves_index, i, tmp[i]);
                        }
                    }
                } else {
                    // is 2                         
                    for (int i = 0; i < 8; i++)
                    {
                        tmp[i] = smem_get_soa_leaves_by_offset(1, i); 
                    }
                }
            }
            else 
            {                             
                mtree_convert_with_leaf_hash(tmp);
            }
            
            SHA3_update_state_with_uin32x8_dynamic(r0_current_round, gl_r0_st, gl_r0_last_bit, tmp);
        }
    }
    gl_r0_round = end_index;
    if ( gl_r0_round == 8)
    {        
        KeccakV2::fast_path_hash_final_state(gl_r0_st, gl_r0_last_bit);

        bigint_u256_t tmp;
        #define MKCODE_mtree_do_sha3(i)\
            tmp.uint32[i] = KeccakV2::fast_path_hash_final_by_u32<i>(gl_r0_st)
        
        ALEO_ITER_8_CODE(MKCODE_mtree_do_sha3)

        #undef MKCODE_mtree_do_sha3

        SHA3_update_state_with_uin32x8_dynamic(gl_r1_round, gl_r1_st, gl_r1_last_bit, tmp.uint32);
        gl_r1_round++;

        svm_mtree_round1_try_final_action(SVM_PARAM, MTREE_PARAM);        
    }
}

static ALEO_ADI void __mtree_deal_uint16_bits_array(SVM_PARAM_DEF, MTREE_PARAM_DEF, uint16_t v16,  uint32_t partial = 8)
{    
    constexpr uint16_t mask = 0xAAAA;
    if ( ( (v16 ^ mask ) & mask) == mask )
    {
        uint8_t bits_array;
        bits_array = svm_mtree_round0_bits_array_to_uint8(v16, 8);
        #ifdef DUMP_LEAF
        {
            for (int n = 0; n < partial; n++ )
            {
                bool leaf = (bits_array >> n) & 1 ;
                hash_bhp256_debug_dump_leaf(leaf);
            }
        }
        #endif 
        //tr_dbg("svm_mtree_round0_final_action_with_16bits_array , bits array u8=0x%02x\n", bits_array);
        svm_mtree_round0_final_action_with_8bits_array(SVM_PARAM, MTREE_PARAM, bits_array);
    } else {
        __mtree_deal_uint16_bits_array_partitial(SVM_PARAM, MTREE_PARAM, v16, 0, 8);
    }
}




static ALEO_ADI void mtree_do_sha3(SVM_PARAM_DEF, MTREE_PARAM_DEF)
{
#if 1  
    /*
    先处理 pad 区 
    处理 share mem 区
    处理 r0_bits_cache 区
    
    */

    uint8_t real_cache_len; 
    uint8_t total_pad_len; 
    uint8_t already_pad_len;
    uint16_t pad_bit_array;
    __r0_bits_cache_meta_unpack(SVM_PARAM,
            real_cache_len, total_pad_len, already_pad_len, pad_bit_array);
        

    tr_dbg("mtree_do_sha3, real_cache_len=%u , total_pad_len=%u already_pad_len=%u pad_bit_array=%u\n",
                real_cache_len, total_pad_len, already_pad_len, pad_bit_array
            );

    if ( total_pad_len > 0 )
    {
        if ( already_pad_len == total_pad_len ) {
            __mtree_deal_uint16_bits_array_partitial(SVM_PARAM, MTREE_PARAM, pad_bit_array, 8-total_pad_len, 8);
            total_pad_len = already_pad_len = pad_bit_array = 0;
        } else if( already_pad_len < total_pad_len) {            
            __mtree_deal_uint16_bits_array_partitial(SVM_PARAM, MTREE_PARAM, pad_bit_array, gl_r0_round, gl_r0_round + already_pad_len);
            total_pad_len = total_pad_len - already_pad_len;
            already_pad_len = 0;
            pad_bit_array = 0;
            __r0_bits_cache_meta_pack(SVM_PARAM, real_cache_len, total_pad_len, already_pad_len, pad_bit_array);
            return;
        }
    }

    {
        uint32_t max_full_leaves;
        uint32_t len_in_smem;
        uint32_t len_in_global_mem;
        RoundZeroMem::get_leaf_len_in_share_global_memory(max_full_leaves, len_in_smem, len_in_global_mem);
        
        uint32_t total_len = len_in_smem + len_in_global_mem;
        tr_dbg("total leaves(u32 count) len = %u  len_in_smem=%u len_in_global_mem=%u\n", 
                total_len, len_in_smem, len_in_global_mem);

        uint32_t total_loop_len = total_len;
        if ( real_cache_len>=8) 
        {
            total_loop_len++;
        }
        for ( int offset = 0; offset < total_loop_len; offset++)
        {
            uint32_t r0_bits_array;
            uint32_t m_loop = 2;
            if ( offset < len_in_smem ) 
            {                
                r0_bits_array = gl_static_shared_memory[ (SMEM_CACHE_SIZE_IN_U32 - 1 - offset) * MY_BLOCK_SIZE + (idx & MY_BLOCK_MASK) ];
            } else if ( offset < total_len) {
                r0_bits_array = svm_get_full_leaves_by_offset(SVM_PARAM, offset - len_in_smem);
            } else {
                r0_bits_array = r0_bits_cache;
                m_loop = 1;
            }
            for ( int m = 0; m < m_loop ; m++)
            {
                uint16_t v16 =  (uint16_t)( r0_bits_array >> (m*16));
                __mtree_deal_uint16_bits_array(SVM_PARAM, MTREE_PARAM, v16);
            }            
        }
        len_in_global_mem = len_in_smem = 0;
        RoundZeroMem::set_leaf_len_in_share_global_memory(len_in_smem, len_in_global_mem);
    }    
        
    if ( real_cache_len >= 8 )
    {        
        real_cache_len -= 8;
        r0_bits_cache = r0_bits_cache >> 16;
    } 
    
    if ( real_cache_len == 0 ) 
    {
        __r0_bits_cache_meta_pack(SVM_PARAM, real_cache_len, total_pad_len, already_pad_len, pad_bit_array);
        return;
    }    

    uint16_t v16 = r0_bits_cache;    
    constexpr uint16_t mask = 0xAAAA;
    if ( ( (v16 ^ mask ) & mask) == mask )
    {        
        //没有 full leaf 
        // do nothing 
    } else {        
        __mtree_deal_uint16_bits_array_partitial(SVM_PARAM, MTREE_PARAM, v16, 0, real_cache_len);
        total_pad_len = 8 - real_cache_len;
        real_cache_len = 0;
        already_pad_len = 0;
        pad_bit_array = 0;
        r0_bits_cache = 0;
    }
    
    __r0_bits_cache_meta_pack(SVM_PARAM, real_cache_len, total_pad_len, already_pad_len, pad_bit_array);
    //printf("mtree_do_sha3 step9, r0_bits_cache_len=%08x\n", r0_bits_cache_len);
#endif 
}

#define LOAD_LEAF_FROM_SMEM(arr_u32, offset) do {\
    for(int n = 0; n < 8; n++)\
        arr_u32[n] = smem_get_soa_leaves_by_offset(offset, n);\
} while(0)

#define LOAD_LEAF_FROM_GLOBAL_MEM_ASC(arr_u32, offset) do {\
    for(int n = 0; n < 8; n++)\
        arr_u32[n] = svm_mtree_pop_soa_leaf_u32_order_asc(SVM_PARAM, offset, n);\
} while(0)

#define R0_PATH_HASH_UPDATE_FROM_GLOBL_MEMORY(round, leaf_u32_offset) do {\
    uint32_t leaf_u32 = svm_mtree_pop_soa_leaf_u32_order_asc(SVM_PARAM, leaf_offset, leaf_u32_offset);\
    path_hash_update_state_with_u32<round, leaf_u32_offset>(gl_r0_st, gl_r0_last_bit, leaf_u32);\
} while(0)

static __device__ __forceinline__ uint64_t revert_u64_to_big_endian(uint64_t data)
{
    static unsigned char lookup[16] = {
        0x0, 0x8, 0x4, 0xc, 0x2, 0xa, 0x6, 0xe,
        0x1, 0x9, 0x5, 0xd, 0x3, 0xb, 0x7, 0xf };

    uint8_t *p = (uint8_t *)&data;
    uint64_t ret = 0;
    uint8_t *r8 = (uint8_t*)&ret;
    for(int n=0; n<8;n++, p++)
    {
        r8[7-n] = (lookup[ (*p) & 0b1111] << 4) | lookup[(*p)>>4];                
    }
    return ret;
}

#define EMPTY_HASH_ORIGIN_U32_0  (0x7f2c15ba)
#define EMPTY_HASH_ORIGIN_U32_1  (0x2163b75b)
#define EMPTY_HASH_ORIGIN_U32_2  (0xf3bd7f77)
#define EMPTY_HASH_ORIGIN_U32_3  (0x8ddf1395)
#define EMPTY_HASH_ORIGIN_U32_4  (0x85107a9e)
#define EMPTY_HASH_ORIGIN_U32_5  (0xa07a855e)
#define EMPTY_HASH_ORIGIN_U32_6  (0x7520778d)
#define EMPTY_HASH_ORIGIN_U32_7  (0xcd440cb4)

static ALEO_ADI uint64_t svm_merkle_tree_final_step(SVM_PARAM_DEF,  MTREE_PARAM_DEF, uint32_t padding_depth)
{
    union {
        uint32_t root_hash[8];
        uint64_t v64s[4];
    };    

    LOAD_LEAF_FROM_GLOBAL_MEM_ASC(root_hash, 0);    

    #ifdef DUMP_LEAF
    {
        bigint_u256_t tmp;
        for (int n =0; n<8; n++)
            tmp.uint32[n] = root_hash[n];
        bigint256_dump_with_prefix("last root=","", &tmp);
    }
    #endif

    for ( int n = 0; n < padding_depth; n++ )
    {
        memset(gl_r0_st, 0 , sizeof(Sha3StateV2)*25);
        gl_r0_last_bit = 1;
        
        
        SHA3_update_state_with_uin32x8(0, gl_r0_st, gl_r0_last_bit, root_hash);
        SHA3_update_state_with_macro(1, gl_r0_st, gl_r0_last_bit, EMPTY_HASH_ORIGIN_U32);
        SHA3_update_state_with_macro(2, gl_r0_st, gl_r0_last_bit, EMPTY_HASH_ORIGIN_U32);
        SHA3_update_state_with_macro(3, gl_r0_st, gl_r0_last_bit, EMPTY_HASH_ORIGIN_U32);
        SHA3_update_state_with_macro(4, gl_r0_st, gl_r0_last_bit, EMPTY_HASH_ORIGIN_U32);
        SHA3_update_state_with_macro(5, gl_r0_st, gl_r0_last_bit, EMPTY_HASH_ORIGIN_U32);
        SHA3_update_state_with_macro(6, gl_r0_st, gl_r0_last_bit, EMPTY_HASH_ORIGIN_U32);
        SHA3_update_state_with_macro(7, gl_r0_st, gl_r0_last_bit, EMPTY_HASH_ORIGIN_U32);
        
        

        KeccakV2::fast_path_hash_final_state(gl_r0_st, gl_r0_last_bit);

        root_hash[0] = KeccakV2::fast_path_hash_final_by_u32<0>(gl_r0_st);
        root_hash[1] = KeccakV2::fast_path_hash_final_by_u32<1>(gl_r0_st);
        root_hash[2] = KeccakV2::fast_path_hash_final_by_u32<2>(gl_r0_st);
        root_hash[3] = KeccakV2::fast_path_hash_final_by_u32<3>(gl_r0_st);
        root_hash[4] = KeccakV2::fast_path_hash_final_by_u32<4>(gl_r0_st);
        root_hash[5] = KeccakV2::fast_path_hash_final_by_u32<5>(gl_r0_st);
        root_hash[6] = KeccakV2::fast_path_hash_final_by_u32<6>(gl_r0_st);
        root_hash[7] = KeccakV2::fast_path_hash_final_by_u32<7>(gl_r0_st);


    }

    uint64_t target = revert_u64_to_big_endian(v64s[0]);
    uint64_t real_target = ((uint64_t)-1) / target;
    
    tr_dbg("real_target=%llu  target=%llu  dest_target=%llu\n", real_target, target, d_target);

    return real_target;
    

}



#if 0
__device__ void merkle_tree_run(SVM_PARAM_DEF, MTREE_PARAM_DEF);
__device__ void merkle_tree_run_round_23(SVM_PARAM_DEF, MTREE_PARAM_DEF);
__device__ bool merkle_tree_run_round_45(SVM_PARAM_DEF, MTREE_PARAM_DEF);


#else 


static ALEO_ADI void merkle_tree_run(SVM_PARAM_DEF, MTREE_PARAM_DEF)
{

    /*
        MAX_LEAVES_COMPRESS = 5120 , 所以
        chunk 的 soa_leaves_stack 内存布局:

    --------------------------------------
    |0|1|2|3|...|soa_leaf_len-1|....|5119|
    --------------------------------------    

    full_leaves: 的内存布局: 3200个字节， 共存放 MAX_FULL_LEAVES = (51200) 个leaf
    -----------------------------------
    |0|1|2|3|...|leaf_len-1|....|51199|
    -----------------------------------

    从后往前计算
    */
     

    /*
    虚拟的 tree 里：

    [0 .. num_nodes ) 是 empty hash
    [num_nodes .. num_nodes + leaf_len ) 是 做过 leaf hash 的 leaves
    [num_nodes + leaf_len ..  262144 ) 是 empty hash 

    */


    /*
        第一轮计算，如果 full_leaves_len = 262144 , 那么 start  = 4681, end = 37449, num_full_nodes = 32768
        计算从  4681*8+1 到 (end-1)*8+1  这个范围 [37449, 299593 ) 的内的 path hash ， 每个path hash 对 8个 leaf 做hash
        虚拟 tree 的长度位 262144 , 就是 [37449, 299592 ] 这个范围内的 leaf 做计算

        结果存到虚拟  tree 的 [ start , end ] = [ 4681,  37449 )  区间

        第二轮计算， start = 585 , end=4681 , num_full_nodes = 4096
        计算从  585*8+1 到 (4681-1)*8+1  这个范围 [4681, 37449 )  的内的 path hash ， 每个path hash 对 8个 leaf 做hash
        结果存到虚拟  tree 的 [ 585 , 4681 )  区间

        第3轮计算， start = 73 , end=585

        第4轮计算， start = 9, end=73

        第5轮计算， start = 1, end=9

        第6轮计算， start = 0, end=1            
    */

    uint8_t real_cache_len; 
    uint8_t total_pad_len; 
    uint8_t already_pad_len;
    uint16_t pad_bit_array;
    __r0_bits_cache_meta_unpack(SVM_PARAM,
            real_cache_len, total_pad_len, already_pad_len, pad_bit_array);

    // printf("merkle_tree_run , real_cache_len=%x, total_pad_len=%x, already_pad_len=%x, pad_bit_array=%x\n", 
    //     real_cache_len, total_pad_len, already_pad_len, pad_bit_array);
    // printf("merkle_tree_run   meta=%x gl_r0_round=%u  gl_r1_round=%u\n", 
    //         smem_get_soa_leaves_len(),
    //         gl_r0_round, gl_r1_round);   

    if ( already_pad_len < total_pad_len )
    {
        //pad round 0
        //printf("merkle_tree_run pad 1\n");

        for (; gl_r0_round < 8; gl_r0_round++)
        {
            SHA3_update_state_with_uin32x8_dynamic(gl_r0_round, gl_r0_st, gl_r0_last_bit, CONST_empty_hash_r0_U32X8);
        }
        svm_mtree_round0_try_final_action(SVM_PARAM, MTREE_PARAM);
    } 

    if ( real_cache_len > 16) 
    {
        //printf("merkle_tree_run deal real_cache_len 1=%u \n", real_cache_len);
        //类似稀疏矩阵
        __mtree_deal_uint16_bits_array(SVM_PARAM, MTREE_PARAM, (uint16_t)r0_bits_cache );
        real_cache_len -= 16;
        r0_bits_cache = r0_bits_cache >> 16;
    }
    if ( real_cache_len > 0 )
    {
        //printf("merkle_tree_run deal real_cache_len 2=%u \n", real_cache_len);
        __mtree_deal_uint16_bits_array(SVM_PARAM, MTREE_PARAM, (uint16_t)r0_bits_cache, real_cache_len );
    }

    //printf("merkle_tree_run deal gl_r1_round=%u \n", gl_r1_round);
    if ( gl_r1_round > 0 && gl_r1_round < 8)
    {
        for (; gl_r1_round < 8; gl_r1_round++)
        {
            SHA3_update_state_with_uin32x8_dynamic(gl_r1_round, gl_r1_st, gl_r1_last_bit, CONST_empty_hash_r1_U32X8);
        }
        svm_mtree_round1_try_final_action(SVM_PARAM, MTREE_PARAM);
    }
        
}


static ALEO_ADI void merkle_tree_run_round_23(SVM_PARAM_DEF, MTREE_PARAM_DEF)
{
    
/*
    r2 的输入：
    虚拟输入 leaf：
    [0, 8 ), [8, 48) [48,208), [208,.... 632)
    实际输入:
    [0, 8 ), [8, 48), 省略,     [48, 472)

    输出：


    虚拟save:
    [0, 1 ), [1, 6) [6, 26), [26,.... 79)
    实际save:
    [0, 1), [1,6), 省略,  [6, 59) ....
*/
/*
r3 的输入：            
    虚拟
    [0, 1 ), [1, 6) [6, 26), [26,.... 79)
    实际:
    [0, 1), [1,6), 省略,  [6, 59) ....

*/

    uint32_t r2_input_leaf_len = svm_get_soa_leaves_length(SVM_PARAM);
    //printf("merkle_tree_run_round_23 input leaf len=%u\n", r2_input_leaf_len);

    svm_set_soa_leaves_length(SVM_PARAM, 0);

    memset(gl_r1_st, 0 , sizeof(Sha3StateV2)*25);    
    gl_r1_round = 0;    
    gl_r1_last_bit = 1;

    int r2_input_leaf_offset ;
    for ( r2_input_leaf_offset = 0; r2_input_leaf_offset < 48; r2_input_leaf_offset += 8)
    {        
        memset(gl_r0_st, 0 , sizeof(Sha3StateV2)*25);
        gl_r0_last_bit = 1;
        gl_r0_round = 0;

        int leaf_offset = r2_input_leaf_offset;
        ALEO_ITER_8_CODE_WITH_2_PARAMS(R0_PATH_HASH_UPDATE_FROM_GLOBL_MEMORY, 0)
        leaf_offset++;
        ALEO_ITER_8_CODE_WITH_2_PARAMS(R0_PATH_HASH_UPDATE_FROM_GLOBL_MEMORY, 1)
        leaf_offset++;
        ALEO_ITER_8_CODE_WITH_2_PARAMS(R0_PATH_HASH_UPDATE_FROM_GLOBL_MEMORY, 2)
        leaf_offset++;        
        ALEO_ITER_8_CODE_WITH_2_PARAMS(R0_PATH_HASH_UPDATE_FROM_GLOBL_MEMORY, 3)
        leaf_offset++;
        ALEO_ITER_8_CODE_WITH_2_PARAMS(R0_PATH_HASH_UPDATE_FROM_GLOBL_MEMORY, 4)
        leaf_offset++;
        ALEO_ITER_8_CODE_WITH_2_PARAMS(R0_PATH_HASH_UPDATE_FROM_GLOBL_MEMORY, 5)
        leaf_offset++;
        ALEO_ITER_8_CODE_WITH_2_PARAMS(R0_PATH_HASH_UPDATE_FROM_GLOBL_MEMORY, 6)
        leaf_offset++;
        ALEO_ITER_8_CODE_WITH_2_PARAMS(R0_PATH_HASH_UPDATE_FROM_GLOBL_MEMORY, 7)

        {
            KeccakV2::fast_path_hash_final_state(gl_r0_st, gl_r0_last_bit);

            //uint32_t tmp[8];
            bigint_u256_t temp_leaf;
            uint32_t *tmp = &temp_leaf.uint32[0];
            #define MKCODE_mtree_do_sha3(i)\
                temp_leaf.uint32[i] = KeccakV2::fast_path_hash_final_by_u32<i>(gl_r0_st)
            
            ALEO_ITER_8_CODE(MKCODE_mtree_do_sha3)            
            #undef MKCODE_mtree_do_sha3
        
            SHA3_update_state_with_uin32x8_dynamic(gl_r1_round, gl_r1_st, gl_r1_last_bit, tmp);            

            /*
            round2 的虚拟输出： [1, 6) [6, 26), 中 [6,26) 时重复的 hash.psd2 数据。这里暂存到 share memory
            */
            
            if ( gl_r1_round >= 1 && gl_r1_round < 6 )
            {
                //save to share memory
                for(int n = 0; n < 8; n++)
                    smem_set_soa_leaves_by_offset(gl_r1_round, n, tmp[n]);                
            }
            gl_r1_round++;
        }      
        
    }
    //printf("merkle_tree_run_round_23 step1.  gl_r1_round=%u\n", gl_r1_round);

    //for ( r2_input_leaf_offset = 48; r2_input_leaf_offset < 64; r2_input_leaf_offset+=8)
    {
        //对应 round3 的：  [0, 6) [6, 8)
        uint32_t tmp[8];        
        // for(int n = 0; n < 8; n++)
        //     tmp[n] = smem_get_soa_leaves_by_offset(1, n);

        LOAD_LEAF_FROM_SMEM(tmp, 1);
        SHA3_update_state_with_uin32x8_dynamic(gl_r1_round, gl_r1_st, gl_r1_last_bit, tmp);
        gl_r1_round++;
        
        // for(int n = 0; n < 8; n++)
        //     tmp[n] = smem_get_soa_leaves_by_offset(2, n);
        LOAD_LEAF_FROM_SMEM(tmp, 2);
        SHA3_update_state_with_uin32x8_dynamic(gl_r1_round, gl_r1_st, gl_r1_last_bit, tmp);
        gl_r1_round++;

        svm_mtree_round1_try_final_action(SVM_PARAM, MTREE_PARAM);
    }

    //round3 : 处理 [8, 16) 
    #define ROUND3_UPDATE_FROM_SHARE_MEMORY(offset) do {\
            uint32_t tmp[8];   \
            LOAD_LEAF_FROM_SMEM(tmp, offset);\
            SHA3_update_state_with_uin32x8_dynamic(gl_r1_round, gl_r1_st, gl_r1_last_bit, tmp);\
            gl_r1_round++;\
        } while(0)

    {
        memset(gl_r1_st, 0 , sizeof(Sha3StateV2)*25);    
        gl_r1_round = 0;    
        gl_r1_last_bit = 1;
        
        ROUND3_UPDATE_FROM_SHARE_MEMORY(3);
        ROUND3_UPDATE_FROM_SHARE_MEMORY(4);
        ROUND3_UPDATE_FROM_SHARE_MEMORY(5);
        ROUND3_UPDATE_FROM_SHARE_MEMORY(1);
        ROUND3_UPDATE_FROM_SHARE_MEMORY(2);
        ROUND3_UPDATE_FROM_SHARE_MEMORY(3);
        ROUND3_UPDATE_FROM_SHARE_MEMORY(4);
        ROUND3_UPDATE_FROM_SHARE_MEMORY(5);
        svm_mtree_round1_try_final_action(SVM_PARAM, MTREE_PARAM);
    }

    //处理 [16, 24)
    {
        memset(gl_r1_st, 0 , sizeof(Sha3StateV2)*25);    
        gl_r1_round = 0;    
        gl_r1_last_bit = 1;
        
        ROUND3_UPDATE_FROM_SHARE_MEMORY(1);
        ROUND3_UPDATE_FROM_SHARE_MEMORY(2);
        ROUND3_UPDATE_FROM_SHARE_MEMORY(3);
        ROUND3_UPDATE_FROM_SHARE_MEMORY(4);
        ROUND3_UPDATE_FROM_SHARE_MEMORY(5);
        ROUND3_UPDATE_FROM_SHARE_MEMORY(1);
        ROUND3_UPDATE_FROM_SHARE_MEMORY(2);
        ROUND3_UPDATE_FROM_SHARE_MEMORY(3);
        svm_mtree_round1_try_final_action(SVM_PARAM, MTREE_PARAM);
    }

    //处理 [24, 32)
    {
        memset(gl_r1_st, 0 , sizeof(Sha3StateV2)*25);    
        gl_r1_round = 0;    
        gl_r1_last_bit = 1;

        memset(gl_r0_st, 0 , sizeof(Sha3StateV2)*25);    
        gl_r0_round = 0;    
        gl_r0_last_bit = 1;
                
        ROUND3_UPDATE_FROM_SHARE_MEMORY(4);
        ROUND3_UPDATE_FROM_SHARE_MEMORY(5);
        //对应 r3 的  6 条hash
        for (r2_input_leaf_offset = 48; r2_input_leaf_offset < r2_input_leaf_len && r2_input_leaf_offset< 96; r2_input_leaf_offset+=1)
        {
            uint32_t leaf[8];
            LOAD_LEAF_FROM_GLOBAL_MEM_ASC(leaf, r2_input_leaf_offset);
            SHA3_update_state_with_uin32x8_dynamic(gl_r0_round, gl_r0_st, gl_r0_last_bit, leaf);            

            gl_r0_round++;
            svm_mtree_round0_try_final_action(SVM_PARAM, MTREE_PARAM);
        }

        if ( r2_input_leaf_offset != 96 )
        {
            //需要补充            
            for(; gl_r0_round < 8; gl_r0_round++)
            {
                SHA3_update_state_with_uin32x8_dynamic(gl_r0_round, gl_r0_st, gl_r0_last_bit, CONST_empty_hash_r2_U32X8);
            }
            svm_mtree_round0_try_final_action(SVM_PARAM, MTREE_PARAM);
        }
    }

    // printf("round23 finish(24-32) r2_input_leaf_offset=%u r2_input_leaf_len=%u gl_r0_round=%u gl_r1_round=%u\n", 
    //         r2_input_leaf_offset, r2_input_leaf_len, gl_r0_round, gl_r1_round);

    for (; r2_input_leaf_offset < r2_input_leaf_len; r2_input_leaf_offset+=1)
    {
        uint32_t leaf[8];
        LOAD_LEAF_FROM_GLOBAL_MEM_ASC(leaf, r2_input_leaf_offset);
        SHA3_update_state_with_uin32x8_dynamic(gl_r0_round, gl_r0_st, gl_r0_last_bit, leaf);

        gl_r0_round++;
        svm_mtree_round0_try_final_action(SVM_PARAM, MTREE_PARAM);
    }

    // printf("round23 finish r2_input_leaf_offset=%u r2_input_leaf_len=%u gl_r0_round=%u gl_r1_round=%u\n", 
    //         r2_input_leaf_offset, r2_input_leaf_len, gl_r0_round, gl_r1_round);
    
    if ( gl_r0_round > 0 )
    {
        for ( ; gl_r0_round < 8; gl_r0_round++)
        {
            SHA3_update_state_with_uin32x8_dynamic(gl_r0_round, gl_r0_st, gl_r0_last_bit, CONST_empty_hash_r2_U32X8);
        }
        svm_mtree_round0_try_final_action(SVM_PARAM, MTREE_PARAM);
    }
    
    //printf("round 2 finish, gl_r1_round=%u\n", gl_r1_round);
    
    if ( gl_r1_round > 0 )
    {
        for (; gl_r1_round < 8; gl_r1_round++)
        {
            SHA3_update_state_with_uin32x8_dynamic(gl_r1_round, gl_r1_st, gl_r1_last_bit, CONST_empty_hash_r3_U32X8);
        }
        svm_mtree_round1_try_final_action(SVM_PARAM, MTREE_PARAM);
    }
    
    #undef ROUND3_UPDATE_FROM_SHARE_MEMORY
}

/*
返回值：

true：  leaf 为 pow(8,5)
false： leaf 为 pow(8,6)

*/
static ALEO_ADI bool merkle_tree_run_round_45(SVM_PARAM_DEF, MTREE_PARAM_DEF)
{
    uint32_t r4_input_leaf_len = svm_get_soa_leaves_length(SVM_PARAM);
    uint32_t r4_input_leaf_offset;
    uint32_t tmp[8];

    //printf("merkle_tree_run_round_45 input leaf len=%u\n", r4_input_leaf_len);

    svm_set_soa_leaves_length(SVM_PARAM, 0);

    memset(gl_r1_st, 0, sizeof(Sha3StateV2)*25);
    gl_r1_round = 0;
    gl_r1_last_bit = 1;    

    bool no_round5 = false;
    if ( r4_input_leaf_len <= 8 )
    {
        no_round5 = true;
    }

    for (r4_input_leaf_offset = 0; (r4_input_leaf_offset+8) <= r4_input_leaf_len; r4_input_leaf_offset+=8)
    {
        memset(gl_r0_st, 0 , sizeof(Sha3StateV2)*25);    
        gl_r0_round = 0;    
        gl_r0_last_bit = 1;

        int leaf_offset = r4_input_leaf_offset;
        ALEO_ITER_8_CODE_WITH_2_PARAMS(R0_PATH_HASH_UPDATE_FROM_GLOBL_MEMORY, 0)        

        leaf_offset++;
        ALEO_ITER_8_CODE_WITH_2_PARAMS(R0_PATH_HASH_UPDATE_FROM_GLOBL_MEMORY, 1)     

        leaf_offset++;
        ALEO_ITER_8_CODE_WITH_2_PARAMS(R0_PATH_HASH_UPDATE_FROM_GLOBL_MEMORY, 2)

        leaf_offset++;
        ALEO_ITER_8_CODE_WITH_2_PARAMS(R0_PATH_HASH_UPDATE_FROM_GLOBL_MEMORY, 3)

        leaf_offset++;
        ALEO_ITER_8_CODE_WITH_2_PARAMS(R0_PATH_HASH_UPDATE_FROM_GLOBL_MEMORY, 4)

        leaf_offset++;
        ALEO_ITER_8_CODE_WITH_2_PARAMS(R0_PATH_HASH_UPDATE_FROM_GLOBL_MEMORY, 5)

        leaf_offset++;
        ALEO_ITER_8_CODE_WITH_2_PARAMS(R0_PATH_HASH_UPDATE_FROM_GLOBL_MEMORY, 6)

        leaf_offset++;
        ALEO_ITER_8_CODE_WITH_2_PARAMS(R0_PATH_HASH_UPDATE_FROM_GLOBL_MEMORY, 7)

        KeccakV2::fast_path_hash_final_state(gl_r0_st, gl_r0_last_bit);
                
        #define MKCODE_mtree_do_sha3(i)\
            tmp[i] = KeccakV2::fast_path_hash_final_by_u32<i>(gl_r0_st)
        
        ALEO_ITER_8_CODE(MKCODE_mtree_do_sha3)            
        #undef MKCODE_mtree_do_sha3      
    
        if ( !no_round5 )
        {
            SHA3_update_state_with_uin32x8_dynamic(gl_r1_round, gl_r1_st, gl_r1_last_bit, tmp);
            gl_r1_round++;
            svm_mtree_round1_try_final_action(SVM_PARAM, MTREE_PARAM);        
        }
    }

    // printf("round 5 test1 gl_r1_round=%u\n", gl_r1_round);
    // printf("round 5 test1 r4_input_leaf_len=%u\n", r4_input_leaf_len);

    if ( r4_input_leaf_offset != r4_input_leaf_len )
    {
        memset(gl_r0_st, 0, sizeof(Sha3StateV2)*25);
        gl_r0_round = 0;
        gl_r0_last_bit = 1;

        for (; r4_input_leaf_offset < r4_input_leaf_len; r4_input_leaf_offset++, gl_r0_round++)
        {            
            uint32_t tmp[8];
            LOAD_LEAF_FROM_GLOBAL_MEM_ASC(tmp, r4_input_leaf_offset);
            SHA3_update_state_with_uin32x8_dynamic(gl_r0_round, gl_r0_st, gl_r0_last_bit, tmp);
        }
        for ( ; gl_r0_round < 8; gl_r0_round++)
        {
            SHA3_update_state_with_uin32x8_dynamic(gl_r0_round, gl_r0_st, gl_r0_last_bit, CONST_empty_hash_r4_U32X8);
        }

        KeccakV2::fast_path_hash_final_state(gl_r0_st, gl_r0_last_bit);
                
        #define MKCODE_mtree_do_sha3(i)\
            tmp[i] = KeccakV2::fast_path_hash_final_by_u32<i>(gl_r0_st)            
        
        ALEO_ITER_8_CODE(MKCODE_mtree_do_sha3)            
        #undef MKCODE_mtree_do_sha3
    
        if ( !no_round5 ) {
            SHA3_update_state_with_uin32x8_dynamic(gl_r1_round, gl_r1_st, gl_r1_last_bit, tmp);
            gl_r1_round++;
            svm_mtree_round1_try_final_action(SVM_PARAM, MTREE_PARAM);
        }
    }

    if ( no_round5 )
    {
        //printf("round 5 , no round 5 ,return\n");
        // pow(8,5) 不需要再做了        
        uint32_t len = svm_get_soa_leaves_length(SVM_PARAM);
        for (int n = 0; n < 8; n++)
            svm_mtree_push_soa_leaf_u32_order_asc(SVM_PARAM, len, n, tmp[n]);            
        svm_set_soa_leaves_length(SVM_PARAM, len + 1);
        return no_round5;
    }

    if ( gl_r1_round > 0 )
    {
        //printf("round 5 test3\n");

        for (; gl_r1_round < 8; gl_r1_round++)
        {
            SHA3_update_state_with_uin32x8_dynamic(gl_r1_round, gl_r1_st, gl_r1_last_bit, CONST_empty_hash_r5_U32X8);
        }
        svm_mtree_round1_try_final_action(SVM_PARAM, MTREE_PARAM);
    }

    return no_round5;
}

#endif

#undef tr_dbg