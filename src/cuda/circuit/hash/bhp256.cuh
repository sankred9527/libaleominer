#pragma once
#include "cuda/cu_bigint.cuh"
#include "cuda/cu_synthesis.cuh"
#include "circuit_hash_help.cuh"
#include "cuda/cu_smem.cuh"
#include "hash_bhp256_bases.cuh"
#include "cuda/circuit/field/circuit_field.cuh"
#include "cuda/circuit/groups/circuit_groups.cuh"
#include "cuda/circuit/integer/circuit_integer.cuh"
#include "cuda/cu_mtree_leaf.cuh"


typedef struct {
    bigint_u256_t input_bits;
    uint32_t input_bits_len;
} HashBhp256_Context;

#if 0
#define cudbg(...) do { printf(__VA_ARGS__); } while(0)
#define cuerr(...) do { printf(__VA_ARGS__); } while(0)
#else
#define cudbg(...) do {}while(0)
#define cuerr(...) do {}while(0)
#endif



class HashBhp256
{
public:
    static constexpr uint8_t NUM_WINDOWS = 8;
    static constexpr uint8_t WINDOW_SIZE = 57;
    static constexpr uint64_t BHP_CHUNK_SIZE = 3;
    static constexpr uint32_t DOMAIN_BITS_LEN = 188;
    static constexpr uint64_t HASH_CHUNK_SIZE = WINDOW_SIZE * BHP_CHUNK_SIZE;

    //little endian, 188 bits
    // static  const bigint_u256_t domain = {
    //     .uint32 = { 0x00000000, 0x00000000, 0x00000000, 0xcac6c000, 0x642120a4, 0x08236a6f, 0x00000000, 0x00000000 }
    // };

    static constexpr bigint_u256_t one_half_bigint = { 
        //.uint32 = { 0x00000001, 0x8508c000, 0x68000000, 0xacd53b7f, 0x2e1bd800, 0x305a268f, 0x4d1652ab, 0x0955b2af }
        .uint64 = { 0xc396fffffffffffaULL, 0xe60136071ffffff9ULL, 0xbbc63149d6b1dff7ULL, 0x0ffb9fc862f41ff9ULL }
    };

    static constexpr bigint_u256_t coeff_a_bigint = { 
        //.uint32 = { 0x37e8f552, 0xedb748da, 0x4d4753dd, 0x93952ad8, 0xece3971b, 0x26949031, 0xada9010a, 0x08d26e3f }
        .uint64 = { 0xbf840c077464fa2eULL, 0x5ef226bda9de7456ULL, 0xade20e8654281b26ULL, 0x0bd9f15623150c79ULL }
    };

    static constexpr bigint_u256_t coeff_b_bigint = { 
        //.uint32 = { 0xc8170aad, 0x1c5a3725, 0x82b8ac23, 0xc6154c26, 0x6f5418e5, 0x3a1fbcec, 0xec83a44c, 0x09d8f71e }
        .uint64 = { 0x647773f88b9b05efULL, 0xc95d541fe6218bc7ULL, 0x468aadea1e26d500ULL, 0x11908a6153f9ae4fULL }
    };


    static ALEO_ADI void field_div_unchecked(uint64_t self[QBigInt::BIGINT_LEN], 
                    uint64_t other[QBigInt::BIGINT_LEN], 
                    uint64_t dest[QBigInt::BIGINT_LEN])
    {    
        if (QBigInt::is_zero(other)) 
        {
            QBigInt::copy(other, CirCuit_Fields_Base::get_field_one().uint64 );
        }

        QBigInt::copy(dest, self);
        CirCuit_Fields_Base::field_div(dest, other);
    }

    static ALEO_ADI void montgomery_add_qbigint(SVM_PARAM_CDEF, 
                    uint64_t this_x[QBigInt::BIGINT_LEN], uint64_t this_y[QBigInt::BIGINT_LEN], 
                    uint64_t that_x[QBigInt::BIGINT_LEN], uint64_t that_y[QBigInt::BIGINT_LEN], 
                    uint64_t sum_x[QBigInt::BIGINT_LEN], uint64_t sum_y[QBigInt::BIGINT_LEN])
    {
        uint64_t lambda[QBigInt::BIGINT_LEN]; //也用做临时变量

        CirCuit_Fields_Base::field_sub(that_y, this_y);        
        QBigInt::copy(lambda, that_x);
        CirCuit_Fields_Base::field_sub(lambda, this_x);
        CirCuit_Fields_Base::field_div(that_y, lambda);
        
        //lambda is that_y 
        QBigInt::copy(lambda, that_y);
        if ( !is_const ) 
            svm_push_leaf_full_with_convert(SVM_PARAM, lambda);
        

        QBigInt::bigint_mont_mul_assign_no_unroll(lambda, lambda, sum_x);
        {            
            uint64_t tmp[QBigInt::BIGINT_LEN] = {
                HashBhp256::coeff_b_bigint.uint64[0],
                HashBhp256::coeff_b_bigint.uint64[1],
                HashBhp256::coeff_b_bigint.uint64[2],
                HashBhp256::coeff_b_bigint.uint64[3]
            };
            
                
            QBigInt::bigint_mont_mul_assign_no_unroll(tmp, sum_x, sum_x);
        }

        {
            uint64_t tmp[QBigInt::BIGINT_LEN] = {
                HashBhp256::coeff_a_bigint.uint64[0],
                HashBhp256::coeff_a_bigint.uint64[1],
                HashBhp256::coeff_a_bigint.uint64[2],
                HashBhp256::coeff_a_bigint.uint64[3]
            };
            
            CirCuit_Fields_Base::field_sub(sum_x, tmp);
        }
        CirCuit_Fields_Base::field_sub(sum_x, this_x);
        CirCuit_Fields_Base::field_sub(sum_x, that_x);
        if ( !is_const ) 
            svm_push_leaf_full_with_convert(SVM_PARAM, sum_x);
        //CirCuit_Fields_Base::field_reduce(sum_x);    
        
        QBigInt::copy(sum_y, sum_x);
        CirCuit_Fields_Base::field_sub(sum_y, this_x);        
        QBigInt::bigint_mont_mul_assign_no_unroll(lambda, sum_y, sum_y);        
        QBigInt::add_ptx(this_y, sum_y, sum_y);
        CirCuit_Fields_Base::field_reduce(sum_y);

        CirCuit_Fields_Base::field_neg(sum_y);
        if ( !is_const ) 
            svm_push_leaf_full_with_convert(SVM_PARAM, sum_y);
    }

#if 0
    static ALEO_ADI void montgomery_add(SVM_PARAM_CDEF, bigint_u256_t this_x, bigint_u256_t this_y, 
                    bigint_u256_t that_x, bigint_u256_t that_y, bigint_u256_t &sum_x, bigint_u256_t &sum_y)
    {
        bigint_u256_t lambda; //也用做临时变量

        CirCuit_Fields_Base::field_sub(that_y, this_y);
        lambda = that_x;
        CirCuit_Fields_Base::field_sub(lambda, this_x);
        CirCuit_Fields_Base::field_div(that_y, lambda);
        
        //lambda is that_y 
        //todo: witeness lambda
        lambda = that_y;
        if ( !is_const ) 
            svm_push_leaf_full_with_convert(SVM_PARAM, lambda);

        bigint_mont_mul_assign(lambda, lambda, sum_x);            
        bigint_mont_mul_assign(HashBhp256::coeff_b_bigint, sum_x, sum_x);
        CirCuit_Fields_Base::field_sub(sum_x, HashBhp256::coeff_a_bigint);
        CirCuit_Fields_Base::field_sub(sum_x, this_x);
        CirCuit_Fields_Base::field_sub(sum_x, that_x);
        if ( !is_const ) 
            svm_push_leaf_full_with_convert(SVM_PARAM, sum_x);
        //CirCuit_Fields_Base::field_reduce(sum_x);    

        sum_y = sum_x;     
        CirCuit_Fields_Base::field_sub(sum_y, this_x);
        bigint_mont_mul_assign(lambda, sum_y, sum_y);    
        bigint_add_carry_ptx(this_y, sum_y, sum_y);
        CirCuit_Fields_Base::field_reduce(sum_y);
        
        lambda = sum_y;
        CirCuit_Fields::operator_neg(lambda, sum_y);    
        if ( !is_const ) 
            svm_push_leaf_full_with_convert(SVM_PARAM, sum_y);
    }
#endif    

    static ALEO_ADI uint32_t hash_bhp256_plaintext_to_input(SVM_PARAM_DEF, nv_uint128 src, uint32_t bits_len, bool is_signed, bigint_u256_t &dest)
    {
        uint32_t bit_offset = 0;        
        union {
            uint64_t src_bits_len;
            uint8_t u8_src_bits_len[8];
        };
        //这里hack 了下， 因为可以提前预判 出 bhp256 ： hash_uncompressed 中inputs的 head固定是 26 bit
        src_bits_len = bits_len + 26; 

        /* 参考: snarkVM-miner/circuit/algorithms/src/bhp/hash_uncompressed.rs
            U64::constant(console::U64::new(input.len() as u64)).write_bits_le(&mut preimage);
        */
        for (int n = 0; n < 8; n++)
        {
            bit_offset = append_to_field(&dest, bit_offset, u8_src_bits_len[n], 8, 0);
        }
                
        // 固定设置2 bit false 长度
        bit_offset += 2 ;

        LiteralVariantAsU8 variant = circuit_literal_var_type_from_bits_signed(bits_len, is_signed);
        //固定设置 8 bit
        bit_offset = append_to_field(&dest, bit_offset, variant, 8, 0);

        union 
        {
            uint16_t b_v16;
            uint8_t b_v8[2];
        };
        b_v16 = bits_len;
        //固定设置 16 bit
        bit_offset = append_to_field(&dest, bit_offset, b_v8[0], 8, 0); // low bit
        bit_offset = append_to_field(&dest, bit_offset, b_v8[1], 8, 0); // high bit
        for (uint32_t n = 0; n < bits_len/8; n++)
        {
            bit_offset = append_to_field(&dest, bit_offset,  src.v8[n], 8, 0);
        }

        //固定长度的bit :  2 + 8 + 16 = 26 bits

        return bit_offset;
    }

    static ALEO_ADI void get_bhp256_chunk_3bits(uint32_t current_bit_offset, 
            uint64_t preimage_bits[QBigInt::BIGINT_LEN], uint32_t preimage_bits_len, 
            bool bits[BHP_CHUNK_SIZE], uint8_t &pad)
    {
        //每次 获取 3个bit
        if ( current_bit_offset <= 185 )
        {
            //从 domain 里获取            
            for ( int n = 0; n < BHP_CHUNK_SIZE ; n++)
            {                                
                uint32_t byte_offset = (current_bit_offset + n ) / 8;
                uint32_t bit_offset = (current_bit_offset + n ) % 8;
                uint32_t data = hash_bhp256_domain.uint8[byte_offset];
                data = data >> bit_offset;
                bits[n] = data & 1;
            }
        } else if ( current_bit_offset == 186 ) {
            for ( int n = 0; n < 2 ; n++)
            {                                
                uint32_t byte_offset = (current_bit_offset + n ) / 8;
                uint32_t bit_offset = (current_bit_offset + n ) % 8;
                uint32_t data = hash_bhp256_domain.uint8[byte_offset];
                data = data >> bit_offset;
                bits[n] = data & 1;
            }
            
            bits[2] = QBigInt::get_u8_by_offset(preimage_bits, 0) & 1;
        } else if ( current_bit_offset >= DOMAIN_BITS_LEN )
        {
            uint32_t this_bits_offset = current_bit_offset - DOMAIN_BITS_LEN ;
            if ( this_bits_offset + 3 <= preimage_bits_len )
            {
                for ( int n = 0; n < BHP_CHUNK_SIZE ; n++)
                {                                
                    uint32_t byte_offset = (this_bits_offset + n ) / 8;
                    uint32_t bit_offset = (this_bits_offset + n ) % 8;
                    uint32_t data = QBigInt::get_u8_by_offset(preimage_bits, byte_offset);
                    data = data >> bit_offset;
                    bits[n] = data & 1;
                }
            } else {
                //最后一轮了
                if ( this_bits_offset < preimage_bits_len )
                {
                    pad = (this_bits_offset + 3) - preimage_bits_len;                    
                    if ( pad == 1 )
                    {
                        uint32_t byte_offset = this_bits_offset / 8;
                        uint32_t bit_offset = this_bits_offset % 8;
                        uint32_t data = QBigInt::get_u8_by_offset(preimage_bits, byte_offset);
                        data = data >> bit_offset;

                        bits[0] = data & 1;

                        byte_offset = (this_bits_offset + 1 ) / 8;
                        bit_offset = ( this_bits_offset + 1 ) % 8;
                        data = QBigInt::get_u8_by_offset(preimage_bits, byte_offset);
                        data = data >> bit_offset;
                        bits[1] = (data >> 1 ) & 1;
                        bits[2] = 0;
                    } else if (pad == 2 )
                    {
                        uint32_t byte_offset = this_bits_offset / 8;
                        uint32_t bit_offset = this_bits_offset % 8;
                        uint32_t data = QBigInt::get_u8_by_offset(preimage_bits, byte_offset);
                        data = data >> bit_offset;

                        bits[0] = data & 1;
                        bits[1] = 0;
                        bits[2] = 0;
                    }

                } else {
                    //printf("bug\n");
                }
            }
        }        
    }

    /*
    
    //513
    let num_hasher_bits = NUM_WINDOWS as usize * WINDOW_SIZE as usize * BHP_CHUNK_SIZE;

    // 252
    let num_data_bits = E::BaseField::size_in_data_bits();
    
    // 261
    let max_input_bits_per_iteration = num_hasher_bits - num_data_bits;    
    */

   /*
    最多 733 个leaf ， 可以一次处理
   */

static ALEO_ADI void exec_hash(SVM_PARAM_DEF, nv_uint128 src, uint32_t gl_bits, bool gl_is_signed)
{
    
    //printf("enter %s, src= %016lx %016lx\n", __func__, (uint64_t)src.stdint, (uint64_t)(src.stdint>>64)  );
    
    bigint_u256_t preimage_bits = { .uint8 = { 0 } };
    uint32_t preimage_bits_len = hash_bhp256_plaintext_to_input(SVM_PARAM, src, gl_bits, gl_is_signed, preimage_bits);    

    //printf("preimage_bits_len=%u\n", preimage_bits_len);
    // dump_bigint_bits(HashBhp256::domain, DOMAIN_BITS_LEN);
    // dump_bigint_bits(preimage_bits, preimage_bits_len);

    /*
    preimage bits 总长度 ( little endian )：

    188 个 domain bits , constant
    64 个 input.len bits , constant
    26 个 input 里的 constant bits

    上面合计 278 个 constant bits

    ？？ 个 input的 non-contant bits    

    BHP_CHUNK_SIZE = 3 

    需要把总长度 pad 到 mod 3 == 0 
    */

   /*
   优化策略： 
   因为 domain 是 const 不变， 长度为 188 ， 188 / 3 = 62 
   从1 开始计数， 1-62 轮循环的结果一定是 常量 

   1-62轮 ：

   1-57  为 round_0_level = 0 的情况 ， 对应  base[0] 里的 57个元素
   58-62 为 round_0_level = 1 的情况,   对应  base[1] 里的 0-4 共计 5个元素

   62*3 = 186 bit ， 那么我们的代码从 total preimage 的 186 bit 开始处理 ( 0 为 start index )

   此刻， sum  =
   314011997016217491554346690112167059081139680159083443805149840132711096692field.constant 8025620320679017648328436315455122655603211872854395469218010714716188307992field.constant
   对应field:
    bigint256=0x1d026138db8709f3, 0x4c58de9deb86d218, 0x02242fef3627fb02, 0x0f9c0a30dcbea5ad,
    bigint256=0x6f7cce58a524a11f, 0x7c6e81b70ff3d3cc, 0x1e67bf01e84f6f83, 0x057c1066a4a48ebf,   
   */

   //开始 hash_uncompressed
    

    //printf("total_preimage_len=%u preimage_bits_len=%u\n", preimage_bits_len+DOMAIN_BITS_LEN, preimage_bits_len);

    int iter_cnt = 1;
    uint32_t round_lvl_1 = 0; //每 171 个 bit 为 round_lv_1  , 可能不足 171    
    uint32_t round_lvl_2 = 0; // 如果 round_lvl_1 有 171 个bit ，那么 round_lvl_2 循环 171/3=57次， 否则 N bits/  3

    bigint_u256_t sum_x_of_92;
    bigint_u256_t sum_y_of_92;

    {
        uint32_t v = Circuit_Integer::get_trailing_zero(gl_bits) - Circuit_Integer::get_trailing_zero(8);
        
        uint64_t *p = &( bhp256_pre_caculate_sum[2*v + (uint32_t)gl_is_signed][0] );

        for ( int n = 0; n < 4; n++)
        {
            sum_x_of_92.uint64[n] = p[n];
        }

        for ( int n = 0; n < 4; n++)
        {
            sum_y_of_92.uint64[n] = p[n+4];
        }

    }

    smem_save_circuit_params(0, sum_x_of_92); //save: sum .x 
    smem_save_circuit_params(1, sum_y_of_92); //save: sum .y
    smem_save_circuit_params(2, preimage_bits);
        
    /* 
    188+64+26 = 278
    
    278/3 = 92 ， 从 93 轮开始

    92*3 = 276 , 从276 bit开始
    */
    uint32_t meta = 276 | (preimage_bits_len<<16);
    meta = (gl_is_signed<<31) | meta ;
    smem_set_circuit_meta(meta); //低 16 bit 存放 current bit offset  , 高 16bit 存放 total bit_offset

}

    /*
    返回： 
    true : 需要执行fold
    false ： no fold 
    */
static ALEO_ADI bool exec_hash_iter(SVM_PARAM_DEF)
{
    constexpr uint32_t init_iter_start = 63;

    uint64_t sum_x[QBigInt::BIGINT_LEN];
    uint64_t sum_y[QBigInt::BIGINT_LEN];
    uint64_t preimage_bits[QBigInt::BIGINT_LEN];

    smem_load_circuit_params(0, sum_x); 
    smem_load_circuit_params(1, sum_y); 
    smem_load_circuit_params(2, preimage_bits);
    
    uint32_t meta;
    meta = smem_get_circuit_meta();
    
    uint32_t current_offset =  meta & UINT16_MAX;
    uint32_t preimage_bits_len = (meta >> 16)  & 0x7FFF;    
    bool gl_is_signed = (meta>>31) & 1;

    uint32_t total_preimage_len = preimage_bits_len + DOMAIN_BITS_LEN;

    bool bits[3];
    //for(; current_offset < total_preimage_len ; iter_cnt++) 
    {
        
        uint32_t iter_cnt = (current_offset / BHP_CHUNK_SIZE) + 1;
        uint8_t pad = 0;
        get_bhp256_chunk_3bits(current_offset, preimage_bits, preimage_bits_len, bits, pad);        
        current_offset += BHP_CHUNK_SIZE;
        {
            uint32_t round_lvl_0 = current_offset / HASH_CHUNK_SIZE;
            uint32_t round_lvl_1 = ( current_offset % HASH_CHUNK_SIZE ) / BHP_CHUNK_SIZE;
            #if 0
            printf("lvl_0=%u lvl_1=%u current_offset=%u itercnt=%u bits=%u %u %u , pad=%u\n", 
                    round_lvl_0, round_lvl_1, current_offset, iter_cnt, bits[0], bits[1], bits[2], pad);
            #endif
        }
        
        // QBigInt::dump("sum_x=", sum_x);
        // QBigInt::dump("sum_y=", sum_y);

        bool bit_0_and_1 = bits[0] & bits[1];
        if ( iter_cnt > 93 && pad < 2 ) {
            //做隔离见证
            svm_push_leaf_one_bit_bool(SVM_PARAM, bit_0_and_1);
        }

        // domain 长度为 188 bit ，对应 186/3 = 62个完整的 BHP_CHUNK_SIZE        

        uint64_t montgomery_x[QBigInt::BIGINT_LEN];
        uint64_t montgomery_y[QBigInt::BIGINT_LEN];
        QBigInt::copy(montgomery_x, hash_bhp256_bases[(iter_cnt-init_iter_start)*8].uint64);
        QBigInt::copy(montgomery_y, hash_bhp256_bases[(iter_cnt-init_iter_start)*8+4].uint64);        

        #if 1
        if ( bits[0] ) 
        {
            QBigInt::add_ptx(montgomery_x, hash_bhp256_bases[(iter_cnt-init_iter_start)*8 + 1].uint64, montgomery_x);
            CirCuit_Fields_Base::field_reduce(montgomery_x);            
        }
        if ( bits[1] ) 
        {
            QBigInt::add_ptx(montgomery_x, hash_bhp256_bases[(iter_cnt-init_iter_start)*8 + 2].uint64, montgomery_x);
            CirCuit_Fields_Base::field_reduce(montgomery_x);
        }
        if ( bit_0_and_1 )
        {
            QBigInt::add_ptx(montgomery_x, hash_bhp256_bases[(iter_cnt-init_iter_start)*8 + 3].uint64, montgomery_x);
            CirCuit_Fields_Base::field_reduce(montgomery_x);
        }
        
        if ( bits[0] ) 
        {
            
            QBigInt::add_ptx(montgomery_y, hash_bhp256_bases[(iter_cnt-init_iter_start)*8 + 5].uint64, montgomery_y);
            CirCuit_Fields_Base::field_reduce(montgomery_y);            
        }
        if ( bits[1] )
        {
            QBigInt::add_ptx(montgomery_y, hash_bhp256_bases[(iter_cnt-init_iter_start)*8 + 6].uint64, montgomery_y);                
            CirCuit_Fields_Base::field_reduce(montgomery_y);            
        }
        if ( bit_0_and_1 )
        {
            QBigInt::add_ptx(montgomery_y, hash_bhp256_bases[(iter_cnt-init_iter_start)*8 + 7].uint64, montgomery_y);        
            CirCuit_Fields_Base::field_reduce(montgomery_y);
        }

        CirCuit_Fields_Base::field_reduce(montgomery_x);
        CirCuit_Fields_Base::field_reduce(montgomery_y);
        #endif 
        if ( bits[2] ) {
            CirCuit_Fields_Base::field_neg(montgomery_y);            
        }            
        
        uint8_t is_const;
        if ( iter_cnt < 93 ) {
            //都是constant bit ，不需要隔离见证
            is_const = 1;
        } else {
            //开始做隔离见证
            is_const = 0;
            svm_push_leaf_full_with_convert(SVM_PARAM, montgomery_y);
        }

        // QBigInt::dump("mx step2=", montgomery_x);
        // QBigInt::dump("my step2=", montgomery_y);
                
        if (iter_cnt % 57 == 1 )
        {
            QBigInt::copy(sum_x, montgomery_x);
            QBigInt::copy(sum_y, montgomery_y);            
        } else  {
            uint64_t new_sum_x[QBigInt::BIGINT_LEN], new_sum_y[QBigInt::BIGINT_LEN];
            montgomery_add_qbigint(SVM_PARAM_C, sum_x, sum_y, montgomery_x, montgomery_y, new_sum_x, new_sum_y);
            QBigInt::copy(sum_x, new_sum_x);
            QBigInt::copy(sum_y, new_sum_y);
        }

        // QBigInt::dump("sum_x 2=", sum_x);
        // QBigInt::dump("sum_y 2=", sum_y);

        smem_save_circuit_params(0, sum_x); //save: sum .x 
        smem_save_circuit_params(1, sum_y); //save: sum .y        
        meta = current_offset | (preimage_bits_len<<16);
        meta = (gl_is_signed<<31) | meta;
        smem_set_circuit_meta(meta);

        if ( (iter_cnt % 57 == 0 ) || current_offset >= total_preimage_len )
        {
            // 内层迭代完毕
            uint64_t edwards_x[QBigInt::BIGINT_LEN] ,  edwards_y[QBigInt::BIGINT_LEN];
            
            #if 1
            field_div_unchecked(sum_x, sum_y, edwards_x);
            
            uint64_t tmp1[QBigInt::BIGINT_LEN];
            uint64_t tmp2[QBigInt::BIGINT_LEN];
            QBigInt::copy(tmp1, sum_x);
            QBigInt::copy(tmp2, CirCuit_Fields_Base::get_field_one().uint64 );

            CirCuit_Fields_Base::field_sub_assign(tmp1, tmp2);                                            
            QBigInt::add_ptx(tmp2, sum_x, tmp2);

            field_div_unchecked(tmp1, tmp2, edwards_y);

            #else
            
            QBigInt::copy(edwards_x, sum_x);
            QBigInt::copy(edwards_y, sum_y);
            #endif 

            svm_push_leaf_full_with_convert(SVM_PARAM, edwards_x);
            svm_push_leaf_full_with_convert(SVM_PARAM, edwards_y);

            //share memory不够，存入 global memory
            for(int n = 0; n < 8; n++) {
                svm_mtree_push_soa_leaf_u32_order_desc(SVM_PARAM, 2, n, QBigInt::get_by_u32(edwards_x, n));
                svm_mtree_push_soa_leaf_u32_order_desc(SVM_PARAM, 3, n, QBigInt::get_by_u32(edwards_y, n));
            }
            
            return true ;       
        } else {
            return false;
        }


    }    

}    

    
/*
返回值：

false : total finish ,  不需要继续了
true  ： 还需要继续

*/
static ALEO_ADI bool exec_hash_fold(SVM_PARAM_DEF)
{
    uint64_t sum_x[QBigInt::BIGINT_LEN] , sum_y[QBigInt::BIGINT_LEN];
    smem_load_circuit_params(0, sum_x); 
    smem_load_circuit_params(1, sum_y); 

    uint32_t meta;
    meta = smem_get_circuit_meta();
    
    uint32_t current_offset =  meta & UINT16_MAX;
    uint32_t preimage_bits_len = (meta >> 16)  & 0x7FFF;    
    bool gl_is_signed = (meta>>31) & 1;

    uint32_t total_preimage_len = preimage_bits_len + DOMAIN_BITS_LEN;
    
    uint64_t edwards_x[QBigInt::BIGINT_LEN], edwards_y[QBigInt::BIGINT_LEN];
    for(int n = 0; n < 8; n++) {
        QBigInt::set_by_u32(edwards_x, svm_mtree_pop_soa_leaf_u32_order_desc(SVM_PARAM, 2, n), n);
        QBigInt::set_by_u32(edwards_y, svm_mtree_pop_soa_leaf_u32_order_desc(SVM_PARAM, 3, n), n);
    }

    uint32_t iter_cnt = (current_offset / BHP_CHUNK_SIZE);
    //printf("%s , iter_cnt=%u\n", __func__, iter_cnt);

    if ( iter_cnt  <= 114  && iter_cnt > 57 )
    {
        //printf("iter round 0 finish\n");
        /* 
        第二轮循环结束 , 第一轮的 acc 值是 fix 值，不会变
        */
        uint64_t acc_x[QBigInt::BIGINT_LEN] = { 
            0x75edf658a7261bafULL, 0x7c6ed5348ebdb5a1ULL, 0x6b300c9c55a9fa4fULL, 0x5bad2122e0d253ULL
        };                
        uint64_t acc_y[QBigInt::BIGINT_LEN] = { 
            0x92a4c6417ae20c00ULL,  0x44d61dc214482308ULL, 0x464989b0403a7ca5ULL, 0x10ecc03549dfeaa8ULL
        }; 
        
        /*
        对应: snarkVM-miner/circuit/algorithms/src/bhp/hasher/hash_uncompressed.rs 里的
            acc + group
        */
        
        // 此刻 iter_cnt = 114 , 57*2 , acc is const , edwards is not const
        uint8_t is_const = 0b01;  
        Circuit_Groups::add(SVM_PARAM_C, acc_x, acc_y, edwards_x, edwards_y);

        //share memory 不足，save在global memory里
        if ( current_offset < total_preimage_len )
        {                    
            for(int i = 0; i<8; i++)
                svm_mtree_push_soa_leaf_u32_order_desc(SVM_PARAM, 0, i,  QBigInt::get_by_u32(acc_x, i) );
            for(int i = 0; i<8; i++)
                svm_mtree_push_soa_leaf_u32_order_desc(SVM_PARAM, 1, i,  QBigInt::get_by_u32(acc_y, i) );
        } else {
            //printf("iter round all finish 1\n");
            //总迭代结束
            //cast 只需要 acc_x 
            smem_save_circuit_params(0, acc_x);
            return false;
        }           
    } else {
        // 此刻 iter_cnt = 57*3 , acc/edwards are not const
        
        //printf("iter round 0 finish\n");

        uint64_t acc_x[QBigInt::BIGINT_LEN], acc_y[QBigInt::BIGINT_LEN];
        for(int i = 0; i<8; i++)
        {
            QBigInt::set_by_u32(acc_x, svm_mtree_pop_soa_leaf_u32_order_desc(SVM_PARAM, 0, i), i);
            QBigInt::set_by_u32(acc_y, svm_mtree_pop_soa_leaf_u32_order_desc(SVM_PARAM, 1, i), i);
        }                    

        uint8_t is_const = 0b00;  
        Circuit_Groups::add(SVM_PARAM_C, acc_x, acc_y, edwards_x, edwards_y);                

        //printf("iter round all finish 2\n");
        //总迭代结束
        //cast 只需要 acc_x 
        smem_save_circuit_params(0, acc_x);
        //smem_save_circuit_params(1, acc_y);
        return false;
    }

    return true;
}
    
    static ALEO_ADI void exec_hash_cast_lossy(SVM_PARAM_DEF)
    {
        uint64_t digest_x[QBigInt::BIGINT_LEN];
        smem_load_circuit_params(0, digest_x);
        //smem_load_circuit_params(1, digest_y);

        CirCuit_Fields::field_to_bigint_with_witness(SVM_PARAM, 0, digest_x, digest_x);
        // bigint_from_field(&digest_x);

        smem_save_circuit_params(0, digest_x);

    }
};


#undef cudbg
#undef cuerr