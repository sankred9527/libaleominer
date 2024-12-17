#pragma once
#include <type_traits>
#include "cuda/cu_mtree_leaf.cuh"
#include "cuda/circuit/circuit_common.h"
#include "cuda/circuit/circuit_type.cuh"
#include "circuit_field_base.cuh"

class CirCuit_Fields
{
public:

    static ALEO_ADI void operator_is_eq(SVM_PARAM_CDEF,
                    uint64_t operand_0[QBigInt::BIGINT_LEN], 
                    uint64_t operand_1[QBigInt::BIGINT_LEN],
                    bool &ret
                    )
    {
        operator_is_not_eq(SVM_PARAM_C, operand_0, operand_1, ret);
        ret = !ret;
    }

    static ALEO_ADI void operator_is_not_eq(SVM_PARAM_CDEF,
                    uint64_t operand_0[QBigInt::BIGINT_LEN], 
                    uint64_t operand_1[QBigInt::BIGINT_LEN],
                    bool &ret
                    )
    {
        ret = (QBigInt::compare(operand_0, operand_1) != 0);

        svm_push_leaf_one_bit_bool(SVM_PARAM, ret);

        {
            CirCuit_Fields_Base::field_neg(operand_1);
            CirCuit_Fields_Base::field_add_assign(operand_0, operand_1);
        }

        if ( bigint_inverse_for_quick_bigint(operand_0) )
        {
            //QBigInt::bigint_from_field(operand_0);
            svm_push_leaf_full_with_convert_quick_bigint(SVM_PARAM, operand_0);
            //svm_push_leaf_full(SVM_PARAM, operand_0);
        } else {
            svm_push_leaf_one_bit_bool(SVM_PARAM, true);
        }
    }

    static ALEO_ADI void  operator_ternary(SVM_PARAM_CDEF, 
                    bool condition, uint64_t src[QBigInt::BIGINT_LEN], 
                    uint64_t other[QBigInt::BIGINT_LEN], uint64_t dest[QBigInt::BIGINT_LEN])
    {
        if ( condition ) {
            QBigInt::copy(dest, src);
        } else {
            QBigInt::copy(dest, other);
        }
        svm_push_leaf_full_with_convert(SVM_PARAM, dest);
    }

    static ALEO_ADI void  operator_ternary(SVM_PARAM_CDEF, 
                    bool condition, bigint_u256_t src, bigint_u256_t other, bigint_u256_t &dest)
    {
        dest = condition? src:other;
        svm_push_leaf_full_with_convert(SVM_PARAM, dest);
    }

    static ALEO_ADI  void  operator_div(SVM_PARAM_CDEF, 
                uint64_t src[QBigInt::BIGINT_LEN], 
                uint64_t other[QBigInt::BIGINT_LEN], 
                uint64_t dest[QBigInt::BIGINT_LEN]
                )
    {        
        if ( bigint_inverse_for_quick_bigint(other) ) {
            svm_push_leaf_full_with_convert(SVM_PARAM, other);

            operator_mul(SVM_PARAM_C, src, other, dest);
        }
        else {
            //printf("field inv failed\n");
        }
    }  

    static ALEO_ADI  void  operator_mul(SVM_PARAM_CDEF, 
                bigint_u256_t &src, bigint_u256_t &other, bigint_u256_t &dest)
    {        
        bigint_mont_mul_assign_no_unroll(src, other , dest);
        //bigint_mont_mul_assign(src, other , dest);
        if ( SVM_SAVE ) {
            svm_push_leaf_full_with_convert(SVM_PARAM, dest);
        }
    }

    static ALEO_ADI  void  operator_mul(SVM_PARAM_CDEF, 
                uint64_t src[QBigInt::BIGINT_LEN], uint64_t other[QBigInt::BIGINT_LEN], uint64_t dest[QBigInt::BIGINT_LEN])
    {
        QBigInt::bigint_mont_mul_assign_no_unroll(src, other, dest);     
        if ( SVM_SAVE ) {
            svm_push_leaf_full_with_convert_quick_bigint(SVM_PARAM, dest);
        }
    }

    static ALEO_ADI void operator_cast_lossy(SVM_PARAM_CDEF, uint64_t src[QBigInt::BIGINT_LEN], nv_uint128 &dest)
    {
        field_to_bigint_with_witness(SVM_PARAM_C, src, src);
        dest.stdint = src[1];
        dest.stdint = (dest.stdint << 64) | src[0];
    }
    
    static ALEO_ADI void operator_cast_lossy(SVM_PARAM_CDEF, bigint_u256_t src, nv_uint128 &dest)
    {
        /*
        cast.lossy r12 into r14 as u8;
        cast.lossy r12 into r15 as i8;
        cast.lossy r12 into r16 as u16;
        cast.lossy r12 into r17 as i16;
        cast.lossy r13 into r18 as i32;
        cast.lossy r13 into r19 as u32;
        cast.lossy r13 into r20 as u64;
        cast.lossy r13 into r21 as i128;

        is.eq r2 r15 into r22;
        is.eq r10 r21 into r23;
        */
        field_to_bigint_with_witness(SVM_PARAM_C, &src, &src);
        dest.stdint = src.uint128[0];
    }


////////////////
    static ALEO_ADI void operator_less_than_prepare(SVM_PARAM_CDEF, 
                        bigint_u256_t &first, bigint_u256_t &other)
    {
        field_to_bigint_with_witness(SVM_PARAM_C, &first, &first);

        smem_save_circuit_params(0, first);
        smem_save_circuit_params(1, other);
        
    }

    static ALEO_ADI void operator_less_than_prepare(SVM_PARAM_CDEF, 
                        uint64_t first[QBigInt::BIGINT_LEN],
                        uint64_t other[QBigInt::BIGINT_LEN])
    {
        field_to_bigint_with_witness(SVM_PARAM_C, first, first);

        smem_save_circuit_params(0, first);
        smem_save_circuit_params(1, other);
        
    }

    static ALEO_ADI void operator_less_than_prepare_2(SVM_PARAM_CDEF)
    {    
        bigint_u256_t other;
        smem_load_circuit_params(1, other);
        field_to_bigint_with_witness(SVM_PARAM_C, &other, &other);
        smem_save_circuit_params(1, other);
    }

    static ALEO_ADI void operator_less_than_run(SVM_PARAM_CDEF, bool &dest)
    {
        bigint_u256_t first, other;

        smem_load_circuit_params(0, first);
        smem_load_circuit_params(1, other);

        /*
        一共循环 BLS12_377_FR_PARAMETER_MODULUS_BITS = 253 次， 每次写入 2个 one bit leaf
        
        */
        bool is_less_than = false;        
        
        union 
        {
            uint16_t v16s[32];
            __uint128_t v128s[4];
        };

        for (int n = 0 ; n < 4 ; n++ )
        {
            v128s[n] = 0;
        }
        
        //32次迭代
        uint32_t index = 0;
        for(int n = 0; n < sizeof(bigint_u256_t); n++ )
        {
            uint8_t p1 = BIGINT256_GET_U8(&first,n);
            uint8_t p2 = BIGINT256_GET_U8(&other,n);

            uint16_t v16 = 0;
            for( int m = 0; m < 8; m++ ) {
                if (index++ == BLS12_377_FR_PARAMETER_MODULUS_BITS )
                    break;

                bool bit1 = (p1 >> m) & 1;
                bool bit2 = (p2 >> m) & 1;

                bool condition = bit1 ^ bit2 ;
                is_less_than = condition? bit2: is_less_than;
                                                
                uint16_t tmp = is_less_than;
                tmp = (tmp<<1) | condition;

                v16 =  (tmp << (m*2)) | v16;
            }
            v16s[n] = v16;
        }
        /*
        总共 save 253*2 = 506 个 leaf
        */
        for ( int n = 0; n < 4; n++)
        {
            uint32_t cnt;            
            cnt = (n==3)?122:128;
            svm_push_leaf_bit_with_u128(SVM_PARAM, v128s[n], cnt);
        }        

        dest = is_less_than;    
    }
////////////////


////////////////

    static ALEO_ADI void field_to_bigint_with_witness(SVM_PARAM_CDEF, 
                uint64_t data[QBigInt::BIGINT_LEN], uint64_t dest[QBigInt::BIGINT_LEN])
    {
        QBigInt::copy(dest, data);
        QBigInt::bigint_from_field(dest);
        if ( SVM_SAVE ) {            
            svm_push_leaf_bit_with_bls12_377_bigint(SVM_PARAM, dest);
        }
        
        if ( is_const )
            return; 

        uint64_t modulus_minus_one[QBigInt::BIGINT_LEN];
        CirCuit_Fields_Base::get_field_neg_one(modulus_minus_one);

        QBigInt::bigint_from_field(modulus_minus_one);

        // 执行rust 逻辑 :  is_less_than_or_equal_constant
        //一共比较 BLS12_377_FR_PARAMETER_MODULUS_BITS 个bit  

        uint64_t tmp_cache[QBigInt::BIGINT_LEN];
        bool rest_is_less = false;            
        for ( int n = 0; n < sizeof(bigint_u256_t); n++ )
        {
            //bit of var 
            uint8_t byte_var = QBigInt::get_u8_by_offset(dest, n);
            //byte of constant
            uint8_t byte_const = QBigInt::get_u8_by_offset(modulus_minus_one, n);

            uint8_t tmp_u8 = 0;
            for ( int m = 0; m < 8; m++ ) 
            {
                uint8_t index = n*8 + m;
                if ( index >= BLS12_377_FR_PARAMETER_MODULUS_BITS)
                    break;

                bool bit_var = (byte_var>>m) & 1;
                bool bit_const = (byte_const>>m) & 1;
                if ( bit_const ) {
                    //bit and
                    rest_is_less = bit_var & rest_is_less;
                } else {
                    //bit or
                    rest_is_less = bit_var | rest_is_less;
                }

                //save to witness
                if ( SVM_SAVE )
                {
                    tmp_u8 = ( ((uint8_t)rest_is_less) << m) | tmp_u8;
                }
            }
            QBigInt::set_by_u8(tmp_cache, tmp_u8, n);
        }
        //第一个是 rest_is_less 是const， 不存入 witness            
        QBigInt::div2(tmp_cache);

        // 一共252 个 bit ， BLS12_377_FR_PARAMETER_MODULUS_BITS - 1 = 252
        svm_push_leaf_bit_with_u64(SVM_PARAM, tmp_cache[0]);
        svm_push_leaf_bit_with_u64(SVM_PARAM, tmp_cache[1]);
        svm_push_leaf_bit_with_u64(SVM_PARAM, tmp_cache[2]);
        svm_push_leaf_bit_with_u64(SVM_PARAM, tmp_cache[3], 60);
    }


    static ALEO_ADI void field_to_bigint_with_witness(SVM_PARAM_CDEF, bigint_u256_t *data, bigint_u256_t *dest)
    {
        *dest = *data;

        bigint_from_field(dest);
            
        if ( SVM_SAVE ) {            
            svm_push_leaf_bit_with_bls12_377_bigint(SVM_PARAM, *dest);
        }
        
        if ( is_const )
            return;    

        bigint_u256_t modulus_minus_one = CirCuit_Fields_Base::get_field_neg_one();
        bigint_from_field(&modulus_minus_one);
        

        // 执行rust 逻辑 :  is_less_than_or_equal_constant
        //一共比较 BLS12_377_FR_PARAMETER_MODULUS_BITS 个bit  
        bigint_u256_t tmp_cache;
        bool rest_is_less = false;            
        for ( int n = 0; n < sizeof(bigint_u256_t); n++ )
        {
            //bit of var 
            uint8_t byte_var = BIGINT256_GET_U8(dest, n); 
            //byte of constant
            uint8_t byte_const = BIGINT256_GET_U8(&modulus_minus_one, n);

            uint8_t tmp_u8 = 0;
            for ( int m = 0; m < 8; m++ ) 
            {
                uint8_t index = n*8 + m;
                if ( index >= BLS12_377_FR_PARAMETER_MODULUS_BITS)
                    break;

                bool bit_var = (byte_var>>m) & 1;
                bool bit_const = (byte_const>>m) & 1;
                if ( bit_const ) {
                    //bit and
                    rest_is_less = bit_var & rest_is_less;
                } else {
                    //bit or
                    rest_is_less = bit_var | rest_is_less;
                }

                //save to witness
                if ( SVM_SAVE )
                {
                    tmp_u8 = ( ((uint8_t)rest_is_less) << m) | tmp_u8;
                }
            }
            tmp_cache.uint8[n] = tmp_u8;
        }
        #if 1
        //第一个是 rest_is_less 是const， 不存入 witness            
        bigint256_div2(&tmp_cache);
        
        svm_push_leaf_bit_with_u128(SVM_PARAM, tmp_cache.uint128[0]);
        svm_push_leaf_bit_with_u128(SVM_PARAM, tmp_cache.uint128[1], 124);        
        
        #endif 
    }

    static ALEO_ADI void operator_neg(uint64_t self[QBigInt::BIGINT_LEN], uint64_t dest[QBigInt::BIGINT_LEN])
    {
    /*
    neg r12 into r14;
    is.eq r13 r14 into r15;

    add r12 r14 into r16;
    is.eq r13 r16 into r17;

    */    
        if ( QBigInt::is_zero(self) ) 
        {
            dest = self;
            return;
        }        
        QBigInt::init_from_bigint(dest, BLS12_377_FR_PARAMETER_MODULUS);
        QBigInt::sub_ptx(dest, self);
    }


    static ALEO_ADI void operator_pow_iter_once(SVM_PARAM_CDEF, uint32_t n, uint32_t m, 
                bigint_u256_t self, bigint_u256_t exponent, bigint_u256_t &output)
    {
        bool k = ( BIGINT256_GET_U8(&exponent, n) >> m) & 1 ;
        
        
        CirCuit_Fields::operator_mul(SVM_PARAM_C, output, output, output);
        

        bigint_u256_t tmp = { .bytes = {0} } ;
        CirCuit_Fields::operator_mul(SVM_PARAM_C, output, self, tmp);
        
        is_const = 0;
        CirCuit_Fields::operator_ternary(SVM_PARAM_C, k, tmp, output, output);
    }

    static ALEO_ADI void operator_pow_prepare(SVM_PARAM_CDEF,
                bigint_u256_t self, 
                bigint_u256_t exponent)
    {
        bigint_u256_t exponent_raw;
        field_to_bigint_with_witness(SVM_PARAM, 0, &exponent, &exponent_raw);        

        smem_save_circuit_params(0, self);
        smem_save_circuit_params(1, exponent_raw);

        constexpr uint32_t skip = sizeof(bigint_u256_t)*8 - BLS12_377_FR_PARAMETER_MODULUS_BITS;
        bigint_u256_t output = CirCuit_Fields_Base::get_field_one();                
        uint32_t n = sizeof(bigint_u256_t) -1 ;
        uint32_t m = 7 - skip;        

        is_const = 1;
        operator_pow_iter_once(SVM_PARAM_C, n, m, self, exponent_raw, output);
        smem_save_circuit_params(2, output);
        
        m--;
        uint32_t meta = (n <<16)  | m;
        smem_set_circuit_meta(meta);
    }
                
    static ALEO_ADI bool operator_pow_run(SVM_PARAM_CDEF)
    {
        bigint_u256_t self , exponent, output;

        smem_load_circuit_params(0, self);
        smem_load_circuit_params(1, exponent);
        smem_load_circuit_params(2, output);

        uint32_t n, m, meta;
        meta = smem_get_circuit_meta();
        n = meta >> 16;
        m = meta & 0xFFFF;        
        
        /*
            每次迭代最多 3个 full hash， 2轮一共 6个
            缓冲区长度为 8 
        */
        for(uint32_t loop = 0 ; loop < 2; loop++)
        {
            operator_pow_iter_once(SVM_PARAM_C, n, m, self, exponent, output);
            smem_save_circuit_params(2, output);

            if ( m == 0 ) 
            {
                if ( n == 0 )
                    return true;
                n--;
                m = 7;
            } else
                m--;    
        }

        meta = (n <<16) | m;
        smem_set_circuit_meta(meta);        
        
        return false;        
    } 


    static ALEO_ADI void operator_add(SVM_PARAM_CDEF, 
                    uint64_t src[QBigInt::BIGINT_LEN],
                    uint64_t other[QBigInt::BIGINT_LEN],
                    uint64_t dest[QBigInt::BIGINT_LEN]
                )
    {
        QBigInt::add_ptx(src, other, dest);
    }   

    static ALEO_ADI void operator_inv(SVM_PARAM_CDEF, uint64_t self[QBigInt::BIGINT_LEN])
    {
        if ( bigint_inverse_for_quick_bigint(self) ) {
            svm_push_leaf_full_with_convert(SVM_PARAM, self);
        }
        else {
            //printf("field inv failed\n");
        }
    }

};