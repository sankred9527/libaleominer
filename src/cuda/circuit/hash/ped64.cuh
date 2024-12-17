#pragma once
#include "cuda/cu_bigint.cuh"
#include "cuda/cu_synthesis.cuh"
#include "circuit_hash_help.cuh"
#include "cuda/cu_smem.cuh"
#include "cuda/circuit/field/circuit_field.cuh"
#include "cuda/circuit/groups/circuit_groups.cuh"
#include "cuda/circuit/integer/circuit_integer.cuh"
#include "ped64_base.cuh"

class HashPed64
{
public:
    static constexpr uint32_t NUM_BITS = 64;

    /*
        why 26?   hash_ped64_plaintext_to_input:  2 + 8 + 16
    */
    static constexpr uint32_t const_bits_len = 26;

    static ALEO_ADI uint32_t hash_ped64_plaintext_to_input(SVM_PARAM_DEF, nv_uint128 src, uint32_t bits_len, bool is_signed, uint64_t dest[QBigInt::BIGINT_LEN])
    {
        uint32_t bit_offset = 0;

        // fixed 2 bit false length
        bit_offset += 2 ;

        LiteralVariantAsU8 variant = circuit_literal_var_type_from_bits_signed(bits_len, is_signed);        
        //fixed  8 bit length
        bit_offset = append_to_field_for_qbigint(dest, bit_offset, variant, 8, 0);
        union 
        {
            uint16_t b_v16;
            uint8_t b_v8[2];
        };
        b_v16 = bits_len;
        //fixed 16 bit length
        bit_offset = append_to_field_for_qbigint(dest, bit_offset, b_v8[0], 8, 0); // low bit
        bit_offset = append_to_field_for_qbigint(dest, bit_offset, b_v8[1], 8, 0); // high bit

        for (uint32_t n = 0; n < bits_len/8; n++)
        {
            bit_offset = append_to_field_for_qbigint(dest, bit_offset,  src.v8[n], 8, 0);
        }
        return bit_offset;
    }

    static ALEO_ADI void prepare(SVM_PARAM_DEF, nv_uint128 src, uint32_t bits_len, bool is_signed)
    {        
        uint64_t input_bits[QBigInt::BIGINT_LEN] = { 0 };
        hash_ped64_plaintext_to_input(SVM_PARAM, src, bits_len, is_signed, input_bits);

        //26 bit + n bits + pad bits 
        uint32_t meta = (bits_len << 16 ) | is_signed;
        smem_set_circuit_meta(meta);

        uint32_t iter_cnt = 0;
        uint64_t acc_x[QBigInt::BIGINT_LEN] = {0} , acc_y[QBigInt::BIGINT_LEN] = {0};

        #if 1
        {
            uint32_t v = Circuit_Integer::get_trailing_zero(bits_len) - Circuit_Integer::get_trailing_zero(8);
            
            uint64_t *p = &( ped64_pre_caculate[2*v + (uint32_t)is_signed][0] );

            for ( int n = 0; n < 4; n++)
            {
                acc_x[n] = p[n];
            }

            for ( int n = 0; n < 4; n++)
            {
                acc_y[n] = p[n+4];
            }

            iter_cnt = const_bits_len;
            input_bits[0] = input_bits[0] >> iter_cnt;

        } 
        #else
        {
            //前面26次，不需要save  private hash
            for( iter_cnt = 0; iter_cnt < const_bits_len; iter_cnt++)
            {
                __iter_internal(SVM_PARAM, iter_cnt, input_bits[0], bits_len, acc_x, acc_y);
                input_bits[0] = input_bits[0] >> 1;

                // printf("prepare iter=%u \n", iter_cnt);            
            }
        }
        #endif

        // QBigInt::dump("acc_x=", acc_x);
        // QBigInt::dump("acc_y=", acc_y); 
        // printf("input=%016lx\n", input_bits[0]);       


        // max input_bits len of ped64 <=  64 bit , only need input_bits[0]
        smem_save_circuit_params_by_u32(0, 0, input_bits[0]);
        smem_save_circuit_params_by_u32(0, 1, input_bits[0]>>32);
        smem_save_circuit_params_by_u32(0, 2, iter_cnt); //iter count

        smem_save_circuit_params(1, acc_x);
        smem_save_circuit_params(2, acc_y);

        
    }

    static ALEO_ADI void __iter_internal(SVM_PARAM_DEF, uint32_t iter_cnt, 
                    uint64_t input_v64, uint32_t bits_len, 
                    uint64_t acc_x[QBigInt::BIGINT_LEN], uint64_t acc_y[QBigInt::BIGINT_LEN]                    
                    )
    {        
        bool condition = input_v64 & 1;
        
        uint64_t group_x[QBigInt::BIGINT_LEN];
        uint64_t group_y[QBigInt::BIGINT_LEN];
        if ( condition )
        {
            QBigInt::copy(group_x, ped64_base[iter_cnt*2].uint64);
            QBigInt::copy(group_y, ped64_base[iter_cnt*2+1].uint64);            
        } else {
            //this is Group::zero() in rust        
            group_x[0] = group_x[1] = group_x[2] = group_x[3] = 0;            
            CirCuit_Fields_Base::set_field_one(group_y);
        }
        //如果 condition_is_const = true ， 那么  group_x/y 为  constant
        uint8_t is_const = 0;
        {            

            // is_const 的 0 bit 存放 acc 是否为const ， bool
            if ( iter_cnt <= const_bits_len ) {
                //acc is constant
                is_const = 1;
            }

            /*
            开始判断 group_x/y 是否为 const
            */
            uint8_t condition_is_const;
            if ( iter_cnt < const_bits_len || iter_cnt >= (const_bits_len+bits_len) )
                condition_is_const = 1;
            else
                condition_is_const = 0;
            
            is_const = (condition_is_const << 1) | is_const;
        }
        
        Circuit_Groups::add(SVM_PARAM, is_const, acc_x, acc_y, group_x, group_y);
    }

    static ALEO_ADI bool iter(SVM_PARAM_DEF) 
    {
        uint32_t iter_cnt = smem_load_circuit_params_by_u32(0, 2);
        union { 
            uint32_t input_v32[2];
            uint64_t input_v64;
        };
        uint64_t acc_x[QBigInt::BIGINT_LEN], acc_y[QBigInt::BIGINT_LEN];
    
        input_v32[0] = smem_load_circuit_params_by_u32(0, 0);
        input_v32[1] = smem_load_circuit_params_by_u32(0, 1);
        smem_load_circuit_params(1, acc_x);
        smem_load_circuit_params(2, acc_y);

        uint32_t bits_len = smem_get_circuit_meta() >> 16;                
        
        __iter_internal(SVM_PARAM, iter_cnt, input_v64, bits_len, acc_x, acc_y);

        iter_cnt++;
        input_v64 = input_v64 >> 1;
        
        if ( iter_cnt == (const_bits_len+bits_len) )
        {
            for (; iter_cnt < NUM_BITS; iter_cnt++)
            {
                __iter_internal(SVM_PARAM, iter_cnt, input_v64, bits_len, acc_x, acc_y);
            }
            CirCuit_Fields::field_to_bigint_with_witness(SVM_PARAM, 0, acc_x, acc_x);
            smem_save_circuit_params(0, acc_x);
            //should finish
            return true;
        } else {
        
            smem_save_circuit_params(1, acc_x);
            smem_save_circuit_params(2, acc_y);        

            smem_save_circuit_params_by_u32(0, 0, input_v32[0]);
            smem_save_circuit_params_by_u32(0, 1, input_v32[1]);
            
            smem_save_circuit_params_by_u32(0, 2, iter_cnt);
            return false;
        }        
    }
};