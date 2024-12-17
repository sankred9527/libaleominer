
#include "cu_mtree_public_hash.cuh"
#include "libdevcore/ChaChaCommon.h"
#include "cu_solution.cuh"
#include "cu_chacha20rng.cuh"
#include "cu_synthesis_reg.cuh"

__device__ uint64_t svm_generate_constant_from_rand(SVM_PARAM_DEF, uint64_t counter)
{

    //参考： rand_core-0.6.4/src/block.rs
    /* 生成算法：
        boolean ：  (rng.next_u32() as i32) < 0
        I8:  rng.next_u32() as u8
        I16: rng.next_u32() as u16
        I32: rng.next_u32()
        I64: rng.next_u64()
        I128:         
            let x = u128::from(rng.next_u64());
            let y = u128::from(rng.next_u64());
            (y << 64) | x
        Field: 参靠snarkVM源码里的 macro_rules! impl_primefield_standard_sample 
    */

    chacha_state_t m_chacha_rng;    
    

    /*
        一共 14个 input ，参考 snarkVM-miner/ledger/puzzle/epoch/src/synthesis/helpers/register_table.rs 里的 input_register_types的初始化值        
    */
    uint64_t solution_id = Solution::solution_new(counter);
    chacha_init_from_u64(&m_chacha_rng, solution_id);

    /*
    第一个public hash 是 true 
    同时生成 501 个public hash
    */
    svm_push_leaf_one_bit_bool(SVM_PARAM, true);
    
    for(int n = 0; n<2; n++) 
    {
        uint32_t v = CuChaCha20Rng::chacha_rand_next_u32(&m_chacha_rng);
        bool to_set =  ((int32_t)v)<0;
        svm_push_reg_implement_new(SVM_PARAM, to_set);
        
        svm_push_leaf_one_bit_bool(SVM_PARAM, to_set );
        
    }

    for(int n = 0; n<2; n++) 
    {
        uint32_t v = CuChaCha20Rng::chacha_rand_next_u32(&m_chacha_rng);
        char1 to_set;
        to_set.x = v & 0xff;
        svm_push_reg_implement_new(SVM_PARAM, to_set);        

        svm_push_leaf_bit_with_u8(SVM_PARAM, to_set.x);
        
    }


    for(int n = 0; n<2; n++) 
    {
        uint32_t v = CuChaCha20Rng::chacha_rand_next_u32(&m_chacha_rng);
        short1 to_set;
        to_set.x = v & 0xffff;
        svm_push_reg_implement_new(SVM_PARAM, to_set);

        svm_push_leaf_bit_with_u16(SVM_PARAM, to_set.x);
    }

    for(int n = 0; n<2; n++)
    {
        int1 to_set;
        to_set.x = CuChaCha20Rng::chacha_rand_next_u32(&m_chacha_rng);
        svm_push_reg_implement_new(SVM_PARAM, to_set);
        svm_push_leaf_bit_with_u32(SVM_PARAM, to_set.x);
    }

    for(int n = 0; n<2; n++)
    {
        uint64_t raw = CuChaCha20Rng::chacha_rand_next_u64(&m_chacha_rng);
        int2 v ;
        v.x = raw & 0xffffffff;
        v.y = raw >> 32;
        svm_push_reg_implement_new(SVM_PARAM, v);
        
        svm_push_leaf_bit_with_u64(SVM_PARAM, raw);
    }

    for(int n = 0; n<2; n++)
    {
        bigint_u128_t raw = CuChaCha20Rng::chacha_rand_next_u128(&m_chacha_rng);

        svm_push_reg_implement_new(SVM_PARAM, raw.int4s[0]);

        //printf("leaf u128=%08x %08x %08x %08x\n", raw.uint4s[0].x , raw.uint4s[0].y , raw.uint4s[0].z , raw.uint4s[0].w   );
        svm_push_leaf_bit_with_u128(SVM_PARAM, raw.uint128[0]);
    }

    for(int n = 0; n < 2; n++)    
        while (1)
        {
            bigint_u256_t tmp_field;
            for(int m = 0;  m < 4; m++)
            {
                tmp_field.uint64[m] = CuChaCha20Rng::chacha_rand_next_u64(&m_chacha_rng);
            }

            tmp_field.uint64[3] &= ((uint64_t)-1) >> BLS12_377_FR_PARAMETER_REPR_SHAVE_BITS ;        
            if ( aleo_u256_compare_less_than(&tmp_field, &BLS12_377_FR_PARAMETER_MODULUS) )
            {   
                //bigint256_dump_with_prefix("save field=", "", &tmp_field);
                //svm_push_reg_implement(SVM_PARAM, CT_FIELD, tmp_field);
                svm_push_reg_implement_new(SVM_PARAM, tmp_field);

                bigint_from_field(&tmp_field);
                svm_push_leaf_full(SVM_PARAM, tmp_field);
                break;
            } else {
                //printf("not valid\n");
            }       
        }


    return solution_id;
}