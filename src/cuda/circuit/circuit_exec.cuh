#pragma once
#include <cstdint>
#include "cuda/cu_program.cuh"

__device__ inline static uint64_t operand_data_to_u64(void* data)
{
    uint64_t *p = (uint64_t*)data;
    return *p;
}

ALEO_ADI static CircuitTypes __circuit_get_operand_type(SVM_PARAM_DEF, VmOperand *operands)
{
    CircuitTypes ct;
    if ( operands[0].op_type == Op_Register) {
        uint8_t reg_index = (uint8_t)operand_data_to_u64(operands[0].data);
                
        uint32_t value;
        svm_load_register_raw(SVM_PARAM, reg_index, ct, value);        
    }
    else {        
        //常量
        ct = (CircuitTypes)operands[0].var_type;
    }
    return ct;
}

template <typename T>
ALEO_ADI static T __circuit_get_param(SVM_PARAM_DEF, VmOperand *operands[], uint8_t op_num, uint8_t &is_const)
{
    T to_reg;
#if 0
    if ( operands[op_num]->op_type == Op_Register) {
        uint8_t reg_index = (uint8_t)operand_data_to_u64(operands[op_num]->data);
        
        CircuitTypes ct;
        uint32_t value;
        svm_load_register_raw(SVM_PARAM, reg_index, ct, value);
        to_reg = svm_load_reg_implement<T>(SVM_PARAM, value);
    }
    else {
        
        //常量
        memcpy( &(to_reg), operands[op_num]->data, sizeof(T) );

        if (op_num == 0 )
            is_const = 1;
        else if ( op_num == 1) {
            is_const = 0b10 | is_const;
        } else if ( op_num == 2) {
            is_const = 0b100 | is_const;
        }
    }
#endif 
    return to_reg;
}



/*
目前的 epoch hash里 ， hash.psd2是一个很特殊的指令，进在程序开始 连续出现 10次 单独优化下
  
前面 5条 if.eq 产生 11个 private hash

每条 hash.psd2 产生 1280 个 private hash , 其中约 270个 full hash

    hash.psd2 r12 into r20 as u8;
    hash.psd2 r13 into r21 as u8;

    hash.psd2 r12 into r22 as u16;
    hash.psd2 r13 into r23 as u16;

    hash.psd2 r12 into r24 as u32;
    hash.psd2 r13 into r25 as u32;

    hash.psd2 r12 into r26 as u64;
    hash.psd2 r13 into r27 as u64;

    hash.psd2 r12 into r28 as u128;
    hash.psd2 r13 into r29 as u128;
    
*/


