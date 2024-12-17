
#pragma once 
#include <cstdint>
#include "cu_common.h"
#include "cu_aleo_globals.cuh"
#include "cu_synthesis.cuh"
#include "cu_smem.cuh"


//必须和rust 的 cuinterface.rs 里的顺序一致
enum VmOperator {
    OPCODE_Neg = 0,
    OPCODE_AbsWrapped,
    OPCODE_Not,
    OPCODE_Xor,
    OPCODE_Nor,
    OPCODE_Nand,
    OPCODE_Or,
    OPCODE_And,
    OPCODE_IsEq,
    OPCODE_IsNotEq,
    OPCODE_Add,
    OPCODE_AddWrapped,
    OPCODE_SubWrapped,
    OPCODE_HashPsd2,
    OPCODE_HashBhp256,
    OPCODE_HashPed64,
    OPCODE_Mul,
    OPCODE_MulWrapped,
    OPCODE_Lt,
    OPCODE_Lte,
    OPCODE_Gt,
    OPCODE_Gte,
    OPCODE_CastLossy,
    OPCODE_ShlWrapped,
    OPCODE_ShrWrapped,
    OPCODE_Ternary,
    OPCODE_Square,
    OPCODE_Pow,
    OPCODE_PowWrapped,
    OPCODE_DivWrapped,
    OPCODE_RemWrapped,
    OPCODE_Modulo,
    OPCODE_Div,
    OPCODE_Inv,
    OPCODE_MAX_NUM
};


struct VmOperand {
    uint8_t size;
    uint8_t op_type; //寄存器或者常量
    uint8_t var_type; //变量类型: int , field
    uint8_t pad[5];
    uint8_t data[];
};

//8个字节
struct VmInstruction {
    uint32_t size;     
    uint8_t operator_id;
    uint8_t operands_num;
    uint8_t as_type_id;
    uint8_t dest; //目的地址寄存器编号
    uint8_t operands[];
} ;

#define SYNTHESIS_VM_MAX_OPERANDS (3)


__device__ void program_instruction_to_str(VmInstruction *instruction, char *buf, size_t buf_len);

/*
返回指向 constant 区域的 指针，不能修改
*/
static ALEO_ADI VmInstruction *program_fetch_one_instruction_from_const(SVM_PARAM_DEF)
{
    
    VmInstruction *instruction = nullptr;
    uint32_t program_index = svm_get_program_index(SVM_PARAM);
        
    if ( program_index >= d_program_size)
    {        
        return nullptr;
    }

    instruction = (VmInstruction*)(&d_program_bin[program_index]);

    program_index += instruction->size;    
    svm_set_program_index(SVM_PARAM, program_index);
    
    return instruction;
    
}


static __device__ __forceinline__ void program_fetch_instruction_operands(VmInstruction *instruction, 
            VmOperand *operands[SYNTHESIS_VM_MAX_OPERANDS])
{
    if ( instruction->operands_num > SYNTHESIS_VM_MAX_OPERANDS || instruction->operands_num < 1 ) {
        return;
    }

    //最多3个操作数        
        
    uint8_t *operand_start = (uint8_t*)(instruction+1);
    size_t offset = 0;
    for (int n = 0; n < instruction->operands_num && n < SYNTHESIS_VM_MAX_OPERANDS; n++ ) {
        operands[n] = (VmOperand*)(operand_start + offset); 
        offset += operands[n]->size;
    }
    
}        