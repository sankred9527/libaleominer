#pragma once
#include "cuda/cu_synthesis_reg.cuh"
#include "cuda/cu_mtree_leaf.cuh"
#include "hash/poseidon.cuh"
#include "hash/ped64.cuh"
#include "hash/bhp256.cuh"
#include "circuit_exec.cuh"
#include "circuit_common.h"

#include "cuda/cu_program.cuh"
#include "cuda/circuit/integer/circuit_integer.cuh"
#include "cuda/circuit/boolean/circuit_bool.cuh"
#include "cuda/circuit/field/circuit_field.cuh"
#include "cuda/circuit/hash/poseidon.cuh"

#define DEFAULT_SMEM_LEAVES_LEN 7 

#define FAST_COMPILE 0

#if 0
#define cudbg(...) do { if (idx==0) printf(__VA_ARGS__); } while(0)
#define cuerr(...) do { if (idx==0) printf(__VA_ARGS__); } while(0)
#else
#define cudbg(...) do {}while(0)
#define cuerr(...) do {}while(0)
#endif


#define CIRCUIT_RUN_PARAM_DEF VmInstruction *instruction, CirCuit_Exec_Step &exec_step
#define CIRCUIT_RUN_PARAM  instruction, exec_step

#define MK_REG_1_by_type(n, operands, type_name, is_const) \
    type_name reg1;\
    if ( operands[n]->op_type == Op_Register) {\
        uint8_t reg_index = (uint8_t)operand_data_to_u64(operands[n]->data);\
        CircuitTypes ct;\
        uint32_t value;\
        svm_load_register_raw(SVM_PARAM, reg_index, ct, value);\
        uint8_t bits = circuit_type_to_bits_len(ct);\
        reg1 = svm_load_reg_implement<type_name>(SVM_PARAM, value, bits);\
    } else {\
        /*常量*/ \
        memset(&(reg1), 0 , sizeof(type_name));\
        memcpy( &(reg1), operands[n]->data, operands[n]->size - sizeof(struct VmOperand));\
        is_const = (1<<n) | is_const; \
    }

#define MK_REG_1_for_bigint(n, operands, is_const) \
    uint64_t reg1[QBigInt::BIGINT_LEN] = { 0 };\
    if ( operands[n]->op_type == Op_Register) {\
        uint8_t reg_index = (uint8_t)operand_data_to_u64(operands[n]->data);\
        CircuitTypes ct;\
        uint32_t value;\
        svm_load_register_raw(SVM_PARAM, reg_index, ct, value);\
        svm_load_reg_implement_for_qbigint(SVM_PARAM, reg1, value);\
    } else {\
        /*常量*/ \
        uint64_t *p = (uint64_t*)(operands[n]->data);\
        reg1[0] = p[0];reg1[1] = p[1];reg1[2] = p[2];reg1[3] = p[3];\
        is_const = (1<<n) | is_const; \
    }


#define __EXEC_INSTRUCTION_WITH_1_REG(instruction, expand_code ) do{\
    VmOperand *operands[SYNTHESIS_VM_MAX_OPERANDS] = { nullptr };\
    program_fetch_instruction_operands(instruction, operands);\
    uint8_t is_const = 0;\
    if ( operands[0]->op_type == Op_Register) {\
        uint8_t reg_index = (uint8_t)operand_data_to_u64(operands[0]->data);\
        CircuitTypes ct;\
        uint32_t value;\
        svm_load_register_raw(SVM_PARAM, reg_index, ct, value);\
        if ( ct == CT_BOOLEAN )\
        {\
            bool reg0 = svm_load_reg_implement_for_bool(SVM_PARAM, value);\
            uint32_t bits = 1; bool is_signed = false;\
            expand_code;\
        } else if ( ct == CT_FIELD )\
        {\
            bigint_u256_t reg0 = svm_load_reg_implement_for_bigint256(SVM_PARAM, value);\
            uint32_t bits = 256; bool is_signed = false;\
            expand_code;\
        } else {\
            uint32_t bits = circuit_type_to_bits_len(ct);\
            bool is_signed = circuit_type_is_signed_int(ct);\
            nv_uint128 reg0 = svm_load_reg_implement_for_int(SVM_PARAM, value, bits);\
            expand_code;\
        }\
    } else {\
        /*not support now */\
    }\
} while (0)


#define QUICK_EXEC_INSTRUCTION_WITH_1_REG(instruction, expand_code ) do{\
    VmOperand *operands[SYNTHESIS_VM_MAX_OPERANDS] = { nullptr };\
    program_fetch_instruction_operands(instruction, operands);\
    uint8_t is_const = 0;\
    if ( operands[0]->op_type == Op_Register) {\
        uint8_t reg_index = (uint8_t)operand_data_to_u64(operands[0]->data);\
        CircuitTypes ct;\
        uint32_t value;\
        svm_load_register_raw(SVM_PARAM, reg_index, ct, value);\
        if ( ct == CT_BOOLEAN )\
        {\
            bool reg0 = svm_load_reg_implement_for_bool(SVM_PARAM, value);\
            uint32_t bits = 1; bool is_signed = false;\
            expand_code;\
        } else if ( ct == CT_FIELD )\
        {\
            uint64_t reg0[QBigInt::BIGINT_LEN];\
            svm_load_reg_implement_for_qbigint(SVM_PARAM, reg0, value);\
            uint32_t bits = 256; bool is_signed = false;\
            expand_code;\
        } else {\
            uint32_t bits = circuit_type_to_bits_len(ct);\
            bool is_signed = circuit_type_is_signed_int(ct);\
            nv_uint128 reg0 = svm_load_reg_implement_for_int(SVM_PARAM, value, bits);\
            expand_code;\
        }\
    } else {\
        /*not support now */\
    }\
} while (0)

#define QUICK_EXEC_INSTRUCTION_WITH_OPERAND_AND_2_REG(instruction, operands, start_operands_index, expand_code ) do{\
    uint8_t is_const = 0;\
    if ( operands[start_operands_index]->op_type == Op_Register) {\
        uint8_t reg_index = (uint8_t)operand_data_to_u64(operands[start_operands_index]->data);\
        CircuitTypes ct;\
        uint32_t value;\
        svm_load_register_raw(SVM_PARAM, reg_index, ct, value);\
        if ( ct == CT_BOOLEAN )\
        {\
            bool reg0 = svm_load_reg_implement_for_bool(SVM_PARAM, value);\
            MK_REG_1_by_type( (start_operands_index+1), operands, bool, is_const)\
            uint32_t bits = 1; bool is_signed = false;\
            expand_code;\
        } else if ( ct == CT_FIELD )\
        {\
            uint64_t reg0[QBigInt::BIGINT_LEN] = {0};\
            svm_load_reg_implement_for_qbigint(SVM_PARAM, reg0, value);\
            MK_REG_1_for_bigint( (start_operands_index+1), operands, is_const)\
            uint32_t bits = 256; bool is_signed = false; \
            expand_code;\
        } else {\
            uint32_t bits = circuit_type_to_bits_len(ct);\
            bool is_signed = circuit_type_is_signed_int(ct);\
            nv_uint128 reg0 = svm_load_reg_implement_for_int(SVM_PARAM, value, bits);\
            MK_REG_1_by_type( (start_operands_index+1), operands, nv_uint128, is_const)\
            expand_code;\
        }\
    } else {\
        /*not support now */\
    }\
} while (0)

#define QUICK_EXEC_INSTRUCTION_WITH_2_REG(instruction, expand_code ) do{\
    VmOperand *operands[SYNTHESIS_VM_MAX_OPERANDS] = { nullptr };\
    program_fetch_instruction_operands(instruction, operands);\
    QUICK_EXEC_INSTRUCTION_WITH_OPERAND_AND_2_REG(instruction, operands, 0, expand_code);\
} while (0)

#define svm_push_reg_as_variant_int(nvint) __svm_push_reg_as_variant_int(SVM_PARAM, nvint, bits, is_signed)

static ALEO_ADI void __svm_push_reg_as_variant_int(SVM_PARAM_DEF, nv_uint128 nvint , uint8_t bits, bool is_signed)
{
    switch (bits>>3)
    {
        case 1:
            if ( is_signed ) {
                char1 dest;
                dest.x = nvint.stdint & 0xff;
                svm_push_reg_implement_new(SVM_PARAM, dest);
            } else {
                uchar1 dest;
                dest.x = nvint.stdint & 0xff;
                svm_push_reg_implement_new(SVM_PARAM, dest);
            }
            break;
        case 2:
            if ( is_signed ) {
                short1 dest;
                dest.x = nvint.stdint & UINT16_MAX;
                svm_push_reg_implement_new(SVM_PARAM, dest);
            } else {
                ushort1 dest;
                dest.x = nvint.stdint & UINT16_MAX;
                svm_push_reg_implement_new(SVM_PARAM, dest);
            }
            break;
        case 4:
            if ( is_signed ) {
                int1 dest;
                dest.x = nvint.nvint.x;
                svm_push_reg_implement_new(SVM_PARAM, dest);
            } else {
                uint1 dest;
                dest.x = nvint.nvint.x;
                svm_push_reg_implement_new(SVM_PARAM, dest);
            }
            break;
        case 8:
            if ( is_signed ) {
                int2 dest;
                dest.x = nvint.nvint.x;
                dest.y = nvint.nvint.y;
                svm_push_reg_implement_new(SVM_PARAM, dest);
            } else {
                uint2 dest;
                dest.x = nvint.nvint.x;
                dest.y = nvint.nvint.y;
                svm_push_reg_implement_new(SVM_PARAM, dest);
            }
            break;
        case 16:
            if ( is_signed ) {
                svm_push_reg_implement_new(SVM_PARAM, nvint.nvint_signed);
            } else {                
                svm_push_reg_implement_new(SVM_PARAM, nvint.nvint);
            }
            break;
        default:
            break;
    }
}

#if !FAST_COMPILE

template <typename T>
static ALEO_ADI void __run_OPCODE_Xor(SVM_PARAM_CDEF, T reg0, T reg1, uint8_t bits, bool is_signed)
{
    if constexpr (std::is_same_v<T, bool> ) {
        bool dest;
        Circuit_Bool::operator_xor(SVM_PARAM_C, reg0, reg1, dest);
        svm_push_reg_implement_new<bool>(SVM_PARAM, dest);        
    } 

    if constexpr (std::is_same_v<T, nv_uint128> ) {
        nv_uint128 dest;
        Circuit_Integer::operator_xor(SVM_PARAM_C, bits, is_signed, reg0, reg1, dest);
        svm_push_reg_as_variant_int(dest);
    }
    
}

static ALEO_ADI bool run_OPCODE_Xor(SVM_PARAM_DEF, CIRCUIT_RUN_PARAM_DEF)
{
/*

    xor r2 r3 into r14;
    xor r2 r2 into r15;
    xor r0 r1 into r16;
    xor r0 r0 into r17;
    is.eq r14 r15 into r18;
    is.eq r16 r17 into r19;
    xor r10 r11 into r20;
    is.eq r10 r20 into r21;
*/

    QUICK_EXEC_INSTRUCTION_WITH_2_REG(
        instruction, 
        __run_OPCODE_Xor(SVM_PARAM_C, reg0, reg1, bits, is_signed)
    );

    return true;        
}


template <typename T>
static ALEO_ADI void __run_OPCODE_Or(SVM_PARAM_CDEF, T reg0, T reg1, uint8_t bits, bool is_signed)
{
    if constexpr (std::is_same_v<T, bool> ) {
        bool dest;
        Circuit_Bool::operator_or(SVM_PARAM_C, reg0, reg1, dest);
        svm_push_reg_implement_new<bool>(SVM_PARAM, dest);        
    } 

    if constexpr (std::is_same_v<T, nv_uint128> ) {
        nv_uint128 dest;
        Circuit_Integer::operator_or(SVM_PARAM_C, bits, is_signed, reg0, reg1, dest);
        svm_push_reg_as_variant_int(dest);
    }
    
}

static ALEO_ADI bool run_OPCODE_Or(SVM_PARAM_DEF, CIRCUIT_RUN_PARAM_DEF)
{
/*
    or r2 r3 into r14;
    or r2 r2 into r15;
    or r0 r1 into r16;
    or r0 r0 into r17;
    is.eq r14 r15 into r18;
    is.eq r16 r17 into r19;
    or r10 r11 into r20;
    is.eq r10 r20 into r21;

*/

    QUICK_EXEC_INSTRUCTION_WITH_2_REG(
        instruction, 
        __run_OPCODE_Or(SVM_PARAM_C, reg0, reg1, bits, is_signed)
    );

    return true;        
}

template <typename T>
static ALEO_ADI void __run_OPCODE_Nor(SVM_PARAM_CDEF, T reg0, T reg1, uint8_t bits, bool is_signed)
{
    if constexpr (std::is_same_v<T, bool> ) {
        bool dest;
        Circuit_Bool::operator_nor(SVM_PARAM_C, reg0, reg1, dest);
        svm_push_reg_implement_new<bool>(SVM_PARAM, dest);        
    }
}

static ALEO_ADI bool run_OPCODE_Nor(SVM_PARAM_DEF, CIRCUIT_RUN_PARAM_DEF)
{
/*
    nor r0 r1 into r14;
    is.eq r2 r3 into r15;
    nor r14 r15 into r16;
    is.eq r14 r16 into r17;
*/    
    QUICK_EXEC_INSTRUCTION_WITH_2_REG(
        instruction, 
        __run_OPCODE_Nor(SVM_PARAM_C, reg0, reg1, bits, is_signed)
    );
    return true;        
}

template <typename T>
static ALEO_ADI void __run_OPCODE_Nand(SVM_PARAM_CDEF, T reg0, T reg1, uint8_t bits, bool is_signed)
{
    if constexpr (std::is_same_v<T, bool> ) {
        bool dest;
        Circuit_Bool::operator_nand(SVM_PARAM_C, reg0, reg1, dest);
        svm_push_reg_implement_new<bool>(SVM_PARAM, dest);        
    }
}

static ALEO_ADI bool run_OPCODE_Nand(SVM_PARAM_DEF, CIRCUIT_RUN_PARAM_DEF)
{
/*
    nand r0 r1 into r14;
    is.eq r2 r3 into r15;
    nand r14 r15 into r16;
    is.eq r14 r16 into r17;
*/  

    QUICK_EXEC_INSTRUCTION_WITH_2_REG(
        instruction, 
        __run_OPCODE_Nand(SVM_PARAM_C, reg0, reg1, bits, is_signed)
    );
    return true;        
}

template <typename T>
static ALEO_ADI void __run_OPCODE_And(SVM_PARAM_CDEF, T reg0, T reg1, uint8_t bits, bool is_signed)
{
    if constexpr (std::is_same_v<T, bool> ) {
        bool dest;
        Circuit_Bool::operator_and(SVM_PARAM_C, reg0, reg1, dest);
        svm_push_reg_implement_new<bool>(SVM_PARAM, dest);        
    }

    if constexpr (std::is_same_v<T, nv_uint128> ) {
        nv_uint128 dest;
        Circuit_Integer::operator_and(SVM_PARAM_C, bits, is_signed, reg0, reg1, dest);
        svm_push_reg_as_variant_int(dest);
    }
}

static ALEO_ADI bool run_OPCODE_And(SVM_PARAM_DEF, CIRCUIT_RUN_PARAM_DEF)
{
/*
    and r0 r1 into r14;
    is.eq r2 r3 into r15;
    and r10 r11 into r16;
    is.eq r11 r16 into r17;
*/  

    QUICK_EXEC_INSTRUCTION_WITH_2_REG(
        instruction, 
        __run_OPCODE_And(SVM_PARAM_C, reg0, reg1, bits, is_signed)
    );
    return true;        
}

/////////



static ALEO_ADI bool run_OPCODE_ShlWrapped(SVM_PARAM_DEF, CIRCUIT_RUN_PARAM_DEF)
{
    if ( exec_step == CEXEC_STEP_INIT )
    {    
        VmOperand *operands[SYNTHESIS_VM_MAX_OPERANDS] = { nullptr };
        program_fetch_instruction_operands(instruction, operands);
                                
        if ( operands[0]->op_type != Op_Register) 
            return true;
        
        uint8_t reg_index = (uint8_t)operand_data_to_u64(operands[0]->data);
        CircuitTypes ct;
        uint32_t value;
        svm_load_register_raw(SVM_PARAM, reg_index, ct, value);
        
        if ( ct == CT_BOOLEAN || ct == CT_FIELD )
            return true;
        
        uint32_t bits = circuit_type_to_bits_len(ct);
        bool is_signed = circuit_type_is_signed_int(ct);
        nv_uint128 reg0 = svm_load_reg_implement_for_int(SVM_PARAM, value, bits);

        reg_index = (uint8_t)operand_data_to_u64(operands[1]->data);
        svm_load_register_raw(SVM_PARAM, reg_index, ct, value);
        
        if ( ct == CT_BOOLEAN || ct == CT_FIELD )
            return true;

        uint32_t rhs_bits = circuit_type_to_bits_len(ct);
        nv_uint128 reg1 = svm_load_reg_implement_for_int(SVM_PARAM, value, rhs_bits);
        uint32_t rhs = reg1.stdint;
        uint8_t is_const = 0;
        
        nv_uint128 dest;
        RoundZeroMem::set_max_allow_full_leaves_in_smem(6);
        Circuit_Integer::operator_shl_wrap(SVM_PARAM_C, bits, is_signed, reg0, rhs, rhs_bits, dest);
        if ( bits < 128 )
        {
            svm_push_reg_as_variant_int(dest);
            return true;
        } else {
            exec_step = CEXEC_STEP_1;
            return false;
        }
    } else if ( exec_step == CEXEC_STEP_1 ) {
        RoundZeroMem::set_max_allow_full_leaves_in_smem(6);
        Circuit_Integer::operator_shl_wrap_final_128bit_step1(SVM_PARAM);
        exec_step = CEXEC_STEP_2;
        return false;
    } else if ( exec_step == CEXEC_STEP_2 ) {

        bool is_signed;
        uint32_t bits = 128;

        RoundZeroMem::set_max_allow_full_leaves_in_smem(3);
        Circuit_Integer::operator_shl_wrap_final_128bit_step2(SVM_PARAM);
        nv_uint128 dest;
        
        {
            bigint_u256_t tmp;
            smem_load_circuit_params(0, tmp);
            dest.stdint = tmp.uint128[0];
            is_signed = tmp.uint128[1];
        }
        
        //__svm_push_reg_as_variant_int(SVM_PARAM, dest, gl_bits, gl_is_signed);
        svm_push_reg_as_variant_int(dest);
        return true;
    }
    return true;
}


static ALEO_ADI bool run_OPCODE_ShrWrapped(SVM_PARAM_DEF, CIRCUIT_RUN_PARAM_DEF)
{
    if ( exec_step == CEXEC_STEP_INIT)
    {
        VmOperand *operands[SYNTHESIS_VM_MAX_OPERANDS] = { nullptr };
        program_fetch_instruction_operands(instruction, operands);

        if ( operands[0]->op_type != Op_Register) 
            return true;
        
        uint8_t reg_index = (uint8_t)operand_data_to_u64(operands[0]->data);
        CircuitTypes ct;
        uint32_t value;
        svm_load_register_raw(SVM_PARAM, reg_index, ct, value);
        
        if ( ct == CT_BOOLEAN || ct == CT_FIELD )
            return true;
        
        uint32_t bits = circuit_type_to_bits_len(ct);
        bool is_signed = circuit_type_is_signed_int(ct);
        nv_uint128 reg0 = svm_load_reg_implement_for_int(SVM_PARAM, value, bits);

        reg_index = (uint8_t)operand_data_to_u64(operands[1]->data);
        svm_load_register_raw(SVM_PARAM, reg_index, ct, value);

        if ( ct == CT_BOOLEAN || ct == CT_FIELD )
            return true;

        uint32_t rhs_bits = circuit_type_to_bits_len(ct);
        nv_uint128 reg1 = svm_load_reg_implement_for_int(SVM_PARAM, value, rhs_bits);

        exec_step = CEXEC_STEP_1;
        uint8_t is_const = 0;
        if ( bits == 128) {
            RoundZeroMem::set_max_allow_full_leaves_in_smem(6);
            // unsigned 直接结束， signed 还需要继续下一步 
            Circuit_Integer::operator_shr_wrap_bits_is_128_prepare(SVM_PARAM_C, bits, is_signed, reg0, reg1.stdint, rhs_bits);
            return false;
        }
        else {
            nv_uint128 dest;
            Circuit_Integer::operator_shr_wrap_bits_less_128(SVM_PARAM_C, bits, is_signed, reg0, reg1.nvint.x, rhs_bits, dest);
            svm_push_reg_as_variant_int(dest);
            return true;
        }
        
    } else if ( exec_step == CEXEC_STEP_1) {
        //only 128bit 
        RoundZeroMem::set_max_allow_full_leaves_in_smem(6);
        Circuit_Integer::operator_shr_wrap_bits_is_128_prepare_finish(SVM_PARAM);

        /*
            unsigned int 128bit :  return true, finish 
            signed : continue work
        */

        exec_step = CEXEC_STEP_2;
        return false;
        // if ( bits == 128 && is_signed) {
        //     exec_step = CEXEC_STEP_2;
        //     Circuit_Integer::operator_shr_wrap_bits_is_128_exec_step1(SVM_PARAM);
        //     return false;
        // }        
        // return true;
    } else if ( exec_step == CEXEC_STEP_2)
    {                
        RoundZeroMem::set_max_allow_full_leaves_in_smem(4);
        bool ret = Circuit_Integer::operator_shr_wrap_bits_is_128_exec_step1(SVM_PARAM);
        if ( ret )
        {
            //仅仅 unsigned 走这个分支
            constexpr uint32_t bits = 128;        
            constexpr bool is_signed = false;
            bigint_u256_t tmp;            
            smem_load_circuit_params(0, tmp);
            nv_uint128 dest;
            dest.stdint = tmp.uint128[0];                
            svm_push_reg_as_variant_int(dest);
        } else {
            exec_step = CEXEC_STEP_3;
        }
        return ret;                  
    } else  if ( exec_step == CEXEC_STEP_3)
    {
        RoundZeroMem::set_max_allow_full_leaves_in_smem(1);
        Circuit_Integer::operator_shr_wrap_bits_is_128_exec_step2(SVM_PARAM);

        bigint_u256_t tmp;
        smem_load_circuit_params(0, tmp);
        nv_uint128 dest;
        dest.stdint = tmp.uint128[0];

        uint32_t bits = 128;
        bool is_signed = true;
        svm_push_reg_as_variant_int(dest);    
        return true;
    }
    return true;
}


template <typename T>
static ALEO_ADI bool __run_OPCODE_DivWrapped(SVM_PARAM_CDEF, T reg0, T reg1, uint32_t bits, bool is_signed)
{
    if constexpr (std::is_same_v<T, nv_uint128> ) {
        nv_uint128 dest;

        if ( is_signed )
        {            
            Circuit_Integer::operator_div_wrap_signed_prepare(SVM_PARAM_C, bits, is_signed, reg0, reg1);
            return false;            
        } else {
            nv_uint128 dest;            
            Circuit_Integer::operator_div_wrap_unsigned(SVM_PARAM_C, bits, is_signed, reg0, reg1, dest);
            svm_push_reg_as_variant_int(dest);
            return true;
        }
    } else 
        return true;
}


static ALEO_ADI bool run_OPCODE_DivWrapped(SVM_PARAM_DEF, CIRCUIT_RUN_PARAM_DEF)
{
    RoundZeroMem::set_max_allow_full_leaves_in_smem(DEFAULT_SMEM_LEAVES_LEN);
    if ( exec_step == CEXEC_STEP_INIT) 
    {
        exec_step = CEXEC_STEP_1;

        QUICK_EXEC_INSTRUCTION_WITH_2_REG(
            instruction, 
            return __run_OPCODE_DivWrapped(SVM_PARAM_C, reg0, reg1, bits, is_signed)
        );
    } else if ( exec_step == CEXEC_STEP_1)  {
        //处理 signed 分支
        exec_step = CEXEC_STEP_2;
        return Circuit_Integer::operator_div_wrap_signed_exec_1(SVM_PARAM);
    } else if ( exec_step == CEXEC_STEP_2 ) {
        bool ret = Circuit_Integer::operator_div_wrap_signed_exec_2(SVM_PARAM);
        if ( ret )
        {
            bigint_u256_t tmp;
            smem_load_circuit_params(0, tmp);
            nv_uint128 dest;
            dest.stdint = tmp.uint128[0];
            uint32_t bits = smem_load_circuit_params_by_u32(1, 0);
            bool is_signed = true;
            svm_push_reg_as_variant_int(dest);
        }
        return ret;
    }

    return true;
}


template <typename T>
static ALEO_ADI bool __run_OPCODE_RemWrapped(SVM_PARAM_CDEF, T reg0, T reg1, uint32_t bits, bool is_signed)
{
    if constexpr (std::is_same_v<T, nv_uint128> ) {
        nv_uint128 dest;

        if ( is_signed )
        {
            Circuit_Integer::operator_rem_wrap_signed_prepare(SVM_PARAM_C, bits, is_signed, reg0, reg1);
            return false;            
        } else {
            nv_uint128 dest;            
            Circuit_Integer::operator_rem_wrap_unsigned(SVM_PARAM_C, bits, is_signed, reg0, reg1, dest);
            svm_push_reg_as_variant_int(dest);
            return true;
        }
    } else
        return true;
}


static ALEO_ADI bool run_OPCODE_RemWrapped(SVM_PARAM_DEF, CIRCUIT_RUN_PARAM_DEF)
{
    RoundZeroMem::set_max_allow_full_leaves_in_smem(DEFAULT_SMEM_LEAVES_LEN);
    if ( exec_step == CEXEC_STEP_INIT)     
    {
        exec_step = CEXEC_STEP_1;

        QUICK_EXEC_INSTRUCTION_WITH_2_REG(
            instruction, 
            return __run_OPCODE_RemWrapped(SVM_PARAM_C, reg0, reg1, bits, is_signed)
        );
    } else if ( exec_step == CEXEC_STEP_1)  {        
        //处理 signed 分支
        exec_step = CEXEC_STEP_2;
        Circuit_Integer::operator_rem_wrap_signed_exec_1(SVM_PARAM);

        bigint_u256_t tmp;
        smem_load_circuit_params(0, tmp);
        nv_uint128 dest;
        dest.stdint = tmp.uint128[0];
        uint32_t bits = smem_load_circuit_params_by_u32(1, 0);
        bool is_signed = true;
        svm_push_reg_as_variant_int(dest);
    }

    return true;
}

template <typename T>
static ALEO_ADI void __run_OPCODE_Div(SVM_PARAM_CDEF, T &reg0, T &reg1)
{
    if constexpr ( (std::is_array_v<T> &&     
            std::is_same_v<std::remove_extent_t<T>, uint64_t> && 
            std::extent_v<T> == 4) )      
    {
        uint64_t dest[QBigInt::BIGINT_LEN];
        CirCuit_Fields::operator_div(SVM_PARAM_C, reg0, reg1, dest);
        svm_push_reg_implement_new_for_qbigint(dest);
    }
}

static ALEO_ADI bool run_OPCODE_Div(SVM_PARAM_DEF, CIRCUIT_RUN_PARAM_DEF)
{    
    QUICK_EXEC_INSTRUCTION_WITH_2_REG(
        instruction, 
        __run_OPCODE_Div(SVM_PARAM_C, reg0, reg1)
    );
    return true;    
}


template <typename T>
static ALEO_ADI void __run_OPCODE_Mul(SVM_PARAM_CDEF, T &reg0, T &reg1)
{
    if constexpr ( (std::is_array_v<T> &&     
            std::is_same_v<std::remove_extent_t<T>, uint64_t> && 
            std::extent_v<T> == 4) )    
    {
        uint64_t dest[QBigInt::BIGINT_LEN];
        CirCuit_Fields::operator_mul(SVM_PARAM_C, reg0, reg1, dest);
        //svm_push_reg_implement_new(SVM_PARAM, dest);
        svm_push_reg_implement_new_for_qbigint(dest);
    }
}

static ALEO_ADI bool run_OPCODE_Mul(SVM_PARAM_DEF, CIRCUIT_RUN_PARAM_DEF)
{    
    QUICK_EXEC_INSTRUCTION_WITH_2_REG(
        instruction, 
        __run_OPCODE_Mul(SVM_PARAM_C, reg0, reg1)
    );
    return true;    
}


template <typename T>
static ALEO_ADI void __run_OPCODE_Square(SVM_PARAM_CDEF, T &reg0)
{    
    if constexpr ( (std::is_array_v<T> &&     
            std::is_same_v<std::remove_extent_t<T>, uint64_t> && 
            std::extent_v<T> == 4) )
    {
        uint64_t dest[QBigInt::BIGINT_LEN];
        CirCuit_Fields::operator_mul(SVM_PARAM_C, reg0, reg0, dest);
        //svm_push_reg_implement_new(SVM_PARAM, dest);
        svm_push_reg_implement_new_for_qbigint(dest);
    }
    
}

static ALEO_ADI bool run_OPCODE_Square(SVM_PARAM_DEF, CIRCUIT_RUN_PARAM_DEF)
{    
    QUICK_EXEC_INSTRUCTION_WITH_1_REG(instruction,
        __run_OPCODE_Square(SVM_PARAM_C, reg0)
    );    
    return true;    
}

static ALEO_ADI bool run_OPCODE_Pow(SVM_PARAM_DEF, CIRCUIT_RUN_PARAM_DEF)
{
    if ( exec_step == CEXEC_STEP_INIT)
    {
        RoundZeroMem::set_max_allow_full_leaves_in_smem(3);

        VmOperand *operands[SYNTHESIS_VM_MAX_OPERANDS] = { nullptr };
        program_fetch_instruction_operands(instruction, operands);
                    
        if ( operands[0]->op_type != Op_Register) 
            return true;
        
        uint8_t reg_index = (uint8_t)operand_data_to_u64(operands[0]->data);
        CircuitTypes ct;
        uint32_t value;
        svm_load_register_raw(SVM_PARAM, reg_index, ct, value);
        
        if ( ct != CT_FIELD )
            return true;
        bigint_u256_t reg0 = svm_load_reg_implement_for_bigint256(SVM_PARAM, value);
        
        reg_index = (uint8_t)operand_data_to_u64(operands[1]->data);
        svm_load_register_raw(SVM_PARAM, reg_index, ct, value);
        
        if ( ct != CT_FIELD )
            return true;
        bigint_u256_t reg1 = svm_load_reg_implement_for_bigint256(SVM_PARAM, value);
        
        uint8_t is_const = 0;                
        CirCuit_Fields::operator_pow_prepare(SVM_PARAM_C, reg0, reg1);

        exec_step = CEXEC_STEP_1;

        return false;
    } else {
        RoundZeroMem::set_max_allow_full_leaves_in_smem(6);

        uint8_t is_const = 0;
        
        bool ret = CirCuit_Fields::operator_pow_run(SVM_PARAM_C);
        if ( ret ) {
            bigint_u256_t dest;
            smem_load_circuit_params(2, dest);
            svm_push_reg_implement_new(SVM_PARAM, dest);
        }
        return ret;
    } 
}

template <typename T>
static ALEO_ADI void __run_OPCODE_AddWrapped(SVM_PARAM_CDEF, T reg0, T reg1, uint8_t bits, bool is_signed)
{
    if constexpr (std::is_same_v<T, nv_uint128> ) {
        nv_uint128 dest;
        Circuit_Integer::operator_add_wrap(SVM_PARAM_C, bits, is_signed, reg0, reg1, dest);
        svm_push_reg_as_variant_int(dest);        
    }
}

static ALEO_ADI bool run_OPCODE_AddWrapped(SVM_PARAM_DEF, CIRCUIT_RUN_PARAM_DEF)
{
    QUICK_EXEC_INSTRUCTION_WITH_2_REG(
        instruction, 
        __run_OPCODE_AddWrapped(SVM_PARAM_C, reg0, reg1, bits, is_signed)
    );
    return true;   
}

template <typename T>
static ALEO_ADI void __run_OPCODE_SubWrapped(SVM_PARAM_CDEF, T reg0, T reg1, uint8_t bits, bool is_signed)
{
    if constexpr (std::is_same_v<T, nv_uint128> ) {
        nv_uint128 dest;
        Circuit_Integer::operator_sub_wrap(SVM_PARAM_C, bits, is_signed, reg0, reg1, dest);
        svm_push_reg_as_variant_int(dest);        
    }
}

static ALEO_ADI bool run_OPCODE_SubWrapped(SVM_PARAM_DEF, CIRCUIT_RUN_PARAM_DEF)
{
    QUICK_EXEC_INSTRUCTION_WITH_2_REG(
        instruction, 
        __run_OPCODE_SubWrapped(SVM_PARAM_C, reg0, reg1, bits, is_signed)
    );
    return true;   
}

template <typename T>
static ALEO_ADI void __run_OPCODE_Add(SVM_PARAM_CDEF, T &reg0, T &reg1, uint8_t bits, bool is_signed)
{
    if constexpr ( (std::is_array_v<T> &&     
            std::is_same_v<std::remove_extent_t<T>, uint64_t> && 
            std::extent_v<T> == 4) )
    {
        uint64_t dest[QBigInt::BIGINT_LEN];
        CirCuit_Fields::operator_add(SVM_PARAM_C, reg0, reg1, dest);
        svm_push_reg_implement_new_for_qbigint(dest);
    }
}

static ALEO_ADI bool run_OPCODE_Add(SVM_PARAM_DEF, CIRCUIT_RUN_PARAM_DEF)
{
    QUICK_EXEC_INSTRUCTION_WITH_2_REG(
        instruction, 
        __run_OPCODE_Add(SVM_PARAM_C, reg0, reg1, bits, is_signed)
    );
    return true;   
}

template <typename T>
static ALEO_ADI bool __run_OPCODE_Lt(SVM_PARAM_CDEF, T &reg0, T &reg1, uint8_t bits, bool is_signed, CirCuit_Exec_Step &exec_step)
{

    if constexpr (std::is_same_v<T, bool> ) {
        
    } 
    else
    if constexpr (std::is_same_v<T, nv_uint128> ) {
        bool dest;
        Circuit_Integer::operator_less_than(SVM_PARAM_C, bits, is_signed, reg0, reg1, dest);
        svm_push_reg_implement_new(SVM_PARAM, dest);
        return true;
    }
    else
    if constexpr ( (std::is_array_v<T> &&     
                std::is_same_v<std::remove_extent_t<T>, uint64_t> && 
                std::extent_v<T> == 4) )
    {
        exec_step = CEXEC_STEP_1;
        CirCuit_Fields::operator_less_than_prepare(SVM_PARAM_C, reg0, reg1);
        return false;
    } else {
        static_assert(!std::is_same_v<T, T>, "Unsupported type for __run_OPCODE_Lt");
    }
    
    return true;
}

static ALEO_ADI bool run_OPCODE_Lt(SVM_PARAM_DEF, CIRCUIT_RUN_PARAM_DEF)
{
    if ( exec_step == CEXEC_STEP_INIT) 
    {
        QUICK_EXEC_INSTRUCTION_WITH_2_REG(
            instruction, 
            return __run_OPCODE_Lt(SVM_PARAM_C, reg0, reg1, bits, is_signed, exec_step)
        );
    } else if ( exec_step == CEXEC_STEP_1 ) {
        uint8_t is_const = 0;
        CirCuit_Fields::operator_less_than_prepare_2(SVM_PARAM_C);
        exec_step = CEXEC_STEP_2;
        return false;
    } else if ( exec_step == CEXEC_STEP_2 ) {
        uint8_t is_const = 0;
        bool dest;
        CirCuit_Fields::operator_less_than_run(SVM_PARAM_C, dest);
        svm_push_reg_implement_new(SVM_PARAM, dest);
        return true;
    }
    
    return true;
}

static ALEO_ADI bool run_OPCODE_Gt(SVM_PARAM_DEF, CIRCUIT_RUN_PARAM_DEF)
{
    if ( exec_step == CEXEC_STEP_INIT) 
    {
        QUICK_EXEC_INSTRUCTION_WITH_2_REG(
            instruction, 
            return __run_OPCODE_Lt(SVM_PARAM_C, reg1, reg0, bits, is_signed, exec_step)
        );
    } else if ( exec_step == CEXEC_STEP_1 ) {
        uint8_t is_const = 0;
        CirCuit_Fields::operator_less_than_prepare_2(SVM_PARAM_C);
        exec_step = CEXEC_STEP_2;
        return false;
    } else if ( exec_step == CEXEC_STEP_2 ) {
        uint8_t is_const = 0;
        bool dest;
        CirCuit_Fields::operator_less_than_run(SVM_PARAM_C, dest);
        svm_push_reg_implement_new(SVM_PARAM, dest);
        return true;
    }
    
    return true;    
}

template <typename T>
static ALEO_ADI bool __run_OPCODE_Lte(SVM_PARAM_CDEF, T &reg0, T &reg1, uint8_t bits, bool is_signed, CirCuit_Exec_Step &exec_step)
{
    if constexpr (std::is_same_v<T, bool> ) {
        
    } 
    else
    if constexpr (std::is_same_v<T, nv_uint128> ) {
        bool dest;
        Circuit_Integer::operator_less_than(SVM_PARAM_C, bits, is_signed, reg0, reg1, dest);
        svm_push_reg_implement_new(SVM_PARAM, !dest);

        return true;        
    }
    else
    if constexpr ( (std::is_array_v<T> &&     
                std::is_same_v<std::remove_extent_t<T>, uint64_t> && 
                std::extent_v<T> == 4) )
    {        
        exec_step = CEXEC_STEP_1;
        CirCuit_Fields::operator_less_than_prepare(SVM_PARAM_C, reg0, reg1);
        return false;        
    } else {
        static_assert(!std::is_same_v<T, T>, "Unsupported type for __run_OPCODE_Lte");
    }
    
    return true;
}

static ALEO_ADI bool run_OPCODE_Lte(SVM_PARAM_DEF, CIRCUIT_RUN_PARAM_DEF)
{
    if ( exec_step == CEXEC_STEP_INIT) 
    {
        QUICK_EXEC_INSTRUCTION_WITH_2_REG(
            instruction, 
            return __run_OPCODE_Lte(SVM_PARAM_C, reg1, reg0, bits, is_signed, exec_step)
        );
    } else if ( exec_step == CEXEC_STEP_1 ) {
        uint8_t is_const = 0;
        CirCuit_Fields::operator_less_than_prepare_2(SVM_PARAM_C);
        exec_step = CEXEC_STEP_2;
        return false;
    } else if ( exec_step == CEXEC_STEP_2 ) {
        uint8_t is_const = 0;
        bool dest;
        CirCuit_Fields::operator_less_than_run(SVM_PARAM_C, dest);
        svm_push_reg_implement_new(SVM_PARAM, !dest);
        return true;
    }

    return true;
}

static ALEO_ADI bool run_OPCODE_Gte(SVM_PARAM_DEF, CIRCUIT_RUN_PARAM_DEF)
{
    if ( exec_step == CEXEC_STEP_INIT) 
    {
        QUICK_EXEC_INSTRUCTION_WITH_2_REG(
            instruction, 
            return __run_OPCODE_Lte(SVM_PARAM_C, reg0, reg1, bits, is_signed, exec_step)
        );
    } else if ( exec_step == CEXEC_STEP_1 ) {
        uint8_t is_const = 0;
        CirCuit_Fields::operator_less_than_prepare_2(SVM_PARAM_C);
        exec_step = CEXEC_STEP_2;
        return false;
    } else if ( exec_step == CEXEC_STEP_2 ) {
        uint8_t is_const = 0;
        bool dest;
        CirCuit_Fields::operator_less_than_run(SVM_PARAM_C, dest);
        svm_push_reg_implement_new(SVM_PARAM, !dest);
        return true;
    }

    return true;
}

static ALEO_ADI bool run_OPCODE_CastLossy(SVM_PARAM_DEF, CIRCUIT_RUN_PARAM_DEF)
{
    VmOperand *operands[SYNTHESIS_VM_MAX_OPERANDS] = { nullptr };
    program_fetch_instruction_operands(instruction, operands);
    
    uint8_t is_const = 0;
    uint64_t reg0[QBigInt::BIGINT_LEN];
    if ( operands[0]->op_type == Op_Register) {
        uint8_t reg_index = (uint8_t)operand_data_to_u64(operands[0]->data);
        CircuitTypes ct;
        uint32_t value;
        svm_load_register_raw(SVM_PARAM, reg_index, ct, value);        
        svm_load_reg_implement_for_qbigint(SVM_PARAM, reg0, value);
    } else {
        /*常量*/ 
        memcpy( &(reg0[0]), operands[0]->data, sizeof(reg0) );
    }    

    nv_uint128 dest;
    CirCuit_Fields::operator_cast_lossy(SVM_PARAM_C, reg0, dest);
    CircuitTypes to_type = (CircuitTypes)instruction->as_type_id;

    uint8_t bits=0;
    bool is_signed;

    #define MK_CODE_run_OPCODE_CastLossy(b)\
        case CT_U##b:\
            bits = b;\
            is_signed = false;\
            svm_push_reg_as_variant_int(dest);\
            break;\
        case CT_I##b:\
            bits = b;\
            is_signed = true;\
            svm_push_reg_as_variant_int(dest);\
            break;            

    switch (to_type)
    {
        MK_CODE_run_OPCODE_CastLossy(8)
        MK_CODE_run_OPCODE_CastLossy(16)
        MK_CODE_run_OPCODE_CastLossy(32)
        MK_CODE_run_OPCODE_CastLossy(64)
        MK_CODE_run_OPCODE_CastLossy(128)
        default:
            break;
    }

    #undef MK_CODE_run_OPCODE_CastLossy
    return true;   
}

template <typename T>
static ALEO_ADI void __run_OPCODE_Neg(SVM_PARAM_CDEF, T &reg0, uint8_t bits, bool is_signed)
{
    if constexpr (std::is_same_v<T, bool> ) {
        //do nothing
    }
    else
    if constexpr (std::is_same_v<T, nv_uint128> ) {
        nv_uint128 dest;
        Circuit_Integer::operator_neg(SVM_PARAM_C, bits, is_signed, reg0, dest);
        svm_push_reg_as_variant_int(dest);
    }
    else
    if constexpr ( (std::is_array_v<T> &&     
                std::is_same_v<std::remove_extent_t<T>, uint64_t> && 
                std::extent_v<T> == 4) )    
    {
        uint64_t dest[QBigInt::BIGINT_LEN];
        CirCuit_Fields::operator_neg(reg0, dest);
        svm_push_reg_implement_new_for_qbigint(dest);
    }
    else {
        static_assert(!std::is_same_v<T, T>, "Unsupported type for __run_OPCODE_Neg");
    }
}

static ALEO_ADI bool run_OPCODE_Neg(SVM_PARAM_DEF, CIRCUIT_RUN_PARAM_DEF)
{
    QUICK_EXEC_INSTRUCTION_WITH_1_REG(instruction,
        __run_OPCODE_Neg(SVM_PARAM_C, reg0, bits, is_signed)
    );

    return true;
}

template <typename T>
static ALEO_ADI void __run_OPCODE_HashBhp256(SVM_PARAM_CDEF, T reg0, uint8_t bits, bool is_signed)
{    
    if constexpr (std::is_same_v<T, nv_uint128> ) {        
        HashBhp256::exec_hash(SVM_PARAM, reg0, bits, is_signed);
    }
}

static ALEO_ADI bool run_OPCODE_HashBhp256(SVM_PARAM_DEF, CIRCUIT_RUN_PARAM_DEF)
{    
    if ( exec_step == CEXEC_STEP_INIT ) {
        __EXEC_INSTRUCTION_WITH_1_REG(instruction,
            __run_OPCODE_HashBhp256(SVM_PARAM_C, reg0, bits, is_signed)
        );
        
        exec_step = CEXEC_STEP_1;
        return false;
    } else if ( exec_step == CEXEC_STEP_1 )
    {
        RoundZeroMem::set_max_allow_full_leaves_in_smem(DEFAULT_SMEM_LEAVES_LEN);
        bool should_fold = HashBhp256::exec_hash_iter(SVM_PARAM);
        if ( should_fold ) 
        {
            exec_step = CEXEC_STEP_2;
        }
        return false;

    } else if ( exec_step == CEXEC_STEP_2 ) {
        RoundZeroMem::set_max_allow_full_leaves_in_smem(DEFAULT_SMEM_LEAVES_LEN);
        bool should_continue = HashBhp256::exec_hash_fold(SVM_PARAM);
        if ( should_continue == false )
        {
            //彻底结束， 开始 cast.lossy
            exec_step = CEXEC_STEP_3;
        } else {
            //fold 结束， 继续 iter
            exec_step = CEXEC_STEP_1;
        }
        return false;

    } else if ( exec_step == CEXEC_STEP_3 )
    {
        HashBhp256::exec_hash_cast_lossy(SVM_PARAM);

        uint32_t meta;
        meta = smem_get_circuit_meta();            
        uint32_t preimage_bits_len = (meta >> 16)  & 0x7FFF;    
        bool is_signed = (meta>>31) & 1;
        uint32_t bits = preimage_bits_len - 64 - 26; // hack

        nv_uint128 dest;
        {
            bigint_u256_t tmp;
            smem_load_circuit_params(0, tmp);
            dest.stdint = tmp.uint128[0];
        }
        svm_push_reg_as_variant_int(dest);

        return true;
    }    

    return true;
}

template <typename T>
static ALEO_ADI void __run_OPCODE_HashPed64(SVM_PARAM_CDEF, T reg0, uint8_t bits, bool is_signed)
{    
    if constexpr (std::is_same_v<T, nv_uint128> ) {
        HashPed64::prepare(SVM_PARAM, reg0, bits, is_signed);
    }
}

static ALEO_ADI bool run_OPCODE_HashPed64(SVM_PARAM_DEF, CIRCUIT_RUN_PARAM_DEF)
{
    if ( exec_step == CEXEC_STEP_INIT ) {
        __EXEC_INSTRUCTION_WITH_1_REG(instruction,
            __run_OPCODE_HashPed64(SVM_PARAM_C, reg0, bits, is_signed)
        );
        
        exec_step = CEXEC_STEP_1;
        return false;
    } else if ( exec_step == CEXEC_STEP_1 )
    {
        RoundZeroMem::set_max_allow_full_leaves_in_smem(DEFAULT_SMEM_LEAVES_LEN);
        bool is_instruction_finish = HashPed64::iter(SVM_PARAM);
        
        if ( is_instruction_finish ) 
        {
            uint32_t bits;
            bool is_signed ;
            uint32_t meta = smem_get_circuit_meta();
            bits = meta >> 16;
            is_signed = meta & 0xffff;
            uint64_t tmp[QBigInt::BIGINT_LEN];
            nv_uint128 dest;
            smem_load_circuit_params(0, tmp);
            dest.stdint = tmp[1];
            dest.stdint = (dest.stdint<<64) || tmp[0];
            
            svm_push_reg_as_variant_int(dest);
        }
        return is_instruction_finish;
    }
    return true;
}


template <typename T>
static ALEO_ADI void __run_OPCODE_Inv(SVM_PARAM_CDEF, T reg0, uint8_t bits, bool is_signed)
{
    if constexpr ( (std::is_array_v<T> &&     
                std::is_same_v<std::remove_extent_t<T>, uint64_t> && 
                std::extent_v<T> == 4) )
    {
        CirCuit_Fields::operator_inv(SVM_PARAM_C, reg0);
        svm_push_reg_implement_new_for_qbigint(reg0);

    }    
}

static ALEO_ADI bool run_OPCODE_Inv(SVM_PARAM_DEF, CIRCUIT_RUN_PARAM_DEF)
{
    QUICK_EXEC_INSTRUCTION_WITH_1_REG(instruction,
        __run_OPCODE_Inv(SVM_PARAM_C, reg0, bits, is_signed)
    );

    return true;
}

template <typename T>
static ALEO_ADI void __run_OPCODE_AbsWrapped(SVM_PARAM_CDEF, T reg0, uint8_t bits, bool is_signed)
{
    if constexpr (std::is_same_v<T, nv_uint128> ) {
        nv_uint128 dest;
        Circuit_Integer::operator_abs_wrap(SVM_PARAM_C, bits, is_signed, reg0, dest);
        svm_push_reg_as_variant_int(dest);
    }
}

static ALEO_ADI bool run_OPCODE_AbsWrapped(SVM_PARAM_DEF, CIRCUIT_RUN_PARAM_DEF)
{
    __EXEC_INSTRUCTION_WITH_1_REG(instruction,
        __run_OPCODE_AbsWrapped(SVM_PARAM_C, reg0, bits, is_signed)
    );

    return true;
}

template <typename T>
static ALEO_ADI void __run_OPCODE_Not(SVM_PARAM_CDEF, T reg0, uint8_t bits, bool is_signed)
{
    if constexpr (std::is_same_v<T, nv_uint128> ) {
        nv_uint128 dest;
        Circuit_Integer::operator_not(SVM_PARAM_C, bits, is_signed, reg0, dest);
        svm_push_reg_as_variant_int(dest);
    }

    if constexpr (std::is_same_v<T, bool> ) {
        bool dest;
        Circuit_Bool::operator_not(SVM_PARAM_C, reg0, dest);
        svm_push_reg_implement_new(SVM_PARAM, dest);
    }
}

static ALEO_ADI bool run_OPCODE_Not(SVM_PARAM_DEF, CIRCUIT_RUN_PARAM_DEF)
{
/*
    not r2 into r14;
    not r3 into r15;
    not r0 into r16;
    not r1 into r17;
    is.eq r14 r15 into r18;
    is.eq r16 r17 into r19;
    not r10 into r20;
    is.eq r10 r20 into r21;

*/
    __EXEC_INSTRUCTION_WITH_1_REG(instruction,
        __run_OPCODE_Not(SVM_PARAM_C, reg0, bits, is_signed)
    );
    return true;
}

#endif 

////////////////SIMPLE////////////////////



template <typename T>
static ALEO_ADI void __run_OPCODE_MulWrapped(SVM_PARAM_CDEF, T reg0, T reg1, uint8_t bits, bool is_signed)
{
    if constexpr (std::is_same_v<T, nv_uint128> ) {
        RoundZeroMem::set_max_allow_full_leaves_in_smem(6);

        nv_uint128 dest;
        Circuit_Integer::operator_mul_wrap(SVM_PARAM_C, bits, is_signed, reg0, reg1, dest);
        svm_push_reg_as_variant_int(dest);
    }
}

static ALEO_ADI bool run_OPCODE_MulWrapped(SVM_PARAM_DEF, CIRCUIT_RUN_PARAM_DEF)
{    
    QUICK_EXEC_INSTRUCTION_WITH_2_REG(
        instruction, 
        __run_OPCODE_MulWrapped(SVM_PARAM_C, reg0, reg1, bits, is_signed)
    );
    return true;
}

static ALEO_ADI bool run_OPCODE_PowWrapped(SVM_PARAM_DEF, CIRCUIT_RUN_PARAM_DEF)
{
    if ( exec_step == CEXEC_STEP_INIT)
    {
        VmOperand *operands[SYNTHESIS_VM_MAX_OPERANDS] = { nullptr };
        program_fetch_instruction_operands(instruction, operands);
                            
        if ( operands[0]->op_type != Op_Register) 
            return true;
        
        uint8_t reg_index = (uint8_t)operand_data_to_u64(operands[0]->data);
        CircuitTypes ct;
        uint32_t value;
        svm_load_register_raw(SVM_PARAM, reg_index, ct, value);
        
        
        if ( ct == CT_BOOLEAN || ct == CT_FIELD )
            return true;
        
        uint32_t bits = circuit_type_to_bits_len(ct);
        bool is_signed = circuit_type_is_signed_int(ct);
        nv_uint128 reg0 = svm_load_reg_implement_for_int(SVM_PARAM, value, bits);

        reg_index = (uint8_t)operand_data_to_u64(operands[1]->data);
        svm_load_register_raw(SVM_PARAM, reg_index, ct, value);
        
        if ( ct == CT_BOOLEAN || ct == CT_FIELD )
            return true;

        uint32_t exp_bits = circuit_type_to_bits_len(ct);
        bool exp_is_signed = circuit_type_is_signed_int(ct);
        nv_uint128 reg1 = svm_load_reg_implement_for_int(SVM_PARAM, value, exp_bits);        

        uint8_t is_const = 0;
        Circuit_Integer::operator_pow_wrap_prepare(SVM_PARAM_C, bits, is_signed, reg0, reg1.stdint, exp_bits);

        exec_step = CEXEC_STEP_1;

        return false;
    } else {
        RoundZeroMem::set_max_allow_full_leaves_in_smem(6);

        nv_uint128 dest; 
        bool ret = Circuit_Integer::operator_pow_wrap_run(SVM_PARAM);
        if ( ret ) {
            bigint_u256_t tmp;            
            smem_load_circuit_params(0, tmp);
            dest.stdint = tmp.uint128[1];

            uint8_t bits = smem_load_circuit_params_by_u32(1, 4);
            bool is_signed = smem_load_circuit_params_by_u32(1, 5);
            //__svm_push_reg_as_variant_int(SVM_PARAM, dest, bits, is_signed);
            svm_push_reg_as_variant_int(dest);
        }
        return ret;
        
    } 
}


template <typename T>
static ALEO_ADI void __run_OPCODE_IsEq(SVM_PARAM_CDEF, T &reg0, T &reg1, uint8_t bits, bool is_signed)
{
    if constexpr (std::is_same_v<T, bool> ) {
        bool dest;
        Circuit_Bool::operator_is_eq(SVM_PARAM_C, reg0, reg1, dest);
        svm_push_reg_implement_new<bool>(SVM_PARAM, dest);
    } 
    else
    if constexpr (std::is_same_v<T, nv_uint128> ) {
        bool dest;
        Circuit_Integer::operator_is_eq(SVM_PARAM_C, bits, is_signed, reg0, reg1, dest);
        svm_push_reg_implement_new<bool>(SVM_PARAM, dest);
    }
    else
    if constexpr ( (std::is_array_v<T> &&     
                std::is_same_v<std::remove_extent_t<T>, uint64_t> && 
                std::extent_v<T> == 4) )
    {
        RoundZeroMem::set_max_allow_full_leaves_in_smem(1);

        bool dest;
        CirCuit_Fields::operator_is_eq(SVM_PARAM_C, reg0, reg1, dest);
        svm_push_reg_implement_new<bool>(SVM_PARAM, dest);
    } else {
        static_assert(!std::is_same_v<T, T>, "Unsupported type for __run_OPCODE_IsEq");
    }
    
}

static ALEO_ADI bool run_OPCODE_IsEq(SVM_PARAM_DEF, CIRCUIT_RUN_PARAM_DEF)
{    
    
    QUICK_EXEC_INSTRUCTION_WITH_2_REG(
        instruction, 
        __run_OPCODE_IsEq(SVM_PARAM_C, reg0, reg1, bits, is_signed)
    );

    return true;
}


template <typename T>
static ALEO_ADI void __run_OPCODE_IsNotEq(SVM_PARAM_CDEF, T &reg0, T &reg1, uint8_t bits, bool is_signed)
{
    if constexpr (std::is_same_v<T, bool> ) {
        bool dest;
        Circuit_Bool::operator_is_not_eq(SVM_PARAM_C, reg0, reg1, dest);
        svm_push_reg_implement_new<bool>(SVM_PARAM, dest);
    } 
    else
    if constexpr (std::is_same_v<T, nv_uint128> ) {
        bool dest;
        Circuit_Integer::operator_is_not_eq_v2(SVM_PARAM_C, bits, is_signed, reg0, reg1, dest);
        svm_push_reg_implement_new<bool>(SVM_PARAM, dest);
    }
    else
    if constexpr (std::is_array_v<T> &&     
                std::is_same_v<std::remove_extent_t<T>, uint64_t> && 
                std::extent_v<T> == 4)
    {
        RoundZeroMem::set_max_allow_full_leaves_in_smem(1);

        bool dest;
        CirCuit_Fields::operator_is_not_eq(SVM_PARAM_C, reg0, reg1, dest);
        svm_push_reg_implement_new<bool>(SVM_PARAM, dest);
    } else {
        static_assert(!std::is_same_v<T, T>, "Unsupported type for __run_OPCODE_IsNotEq");
    }
    
}

static ALEO_ADI bool run_OPCODE_IsNotEq(SVM_PARAM_DEF, CIRCUIT_RUN_PARAM_DEF)
{
    QUICK_EXEC_INSTRUCTION_WITH_2_REG(
        instruction, 
        __run_OPCODE_IsNotEq(SVM_PARAM_C, reg0, reg1, bits, is_signed)
    );

    return true;
}


template <typename T>
static ALEO_ADI void __run_OPCODE_Ternary(SVM_PARAM_CDEF, bool condition, T &reg0, T &reg1, uint8_t bits, bool is_signed)
{
    if constexpr (std::is_same_v<T, bool> ) {
        bool dest;
        Circuit_Bool::operator_ternary(SVM_PARAM_C, condition, reg0, reg1, dest);
        svm_push_reg_implement_new(SVM_PARAM, dest);
    }
    else
    if constexpr (std::is_same_v<T, nv_uint128> ) {
        nv_uint128 dest;
        Circuit_Integer::operator_ternary(SVM_PARAM_C, bits, is_signed, condition, reg0, reg1, dest);
        svm_push_reg_as_variant_int(dest);
    }
    else
    if constexpr (std::is_array_v<T> &&     
                std::is_same_v<std::remove_extent_t<T>, uint64_t> && 
                std::extent_v<T> == QBigInt::BIGINT_LEN)
    {
        uint64_t dest[QBigInt::BIGINT_LEN];
        CirCuit_Fields::operator_ternary(SVM_PARAM_C, condition, reg0, reg1, dest);
        svm_push_reg_implement_new_for_qbigint(dest);
    } else {
        static_assert(!std::is_same_v<T, T>, "Unsupported type for __run_OPCODE_Ternary");
    }
}

static ALEO_ADI bool run_OPCODE_Ternary(SVM_PARAM_DEF, CIRCUIT_RUN_PARAM_DEF)
{
    VmOperand *operands[SYNTHESIS_VM_MAX_OPERANDS] = { nullptr };
    program_fetch_instruction_operands(instruction, operands);
        
    if ( operands[0]->op_type == Op_Register) {
        uint8_t reg_index = (uint8_t)operand_data_to_u64(operands[0]->data);
        CircuitTypes ct;
        uint32_t value;
        svm_load_register_raw(SVM_PARAM, reg_index, ct, value);
        if ( ct == CT_BOOLEAN )
        {
            bool reg_condition = svm_load_reg_implement_for_bool(SVM_PARAM, value);
            QUICK_EXEC_INSTRUCTION_WITH_OPERAND_AND_2_REG(instruction, operands, 1, 
                __run_OPCODE_Ternary(SVM_PARAM_C, reg_condition, reg0, reg1, bits, is_signed)
            );
            
        } else {
            cudbg("ternary 1st reg is not bool\n");
        }
    } 
    // 第 0 个参数为 const 的情况暂时不考虑
    return true;
}

static ALEO_ADI bool __run_OPCODE_HashPsd2(SVM_PARAM_DEF, CirCuit_Exec_Step &exec_step)
{
    //bigint_u256_t dest_reg_12 , dest_reg_13;

    if ( exec_step == CEXEC_STEP_1)
    {
        //处理 hash.psd2 r12 
        cudbg("hash.psd2 for r12\n");
        
        CircuitTypes reg_ct;
        uint32_t value;
        uint64_t reg_raw_value[QBigInt::BIGINT_LEN];
        svm_load_register_raw(SVM_PARAM, 12, reg_ct, value);        
        svm_load_reg_implement_for_qbigint(SVM_PARAM, reg_raw_value, value);

        HashPoseidon::prepare_context(SVM_PARAM, reg_raw_value);
        exec_step = CEXEC_STEP_2;
        return false;
    } else if ( exec_step == CEXEC_STEP_2 )
    {
        cudbg("hash.psd2 for r12 , next\n");
        
        //已经拆分过，最多5条 leaf_hash 
        RoundZeroMem::set_max_allow_full_leaves_in_smem(5);
        if ( HashPoseidon::exec_hash(SVM_PARAM) )
        {
            //finished 
            /*
                输出暂时desc 存在 global memory的 soa_leave的 tail 
            */
            uint64_t dest_reg_12[QBigInt::BIGINT_LEN];
            smem_load_circuit_params(1, dest_reg_12);

            //这里已经把 dest_reg_12转为自然数域了
            nv_uint8 dest_uint8;
            CirCuit_Fields::field_to_bigint_with_witness(SVM_PARAM, 0, dest_reg_12, dest_reg_12);
            dest_uint8.stdint = dest_reg_12[0];
            svm_push_reg_implement_new(SVM_PARAM, dest_uint8.nvint);
            for(int n = 0; n < 8; n++)
            {
                svm_mtree_push_soa_leaf_u32_order_desc(SVM_PARAM, 0, n,  QBigInt::get_by_u32(dest_reg_12, n) );
            }

            exec_step = CEXEC_STEP_3;            
        }
        return false;
    } else if ( exec_step == CEXEC_STEP_3)
    {
        //处理 hash.psd2 r13
        cudbg("hash.psd2 for r13\n");
        
        CircuitTypes reg_ct;
        uint32_t value;
        uint64_t reg_raw_value[QBigInt::BIGINT_LEN];
        svm_load_register_raw(SVM_PARAM, 13, reg_ct, value);
        svm_load_reg_implement_for_qbigint(SVM_PARAM, reg_raw_value, value);

        HashPoseidon::prepare_context(SVM_PARAM, reg_raw_value);
        exec_step = CEXEC_STEP_4;
        return false;       
    } else if ( exec_step == CEXEC_STEP_4 )
    {
        cudbg("hash.psd2 for r13 , next\n");
        RoundZeroMem::set_max_allow_full_leaves_in_smem(5);
        if ( HashPoseidon::exec_hash(SVM_PARAM) )
        {
            //finished 
            /*
                输出暂时desc 存在 global memory的 soa_leave的 tail 
            */
            uint64_t dest_reg_13[QBigInt::BIGINT_LEN];
            smem_load_circuit_params(1, dest_reg_13);            
            
            nv_uint8 dest_uint8;
            CirCuit_Fields::field_to_bigint_with_witness(SVM_PARAM, 0, dest_reg_13, dest_reg_13);
            dest_uint8.stdint = dest_reg_13[0];
            svm_push_reg_implement_new(SVM_PARAM, dest_uint8.nvint);
                        
            uint64_t dest_reg_12[QBigInt::BIGINT_LEN];
            for(int n = 0; n < 8; n++)
                QBigInt::set_by_u32(dest_reg_12, 
                        svm_mtree_pop_soa_leaf_u32_order_desc(SVM_PARAM, 0, n), n );
            

            //u16:
            ushort1 u16;
            u16.x = dest_reg_12[0];
            svm_push_reg_implement_new(SVM_PARAM,  u16);
            u16.x = dest_reg_13[0];
            svm_push_reg_implement_new(SVM_PARAM,  u16);

            //u32:
            uint1 u32;
            u32.x = dest_reg_12[0];
            svm_push_reg_implement_new(SVM_PARAM,  u32);
            u32.x = dest_reg_13[0];
            svm_push_reg_implement_new(SVM_PARAM,  u32);

            //u64:
            uint2 u64;
            u64.x = dest_reg_12[0]; 
            u64.y = dest_reg_12[0] >> 32;
            svm_push_reg_implement_new(SVM_PARAM, u64);
            u64.x = dest_reg_13[0]; 
            u64.y = dest_reg_13[0] >> 32;
            svm_push_reg_implement_new(SVM_PARAM, u64);

            //u128:
            uint4 u128;
            u128.x = dest_reg_12[0];
            u128.y = dest_reg_12[0] >> 32;
            u128.z = dest_reg_12[1];
            u128.w = dest_reg_12[1] >> 32;
            svm_push_reg_implement_new(SVM_PARAM,  u128);
            u128.x = dest_reg_13[0];
            u128.y = dest_reg_13[0] >> 32;
            u128.z = dest_reg_13[1];
            u128.w = dest_reg_13[1] >> 32;
            svm_push_reg_implement_new(SVM_PARAM,  u128);

            /*
            省略了 8次 hash.psd2, 共  1280*8 个leaf
            */

            //finish 
            return true;
        } else 
            return false;
    }
    return true;
}

static ALEO_ADI bool run_OPCODE_HashPsd2(SVM_PARAM_DEF, CIRCUIT_RUN_PARAM_DEF)
{
    constexpr const int max_hash_count = 10;
    if ( exec_step == CEXEC_STEP_INIT )
    {
        unsigned int hash_count =  svm_get_hash_psd2_count(SVM_PARAM);
        hash_count += 1;
        svm_set_hash_psd2_count(SVM_PARAM, hash_count);
        if ( hash_count == max_hash_count)
        {        
            // equal 10
            exec_step = CEXEC_STEP_1;
        } else {
            return true;
        }
    }

    //前面10 条 hash.psd2 已经读完了 

    return __run_OPCODE_HashPsd2(SVM_PARAM, exec_step);
}


static ALEO_ADI bool circuit_execute_instruction(SVM_PARAM_DEF, VmInstruction *instruction, CirCuit_Exec_Step &exec_step)
{
    switch ( (VmOperator)(instruction->operator_id) )
    {
        case OPCODE_HashPsd2:        
            return run_OPCODE_HashPsd2(SVM_PARAM, CIRCUIT_RUN_PARAM);        
        case OPCODE_IsEq:
            return run_OPCODE_IsEq(SVM_PARAM, CIRCUIT_RUN_PARAM);
        case OPCODE_MulWrapped:
            return run_OPCODE_MulWrapped(SVM_PARAM, CIRCUIT_RUN_PARAM);    
        case OPCODE_Ternary:
            return run_OPCODE_Ternary(SVM_PARAM, CIRCUIT_RUN_PARAM);                    
        case OPCODE_IsNotEq:
            return run_OPCODE_IsNotEq(SVM_PARAM, CIRCUIT_RUN_PARAM);       
        case OPCODE_PowWrapped:
            return run_OPCODE_PowWrapped(SVM_PARAM, CIRCUIT_RUN_PARAM);                 
#if !FAST_COMPILE
        case OPCODE_HashBhp256:
            return run_OPCODE_HashBhp256(SVM_PARAM, CIRCUIT_RUN_PARAM);
        case OPCODE_HashPed64:
            return run_OPCODE_HashPed64(SVM_PARAM, CIRCUIT_RUN_PARAM);
        case OPCODE_Xor:
            return run_OPCODE_Xor(SVM_PARAM, CIRCUIT_RUN_PARAM);
        case OPCODE_Or:
            return run_OPCODE_Or(SVM_PARAM, CIRCUIT_RUN_PARAM);
        case OPCODE_Not:
            return run_OPCODE_Not(SVM_PARAM, CIRCUIT_RUN_PARAM);            
        case OPCODE_Nor:
            return run_OPCODE_Nor(SVM_PARAM, CIRCUIT_RUN_PARAM);
        case OPCODE_Nand:
            return run_OPCODE_Nand(SVM_PARAM, CIRCUIT_RUN_PARAM);
        case OPCODE_And:
            return run_OPCODE_And(SVM_PARAM, CIRCUIT_RUN_PARAM);            
        case OPCODE_AbsWrapped:
            return run_OPCODE_AbsWrapped(SVM_PARAM, CIRCUIT_RUN_PARAM);
        case OPCODE_SubWrapped:
            return run_OPCODE_SubWrapped(SVM_PARAM, CIRCUIT_RUN_PARAM);
        case OPCODE_AddWrapped:
            return run_OPCODE_AddWrapped(SVM_PARAM, CIRCUIT_RUN_PARAM);                
        case OPCODE_DivWrapped:
            return run_OPCODE_DivWrapped(SVM_PARAM, CIRCUIT_RUN_PARAM);
        case OPCODE_RemWrapped:
            return run_OPCODE_RemWrapped(SVM_PARAM, CIRCUIT_RUN_PARAM);
        case OPCODE_Mul:
            return run_OPCODE_Mul(SVM_PARAM, CIRCUIT_RUN_PARAM);
        case OPCODE_Square:
            return run_OPCODE_Square(SVM_PARAM, CIRCUIT_RUN_PARAM);
        case OPCODE_Pow:
            return run_OPCODE_Pow(SVM_PARAM, CIRCUIT_RUN_PARAM);
        case OPCODE_Neg:
            return run_OPCODE_Neg(SVM_PARAM, CIRCUIT_RUN_PARAM);
        case OPCODE_Add:
            return run_OPCODE_Add(SVM_PARAM, CIRCUIT_RUN_PARAM);
        case OPCODE_Lt:
            return run_OPCODE_Lt(SVM_PARAM, CIRCUIT_RUN_PARAM);
        case OPCODE_Lte:
            return run_OPCODE_Lte(SVM_PARAM, CIRCUIT_RUN_PARAM);
        case OPCODE_Gt:
            return run_OPCODE_Gt(SVM_PARAM, CIRCUIT_RUN_PARAM);
        case OPCODE_Gte:
            return run_OPCODE_Gte(SVM_PARAM, CIRCUIT_RUN_PARAM);        
        case OPCODE_CastLossy:
            return run_OPCODE_CastLossy(SVM_PARAM, CIRCUIT_RUN_PARAM);
        case OPCODE_ShrWrapped:
           return run_OPCODE_ShrWrapped(SVM_PARAM, CIRCUIT_RUN_PARAM);
        case OPCODE_ShlWrapped:
            return run_OPCODE_ShlWrapped(SVM_PARAM, CIRCUIT_RUN_PARAM);
        case OPCODE_Inv:
            return run_OPCODE_Inv(SVM_PARAM, CIRCUIT_RUN_PARAM);
        case OPCODE_Modulo:
            return run_OPCODE_RemWrapped(SVM_PARAM, CIRCUIT_RUN_PARAM);
        case OPCODE_Div:
            return run_OPCODE_Div(SVM_PARAM, CIRCUIT_RUN_PARAM);
#endif            
        default:
            return true;
    } 
    
    /*
    上面的指令 多个操作数类型不同，下面都是操作数和目的寄存器， 类型完全一致的指令
    TODO: div 和 hash 指令
    */    


}

#undef cuerr
#undef cudbg