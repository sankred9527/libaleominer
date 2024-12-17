
#include <cstdint>
#include "cu_program.cuh"
#include "cu_utils.cuh"
#include "circuit/circuit_type.cuh"



__device__  static void __program_operand_to_str(VmOperand *operand, char *buf, size_t buf_len)
{
    //每个数据都是 8字节对齐的, 可以这样直接转换。如果是 i128类型，暂时不支持显示 
    uint64_t v = *(uint64_t*)(operand->data);

    if (operand->op_type == Op_Register) 
    {
        cu_strncat(buf, " r", buf_len);            
        cu_ultoa(buf+cu_strlen(buf), (unsigned long)v);
    } else {

        #define TMP_OP_SHOW(t1, t2) \
            case CT_##t1 :  \
            { \
                char *p = #t1; \
                if ( p[0] == 'I') \
                    cu_sltoa(buf+cu_strlen(buf), (long)v); \
                else \
                    cu_ultoa(buf+cu_strlen(buf), (unsigned long)v); \
                cu_strncat(buf, #t2, buf_len);   \
            } \
            break

        cu_strncat(buf, " ", buf_len);
        switch (operand->var_type) {
            TMP_OP_SHOW(I8, i8);
            TMP_OP_SHOW(U8, u8);
            TMP_OP_SHOW(I16, i16);
            TMP_OP_SHOW(U16, u16);
            TMP_OP_SHOW(I32, i32);
            TMP_OP_SHOW(U32, u32);
            TMP_OP_SHOW(I64, i64);
            TMP_OP_SHOW(U64, u64);
            case CT_I128:
            {                                
                cu_strncat(buf, "notsupport-i128", buf_len);
            }
            break;
            case CT_U128:
            {                                
                cu_strncat(buf, "notsupport-u128", buf_len);
            }
            break;
            default:
                cu_strncat(buf, "notsupport-type", buf_len);
                break;
        }
        #undef TMP_OP_SHOW
    }
}

__device__ static void __program_circuit_type_to_str(CircuitTypes ct_type, char *buf, size_t buf_len)
{
    char *_map[CT_MAX] = { nullptr };
    _map[CT_BOOLEAN] = "bool";
    _map[CT_I8] = "i8";
    _map[CT_U8] = "u8";
    _map[CT_I16] = "i16";
    _map[CT_U16] = "u16";
    _map[CT_I32] = "i32";
    _map[CT_U32] = "u32";
    _map[CT_I64] = "i64";
    _map[CT_U64] = "u64";
    _map[CT_I128] = "i128";
    _map[CT_U128] = "u128";
    _map[CT_FIELD] = "field";
    
    uint8_t ct = (uint8_t)ct_type;
    if ( ct < CT_MAX) {
        cu_strncat(buf, _map[ct], buf_len);
    }
}

static __device__  void __program_opcode_to_str(VmOperator opcode, char *buf, size_t buf_len)
{
    char *_map[OPCODE_MAX_NUM];
    _map[OPCODE_Neg] = "neg";
    _map[OPCODE_Not] = "not";
    _map[OPCODE_Xor] = "xor";
    _map[OPCODE_Nor] = "nor";
    _map[OPCODE_Nand] = "nand";
    _map[OPCODE_Or] = "or";
    _map[OPCODE_And] = "and";
    _map[OPCODE_IsEq] = "is.eq";
    _map[OPCODE_IsNotEq] = "is.neq";
    _map[OPCODE_Add] = "add";
    _map[OPCODE_AbsWrapped] = "abs.w";
    _map[OPCODE_AddWrapped] = "add.w";
    _map[OPCODE_SubWrapped] = "sub.w";
    _map[OPCODE_HashPsd2] = "hash.psd2";
    _map[OPCODE_HashBhp256] = "hash.bhp256";
    _map[OPCODE_HashPed64] = "hash.ped64";
    _map[OPCODE_Mul] = "mul";
    _map[OPCODE_MulWrapped] = "mul.w";
    _map[OPCODE_Lt] = "lt";
    _map[OPCODE_Lte] = "lte";
    _map[OPCODE_Gt] = "gt";
    _map[OPCODE_Gte] = "gte";
    _map[OPCODE_CastLossy] = "cast.lossy";
    _map[OPCODE_ShlWrapped] = "shl.w";
    _map[OPCODE_ShrWrapped] = "shr.w";
    _map[OPCODE_Ternary] = "ternary";
    _map[OPCODE_Square] = "square";
    _map[OPCODE_Pow] = "pow";
    _map[OPCODE_PowWrapped] = "pow.w";
    _map[OPCODE_DivWrapped] = "div.w";
    _map[OPCODE_RemWrapped] = "rem.w";
    _map[OPCODE_Modulo] = "mod";
    _map[OPCODE_Div] = "div";
    _map[OPCODE_Inv] = "inv";
    
    uint8_t code = static_cast<uint8_t>(opcode);
    if ( code < OPCODE_MAX_NUM )
    {
        cu_strncat(buf, _map[code], buf_len);
    }
}


__device__  void svm_operand_to_str(VmOperand *operand, char *buf, size_t buf_len)
{
    //每个数据都是 8字节对齐的, 可以这样直接转换。如果是 i128类型，暂时不支持显示 
    uint64_t v = *(uint64_t*)(operand->data);
    
    if (operand->op_type == Op_Register) 
    {
        cu_strncat(buf, " r", buf_len);                    
        cu_ultoa(buf+cu_strlen(buf), (unsigned long)v);
    } else {
        #define TMP_OP_SHOW(t1, t2) \
            case CT_##t1 :  \
            { \
                char *p = #t1; \
                if ( p[0] == 'I') \
                    cu_sltoa(buf+cu_strlen(buf), (long)v); \
                else \
                    cu_ultoa(buf+cu_strlen(buf), (unsigned long)v); \
                cu_strncat(buf, #t2, buf_len);   \
            } \
            break

        cu_strncat(buf, " ", buf_len);
        switch (operand->var_type) {
            TMP_OP_SHOW(I8, i8);
            TMP_OP_SHOW(U8, u8);
            TMP_OP_SHOW(I16, i16);
            TMP_OP_SHOW(U16, u16);
            TMP_OP_SHOW(I32, i32);
            TMP_OP_SHOW(U32, u32);
            TMP_OP_SHOW(I64, i64);
            TMP_OP_SHOW(U64, u64);
            case CT_I128:
            {                                
                cu_strncat(buf, "notsupport-i128", buf_len);
            }
            break;
            case CT_U128:
            {                                
                cu_strncat(buf, "notsupport-u128", buf_len);
            }
            break;
            default:
                cu_strncat(buf, "notsupport-type", buf_len);
                break;
        }
        #undef TMP_OP_SHOW
    }

}




__device__ void program_instruction_to_str(VmInstruction *instruction, char *buf, size_t buf_len)
{
    size_t len = buf_len;
    VmOperand *operands[3] = { nullptr };
    
    program_fetch_instruction_operands(instruction, operands);
    __program_opcode_to_str((VmOperator)instruction->operator_id, buf, len - cu_strlen(buf));
    
    for(int n = 0; n < 3 ; n++)
    {
        if (operands[n] == nullptr)
            break;    
        svm_operand_to_str(operands[n], buf, len - cu_strlen(buf));    
    }
    
    cu_strncat(buf, " into r", len - cu_strlen(buf));    
    cu_ultoa(buf + cu_strlen(buf), (unsigned long)instruction->dest);
    if ( instruction->as_type_id != (uint8_t)CT_UNKNOWN) {

        cu_strncat(buf, " as ", buf_len);
        __program_circuit_type_to_str((CircuitTypes)instruction->as_type_id, buf, buf_len);
    }
    
}


