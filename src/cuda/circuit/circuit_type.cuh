
#pragma once 


//必须和rust 的 cuinterface.rs 里的顺序一致
enum CircuitTypes{
    CT_BOOLEAN = 0,
    CT_I8,
    CT_U8,
    CT_I16,
    CT_U16,
    CT_I32,
    CT_U32,
    CT_I64,
    CT_U64,
    CT_I128,
    CT_U128,
    CT_FIELD,
    CT_MAX,
    CT_UNKNOWN = 255,
};


enum VmOperandType {
    Op_Register = 0,
    Op_Constant
};


enum LiteralVariantAsU8 {
    VARIANT_Address = 0,
    VARIANT_Boolean,
    VARIANT_Field ,
    VARIANT_Group ,
    VARIANT_I8 ,
    VARIANT_I16 ,
    VARIANT_I32 ,
    VARIANT_I64 ,
    VARIANT_I128 ,
    VARIANT_U8 ,
    VARIANT_U16 ,
    VARIANT_U32 ,
    VARIANT_U64 ,
    VARIANT_U128 ,
    VARIANT_Scalar ,
    VARIANT_Signature ,
    VARIANT_String,
    VARIANT_UNKNOWN
};


static ALEO_ADI enum LiteralVariantAsU8 circuit_literal_var_type_from_bits_signed(uint32_t bits, bool is_signed)
{
    switch (bits)
    {
        case 8:
            return is_signed?VARIANT_I8:VARIANT_U8;
        case 16:
            return is_signed?VARIANT_I16:VARIANT_U16;
        case 32:
            return is_signed?VARIANT_I32:VARIANT_U32;
        case 64:
            return is_signed?VARIANT_I64:VARIANT_U64;
        case 128:
            return is_signed?VARIANT_I128:VARIANT_U128;
        default:
            break;
    }
    return VARIANT_UNKNOWN;
}

static ALEO_ADI uint8_t circuit_type_to_bits_len(CircuitTypes ct)
{
    switch (ct)
    {
    case CT_I8:
    case CT_U8:
        return 8;
    case CT_I16:
    case CT_U16:
        return 16;
    case CT_I32:
    case CT_U32:
        return 32;        
    case CT_I64:
    case CT_U64:
        return 64;
    case CT_I128:
    case CT_U128:
        return 128;                
    default:
        break;
    }
    return 0;
}

static ALEO_ADI bool circuit_type_is_signed_int(CircuitTypes ct)
{
    switch (ct)
    {
    case CT_I8:
    case CT_I16:
    case CT_I32:
    case CT_I64:
    case CT_I128:
        return true;               
    default:
        return false;
    }    
}