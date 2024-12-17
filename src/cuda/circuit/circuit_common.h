#pragma once


#define SVM_SAVE ( !is_const && ctx != nullptr)

#define CIRCUIT_MAKE_INTTYPE_CODE(_ECODE)\
    _ECODE(CT_I8, char1)\
    _ECODE(CT_U8, uchar1)\
    _ECODE(CT_I16, short1)\
    _ECODE(CT_U16, ushort1)\
    _ECODE(CT_I32, int1)\
    _ECODE(CT_U32, uint1)\
    _ECODE(CT_I64, int2)\
    _ECODE(CT_U64, uint2)\
    _ECODE(CT_I128, int4)\
    _ECODE(CT_U128, uint4)

enum CirCuit_Exec_Step
{
    CEXEC_STEP_INIT = 0,
    CEXEC_STEP_1,
    CEXEC_STEP_2,
    CEXEC_STEP_3,
    CEXEC_STEP_4,
    CEXEC_STEP_5,
    CEXEC_STEP_6,
    CEXEC_STEP_7,
    CEXEC_STEP_8
};

