#pragma once
#include "cuda/circuit/circuit_common.h"
#include "cuda/cu_common.h"
#include "cuda/cu_synthesis.cuh"
#include "cuda/cu_synthesis_reg.cuh"


class Circuit_Bool
{
public:
    static ALEO_ADI void operator_is_eq(SVM_PARAM_CDEF, bool reg0, bool reg1, bool &dest)
    {        
        Circuit_Bool::operator_is_not_eq(SVM_PARAM_C, reg0, reg1, dest);
        dest = !dest;
    }

    static ALEO_ADI void operator_is_not_eq(SVM_PARAM_CDEF, bool reg0, bool reg1, bool &dest)
    {
        Circuit_Bool::operator_xor(SVM_PARAM_C, reg0, reg1, dest);
        dest = (dest != false );
    }

    static ALEO_ADI void operator_xor(SVM_PARAM_CDEF, bool reg0, bool reg1, bool &dest)
    {
        dest = reg0 ^ reg1;
        if ( SVM_SAVE ) {
            svm_push_leaf_one_bit_bool(SVM_PARAM, dest);
        }
    }

    static ALEO_ADI void operator_not(SVM_PARAM_CDEF, bool reg0, bool &dest)
    {
        dest = !reg0;
    }

    static ALEO_ADI void operator_and(SVM_PARAM_CDEF, bool reg0, bool reg1, bool &dest)
    {
        dest = reg0 & reg1;
        if ( SVM_SAVE ) {
            svm_push_leaf_one_bit_bool(SVM_PARAM, dest);
        }
    }

    static ALEO_ADI void operator_nand(SVM_PARAM_CDEF, bool reg0, bool reg1, bool &dest)
    {
        dest = !(reg0 & reg1);
        if ( SVM_SAVE ) {
            svm_push_leaf_one_bit_bool(SVM_PARAM, dest);
        }
    }

    static ALEO_ADI void operator_nor(SVM_PARAM_CDEF, bool reg0, bool reg1, bool &dest)
    {
        dest = (!reg0) & (!reg1);
        if ( SVM_SAVE ) {
            svm_push_leaf_one_bit_bool(SVM_PARAM, dest);
        }
    }

    static ALEO_ADI void operator_or(SVM_PARAM_CDEF, bool reg0, bool reg1, bool &dest)
    {
        dest = reg0 | reg1;
        if ( SVM_SAVE ) {
            svm_push_leaf_one_bit_bool(SVM_PARAM, dest);
        }
    }

    static ALEO_ADI  void operator_ternary(
        SVM_PARAM_CDEF,
        bool condition,
        bool first,
        bool second,
        bool &dest)
    {
        dest = condition ? first : second;
        if ( SVM_SAVE )
            svm_push_leaf_one_bit_bool(SVM_PARAM, dest);
    }
};