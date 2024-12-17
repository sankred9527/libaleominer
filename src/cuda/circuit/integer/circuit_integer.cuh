#pragma once

#include "cuda/cu_common.h"
#include "cuda/cu_synthesis.cuh"
#include "cuda/cu_synthesis_reg.cuh"
#include "cuda/cu_mtree_leaf.cuh"
#include "cuda/circuit/circuit_common.h"
#include "cuda/cu_mtree_leaf.cuh"
#include "cuda/circuit/field/circuit_field_base.cuh"
#include "cuda/circuit/boolean/circuit_bool.cuh"

#define  CINT_PARAM_CDEF SVM_PARAM_CDEF, uint32_t gl_bits, bool gl_is_signed
#define  CINT_PARAM_DEF SVM_PARAM_DEF, uint32_t gl_bits, bool gl_is_signed
#define  CINT_PARAM_C  SVM_PARAM_C, gl_bits, gl_is_signed
#define  CINT_PARAM  SVM_PARAM, gl_bits, gl_is_signed



class Circuit_Integer
{
public:


    static ALEO_ADI void __revert_bit(nv_uint128 src, nv_uint128 &dest, uint8_t bits)
    {        

        for (int n = 0; n < bits/8; n++)
        {
            uint8_t b = src.v8[n];
            b = ((b * 0x0802LU & 0x22110LU) | (b * 0x8020LU & 0x88440LU)) * 0x10101LU >> 16;
            dest.v8[(bits/8)-1-n] = b;
        }
    }


static ALEO_ADI nv_uint128 __add_nv_int_ptx(const nv_uint128 a, const nv_uint128 b, uint8_t bits, bool &has_carry) 
{
    nv_uint128 result;
    result.stdint = 0;
    if  (bits <= 16)
    {                
        uint32_t ret = a.nvint.x + b.nvint.x;
        has_carry = ( (ret >> bits) != 0 );
        result.nvint.x = ret;
    }
    else if  (bits == 32 )
    {
        result.nvint.x = ptx::add_cc(a.nvint.x, b.nvint.x);
        has_carry = ptx::addc(0,0);
    }
    else if ( bits == 64)
    {
        result.nvint.x = ptx::add_cc(a.nvint.x, b.nvint.x);
        result.nvint.y = ptx::addc_cc(a.nvint.y, b.nvint.y);
        has_carry = ptx::addc(0,0);        
    } else if ( bits == 128)
    {
        result.nvint.x = ptx::add_cc(a.nvint.x, b.nvint.x);
        result.nvint.y = ptx::addc_cc(a.nvint.y, b.nvint.y);
        result.nvint.z = ptx::addc_cc(a.nvint.z, b.nvint.z);
        result.nvint.w = ptx::addc_cc(a.nvint.w, b.nvint.w);
        has_carry = ptx::addc(0,0);        
    }
    return result;
}
    static ALEO_ADI void __karatsuba_multiply_128(CINT_PARAM_CDEF, nv_uint128 first, nv_uint128 second, nv_uint128& dest)
    {
        //大约 200个 leaf
        if ( gl_bits < 128) return;

        nv_uint128 z0;
        {
            nv_uint128 z2, tz;

            // low * low 
            z0.stdint = static_cast<__uint128_t>(first.v64[0]) * static_cast<__uint128_t>(second.v64[0]); 
            // high * high
            z2.stdint = static_cast<__uint128_t>(first.v64[1]) * static_cast<__uint128_t>(second.v64[1]); 

            tz.stdint = ( static_cast<__uint128_t>(first.v64[0]) + static_cast<__uint128_t>(first.v64[1]) ) *  \
                        ( static_cast<__uint128_t>(second.v64[0]) + static_cast<__uint128_t>(second.v64[1]) );            
            
            svm_push_leaf_full(SVM_PARAM, z0.stdint);            
            svm_push_leaf_full(SVM_PARAM, z2.stdint);            
            svm_push_leaf_full(SVM_PARAM, tz.stdint);
        }

        nv_uint128 z1;
        z1.stdint = static_cast<__uint128_t>(first.v64[1]) * static_cast<__uint128_t>(second.v64[0]) + \
                    static_cast<__uint128_t>(first.v64[0]) * static_cast<__uint128_t>(second.v64[1]);
        
        nv_uint128 z_0_plus_scaled_z_1__low, z_0_plus_scaled_z_1__high;
        z_0_plus_scaled_z_1__low.stdint = z1.stdint << ( gl_bits/2 );
        z_0_plus_scaled_z_1__high.stdint = z1.stdint >> ( gl_bits/2 );

        bool has_carry;
        z_0_plus_scaled_z_1__low = __add_nv_int_ptx(z0, z_0_plus_scaled_z_1__low, 128, has_carry);
        if ( has_carry )
        {
            nv_uint128 one; 
            one.stdint = 1;
            z_0_plus_scaled_z_1__high = __add_nv_int_ptx(z_0_plus_scaled_z_1__high, one, 128, has_carry);
        }
        svm_push_leaf_bit_with_u128(SVM_PARAM, z_0_plus_scaled_z_1__low.stdint);
        svm_push_leaf_bit_with_u128(SVM_PARAM, z_0_plus_scaled_z_1__high.stdint, 64+1);
        dest = z_0_plus_scaled_z_1__low;
    }

    static ALEO_ADI void __unsigned_division_via_witness(CINT_PARAM_CDEF, 
                    nv_uint128 dividend_value, nv_uint128 divisor_value, 
                    nv_uint128& dest_quotient, nv_uint128& dest_remainder)
    {
        /*
        如果 bits= 128, 可能产生 800 个 leaf， 可能溢出 share memory,  需要单独执行
        
        */
        if ( gl_is_signed )
            return;
        
        if ( divisor_value.stdint == 0 )
            divisor_value.stdint = 1;
        
        dest_quotient.stdint = dividend_value.stdint / divisor_value.stdint;
        dest_remainder.stdint = dividend_value.stdint % divisor_value.stdint;

        cint_push_var_int(CINT_PARAM, dest_quotient);
        cint_push_var_int(CINT_PARAM, dest_remainder);

        /*
        参考做隔离见证：    
        fn unsigned_division_via_witness(): 

        E::assert_eq(self, quotient.mul_checked(other).add_checked(&remainder));
        quotient = 0x643f9e04f
        other = divisor_value = 4951760157141521099596496896, 0x100000000000000000000000
        
        需要做一次 mul， 一次 add_checked
        mul_checked 最后调用 karatsuba_multiply
        */     
        bigint_u256_t tmp = {.bytes ={0}};   
        if ( gl_bits == 128 ) {
            nv_uint128 dest_tmp;

            __karatsuba_multiply_128(CINT_PARAM_C, dest_quotient, divisor_value, dest_tmp);
            bool tmp_bool;
            dest_tmp = __add_nv_int_ptx(dest_tmp, dest_remainder, gl_bits, tmp_bool);
            cint_push_var_int(CINT_PARAM, dest_tmp);
            operator_less_than(CINT_PARAM_C, dest_remainder, divisor_value, tmp_bool);
        } else {
            nv_uint128 tmp;
            tmp.stdint = dest_quotient.stdint * divisor_value.stdint;
            svm_push_leaf_full(SVM_PARAM, tmp.stdint);
            bool tmp_bool;
            operator_less_than(CINT_PARAM_C, dest_remainder, divisor_value, tmp_bool);
        }
    }

    static ALEO_ADI void cint_push_var_int(CINT_PARAM_DEF, nv_uint128 value)
    {
        svm_push_leaf_bit_with_var_int(SVM_PARAM, value, gl_bits);
    }

    static ALEO_ADI void __circuit_do_bitwise_xor(CINT_PARAM_CDEF, 
        nv_uint128 &self, nv_uint128 &other, nv_uint128 &dest)
    {
        if (gl_bits <= 32) {
            dest.nvint.x = self.nvint.x ^ other.nvint.x;
        } else if (gl_bits == 64) {
            dest.nvint.x = self.nvint.x ^ other.nvint.x;
            dest.nvint.y = self.nvint.y ^ other.nvint.y;
        } else if  (gl_bits == 128) {
            dest.nvint.x = self.nvint.x ^ other.nvint.x;
            dest.nvint.y = self.nvint.y ^ other.nvint.y;
            dest.nvint.z = self.nvint.z ^ other.nvint.z;
            dest.nvint.w = self.nvint.w ^ other.nvint.w;
        }

        if ( SVM_SAVE )
            cint_push_var_int(CINT_PARAM, dest);
    }

    static ALEO_ADI void __circuit_do_bitwise_and(CINT_PARAM_CDEF, 
        nv_uint128 &self, nv_uint128 &other, nv_uint128 &dest)
    {
        if (gl_bits <= 32) {
            dest.nvint.x = self.nvint.x & other.nvint.x;
        } else if (gl_bits == 64) {
            dest.nvint.x = self.nvint.x & other.nvint.x;
            dest.nvint.y = self.nvint.y & other.nvint.y;
        } else if  (gl_bits == 128) {
            dest.nvint.x = self.nvint.x & other.nvint.x;
            dest.nvint.y = self.nvint.y & other.nvint.y;
            dest.nvint.z = self.nvint.z & other.nvint.z;
            dest.nvint.w = self.nvint.w & other.nvint.w;
        }
        if ( SVM_SAVE )
            cint_push_var_int(CINT_PARAM, dest);        
    }

    static ALEO_ADI void __circuit_do_bitwise_or( CINT_PARAM_CDEF, 
        nv_uint128 &self, nv_uint128 &other, nv_uint128 &dest)
    {
        if (gl_bits <= 32) {
            dest.nvint.x = self.nvint.x | other.nvint.x;
        } else if (gl_bits == 64) {
            dest.nvint.x = self.nvint.x | other.nvint.x;
            dest.nvint.y = self.nvint.y | other.nvint.y;
        } else if  (gl_bits == 128) {
            dest.nvint.x = self.nvint.x | other.nvint.x;
            dest.nvint.y = self.nvint.y | other.nvint.y;
            dest.nvint.z = self.nvint.z | other.nvint.z;
            dest.nvint.w = self.nvint.w | other.nvint.w;
        }
        if ( SVM_SAVE )
            cint_push_var_int(CINT_PARAM, dest);        
    } 

    static ALEO_ADI void __circuit_do_bitwise_not(CINT_PARAM_CDEF, 
        nv_uint128 &self, nv_uint128 &dest)
    {
        if (gl_bits < 32) {
            dest.nvint.x = (~self.nvint.x) & ( (1 << gl_bits) - 1 );
        } else if (gl_bits == 32) {
            dest.nvint.x = ~self.nvint.x;
        } else if (gl_bits == 64) {
            dest.nvint.x = ~self.nvint.x;
            dest.nvint.y = ~self.nvint.y;
        } else if  (gl_bits == 128) {
            dest.nvint.x = ~self.nvint.x;
            dest.nvint.y = ~self.nvint.y;
            dest.nvint.z = ~self.nvint.z;
            dest.nvint.w = ~self.nvint.w;
        }
        if ( SVM_SAVE )
            cint_push_var_int(CINT_PARAM, dest);        
    }    

    ALEO_ADI static void to_bigint(nv_uint128 src, bigint_u256_t &dest)
    {
        dest.uint128[0] = src.stdint;
        dest.uint128[1] = 0;
    }

    static ALEO_ADI bool __get_msb_bit(nv_uint128 value, uint32_t bits)
    {
        bool msb = false;
        if (bits <= 32)
        {
            msb = (value.nvint.x >> (bits-1)) & 1;
        }
        else if (bits == 64)
        {
            msb = (value.nvint.y >> 31) & 1;
        }
        else if (bits == 128)
        {
            msb = ( value.nvint.w >> 31 ) & 1;
        }
        return msb;
    }  

    static ALEO_ADI bool cint_vint_is_zero(__uint128_t var_int , uint32_t bits)
    {
        __uint128_t mask = 1;
        mask = ( mask << bits ) - 1;

        return ( (var_int & mask) == 0 );
    }

////////////////

    static ALEO_ADI void operator_or(CINT_PARAM_CDEF, nv_uint128 reg0, nv_uint128 reg1, nv_uint128 &dest)
    {
        __circuit_do_bitwise_or(CINT_PARAM_C, reg0, reg1, dest);
    }

    static ALEO_ADI void operator_xor(CINT_PARAM_CDEF, nv_uint128 reg0, nv_uint128 reg1, nv_uint128 &dest)
    {
        __circuit_do_bitwise_xor(CINT_PARAM_C, reg0, reg1, dest);
    }

    static ALEO_ADI void operator_not(CINT_PARAM_CDEF, nv_uint128 reg0, nv_uint128 &dest)
    {
        is_const = 1; //not instruction don't save witeness
        __circuit_do_bitwise_not(CINT_PARAM_C, reg0, dest);
    }

    static ALEO_ADI void operator_and(CINT_PARAM_CDEF, nv_uint128 reg0, nv_uint128 reg1, nv_uint128 &dest)
    {
        __circuit_do_bitwise_and(CINT_PARAM_C, reg0, reg1, dest);
    }
    

    static ALEO_ADI void operator_is_eq(CINT_PARAM_CDEF, nv_uint128 reg0, nv_uint128 reg1, bool &dest)
    {
        operator_is_not_eq(CINT_PARAM_C, reg0, reg1, dest);
        dest = !dest;
    }

    static ALEO_ADI void operator_is_not_eq_v2(
                CINT_PARAM_CDEF,
                nv_uint128 reg0, nv_uint128 reg1,  bool &ret
            )
    {
        nv_uint128 dest;
        dest.stdint = 0;

        is_const = 1;
        __circuit_do_bitwise_xor(CINT_PARAM_C, reg0, reg1, dest);
        
        ret = !(dest.stdint == 0 );
                
        svm_push_leaf_one_bit_bool(SVM_PARAM, ret);    

        bool flag = reg0.stdint < reg1.stdint ;
        reg0.stdint -= reg1.stdint;
        
        if ( reg0.stdint == 0 )
        {
            svm_push_leaf_one_bit_bool(SVM_PARAM, true);
            return;
        }

        if ( gl_bits <=16 )
        {
            uint32_t *p =  ((uint32_t*)ctx) - 1024*1024;
            int v16 = (__int128_t)reg0.stdint;            
            uint32_t offset ;
            if ( v16 > 0) {
                offset = v16 - 1;
            } else {
                offset = 65534 - v16;
            }

            bigint_u256_t leaf;
            
            for ( int n = 0; n < 8; n++)
            {
                leaf.uint32[n] = p[offset*8 + n];
            }

            svm_push_leaf_full(SVM_PARAM, leaf);
            return ;
            
        } else {
            uint64_t field_self[QBigInt::BIGINT_LEN] = {0};
                    
            QBigInt::merge_from_u128(field_self, reg0.stdint, 0);
            if ( flag )
            {
                field_self[2] = UINT64_MAX;
                field_self[3] = UINT64_MAX;
                QBigInt::add_with_modulus(field_self);
            }
            
            QBigInt::bigint_to_field(field_self);        

            if ( bigint_inverse_for_quick_bigint(field_self)  )
            {
                QBigInt::bigint_from_field(field_self);
                svm_push_leaf_full(SVM_PARAM, field_self);
            } else {            
                svm_push_leaf_one_bit_bool(SVM_PARAM, true);
            }

        }
    }

    static ALEO_ADI void operator_is_not_eq(
                CINT_PARAM_CDEF,
                nv_uint128 reg0, nv_uint128 reg1,  bool &ret
            )
    {
        nv_uint128 dest;
        dest.stdint = 0;

        is_const = 1;
        __circuit_do_bitwise_xor(CINT_PARAM_C, reg0, reg1, dest);
        
        ret = !(dest.stdint == 0 );
                
        svm_push_leaf_one_bit_bool(SVM_PARAM, ret);        

        uint64_t field_self[QBigInt::BIGINT_LEN] = {0};
        
        {
            bool flag = reg0.stdint < reg1.stdint ;
            reg0.stdint -= reg1.stdint;            
            
            QBigInt::merge_from_u128(field_self, reg0.stdint, 0);
            if ( flag )
            {
                field_self[2] = UINT64_MAX;
                field_self[3] = UINT64_MAX;
                QBigInt::add_with_modulus(field_self);
            }        
        }
        
        QBigInt::bigint_to_field(field_self);
        

        if ( bigint_inverse_for_quick_bigint(field_self)  )
        {
            QBigInt::bigint_from_field(field_self);            
            svm_push_leaf_full(SVM_PARAM, field_self);
        } else {            
            svm_push_leaf_one_bit_bool(SVM_PARAM, true);
        }
    }   

        /*
    该算法从  fields/src/fp_256.rs 里的 fn from_bigint() 转换而来

    核心是 ： fn mul_assign(&mut self, other: &Self)
    */    
    static ALEO_ADI void to_field(nv_uint128 src, bigint_u256_t &dest)
    {           
        to_bigint(src, dest);
        bigint_to_field(dest);
    }

    static ALEO_ADI  void operator_neg(
                            CINT_PARAM_CDEF,nv_uint128 src, nv_uint128 &ret)
    {
/*


neg r2 into r14;
neg r4 into r15;
neg r6 into r16;
neg r8 into r17;
neg r10 into r18;

is.eq r3 r14 into r19;
is.eq r5 r15 into r20;
is.eq r7 r16 into r21;
is.eq r9 r17 into r22;
is.eq r11 r18 into r23;

///////////////////


sub.w r2 r2 into r14;                                                                                                           
neg r14 into r15;
is.eq r3 r15 into r16;

sub.w r14 1i8 into r17;
neg r17 into r18;
is.eq r3 r18 into r19;

*/
        if ( gl_is_signed )
        {
            __uint128_t tmp = ~src.stdint + 1;
            bool carry = false;
            /*假定这里的 overflow 检测正确  circuit/types/integers/src/add_checked.rs */
            if ( tmp < 1 || tmp < ~src.stdint)
            {
                carry = true;
            }
            ret.stdint = tmp;
            cint_push_var_int(CINT_PARAM, ret);
            svm_push_leaf_one_bit_bool(SVM_PARAM, carry);
            svm_push_leaf_one_bit_bool(SVM_PARAM, false);
        }   
    }

    template <int BITS>
    static ALEO_ADI  void operator_square_fast(
        CINT_PARAM_CDEF, nv_uint128 self,  nv_uint128 &dest)
    {
        nv_uint128 first;
        nv_uint128 second;        

        //类似 mul_wrap ，但是少一次乘法
        #define MY_SQUARE_FAST(x1, x0, b, b2, z_plus_scale) do {\
            uint##b##_t z0 = (uint##b##_t)(x0)*(uint##b##_t)(x0);\
            uint##b##_t z1_a = (uint##b##_t)(x1)*(uint##b##_t)(x0);\
            if (SVM_SAVE ) \
            {\
                uint64_t tmp[QBigInt::BIGINT_LEN] = { 0 };\
                tmp[0] = z0;\
                svm_push_leaf_full(SVM_PARAM, tmp);\
                tmp[0] = z1_a;\
                svm_push_leaf_full(SVM_PARAM, tmp);\
                svm_push_leaf_full(SVM_PARAM, tmp);\
            }\
            __uint##b2##_t bm = 1ULL<<(b/2);\
            __uint##b2##_t z1 = (__uint##b2##_t)z1_a + (__uint##b2##_t)z1_a;\
            z_plus_scale = (__uint##b2##_t)z0 + (z1*bm);\
            svm_push_leaf_bit_with_u##b2(SVM_PARAM, z_plus_scale, BITS+(BITS/2)+1);\
            ;\
        } while(0)

        if constexpr (BITS == 8)
        {
            __uint16_t z_plus_scale;
            MY_SQUARE_FAST( 
                (self.nvint.x >> 4),
                (self.nvint.x & 0xF),                
                8, 16, z_plus_scale);
            dest.stdint = static_cast<uint8_t>(z_plus_scale);
        }
        else if constexpr (BITS == 16)
        {            
            __uint32_t z_plus_scale;
            MY_SQUARE_FAST( 
                (self.nvint.x >> 8),
                (self.nvint.x & 0xFF),
                16, 32, z_plus_scale);
            dest.stdint = static_cast<uint16_t>(z_plus_scale);
        }
        else if constexpr (BITS == 32)
        {            
            __uint64_t z_plus_scale;
            MY_SQUARE_FAST( 
                (self.nvint.x >> 16),
                (self.nvint.x & 0xFFFF),
                32, 64, z_plus_scale);
            dest.stdint = static_cast<uint32_t>(z_plus_scale);
        }
        else if constexpr (BITS == 64)
        {            
            __uint128_t z_plus_scale;
            MY_SQUARE_FAST(
                (self.nvint.y),
                (self.nvint.x),
                64, 128, z_plus_scale);
            dest.stdint = static_cast<uint64_t>(z_plus_scale);
        }
        else if constexpr (BITS == 128)
        {            
            nv_uint128 x0, x1, z0, z1_a;
            constexpr uint64_t half_mask = UINT64_MAX;

            x0.stdint = self.stdint & half_mask;
            x1.stdint = self.stdint >> (BITS/2);

            z0.stdint = (x0.stdint * x0.stdint);
            z1_a.stdint = (x1.stdint * x0.stdint);

            if ( SVM_SAVE ) {            
                uint64_t tmp[QBigInt::BIGINT_LEN] = { 0 };
                tmp[0] = z0.stdint; 
                tmp[1] = z0.stdint >> 64;
                svm_push_leaf_full(SVM_PARAM, tmp);

                tmp[0] = z1_a.stdint; 
                tmp[1] = z1_a.stdint >> 64;
                svm_push_leaf_full(SVM_PARAM, tmp);                
                svm_push_leaf_full(SVM_PARAM, tmp);
            }

            __uint128_t z1 = z1_a.stdint + z1_a.stdint;
            __uint128_t carry = 0;
            if ( z1 < z1_a.stdint || z1 < z1_a.stdint)
                carry = 1;
            __uint128_t z1_low = z1<<64; 
            __uint128_t z1_high = z1 >> 64;
            if ( carry > 0 )
                z1_high += carry << 64;
            
            dest.stdint = z1_low + z0.stdint;
            if ( dest.stdint < z0.stdint || dest.stdint < z1_low )
                z1_high += 1;
            
            // total bits: 128 + (128/2) + 1;
            svm_push_leaf_bit_with_u128(SVM_PARAM, dest.stdint);
            svm_push_leaf_bit_with_u128(SVM_PARAM, z1_high, 64 + 1);

        }

        #undef MY_SQUARE_FAST
    }

    template <int BITS>
    static ALEO_ADI  void operator_mul_wrap_fast(
        CINT_PARAM_CDEF,nv_uint128 first, nv_uint128 second,  nv_uint128 &dest)
    {

    #define MY_MULW(x1, x0, y1, y0, b, b2, z_plus_scale) do {\
        uint##b##_t z0 = (uint##b##_t)(x0)*(uint##b##_t)(y0);\
        uint##b##_t z1_a = (uint##b##_t)(x1)*(uint##b##_t)(y0);\
        uint##b##_t z1_b = (uint##b##_t)(x0)*(uint##b##_t)(y1);\
        if (SVM_SAVE ) \
        {\
            uint64_t tmp[QBigInt::BIGINT_LEN] = { 0 };\
            tmp[0] = z0;\
            svm_push_leaf_full(SVM_PARAM, tmp);\
            tmp[0] = z1_a;\
            svm_push_leaf_full(SVM_PARAM, tmp);\
            tmp[0] = z1_b;\
            svm_push_leaf_full(SVM_PARAM, tmp);\
        }\
        __uint##b2##_t bm = 1ULL<<(b/2);\
        __uint##b2##_t z1 = (__uint##b2##_t)z1_a + (__uint##b2##_t)z1_b;\
        z_plus_scale = (__uint##b2##_t)z0 + (z1*bm);\
        svm_push_leaf_bit_with_u##b2(SVM_PARAM, z_plus_scale, BITS+(BITS/2)+1);\
    } while(0)
        if constexpr (BITS == 8)
        {
            __uint16_t z_plus_scale;
            MY_MULW( 
                (first.nvint.x >> 4),
                (first.nvint.x & 0xF),
                (second.nvint.x >> 4),
                (second.nvint.x & 0xF),
                8, 16, z_plus_scale);
            dest.stdint = static_cast<uint8_t>(z_plus_scale);
        }
        else if constexpr (BITS == 16)
        {            
            __uint32_t z_plus_scale;
            MY_MULW( 
                (first.nvint.x >> 8),
                (first.nvint.x & 0xFF),
                (second.nvint.x >> 8),
                (second.nvint.x & 0xFF), 
                16, 32, z_plus_scale);
            dest.stdint = static_cast<uint16_t>(z_plus_scale);            
        }
        else if constexpr (BITS == 32)
        {            
            __uint64_t z_plus_scale;
            MY_MULW( 
                (first.nvint.x >> 16),
                (first.nvint.x & 0xFFFF),
                (second.nvint.x >> 16),
                (second.nvint.x & 0xFFFF), 
                32, 64, z_plus_scale);
            dest.stdint = static_cast<uint32_t>(z_plus_scale);
        }
        else if constexpr (BITS == 64)
        {            
            __uint128_t z_plus_scale;
            MY_MULW(
                (first.nvint.y),
                (first.nvint.x),
                (second.nvint.y),
                (second.nvint.x),
                64, 128, z_plus_scale);
            dest.stdint = static_cast<uint64_t>(z_plus_scale);
        }
        else if constexpr (BITS == 128)
        {
            operator_mul_wrap(CINT_PARAM_C, first, second, dest);
        }
        #undef MY_MULW
    } 
    
    static ALEO_ADI  void operator_mul_wrap(
                            CINT_PARAM_CDEF,nv_uint128 reg0, nv_uint128 reg1,  nv_uint128 &ret)
    {
    /*
    mul.w r2 r3 into r14;
    is.eq r2 r14 into r15;
    mul.w r4 r5 into r16;
    is.eq r4 r16 into r17;
    mul.w r6 r7 into r18;
    mul.w r8 r9 into r19;
    is.eq r8 r19 into r20;

    ////////

    mul.w r10 128i128 into r14;
    is.eq r14 r11 into r15;

    mul.w r8 128i64 into r14
    is.eq r14 r9 into r15;
    */
        nv_uint128 x0, x1,y0,y1 , z0, z1_a, z1_b;
        uint64_t half_mask;
        if ( gl_bits <= 64)
            half_mask = (1ULL <<(gl_bits/2)) - 1 ;
        else 
            half_mask = UINT64_MAX;
        
        __uint128_t mask = 1;
        if ( gl_bits <= 64)
            mask = (mask << gl_bits) - 1;
        else 
            mask = (__uint128_t)-1;
        
        x0.stdint = reg0.stdint & half_mask;
        x1.stdint = reg0.stdint >> (gl_bits/2);

        y0.stdint = reg1.stdint & half_mask;
        y1.stdint = reg1.stdint >> (gl_bits/2);        

        z0.stdint = (x0.stdint * y0.stdint);
        z1_a.stdint = (x1.stdint * y0.stdint);
        z1_b.stdint = (x0.stdint * y1.stdint);

        if ( SVM_SAVE ) {            
            uint64_t tmp[QBigInt::BIGINT_LEN] = { 0 };
            if ( gl_bits <= 64)
            {
                tmp[0] = z0.stdint;
                svm_push_leaf_full(SVM_PARAM, tmp);
                tmp[0] = z1_a.stdint;
                svm_push_leaf_full(SVM_PARAM, tmp);
                tmp[0] = z1_b.stdint;
                svm_push_leaf_full(SVM_PARAM, tmp);
            } else {
                tmp[0] = z0.stdint; 
                tmp[1] = z0.stdint >> 64;
                svm_push_leaf_full(SVM_PARAM, tmp);

                tmp[0] = z1_a.stdint; 
                tmp[1] = z1_a.stdint >> 64;
                svm_push_leaf_full(SVM_PARAM, tmp);

                tmp[0] = z1_b.stdint;
                tmp[1] = z1_b.stdint >> 64;
                svm_push_leaf_full(SVM_PARAM, tmp);
            }
        }

        if ( gl_bits <= 64)
        {
            __uint128_t bm = 1ULL << (gl_bits/2);
            __uint128_t z1 = z1_a.stdint + z1_b.stdint;
            nv_uint128 z_plus_scale;
            z_plus_scale.stdint = z0.stdint + (z1*bm);
            //svm_push_leaf_bits_from_pointer(SVM_PARAM, &z_plus_scale.stdint, gl_bits + (gl_bits/2) + 1);
            svm_push_leaf_bit_with_u128(SVM_PARAM, z_plus_scale.stdint, gl_bits + (gl_bits/2) + 1);
            ret.stdint = z_plus_scale.stdint & mask;
        } else {
            __uint128_t z1 = z1_a.stdint + z1_b.stdint;
            __uint128_t carry = 0;
            if ( z1 < z1_a.stdint || z1 < z1_b.stdint)
                carry = 1;
            __uint128_t z1_low = z1<<64; 
            __uint128_t z1_high = z1 >> 64;
            if ( carry > 0 )
                z1_high += carry << 64;
            
            ret.stdint = z1_low + z0.stdint;
            if ( ret.stdint < z0.stdint || ret.stdint < z1_low )
                z1_high += 1;
            
            // total bits: 128 + (128/2) + 1;
            svm_push_leaf_bit_with_u128(SVM_PARAM, ret.stdint);
            svm_push_leaf_bit_with_u128(SVM_PARAM, z1_high, 64 + 1);
        }
    }

    
    static ALEO_ADI  void operator_ternary(
            CINT_PARAM_CDEF,
            bool condition,
            nv_uint128 first,
            nv_uint128 second,
            nv_uint128 &dest)
    {
        bool cond_const = is_const & 1;
        bool first_const = ((is_const>>1) & 1);
        bool second_const = ((is_const>>2) & 1);
        
        if ( cond_const ) {
            //printf("not implement operator_ternaty_full 1\n");
            return;
        } else if ( first_const ) {
            //printf("not implement operator_ternaty_full 2\n");
            return;
        }

        if ( second_const ) {
            nv_uint128 not_condition_int;
            if ( condition )
            {
                not_condition_int.stdint = 0;                
            } else {
                not_condition_int.stdint = static_cast<__uint128_t>(-1);
            }
            __uint128_t mask = ~second.stdint;
            nv_uint128 part1;
            part1.stdint = second.stdint & ( first.stdint | not_condition_int.stdint );
            __uint128_t part2 = mask & ( first.stdint & (~not_condition_int.stdint)  );
            part1.stdint |= part2;
            
            cint_push_var_int(CINT_PARAM, part1);
            dest = condition ? first : second;
        } else {
            //都不是 const 
            dest = condition ? first : second;            
            cint_push_var_int(CINT_PARAM, dest);
        }
    }

    //ret is carry
    static ALEO_ADI bool cint_add_with_carry(__uint128_t v1, __uint128_t v2, nv_uint128 &dest, uint32_t bits)
    {
        bool carry;
        dest.stdint = v1 + v2;
        if ( bits < 128) {
            carry = (dest.stdint >> bits) & 1;
        } else {            
            if ( dest.stdint < v1 || dest.stdint < v2 ) {
                carry = true;
            } else {
                carry = false;
            }
        }
        return carry;
    }

    static ALEO_ADI void operator_sub_wrap(
        CINT_PARAM_CDEF,
        nv_uint128 first,
        nv_uint128 second,
        nv_uint128 &dest)
    {
/*
sub.w r2 r2 into r14;
is.eq r2 r14 into r15;
sub.w r2 r14 into r16;
is.eq r2 r16 into r17;

sub.w r2 0i8 into r18;
is.eq r2 r18 into r19;
sub.w r2 -1i8 into r20;
sub.w r2 r20 into r21;
sub.w r2 127i8 into r22;
sub.w r2 r22 into r23;
is.eq r2 r18 into r24;

======
sub.w r2 r3 into r14;
sub.w r4 r5 into r15;
sub.w r6 r7 into r16;
sub.w r8 r9 into r17;
sub.w r10 r11 into r18;
is.eq r18 r10 into r19;

sub.w r10 0i128 into r20;
======

*/
        __circuit_do_bitwise_not(SVM_PARAM, 1, gl_bits, gl_is_signed, second, second);
        dest.stdint = first.stdint + second.stdint;
        bool carry1 = cint_add_with_carry(first.stdint, second.stdint, dest, gl_bits);
                    
        nv_uint128 dest2;
        bool carry2 = cint_add_with_carry(dest.stdint, 1, dest2, gl_bits);

        dest = dest2;
        cint_push_var_int(CINT_PARAM, dest2);
        svm_push_leaf_one_bit_bool(SVM_PARAM, carry1 | carry2);
    }

    static ALEO_ADI void operator_add_wrap(
        CINT_PARAM_CDEF,
        nv_uint128 first,
        nv_uint128 second,
        nv_uint128 &dest)
    {
        /*
add.w r2 r3 into r14;
add.w r2 r14 into r15;
add.w r2 r15 into r16;
is.eq r2 r16 into r17;
add.w r2 1i8 into r18;
is.eq r2 r18 into r19;

32bit:
add.w r6 r7 into r14;
add.w r6 r14 into r15;
add.w r6 r15 into r16;
is.eq r6 r16 into r17;
add.w r6 100i32 into r18;
is.eq r6 r18 into r19;

64bit:
add.w r8 r9 into r14;
add.w r8 r14 into r15;
add.w r8 r15 into r16;
is.eq r8 r16 into r17;
add.w r8 100i64 into r18;
is.eq r8 r18 into r19;

128 bit:
add.w r10 r11 into r14;
add.w r10 r14 into r15;
add.w r10 r15 into r16;
is.eq r10 r16 into r17;
add.w r10 100i128 into r18;
is.eq r10 r18 into r19;

unsigned 64 bit:

cast.lossy r12 into r14 as u64;
cast.lossy r12 into r15 as u64;
add.w r14 r15 into r16;
add.w r14 r16 into r17;
add.w r14 r17 into r18;
add.w r14 r18 into r19;
add.w r14 r19 into r20;
is.eq r14 r20 into r21;

u128:
cast.lossy r12 into r14 as u128;
cast.lossy r12 into r15 as u128;
add.w r14 r15 into r16;
add.w r14 r16 into r17;
add.w r14 r17 into r18;
add.w r14 r18 into r19;
add.w r14 r19 into r20;
is.eq r14 r20 into r21;

*/
        if ( gl_bits < 128)
        {
            dest.stdint = first.stdint + second.stdint;
            svm_push_leaf_bit_with_u128(SVM_PARAM, dest.stdint, gl_bits + 1);
        } else {
            // 128 bits
            bool carry;            
            dest = __add_nv_int_ptx(first, second, gl_bits, carry);
            cint_push_var_int(CINT_PARAM, dest);
            svm_push_leaf_one_bit_bool(SVM_PARAM,  carry);
        }        
    }




    static ALEO_ADI void operator_less_than(
            CINT_PARAM_CDEF,
            nv_uint128 first,
            nv_uint128 second,
            bool &dest)
    {
    /*
        参考代码： circuit/types/integers/src/compare.rs  =>  fn is_less_than

        因为这个是精简版本实现，没有考虑 coefficient, 所以这里假定 first 和 second 指向不同的数字

        测试指令:
                lt r2 r3 into r14;
                lt r4 r5 into r15;
                lt r6 r7 into r16;
                lt r8 r9 into r17;
                lt r10 r11 into r18;
                lt r12 r13 into r19;

                lt r3 r2 into r20;
                lt r5 r4 into r21;
                lt r7 r6 into r22;
                lt r9 r8 into r23;
                lt r11 r10 into r24;
                lt r13 r12 into r25;

                和
                cast.lossy r12 into r14 as u8;
                cast.lossy r12 into r15 as u8;
                cast.lossy r13 into r16 as u8;
                lte r15 r14 into r17;
                xor r0 r17 into r18;
                lte r15 r16 into r19;
                xor r0 r19 into r20;
                lte r16 r15 into r21;
                xor r0 r21 into r22;

                和 
                cast.lossy r12 into r14 as u8;
                cast.lossy r13 into r15 as u8;
                lt r14 r15 into r16;
                lt r15 r14 into r17;

                cast.lossy r12 into r18 as u128;
                cast.lossy r13 into r19 as u128;
                lt r18 r19 into r20;
                lt r19 r18 into r21;

                lt r4 r5 into r22;
                lt r5 r4 into r23;
                lt r6 r7 into r24;
                is.eq r22 r24 into r25;
                is.eq r23 r24 into r26;

        */

    /*
    追加的测试代码： 


    cast.lossy r12 into r14 as u8;
    cast.lossy r13 into r15 as u8;
    lt r14 r15 into r16;
    lt r14 r14 into r17;

    cast.lossy r12 into r14 as u16;
    cast.lossy r13 into r15 as u16;
    lt r14 r15 into r16;
    lt r15 r14 into r17;

    cast.lossy r12 into r14 as u32;
    cast.lossy r13 into r15 as u32;
    lt r14 r15 into r16;
    lt r15 r14 into r17;
    lt r14 r14 into r18;
    sub.w r14 r14 into r19;
    lt r15 r19 into r20;
    lt r15 0u32 into r21;
    lt r15 10000u32 into r22;

    cast.lossy r12 into r14 as u64;
    cast.lossy r13 into r15 as u64;
    lt r14 r15 into r16;
    lt r15 r14 into r17;
    lt r14 r14 into r18;
    sub.w r14 r14 into r19;
    lt r15 r19 into r20;
    lt r15 0u64 into r21;
    lt r15 10000u64 into r22;


    cast.lossy r12 into r14 as u128;
    cast.lossy r13 into r15 as u128;
    lt r14 r15 into r16;
    lt r15 r14 into r17;
    lt r14 r14 into r18;
    sub.w r14 r14 into r19;
    lt r15 r19 into r20;
    lt r15 0u128 into r21;
    lt r15 10000u128 into r22;

    */    
        
        if (  gl_is_signed )
        {
            bool msb_first = __get_msb_bit(first, gl_bits);
            bool msb_second = __get_msb_bit(second, gl_bits);
            bool same_sign;
            Circuit_Bool::operator_is_eq(SVM_PARAM_C, msb_first, msb_second, same_sign);
            bool self_is_negative_and_other_is_positive;
            Circuit_Bool::operator_and(SVM_PARAM_C, msb_first, !msb_second, self_is_negative_and_other_is_positive);
            
            __circuit_do_bitwise_not(SVM_PARAM, 1, gl_bits, gl_is_signed, second, second);

            nv_uint128 tmp;
            bool carry1 = cint_add_with_carry(first.stdint, second.stdint, tmp, gl_bits);
            bool carry2 = cint_add_with_carry(tmp.stdint, 1, tmp, gl_bits);

            bool carry = carry1 | carry2;            
                        
            cint_push_var_int(CINT_PARAM, tmp);
            svm_push_leaf_one_bit_bool(SVM_PARAM, carry);            
            
            bool ternary = same_sign ? !carry : self_is_negative_and_other_is_positive;
            svm_push_leaf_one_bit_bool(SVM_PARAM, ternary);
            dest = ternary;
        } else {
            __circuit_do_bitwise_not(SVM_PARAM, 1, gl_bits, gl_is_signed, second, second);
            nv_uint128 tmp;
            bool carry1 = cint_add_with_carry(first.stdint, second.stdint, tmp, gl_bits);
            bool carry2 = cint_add_with_carry(tmp.stdint, 1, tmp, gl_bits);
                        
            cint_push_var_int(CINT_PARAM, tmp);
            svm_push_leaf_one_bit_bool(SVM_PARAM, carry1|carry2);
            
            dest =  !(carry1 | carry2);
        }
    }


    static ALEO_ADI void operator_pow_wrap_prepare(
            CINT_PARAM_CDEF,
            nv_uint128 self,
            uint32_t exponent,
            uint8_t exponent_bits
            )
    {
/*
cast.lossy r12 into r14 as u32;
pow.w r8 r14 into r15;
pow.w r10 r14 into r16;
is.eq r9 r15 into r17;
is.eq r11 r16 into r18;

*/        
        //big endian
        uint32_t n = exponent_bits/8 - 1, m = 7;

        {
            bigint_u256_t tmp;
            tmp.uint128[0] = self.stdint;
            tmp.uint128[1] = 1; //存dest
            smem_save_circuit_params(0, tmp);            
        }

        smem_save_circuit_params_by_u32(1, 0, n);
        smem_save_circuit_params_by_u32(1, 1, m);
        smem_save_circuit_params_by_u32(1, 2, exponent);
        smem_save_circuit_params_by_u32(1, 3, exponent_bits);
        smem_save_circuit_params_by_u32(1, 4, gl_bits);
        smem_save_circuit_params_by_u32(1, 5, gl_is_signed);
    }

    static ALEO_ADI bool operator_pow_wrap_run(SVM_PARAM_DEF)
    {
        uint32_t n, m , exponent;
        uint8_t exponent_bits;
        nv_uint128 dest;
        nv_uint128 self;
        uint32_t gl_bits;
        constexpr bool gl_is_signed = false;        

        {
            // bigint_u256_t tmp;
            // smem_load_circuit_params(0, tmp);
            // self.stdint = tmp.uint128[0];
            // dest.stdint = tmp.uint128[1];

            uint64_t tmp[QBigInt::BIGINT_LEN];
            smem_load_circuit_params(0, tmp);

            // self : low, dest: high 
            QBigInt::split_to_u128(tmp, self.stdint, dest.stdint);
        }
        n = smem_load_circuit_params_by_u32(1,0);
        m = smem_load_circuit_params_by_u32(1,1);
        exponent = smem_load_circuit_params_by_u32(1,2);
        exponent_bits = smem_load_circuit_params_by_u32(1, 3);
        gl_bits = smem_load_circuit_params_by_u32(1, 4);

        if  ( gl_bits >= 32)
            smem_set_circuit_is_u8_leaf_hash(1);
        else if ( gl_bits == 16 )
            smem_set_circuit_is_u8_leaf_hash(2);
        else if ( gl_bits == 8 )
            smem_set_circuit_is_u8_leaf_hash(3);

        uint8_t p = ((uint8_t*)(&exponent))[n];
        bool condition = (p>>m) & 1;

        bool result_is_const = ( ( n == (exponent_bits/8 - 1) ) && m == 7 );
        
        if ( result_is_const == 1 ) {
            /*对于：  result = result.mul_wrapped(&result);
            如果 result 还是常量， 直接乘
            */                
            //dest = dest * dest;
        }
        else {
            //Circuit_Integer::operator_mul_wrap(SVM_PARAM, result_is_const, gl_bits, gl_is_signed, dest, dest, dest);
            #if 1
            if ( gl_bits == 8)
                operator_square_fast<8>(SVM_PARAM, result_is_const, gl_bits, gl_is_signed, dest, dest);
            else if ( gl_bits == 16)
                operator_square_fast<16>(SVM_PARAM, result_is_const, gl_bits, gl_is_signed, dest, dest);
            else if ( gl_bits == 32)
                operator_square_fast<32>(SVM_PARAM, result_is_const, gl_bits, gl_is_signed, dest, dest);
            else if ( gl_bits == 64)
                operator_square_fast<64>(SVM_PARAM, result_is_const, gl_bits, gl_is_signed, dest, dest);
            else if ( gl_bits == 128)
                operator_square_fast<128>(SVM_PARAM, result_is_const, gl_bits, gl_is_signed, dest, dest);
            #else 
            if ( gl_bits == 8)
                operator_mul_wrap_fast<8>(SVM_PARAM, result_is_const, gl_bits, gl_is_signed, dest, dest, dest);
            else 
                operator_mul_wrap(SVM_PARAM, result_is_const, gl_bits, gl_is_signed, dest, dest, dest);
            #endif
        }
        
        nv_uint128 tmp;
        {
            #if 1
            if ( gl_bits == 8)
                operator_mul_wrap_fast<8>(SVM_PARAM, result_is_const, gl_bits, gl_is_signed, dest, self, tmp);
            else if ( gl_bits == 16)
                operator_mul_wrap_fast<16>(SVM_PARAM, result_is_const, gl_bits, gl_is_signed, dest, self, tmp);
            else if ( gl_bits == 32)
                operator_mul_wrap_fast<32>(SVM_PARAM, result_is_const, gl_bits, gl_is_signed, dest, self, tmp);
            else if ( gl_bits == 64)
                operator_mul_wrap_fast<64>(SVM_PARAM, result_is_const, gl_bits, gl_is_signed, dest, self, tmp);
            else if ( gl_bits == 128)
                Circuit_Integer::operator_mul_wrap(SVM_PARAM, result_is_const, gl_bits, gl_is_signed, dest, self, tmp);
            #else 

            if ( gl_bits == 8)
                operator_mul_wrap_fast<8>(SVM_PARAM, result_is_const, gl_bits, gl_is_signed, dest, self, tmp);
            else
                Circuit_Integer::operator_mul_wrap(SVM_PARAM, result_is_const, gl_bits, gl_is_signed, dest, self, tmp);

            #endif
        }
        
        
        if ( result_is_const ) {
            //4 表示 second is const
            Circuit_Integer::operator_ternary(SVM_PARAM, 4, gl_bits, gl_is_signed, condition, tmp, dest , dest );
        }                
        else {
            Circuit_Integer::operator_ternary(SVM_PARAM, 0, gl_bits, gl_is_signed, condition, tmp, dest , dest );
        }

        {
            // bigint_u256_t tmp;
            // tmp.uint128[0] = self.stdint;
            // tmp.uint128[1] = dest.stdint;
            // smem_save_circuit_params(0, tmp);

            uint64_t tmp[QBigInt::BIGINT_LEN];
            QBigInt::merge_from_u128(tmp, self.stdint, dest.stdint);
            smem_save_circuit_params(0, tmp);
        }        
        
        if ( m == 0)
        {
            if ( n == 0 )
                return true;            
            m = 7;
            n--;
        } else {
            m--;
        }        

        smem_save_circuit_params_by_u32(1, 0, n);
        smem_save_circuit_params_by_u32(1, 1, m);

        return false;
    }
    
    static ALEO_ADI void operator_abs_wrap(
                        CINT_PARAM_CDEF,
                        nv_uint128 first,
                        nv_uint128 &dest
    )
    {
/*
abs.w r2 into r14;
abs.w r4 into r15;
abs.w r6 into r16;
abs.w r8 into r17;
abs.w r10 into r18;
is.eq r3 r14 into r19;
is.eq r5 r15 into r20;
is.eq r7 r16 into r21;
is.eq r9 r17 into r22;
is.eq r11 r18 into r23;
*/        
        nv_uint128 tmp; 
        tmp.stdint = 0;        
        bool msb = __get_msb_bit(first, gl_bits);

        operator_sub_wrap(CINT_PARAM_C, tmp, first, tmp);
        operator_ternary(CINT_PARAM_C, msb, tmp, first, dest);
    }

    static ALEO_ADI bool operator_div_wrap_signed_prepare(
                        CINT_PARAM_CDEF,
                        nv_uint128 dividend,
                        nv_uint128 divisor                        
    )
    {
        nv_uint128 abs_dividend, abs_divisor;
        operator_abs_wrap(CINT_PARAM_C, dividend, abs_dividend);
        operator_abs_wrap(CINT_PARAM_C, divisor, abs_divisor);

        uint32_t msb1 = __get_msb_bit(dividend, gl_bits);
        uint32_t msb2 = __get_msb_bit(divisor, gl_bits);

        {
            uint64_t tmp[QBigInt::BIGINT_LEN];
            QBigInt::merge_from_u128(tmp, abs_dividend.stdint, abs_divisor.stdint);
            smem_save_circuit_params(0, tmp);
        }
        smem_save_circuit_params_by_u32(1, 0, gl_bits);
        uint32_t meta = (msb2<<1) | msb1;
        smem_save_circuit_params_by_u32(1, 1, meta);
        return false;
    }

    static ALEO_ADI bool operator_div_wrap_signed_exec_1(SVM_PARAM_DEF)
    {
        nv_uint128 abs_dividend, abs_divisor;
        uint32_t gl_bits = smem_load_circuit_params_by_u32(1, 0);
        {
            uint64_t tmp[QBigInt::BIGINT_LEN];
            smem_load_circuit_params(0, tmp);
            
            abs_dividend.stdint = QBigInt::make_u128_with_two_u64(tmp[0], tmp[1]);
            abs_divisor.stdint = QBigInt::make_u128_with_two_u64(tmp[2], tmp[3]);
        }

        nv_uint128 dest_quotient, dest_remainder;
        uint8_t is_const = 0;
        bool gl_is_signed = false;
        __unsigned_division_via_witness(CINT_PARAM_C, abs_dividend, abs_divisor, dest_quotient, dest_remainder);
        
        for(int n = 0; n < 4; n++)
        {
            smem_save_circuit_params_by_u32(0, n, dest_quotient.v32[n]);
        }

        return false;
    }

    static ALEO_ADI bool operator_div_wrap_signed_exec_2(SVM_PARAM_DEF)
    {
        uint32_t gl_bits = smem_load_circuit_params_by_u32(1, 0);
        uint32_t meta = smem_load_circuit_params_by_u32(1, 1);
        bool msb1 = meta & 1;
        bool msb2 = (meta >> 1) & 1;
        bool gl_is_signed = true;
        uint8_t is_const = 0;
        nv_uint128 dest_quotient;

        {
            uint64_t tmp[QBigInt::BIGINT_LEN];
            smem_load_circuit_params(0, tmp);
            dest_quotient.stdint = QBigInt::make_u128_with_two_u64(tmp[0], tmp[1]);
        }

        bool same_sign;
        Circuit_Bool::operator_is_eq(SVM_PARAM_C, msb1, msb2, same_sign);
        nv_uint128 zero;
        zero.stdint = 0;

        nv_uint128 dest;
        operator_sub_wrap(CINT_PARAM_C, zero, dest_quotient, dest);
        operator_ternary(CINT_PARAM_C, same_sign, dest_quotient, dest, dest);
        
        for(int n = 0; n < 4; n++)
        {
            smem_save_circuit_params_by_u32(0, n, dest.v32[n]);
        }

        return true;
    }


    static ALEO_ADI bool operator_div_wrap_unsigned(
                        CINT_PARAM_CDEF,
                        nv_uint128 first,
                        nv_uint128 second,
                        nv_uint128 &dest
    )
    {        
/*
cast.lossy r12 into r14 as u128;
cast.lossy r13 into r15 as u128;
div.w r14 r15 into r16;
div.w r15 r14 into r17;
is.eq r14 r16 into r18;
is.eq r14 r17 into r19;

=========

cast.lossy r12 into r14 as u64;
cast.lossy r13 into r15 as u64;
div.w r14 r15 into r16;
div.w r15 r14 into r17;
is.eq r14 r16 into r18;
is.eq r14 r17 into r19;

*/        

        nv_uint128 dest_remainder;
        __unsigned_division_via_witness(CINT_PARAM_C, first, second, dest, dest_remainder);
        return true;
    }

    static ALEO_ADI bool operator_rem_wrap_signed_prepare(
                        CINT_PARAM_CDEF,
                        nv_uint128 dividend,
                        nv_uint128 divisor                        
    )
    {
/*
rem.w r2 r3 into r14;
rem.w r4 r5 into r15;
rem.w r6 r7 into r16;
rem.w r8 r9 into r17;
rem.w r10 r11 into r18;
is.eq r2 r14 into r19;
is.eq r4 r15 into r20;
is.eq r6 r16 into r21;
is.eq r8 r17 into r22;
is.eq r10 r18 into r23;
*/
        
        nv_uint128 abs_dividend, abs_divisor;
        operator_abs_wrap(CINT_PARAM_C, dividend, abs_dividend);
        operator_abs_wrap(CINT_PARAM_C, divisor, abs_divisor);        

        {
            uint64_t tmp[QBigInt::BIGINT_LEN];
            tmp[0] = abs_dividend.stdint;
            tmp[1] = abs_dividend.stdint >> 64;
            tmp[2] = abs_divisor.stdint;
            tmp[3] = abs_divisor.stdint>>64;
            smem_save_circuit_params(0, tmp);
        }
        smem_save_circuit_params_by_u32(1, 0, gl_bits);
        uint32_t msb = __get_msb_bit(dividend, gl_bits); 
        smem_save_circuit_params_by_u32(1, 1, msb);
                
        return false;
    }

    static ALEO_ADI bool operator_rem_wrap_signed_exec_1(SVM_PARAM_DEF)
    {
        nv_uint128 abs_dividend, abs_divisor;
        uint32_t gl_bits = smem_load_circuit_params_by_u32(1, 0);
        {
            uint64_t tmp[QBigInt::BIGINT_LEN];
            smem_load_circuit_params(0, tmp);
            abs_dividend.stdint = QBigInt::make_u128_with_two_u64(tmp[0], tmp[1]);
            abs_divisor.stdint = QBigInt::make_u128_with_two_u64(tmp[2], tmp[3]);
        }

        nv_uint128 dest_quotient, dest_remainder;
        uint8_t is_const = 0;
        bool gl_is_signed = false;
        __unsigned_division_via_witness(CINT_PARAM_C, abs_dividend, abs_divisor, dest_quotient, dest_remainder);

        gl_is_signed = true;
        nv_uint128 dest;
        {
            nv_uint128 zero;
            zero.stdint = 0;
            operator_sub_wrap(CINT_PARAM_C, zero, dest_remainder, dest);
        }
        
        bool msb = smem_load_circuit_params_by_u32(1,1);
        operator_ternary(CINT_PARAM_C, !msb, dest_remainder, dest, dest);

        
        smem_save_circuit_params_by_u32(0, 0, dest.stdint);
        smem_save_circuit_params_by_u32(0, 1, dest.stdint>>32);
        smem_save_circuit_params_by_u32(0, 2, dest.stdint>>64);
        smem_save_circuit_params_by_u32(0, 3, dest.stdint>>96);
        return true;        
    }

    static ALEO_ADI bool operator_rem_wrap_unsigned(
                        CINT_PARAM_CDEF,
                        nv_uint128 first,
                        nv_uint128 second,
                        nv_uint128 &dest
    )
    {
/*
cast.lossy r12 into r14 as u128;
cast.lossy r13 into r15 as u128;
rem.w r14 r15 into r16;
rem.w r15 r14 into r17;
is.eq r14 r16 into r18;
is.eq r14 r17 into r19;

=========

cast.lossy r12 into r14 as u64;
cast.lossy r13 into r15 as u64;
rem.w r14 r15 into r16;
rem.w r15 r14 into r17;
is.eq r14 r16 into r18;
is.eq r14 r17 into r19;

*/           
        nv_uint128 dest_quotient;
        __unsigned_division_via_witness(CINT_PARAM_C, first, second, dest_quotient, dest);
        return true;        
    }

    static ALEO_ADI uint32_t get_trailing_zero(uint32_t bits)
    {
        switch (bits)
        {
        case 8:
            return 3;
        case 16:        
            return 4;
        case 32:
            return 5;
        case 64:
            return 6;           
        case 128:
            return 7;                        
        default:
            return 0;
        }
    }

    static ALEO_ADI bool operator_shr_wrap_bits_is_128_exec_step1(SVM_PARAM_DEF)
    {
        //只有 signed 才会进入这里 
        nv_uint128 divisor_value ;
        nv_uint128 dividend_value ; 
        bool gl_is_signed ;
        
        {
            bigint_u256_t tmp;
            smem_load_circuit_params(0, tmp);            
            gl_is_signed = smem_load_circuit_params_by_u32(1,5);
            divisor_value.stdint = tmp.uint128[0];
            dividend_value.stdint = tmp.uint128[1];
        }
        
        uint32_t gl_bits = 128;
        nv_uint128 dest_quotient ;
        nv_uint128 dest_reminder ;
        uint8_t is_const = 0;   

        bool old_signed = gl_is_signed;     
        if ( gl_is_signed ) {
            gl_is_signed = false;
        }
        __unsigned_division_via_witness(CINT_PARAM_C, dividend_value, divisor_value, dest_quotient, dest_reminder );
        gl_is_signed = old_signed;

        {
            bigint_u256_t tmp;
            tmp.uint128[0] = dest_quotient.stdint;
            tmp.uint128[1] = dest_reminder.stdint;

            smem_save_circuit_params(0, tmp);
        }
        //total 2112 leaf   
        /*
            if signed int ,should continue, return false
        */
        return !gl_is_signed;
    }

    static ALEO_ADI void operator_shr_wrap_bits_is_128_exec_step2(SVM_PARAM_DEF)
    {
        
        bool msb = smem_load_circuit_params_by_u32(1, 0);    
        nv_uint128 dest_quotient;
        nv_uint128 dest_reminder;
        {
            bigint_u256_t tmp;
            smem_load_circuit_params(0, tmp);
            dest_quotient.stdint = tmp.uint128[0];
            dest_reminder.stdint = tmp.uint128[1];
        }

        nv_uint128 dest;
        uint32_t gl_bits = 128;
        bool gl_is_signed = true;
        uint8_t is_const = 1;
        __circuit_do_bitwise_not(CINT_PARAM_C, dest_quotient, dest);
        is_const = 0;
    
        nv_uint128 one;  one.stdint = 1;
        operator_add_wrap(CINT_PARAM_C, dest, one, dest);
        
        one.stdint = 0;
        bool cond; //计算 ternaty的条件变量 
        operator_is_not_eq(CINT_PARAM_C, dest_reminder, one, cond);
        cond = !cond;

        //计算 ternaty 的 第三个参数
        one.stdint = 1;
        operator_sub_wrap(CINT_PARAM_C, dest, one, dest_reminder);
        // dest_reminder 里存放 ternaty 的 第三个参数
        operator_ternary(CINT_PARAM_C, cond, dest, dest_reminder, dest);
        // dest 存放 : rounded_negated_quotient
        operator_ternary(CINT_PARAM_C, msb, dest, dest_quotient, dest);

        {
            bigint_u256_t tmp;
            tmp.uint128[0] = dest.stdint;
            smem_save_circuit_params(0, tmp);
        }
    }

    static ALEO_ADI void operator_shr_wrap_bits_is_128_prepare(
            CINT_PARAM_CDEF,
            nv_uint128 self,
            uint32_t rhs,
            uint8_t rhs_bits
            )
    {
        bigint_u256_t shift_in_field = { .uint128 = { 1 , 0 } }; //初始值为 1                
        uint32_t mask = rhs & 0x7f; // 128bit's trailing zero is 7 
        uint32_t trailing_zero = get_trailing_zero(gl_bits);
        for (int n = 0; n < 4; n++)
        {
            /* 
                Field::ternary 的 第一轮， 2个参数都是 常量， 所以 n = 0时特殊处理
            */
            shift_in_field.uint128[0] *= shift_in_field.uint128[0];
            if ( n != 0 ) {
                svm_push_leaf_full(SVM_PARAM, shift_in_field );
            }

            bool c = (mask >> (trailing_zero-n-1)) & 1;
            if ( c ) 
                shift_in_field.uint128[0] *= 2;
            if ( n != 0 ) {
                svm_push_leaf_full(SVM_PARAM, shift_in_field );
            }
        }

        {            
            smem_save_circuit_params(0, shift_in_field);
            bigint_u256_t tmp;
            tmp.uint128[0] = self.stdint;
            tmp.uint32[4] = rhs;
            tmp.uint32[5] = gl_is_signed;
            smem_save_circuit_params(1, tmp);
        }
    }

    static ALEO_ADI bool operator_shr_wrap_bits_is_128_prepare_finish(SVM_PARAM_DEF)
    {
        uint32_t gl_bits = 128;
        bool gl_is_signed;
        nv_uint128 self;
        uint32_t rhs;
        bigint_u256_t shift_in_field;
        bool is_const = false;

        {                        
            smem_load_circuit_params(1, shift_in_field);
            self.stdint = shift_in_field.uint128[0];
            rhs = shift_in_field.uint32[4];
            gl_is_signed = shift_in_field.uint32[5];

            smem_load_circuit_params(0, shift_in_field);
        }

        uint32_t mask = rhs & 0x7f; // 128bit's trailing zero is 7 
        uint32_t trailing_zero = get_trailing_zero(gl_bits);
        for (int n = 4; n < trailing_zero; n++)
        {
            /* 
                Field::ternary 的 第一轮， 2个参数都是 常量， 所以 n = 0时特殊处理
            */
            shift_in_field.uint128[0] *= shift_in_field.uint128[0];
            if ( n != 0 ) {
                svm_push_leaf_full(SVM_PARAM, shift_in_field );
            }

            bool c = (mask >> (trailing_zero-n-1)) & 1;
            if ( c ) 
                shift_in_field.uint128[0] *= 2;
            if ( n != 0 ) {
                svm_push_leaf_full(SVM_PARAM, shift_in_field );
            }
        }
        svm_push_leaf_bit_with_u128(SVM_PARAM, shift_in_field.uint128[0]);

        if ( gl_is_signed )         
        {
            nv_uint128 dest;            
            operator_abs_wrap(CINT_PARAM_C, self, dest);

            /*
                0 : 存 divisor_value
                1 : 存 dividend_value
            */
            shift_in_field.uint128[1] = dest.stdint;
            smem_save_circuit_params(0, shift_in_field);

            bool msb = __get_msb_bit(self, gl_bits);
            smem_save_circuit_params_by_u32(1, 0, msb);

            return false;
        } else {
            #if 0
            nv_uint128 divisor_int , dest, dest_remainder;
            divisor_int.stdint = shift_in_field.uint128[0];

            __unsigned_division_via_witness(CINT_PARAM_C, self, divisor_int, dest, dest_remainder);

            shift_in_field.uint128[0] = dest.stdint;
            shift_in_field.uint128[1] = 0;
            smem_save_circuit_params(0, shift_in_field);
            #endif
            /*
                0 : 存 divisor_value
                1 : 存 dividend_value
            */
            shift_in_field.uint128[1] = self.stdint;
            smem_save_circuit_params(0, shift_in_field);

            return false;
        }
    }


    static ALEO_ADI void operator_shr_wrap_bits_less_128(
            CINT_PARAM_CDEF,
            nv_uint128 self,
            uint32_t rhs,
            uint8_t rhs_bits,
            nv_uint128 &dest
            )
    {
        nv_uint128 revert_value;
        revert_value.stdint = 0;
        __revert_bit(self, revert_value, gl_bits);
        uint32_t trailing_zero = get_trailing_zero(gl_bits);
        if ( gl_is_signed)
        {
            bool msb = __get_msb_bit(self, gl_bits);
            for ( int n = 0; n < trailing_zero; n++)
            {
                /*
                circuit/types/integers/src/shr_wrapped.rs 里的： 
                result = Field::ternary(bit, &masked, &result);
                */
                if ( (rhs >> n) & 1 )
                {
                    __uint128_t constant_val = ((__uint128_t)1) << ( ((__uint128_t)1) << n );
                    __uint128_t product = revert_value.stdint * constant_val;
                    __uint128_t mask = constant_val - 1;
                    revert_value.stdint = product + mask * msb;
                }
                svm_push_leaf_full(SVM_PARAM, revert_value.stdint);
            }
            cint_push_var_int(SVM_PARAM, gl_bits*2, gl_is_signed , revert_value);
            __revert_bit(revert_value, dest, gl_bits);
        } else {
            constexpr __uint128_t shift = 1;            
            for ( int n = 0; n < trailing_zero; n++)
            {
                if ( (rhs >> n) & 1 )
                {
                    revert_value.stdint *=  shift << ( shift << n );
                }
                svm_push_leaf_full(SVM_PARAM, revert_value.stdint);                
            }
            cint_push_var_int(SVM_PARAM, gl_bits*2, gl_is_signed , revert_value);
            __revert_bit(revert_value, dest, gl_bits);
        }
    }

static ALEO_ADI void operator_shl_wrap_final_128bit_step1(SVM_PARAM_DEF)
    {
        __uint128_t shift_in_field;
        uint32_t rhs;        
        uint32_t gl_bits = 128;        
        uint32_t mask; // 128bit's trailing zero is 7 

        rhs = smem_load_circuit_params_by_u32(2, 0);

        {
            uint64_t tmp[QBigInt::BIGINT_LEN];
            smem_load_circuit_params(0, tmp);
            shift_in_field = QBigInt::make_u128_with_two_u64(tmp[0], tmp[1]);
        }

        mask = rhs & 0x7f;
        uint32_t trailing_zero = 7;

        for (int n = 4; n < trailing_zero; n++)
        {
            /* 
            Field::ternary 的 第一轮， 2个参数都是 常量， 所以 n = 0时特殊处理
            */
            shift_in_field *= shift_in_field;
            if ( n != 0 ) {
                svm_push_leaf_full(SVM_PARAM, shift_in_field );
            }

            bool c = (mask >> (trailing_zero-n-1)) & 1;
            if ( c ) 
                shift_in_field *= 2;
            if ( n != 0 ) {
                svm_push_leaf_full(SVM_PARAM, shift_in_field );
            }
        }
        svm_push_leaf_bit_with_u128(SVM_PARAM, shift_in_field);

        {
            uint64_t tmp[QBigInt::BIGINT_LEN];
            QBigInt::merge_from_u128(tmp, shift_in_field, 0);
            smem_save_circuit_params(0, tmp);
        }        
    }

    static ALEO_ADI void operator_shl_wrap_final_128bit_step2(SVM_PARAM_DEF)
    {
        nv_uint128 shift_in_field;        
        bool gl_is_signed;
        uint32_t gl_bits = 128;
        nv_uint128 self;
        
        {
            uint64_t tmp[QBigInt::BIGINT_LEN];
            smem_load_circuit_params(0, tmp);
            shift_in_field.stdint = QBigInt::make_u128_with_two_u64(tmp[0], tmp[1]);

            smem_load_circuit_params(1, tmp);
            self.stdint = QBigInt::make_u128_with_two_u64(tmp[0], tmp[1]);

            gl_is_signed = smem_load_circuit_params_by_u32(2, 1);
        }        

        nv_uint128 dest;
        bool is_const = false;
        operator_mul_wrap(CINT_PARAM_C, self, shift_in_field, dest);

        {
            uint64_t tmp[QBigInt::BIGINT_LEN];
            QBigInt::merge_from_u128(tmp, dest.stdint, gl_is_signed);
            smem_save_circuit_params(0, tmp);
        }
    }

    static ALEO_ADI void operator_shl_wrap(
            CINT_PARAM_CDEF,
            nv_uint128 self,
            uint32_t rhs,
            uint8_t rhs_bits,
            nv_uint128 &dest
            )
    {
        uint32_t trailing_zero = get_trailing_zero(gl_bits);
        if ( gl_bits < 128)
        {
            dest.stdint = self.stdint;
            for (int n = 0; n < trailing_zero; n++)
            {
                if ( (rhs >> n) & 1 )
                {
                    __uint128_t constant_val = 1;
                    constant_val = constant_val << (1<<n);
                    dest.stdint *= constant_val;
                }
                svm_push_leaf_full(SVM_PARAM, dest.stdint);
            }
            cint_push_var_int(SVM_PARAM, gl_bits*2, gl_is_signed , dest);
        } else {
            //初始值为 1
            //bigint_u256_t shift_in_field = { .uint128 = { 1 , 0 } }; 
            __uint128_t shift_in_field = 1;
            uint32_t mask = rhs & 0x7f; // 128bit's trailing zero is 7 

            //128bit 的 trailing zero 为 7,  share memory cache 存不下 ,分成2次

            for (int n = 0; n < 4; n++)
            {
                /* 
                Field::ternary 的 第一轮， 2个参数都是 常量， 所以 n = 0时特殊处理
                */
                shift_in_field *= shift_in_field;
                
                if ( n != 0 ) {
                    svm_push_leaf_full(SVM_PARAM, shift_in_field );
                }

                bool c = (mask >> (trailing_zero-n-1)) & 1;
                if ( c ) 
                    shift_in_field *= 2;
                                
                if ( n != 0 ) {
                    svm_push_leaf_full(SVM_PARAM, shift_in_field );
                }
            }

            {
                uint64_t tmp[QBigInt::BIGINT_LEN] = { 0 };
                QBigInt::merge_from_u128(tmp, shift_in_field, 0);
                smem_save_circuit_params(0, tmp);
                
                QBigInt::merge_from_u128(tmp, self.stdint, 0);
                smem_save_circuit_params(1, tmp);

                smem_save_circuit_params_by_u32(2, 0, rhs);
                smem_save_circuit_params_by_u32(2, 1, gl_is_signed);
            }            
        }
    }

};