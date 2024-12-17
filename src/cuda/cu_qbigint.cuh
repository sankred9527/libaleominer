#pragma once
#include "cu_common.h"
#include "ptx.cuh"
#include "bls12_377.cuh"

//quick 256 bit bigint 
class QBigInt 
{
public:
    constexpr static uint32_t BIGINT_LEN = 4;

    static ALEO_ADI __uint128_t make_u128_with_two_u64(uint64_t low, uint64_t high)
    {
        __uint128_t v = high;
        v = (v << 64) | low;
        return v;
    }
    
    static ALEO_ADI bool is_zero(uint64_t data[BIGINT_LEN])
    {
        return (!data[0]) && (!data[1]) && (!data[2]) && (!data[3]) ;
    }

    static ALEO_ADI bool is_one(uint64_t data[BIGINT_LEN])
    {
        return (data[0]==1) && (!data[1]) && (!data[2]) && (!data[3]);
    }

    static ALEO_ADI bool is_even(uint64_t data[BIGINT_LEN])
    {
        return ( data[0] & 1 ) == 0;
    }

    static ALEO_ADI void dump(const char *prefix, uint64_t data[BIGINT_LEN])
    {
        printf("%s",prefix);
        for ( int n = 0; n < BIGINT_LEN; n++)
        {
            printf("0x%016lx ", data[n]);
        }
        printf("\n");
    }

    static ALEO_ADI int compare(uint64_t v1[BIGINT_LEN], uint64_t v2[BIGINT_LEN])
    {        
        for ( int n = BIGINT_LEN-1 ; n>=0; n--)
        {
            if ( v1[n] > v2[n] )
                return 1;
            else if ( v1[n] < v2[n]) 
                return -1;
        }
        return 0;
    }

    static ALEO_ADI void div2(uint64_t data[BIGINT_LEN])
    {
        uint64_t tmp = 0;
        #pragma unroll
        for ( int n = BIGINT_LEN-1; n >= 0; n-- )
        {
            uint64_t b = data[n] << 63;
            data[n] = data[n] >> 1;
            data[n] = tmp | data[n];
            tmp = b;
        }
    }
    

    static ALEO_ADI void split_to_u128(uint64_t data[BIGINT_LEN], __uint128_t &low, __uint128_t &high)
    {
        high = make_u128_with_two_u64(data[2], data[3]);
        low = make_u128_with_two_u64(data[0], data[1]);
    }

    static ALEO_ADI void merge_from_u128(uint64_t data[BIGINT_LEN], __uint128_t low, __uint128_t high)
    {
        data[0] = low;
        data[1] = low >> 64;
        data[2] = high;
        data[3] = high >> 64;
    }

    static ALEO_ADI void set_by_u8(uint64_t data[BIGINT_LEN], uint8_t v8, uint32_t offset)
    {
        //offset 取值 [0, 31]
        uint32_t off_in_u64 = offset / 8 ;
        uint32_t off_inner = (offset % 8)*8;
        uint64_t tmp = data[off_in_u64];
        uint64_t mask = 0xff;        

        mask = mask << off_inner;
        mask = ~mask;
        uint64_t value = static_cast<uint64_t>(v8);
        value = value << off_inner;
        data[off_in_u64] = (data[off_in_u64] & mask) | value;
    }

    static ALEO_ADI uint8_t get_u8_by_offset(uint64_t data[BIGINT_LEN], uint32_t offset)
    {
        //offset 取值 [0, 31]
        uint32_t off_in_u64 = offset / 8 ;
        uint32_t off_inner = (offset % 8 )*8;
        uint64_t tmp = data[off_in_u64];
        uint64_t mask = 0xff;
        mask = mask << off_inner;
        tmp = tmp & mask;
        tmp = tmp >> off_inner;
        return tmp;
    }

    static ALEO_ADI uint32_t get_by_u32(uint64_t to[BIGINT_LEN], uint32_t offset)
    {
        uint32_t off_in_u64 = offset / 2 ;
        uint32_t off_inner = (offset % 2)*32 ;        

        return (uint32_t)( to[off_in_u64] >> off_inner ) ;
    }

    static ALEO_ADI void set_by_u32(uint64_t to[BIGINT_LEN], uint32_t data, uint32_t offset)
    {
        //offset 取值 [0, 7]
        uint32_t off_in_u64 = offset / 2 ;
        uint32_t off_inner = (offset % 2)*32 ;

        uint64_t mask = 0xffffffff00000000ULL;
        mask = mask >> off_inner;
        
        to[off_in_u64] = (to[off_in_u64] & mask ) | ( static_cast<uint64_t>(data) << off_inner );
    }

    static ALEO_ADI void copy(uint64_t to[BIGINT_LEN], uint64_t from[BIGINT_LEN])
    {
        #pragma unroll
        for (int n = 0 ; n< BIGINT_LEN; n++)
        {
            to[n] = from[n];
        }
    }

    static ALEO_ADI bool high_128bit_is_zero(uint64_t data[BIGINT_LEN])
    {
        return data[2] == 0 && data[3] == 0 ;
    }

    static ALEO_ADI void init_from_bigint(uint64_t data[BIGINT_LEN], bigint_u256_t origin)
    {
        #pragma unroll
        for (int n = 0 ; n< BIGINT_LEN; n++)
        {
            data[n] = origin.uint64[n];
        }
    }

    static ALEO_ADI void add_ptx(uint64_t r1[BIGINT_LEN], uint64_t r2[BIGINT_LEN], uint64_t dest[BIGINT_LEN])
    {        
        dest[0] = ptx::u64::add_cc(r1[0], r2[0]);
        dest[1] = ptx::u64::addc_cc(r1[1], r2[1]);
        dest[2] = ptx::u64::addc_cc(r1[2], r2[2]);
        dest[3] = ptx::u64::addc(r1[3], r2[3]);    
    }

    static ALEO_ADI void add_with_modulus(uint64_t data[BIGINT_LEN])
    {
        data[0] = ptx::u64::add_cc(data[0], BLS12_377_FR_PARAMETER_MODULUS_V0);
        data[1] = ptx::u64::addc_cc(data[1], BLS12_377_FR_PARAMETER_MODULUS_V1);
        data[2] = ptx::u64::addc_cc(data[2], BLS12_377_FR_PARAMETER_MODULUS_V2);
        data[3] = ptx::u64::addc(data[3], BLS12_377_FR_PARAMETER_MODULUS_V3);
    }

    static ALEO_ADI void sub_ptx(uint64_t self[BIGINT_LEN], uint64_t other[BIGINT_LEN])
    {                
        self[0] = ptx::u64::sub_cc(self[0], other[0]);
        self[1] = ptx::u64::subc_cc(self[1], other[1]);
        self[2] = ptx::u64::subc_cc(self[2], other[2]);
        self[3] = ptx::u64::subc(self[3], other[3]);
    }

    
    static ALEO_ADI void  bigint_mont_mul_assign_no_unroll( 
                uint64_t src[BIGINT_LEN], uint64_t other[BIGINT_LEN], uint64_t dest[BIGINT_LEN])
    {
        uint64_t carry1 = 0;
        uint64_t carry2 = 0, carry2_1 = 0;
        uint64_t k, tmp1;

        uint64_t v[4] = {0};

        for (int n = 0; n < 4; n++)
        {
            v[0] = ptx::u64::mad_lo_cc(src[0], other[n],  v[0]);
            v[1] = ptx::u64::madc_lo_cc(src[1], other[n],  v[1]);
            v[2] = ptx::u64::madc_lo_cc(src[2], other[n],  v[2]);
            v[3] = ptx::u64::madc_lo_cc(src[3], other[n], v[3]);
            carry1  = ptx::u64::madc_hi(src[3], other[n], 0);

            v[1] = ptx::u64::mad_hi_cc(src[0],  other[n], v[1]);
            v[2] = ptx::u64::madc_hi_cc(src[1], other[n], v[2]);
            v[3] = ptx::u64::madc_hi_cc(src[2], other[n], v[3]);
            carry1  = ptx::u64::addc(carry1,0);

            k = v[0]*BLS12_377_INV;
            ptx::u64::mad_lo_cc(k, BLS12_377_FR_PARAMETER_MODULUS_V0, v[0]);
            carry2 = ptx::u64::madc_hi(k, BLS12_377_FR_PARAMETER_MODULUS_V0, 0);
            
            tmp1 = ptx::u64::mad_lo_cc(k, BLS12_377_FR_PARAMETER_MODULUS_V1, v[1]);
            carry2_1    = ptx::u64::madc_hi(k, BLS12_377_FR_PARAMETER_MODULUS_V1, 0);
            v[0] = ptx::u64::add_cc(tmp1, carry2);
            
            tmp1 = ptx::u64::madc_lo_cc(k, BLS12_377_FR_PARAMETER_MODULUS_V2, v[2]);
            carry2 = ptx::u64::madc_hi(k, BLS12_377_FR_PARAMETER_MODULUS_V2, 0);
            v[1] = ptx::u64::add_cc(tmp1, carry2_1);
            
            tmp1 = ptx::u64::madc_lo_cc(k, BLS12_377_FR_PARAMETER_MODULUS_V3, v[3]);
            carry2_1 = ptx::u64::madc_hi(k, BLS12_377_FR_PARAMETER_MODULUS_V3, 0);
            v[2] = ptx::u64::add_cc(tmp1, carry2);
            
            v[3] = ptx::u64::addc(carry1, carry2_1);

        }

        for ( int n = 0; n < 4; n++)  
            dest[n] = v[n];
        
    }

    static uint64_t ALEO_ADI u64_mac_with_carry_ptx(uint64_t a, uint64_t b, uint64_t c, uint64_t &carry)
    {
        uint64_t ret_low = ptx::u64::mad_lo_cc(b,c,a);
        uint64_t ret_high = ptx::u64::madc_hi(b,c,0);

        ret_low = ptx::u64::add_cc(ret_low, carry);

        //TODO: 可以和后面的指令合并优化
        carry = ptx::u64::addc(ret_high, 0);
                
        return ret_low ;    
    }

    /*
    该算法从  fields/src/fp_256.rs 里的 fn to_bigint() 转换而来
    */
    static ALEO_ADI void bigint_from_field(uint64_t data[BIGINT_LEN])
    {
        uint64_t k;
        uint64_t carry;        

        #define MY_CAL(a,b,c,d) do {\
            k = data[a]*BLS12_377_INV; \
            carry = 0; \
            u64_mac_with_carry_ptx(data[a], k , BLS12_377_FR_PARAMETER_MODULUS_V0, carry); \
            data[b] = u64_mac_with_carry_ptx(data[b], k , BLS12_377_FR_PARAMETER_MODULUS_V1, carry); \
            data[c] = u64_mac_with_carry_ptx(data[c], k , BLS12_377_FR_PARAMETER_MODULUS_V2, carry); \
            data[d] = u64_mac_with_carry_ptx(data[d], k , BLS12_377_FR_PARAMETER_MODULUS_V3, carry); \
            data[a] = carry; \
        } while (0)

        MY_CAL(0,1,2,3);
        MY_CAL(1,2,3,0);
        MY_CAL(2,3,0,1);
        MY_CAL(3,0,1,2);
        #undef MY_CAL
    }

    static ALEO_ADI void bigint_to_field(uint64_t data[BIGINT_LEN])
    {
        if ( is_zero(data))
        {
            return;
        }
        uint64_t R2[BIGINT_LEN] = {
            BLS12_377_FR_PARAMETER_R2.uint64[0], 
            BLS12_377_FR_PARAMETER_R2.uint64[1], 
            BLS12_377_FR_PARAMETER_R2.uint64[2], 
            BLS12_377_FR_PARAMETER_R2.uint64[3]
        };
                    
        bigint_mont_mul_assign_no_unroll(data, R2, data);
    }

};