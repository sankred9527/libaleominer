
#pragma once
#include <stdint.h>
#include "cu_common.h"
#include "ptx.cuh"
#include "bls12_377.cuh"
#include "cu_bigint_define.cuh"
#include "cuda/cu_keccak_v2.cuh"
#include "cuda/cu_qbigint.cuh"

static ALEO_ADI bool bigint256_is_even(bigint_u256_t *self)
{
    return (BIGINT256_GET_BYTES(self, 0) & 0x1) == 0 ;
}

//static ALEO_ADI bool bigint256_is_zero(bigint_u256_t *self)
static __device__  __forceinline__  bool bigint256_is_zero(bigint_u256_t *self)
{
    return self->uint128[0] == 0 && self->uint128[1] == 0;
    // int n = 0;
    // #pragma unroll
    // for  ( n = 0 ; n < BIGINT256_IN_U64; n++ )    
    //     if (BIGINT256_GET_U64(self, n) != 0 ) break;
    
    // return n == BIGINT256_IN_U64;    
}

static ALEO_ADI  bool aleo_u256_compare_less_than(aleo_u256_t *self, aleo_u256_t *other) {
    #pragma unroll
    for (int i = 3; i >= 0; --i) {
        if (self->uint64[i] != other->uint64[i]) {
            return self->uint64[i] < other->uint64[i];
        }
    }
    return false;
}

__device__ void bigint256_dump_with_prefix(const char *prefix, const char *suffix, bigint_u256_t *self);

__device__ inline void bigint256_dump(bigint_u256_t *self)
{
    bigint256_dump_with_prefix("bigint256=", "", self);
}


// 参考代码： snarkVM-miner/utilities/src/biginteger/bigint_256.rs
static ALEO_ADI void bigint256_div2(bigint_u256_t *self)
{
    //printf("entr bigint256_div2\n");
#if 0
    
    if (self->uint128[1] == 0 )
    {
        self->uint128[0] = self->uint128[0] >> 1;
    } else {
        __uint128_t t = self->uint8[16] & 1;
        self->uint128[0] = (t<<127) | (self->uint128[0] >>1);
        self->uint128[1] = self->uint128[1] >> 1;
    }
    
#else    
    __uint128_t t = self->uint8[16] & 1;
    self->uint128[0] = (t<<127) | (self->uint128[0] >>1);
    self->uint128[1] = self->uint128[1] >> 1;

#endif
}


static uint64_t ALEO_ADI u64_mac_with_carry_ptx(uint64_t a, uint64_t b, uint64_t c, uint64_t &carry)
{
#if 1
    uint64_t ret_low = ptx::u64::mad_lo_cc(b,c,a);
    uint64_t ret_high = ptx::u64::madc_hi(b,c,0);

    ret_low = ptx::u64::add_cc(ret_low, carry);

    //TODO: 可以和后面的指令合并优化
    carry = ptx::u64::addc(ret_high, 0);
            
    return ret_low ;
#else 

    return 1;
    //return u64_mac_with_carry(a,b,c,carry);
#endif  
}

static ALEO_ADI void bigint256_sub_noborrow(bigint_u256_t *self, bigint_u256_t *other)
{
    unsigned int borrow = 0;
    // bigint256_dump(self);
    // bigint256_dump(other);

    self->uint64[0] = ptx::u64::sub_cc(self->uint64[0], other->uint64[0]);
    self->uint64[1] = ptx::u64::subc_cc(self->uint64[1], other->uint64[1]);
    self->uint64[2] = ptx::u64::subc_cc(self->uint64[2], other->uint64[2]);
    self->uint64[3] = ptx::u64::subc(self->uint64[3], other->uint64[3]);
}


template <typename T>
__device__ T __add_nv_int_ptx(const T a, const T b, bool &has_carry) 
{
    constexpr int bits = sizeof(T)*8;
    if  constexpr (bits <= 16)
    {
        using UnsignedType = typename std::make_unsigned<decltype(a.x)>::type;
        T result;
        uint32_t ret = 
                static_cast<uint32_t>(static_cast<UnsignedType>(a.x)) + 
                static_cast<uint32_t>(static_cast<UnsignedType>(b.x));

        has_carry = ( (ret >> bits) != 0 );
        result.x = static_cast<UnsignedType>(ret);
        return result;
    }
    if  constexpr (bits == 32 )
    {
        T result;
        using UnsignedType = typename std::make_unsigned<decltype(a.x)>::type;
        result.x = ptx::add_cc(static_cast<UnsignedType>(a.x) , static_cast<UnsignedType>(b.x));
        has_carry = ptx::addc(0,0);
        return result;
    }
    else if constexpr ( bits == 64)
    {
        using UnsignedType = typename std::make_unsigned<decltype(a.x)>::type;
        T result;
        result.x = ptx::add_cc(static_cast<UnsignedType>(a.x) , static_cast<UnsignedType>(b.x));
        result.y = ptx::addc_cc(static_cast<UnsignedType>(a.y) , static_cast<UnsignedType>(b.y));
        has_carry = ptx::addc(0,0);
        return result;    
    } else if constexpr ( bits == 128)
    {
        using UnsignedType = typename std::make_unsigned<decltype(a.x)>::type;
        T result;

        result.x = ptx::add_cc(static_cast<UnsignedType>(a.x) , static_cast<UnsignedType>(b.x));
        result.y = ptx::addc_cc(static_cast<UnsignedType>(a.y) , static_cast<UnsignedType>(b.y));
        result.z = ptx::addc_cc(static_cast<UnsignedType>(a.z) , static_cast<UnsignedType>(b.z));
        result.w = ptx::addc_cc(static_cast<UnsignedType>(a.w) , static_cast<UnsignedType>(b.w));    
        has_carry = ptx::addc(0,0);
        return result;
    }
}

static ALEO_ADI void  bigint_mont_mul_assign_no_unroll( 
            bigint_u256_t src, bigint_u256_t other, bigint_u256_t &dest)
{
    uint64_t carry1 = 0;
    uint64_t carry2 = 0, carry2_1 = 0;
    uint64_t k, tmp1;

    uint64_t v[4] = {0};

    for (int n = 0; n < 4; n++)
    {
        v[0] = ptx::u64::mad_lo_cc(src.uint64[0], other.uint64[n],  v[0]);
        v[1] = ptx::u64::madc_lo_cc(src.uint64[1], other.uint64[n],  v[1]);
        v[2] = ptx::u64::madc_lo_cc(src.uint64[2], other.uint64[n],  v[2]);
        v[3] = ptx::u64::madc_lo_cc(src.uint64[3], other.uint64[n], v[3]);
        carry1  = ptx::u64::madc_hi(src.uint64[3], other.uint64[n], 0);

        v[1] = ptx::u64::mad_hi_cc(src.uint64[0],  other.uint64[n], v[1]);
        v[2] = ptx::u64::madc_hi_cc(src.uint64[1], other.uint64[n], v[2]);
        v[3] = ptx::u64::madc_hi_cc(src.uint64[2], other.uint64[n], v[3]);
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
        dest.uint64[n] = v[n];
    
    //NOTICE： 可以延迟做 reduce 操作
    //CirCuit_Fields::field_reduce(dest);
    
}

static ALEO_ADI void  bigint_mont_mul_assign( 
            bigint_u256_t src, bigint_u256_t other, bigint_u256_t &dest)
{
    uint64_t carry1 = 0;
    uint64_t carry2 = 0, carry2_1 = 0;
    uint64_t k, tmp1;

    uint64_t v0=0,v1=0,v2=0,v3=0;
    
    #define MY_R(n) v##n

    #define MY_MUL(a) do {\
        MY_R(0) = ptx::u64::mad_lo_cc(src.uint64[0], other.uint64[a],  MY_R(0));\
        MY_R(1) = ptx::u64::madc_lo_cc(src.uint64[1], other.uint64[a],  MY_R(1));\
        MY_R(2) = ptx::u64::madc_lo_cc(src.uint64[2], other.uint64[a],  MY_R(2));\
        MY_R(3) = ptx::u64::madc_lo_cc(src.uint64[3], other.uint64[a], MY_R(3));\
        carry1  = ptx::u64::madc_hi(src.uint64[3], other.uint64[a], 0);\
\
        MY_R(1) = ptx::u64::mad_hi_cc(src.uint64[0],  other.uint64[a], MY_R(1));\
        MY_R(2) = ptx::u64::madc_hi_cc(src.uint64[1], other.uint64[a], MY_R(2));\
        MY_R(3) = ptx::u64::madc_hi_cc(src.uint64[2], other.uint64[a], MY_R(3));\
        carry1  = ptx::u64::addc(carry1,0);\
        \
        k = MY_R(0)*BLS12_377_INV; \
        ptx::u64::mad_lo_cc(k, BLS12_377_FR_PARAMETER_MODULUS_V0, MY_R(0));\
        carry2 = ptx::u64::madc_hi(k, BLS12_377_FR_PARAMETER_MODULUS_V0, 0);\
        \
        tmp1 = ptx::u64::mad_lo_cc(k, BLS12_377_FR_PARAMETER_MODULUS_V1, MY_R(1));\
        carry2_1    = ptx::u64::madc_hi(k, BLS12_377_FR_PARAMETER_MODULUS_V1, 0);\
        MY_R(0) = ptx::u64::add_cc(tmp1, carry2);\
        \
        tmp1 = ptx::u64::madc_lo_cc(k, BLS12_377_FR_PARAMETER_MODULUS_V2, MY_R(2));\
        carry2 = ptx::u64::madc_hi(k, BLS12_377_FR_PARAMETER_MODULUS_V2, 0);\
        MY_R(1) = ptx::u64::add_cc(tmp1, carry2_1);\
        \
        tmp1 = ptx::u64::madc_lo_cc(k, BLS12_377_FR_PARAMETER_MODULUS_V3, MY_R(3));\
        carry2_1 = ptx::u64::madc_hi(k, BLS12_377_FR_PARAMETER_MODULUS_V3, 0);\
        MY_R(2) = ptx::u64::add_cc(tmp1, carry2);\
        \
        MY_R(3) = ptx::u64::addc(carry1, carry2_1);\
    } while(0)

    MY_MUL(0);
    MY_MUL(1);
    MY_MUL(2);
    MY_MUL(3);

    #undef MY_MUL
    #undef MY_R
      
    dest.uint64[0] = v0;
    dest.uint64[1] = v1;
    dest.uint64[2] = v2;
    dest.uint64[3] = v3;
    
    //NOTICE： 可以延迟做 reduce 操作
    //CirCuit_Fields::field_reduce(dest);
    
}

static ALEO_ADI void bigint_to_field(bigint_u256_t &src)
{
        
    if ( bigint256_is_zero(&src) ) {        
        return;
    }
        
    bigint_mont_mul_assign(src, BLS12_377_FR_PARAMETER_R2, src);
}


//src小于dest 返回 -1； 大于 返回1； 相等返回 0 
static ALEO_ADI  int8_t bigint_compare(bigint_u256_t &src, bigint_u256_t &dest)
{
    #pragma unroll
    for ( int n = BIGINT256_IN_U64-1; n>=0; n-- )
    {
        if ( src.uint64[n] < dest.uint64[n] ) {
            return -1;
        } else if ( src.uint64[n] > dest.uint64[n] ) {
            return 1;
        }
    }
    return 0;
}

/*
    该算法从  fields/src/fp_256.rs 里的 fn to_bigint() 转换而来
*/
static ALEO_ADI void bigint_from_field(bigint_u256_t *value)
{
    uint64_t k;
    uint64_t carry;
    uint64_t *data = value->uint64;    
    
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

static ALEO_ADI void bigint_add_carry_ptx(bigint_u256_t r1, bigint_u256_t r2, bigint_u256_t &dest)
{
    uint8_t carry = 0;    
    dest.uint64[0] = ptx::u64::add_cc(r1.uint64[0], r2.uint64[0]);
    dest.uint64[1] = ptx::u64::addc_cc(r1.uint64[1], r2.uint64[1]);
    dest.uint64[2] = ptx::u64::addc_cc(r1.uint64[2], r2.uint64[2]);
    dest.uint64[3] = ptx::u64::addc(r1.uint64[3], r2.uint64[3]);    
}

static ALEO_ADI void bigint256_sub_noborrow_ptx(bigint_u256_t &self, bigint_u256_t other)
{
    unsigned int borrow = 0;
    
    self.uint64[0] = ptx::u64::sub_cc(self.uint64[0], other.uint64[0]);
    self.uint64[1] = ptx::u64::subc_cc(self.uint64[1], other.uint64[1]);
    self.uint64[2] = ptx::u64::subc_cc(self.uint64[2], other.uint64[2]);
    self.uint64[3] = ptx::u64::subc(self.uint64[3], other.uint64[3]);    
}

static ALEO_ADI void bigint_add_with_modulus(bigint_u256_t r1, bigint_u256_t &dest)
{
    uint8_t carry = 0;    
    dest.uint64[0] = ptx::u64::add_cc(r1.uint64[0], BLS12_377_FR_PARAMETER_MODULUS_V0);
    dest.uint64[1] = ptx::u64::addc_cc(r1.uint64[1], BLS12_377_FR_PARAMETER_MODULUS_V1);
    dest.uint64[2] = ptx::u64::addc_cc(r1.uint64[2], BLS12_377_FR_PARAMETER_MODULUS_V2);
    dest.uint64[3] = ptx::u64::addc(r1.uint64[3], BLS12_377_FR_PARAMETER_MODULUS_V3);
}


static ALEO_ADI bool bigint_eq_one(bigint_u256_t data)
{
    return data.uint128[1] == 0 && data.uint128[0] == 1 ; 
}


static ALEO_ADI bool bigint_inverse_for_quick_bigint(uint64_t u[QBigInt::BIGINT_LEN] )
{
    uint64_t v[QBigInt::BIGINT_LEN];
    uint64_t b[QBigInt::BIGINT_LEN];
    uint64_t c[QBigInt::BIGINT_LEN] = { 0 };
    
    QBigInt::init_from_bigint(v, BLS12_377_FR_PARAMETER_MODULUS);
    QBigInt::init_from_bigint(b, BLS12_377_FR_PARAMETER_R2);

    if ( QBigInt::is_zero(u) ) 
        return false;
        
    //while ( !QBigInt::is_one(u) && !QBigInt::is_one(v) )
    while ( !QBigInt::high_128bit_is_zero(u) || !QBigInt::high_128bit_is_zero(v) )
    {
        while ( QBigInt::is_even(u) )
        {
            QBigInt::div2(u);
            if ( !QBigInt::is_even(b) )
                QBigInt::add_with_modulus(b);
            QBigInt::div2(b);
        }

        while ( QBigInt::is_even(v) )
        {
            QBigInt::div2(v);
            if ( !QBigInt::is_even(c) )
                QBigInt::add_with_modulus(c);
            QBigInt::div2(c);
        }   
        
        int cond = QBigInt::compare(c,b);
        if ( QBigInt::compare(v, u) < 0)
        {
            QBigInt::sub_ptx(u, v);
            if ( cond == 1)
            {
                QBigInt::add_with_modulus(b);
            }
            QBigInt::sub_ptx(b,c);
        } else {
            QBigInt::sub_ptx(v, u);            
            if ( cond == -1 )
            {
                QBigInt::add_with_modulus(c);
            }
            QBigInt::sub_ptx(c, b);
        }
    }
    
#if 1

    __uint128_t u128;
    __uint128_t v128;

    u128 = u[1];
    u128 = (u128 << 64) | u[0];

    v128 = v[1];
    v128 = (v128 << 64) | v[0];

    while ( (u128 != 1) && (v128 != 1) )
    {
        while ( (u128 & 1) == 0 )
        {
            u128 = u128 >> 1;
            if ( !QBigInt::is_even(b) )
                QBigInt::add_with_modulus(b);
            QBigInt::div2(b);
        }
        
        while ( (v128 & 1) == 0 )
        {            
            v128 = v128 >> 1;
            if ( !QBigInt::is_even(c) )
                QBigInt::add_with_modulus(c);
            QBigInt::div2(c);            
        }

        int cond = QBigInt::compare(c,b);
        if ( v128 < u128 ) {
            u128 = u128 - v128;
            if ( cond == 1)
            {
                QBigInt::add_with_modulus(b);
            }
            QBigInt::sub_ptx(b, c);
        } else {
            v128 = v128 - u128;
            if ( cond == -1 )
            {
                QBigInt::add_with_modulus(c);
            }
            QBigInt::sub_ptx(c, b);
        }
    }

    if ( u128 == 1 )
    {
        QBigInt::copy(u, b);
    } else {
        QBigInt::copy(u, c);
    }
    
#else     
    if ( QBigInt::is_one(u) )
    {        
        for ( int n = 0; n < 4; n++)
            self.uint64[n] = b[n];        
    } else {    
        for ( int n = 0; n < 4; n++)
            self.uint64[n] = c[n];
    }
#endif 

    return true;
}


static ALEO_ADI bool bigint_inverse(bigint_u256_t &self)
{
    uint64_t u[QBigInt::BIGINT_LEN];
    uint64_t v[QBigInt::BIGINT_LEN];
    uint64_t b[QBigInt::BIGINT_LEN];
    uint64_t c[QBigInt::BIGINT_LEN] = { 0 };

    QBigInt::init_from_bigint(u, self);
    QBigInt::init_from_bigint(v, BLS12_377_FR_PARAMETER_MODULUS);
    QBigInt::init_from_bigint(b, BLS12_377_FR_PARAMETER_R2);

    if ( QBigInt::is_zero(u) ) 
        return false;
        
    //while ( !QBigInt::is_one(u) && !QBigInt::is_one(v) )
    while ( !QBigInt::high_128bit_is_zero(u) || !QBigInt::high_128bit_is_zero(v) )
    {
        while ( QBigInt::is_even(u) )
        {
            QBigInt::div2(u);
            if ( !QBigInt::is_even(b) )
                QBigInt::add_with_modulus(b);
            QBigInt::div2(b);
        }

        while ( QBigInt::is_even(v) )
        {
            QBigInt::div2(v);
            if ( !QBigInt::is_even(c) )
                QBigInt::add_with_modulus(c);
            QBigInt::div2(c);
        }   
        
        int cond = QBigInt::compare(c,b);
        if ( QBigInt::compare(v, u) < 0)
        {
            QBigInt::sub_ptx(u, v);
            if ( cond == 1)
            {
                QBigInt::add_with_modulus(b);
            }
            QBigInt::sub_ptx(b,c);
        } else {
            QBigInt::sub_ptx(v, u);            
            if ( cond == -1 )
            {
                QBigInt::add_with_modulus(c);
            }
            QBigInt::sub_ptx(c, b);
        }
    }
    
#if 1

    __uint128_t u128;
    __uint128_t v128;

    u128 = u[1];
    u128 = (u128 << 64) | u[0];

    v128 = v[1];
    v128 = (v128 << 64) | v[0];

    while ( (u128 != 1) && (v128 != 1) )
    {
        while ( (u128 & 1) == 0 )
        {
            u128 = u128 >> 1;
            if ( !QBigInt::is_even(b) )
                QBigInt::add_with_modulus(b);
            QBigInt::div2(b);
        }
        
        while ( (v128 & 1) == 0 )
        {            
            v128 = v128 >> 1;
            if ( !QBigInt::is_even(c) )
                QBigInt::add_with_modulus(c);
            QBigInt::div2(c);            
        }

        int cond = QBigInt::compare(c,b);
        if ( v128 < u128 ) {
            u128 = u128 - v128;
            if ( cond == 1)
            {
                QBigInt::add_with_modulus(b);
            }
            QBigInt::sub_ptx(b, c);
        } else {
            v128 = v128 - u128;
            if ( cond == -1 )
            {
                QBigInt::add_with_modulus(c);
            }
            QBigInt::sub_ptx(c, b);
        }
    }

    if ( u128 == 1 )
    {
        for ( int n = 0; n < 4; n++)
            self.uint64[n] = b[n];
    } else {
        for ( int n = 0; n < 4; n++)
            self.uint64[n] = c[n];
    }
    
#else     
    if ( QBigInt::is_one(u) )
    {        
        for ( int n = 0; n < 4; n++)
            self.uint64[n] = b[n];        
    } else {    
        for ( int n = 0; n < 4; n++)
            self.uint64[n] = c[n];
    }
#endif 

    return true;
}


static ALEO_ADI bool bigint_inverse_old(bigint_u256_t &self)
//static ALEO_ADI bool bigint_inverse(bigint_u256_t &self)
{
    if ( bigint256_is_zero(&self) )
        return false;

    bigint_u256_t one = {
        .uint64 = { 1, 0, 0, 0 }
    };

    bigint_u256_t u = self;
    bigint_u256_t v = BLS12_377_FR_PARAMETER_MODULUS;
    bigint_u256_t b = BLS12_377_FR_PARAMETER_R2;
    bigint_u256_t c = { .bytes = { 0 } };

    
    while ( bigint_compare(u, one) &&  bigint_compare(v, one) )
    {
        while ( bigint256_is_even(&u) )
        {
            bigint256_div2(&u);
            if ( !bigint256_is_even(&b) ) {
                bigint_add_with_modulus(b,b);
            }

            bigint256_div2(&b);
        }

        while ( bigint256_is_even(&v) )
        {            
            bigint256_div2(&v);
            if ( !bigint256_is_even(&c) ) {
                bigint_add_with_modulus(c, c);
            }
            bigint256_div2(&c);
        }

        int8_t cond = bigint_compare(c, b);
        if ( bigint_compare(v, u) < 0 ) {
            bigint256_sub_noborrow_ptx(u, v);
            if ( cond == 1)
            {
                bigint_add_with_modulus(b,b);
            }
            bigint256_sub_noborrow_ptx(b, c);
        } else {
            bigint256_sub_noborrow_ptx(v, u);
            if ( cond == -1 )
            {
                bigint_add_with_modulus(c,c);
            }
            bigint256_sub_noborrow_ptx(c, b);
        }
    }

    if ( !bigint_compare(u, one) ) 
    {        
        memcpy(&self, &b, sizeof(self));
    } else {    
        memcpy(&self, &c, sizeof(self));
    }
    return true;
}


ALEO_ADI static uint4 bigint_u128_mul_with_carry(uint4 input_128_a , uint4 input_128_b, uint4 *high_carry)
{
    union  
    {
        uint4 _ia;
        struct
        {
            uint64_t input_a_lo;
            uint64_t input_a_hi;
        };
    };

    union  
    {
        uint4 _ib;
        struct
        {
            uint64_t input_b_lo;
            uint64_t input_b_hi;
        };
    };


    union 
    {                        
        struct {
            uint64_t result_low;
            uint64_t result_high;
        };
        uint4 result;
    };

    _ia = input_128_a;
    _ib = input_128_b;

    result_low =  input_a_lo * input_b_lo;
    result_high = __umul64hi(input_a_lo , input_b_lo);
    if ( high_carry != nullptr )
    {
        union 
        {                        
            struct {
                uint64_t carry_low;
                uint64_t carry_high;
            };
            uint4 carry;
        };
        

        //计算高位 的 进位
        carry_low = 0;
        carry_high = 0;
        result_high = ptx::u64::mad_lo_cc(input_a_lo, input_b_hi, result_high);
        carry_low += ptx::u64::addc(0,0);
        result_high = ptx::u64::mad_lo_cc(input_b_lo, input_a_hi, result_high);

        carry_low = ptx::u64::madc_hi_cc(input_b_lo, input_a_hi, carry_low);
        carry_high += ptx::u64::addc(0,0);
        carry_low = ptx::u64::mad_hi_cc(input_a_lo, input_b_hi, carry_low);
        carry_high += ptx::u64::addc(0,0);

        carry_low = ptx::u64::mad_lo_cc(input_a_hi, input_b_hi, carry_low);
        carry_high = ptx::u64::madc_hi(input_a_hi, input_b_hi, carry_high);        

        *high_carry = carry;
    } else {
        result_high = ptx::u64::mad_lo(input_a_lo, input_b_hi, result_high);
        result_high = ptx::u64::mad_lo(input_b_lo, input_a_hi, result_high);
    }

    return result;     
}

static ALEO_ADI void bigint_mul(bigint_u256_t *input_a, bigint_u256_t *input_b, bigint_u256_t *dest)
{
    dest->uint4s[0] = bigint_u128_mul_with_carry( input_a->uint4s[0], input_b->uint4s[0],  &(dest->uint4s[1]) );

    bool carry_for_add ;
    uint4 tmp = bigint_u128_mul_with_carry( input_a->uint4s[1], input_b->uint4s[0], nullptr);
    dest->uint4s[1] = __add_nv_int_ptx<uint4>(tmp, dest->uint4s[1], carry_for_add);
    tmp = bigint_u128_mul_with_carry( input_a->uint4s[0], input_b->uint4s[1], nullptr);
    dest->uint4s[1] = __add_nv_int_ptx<uint4>(tmp, dest->uint4s[1], carry_for_add);    
}

__device__ void btest1(uint64_t r1);


#ifdef ENABLE_TEST
__global__ void test_bigint256_div2(aleo_u256_t *v);

__global__ void test_bigint256_sub_noborrow(aleo_u256_t *self, aleo_u256_t *other);

__global__ void test_bigint256_from_bigint(aleo_u256_t *self, aleo_u256_t *dest);

__global__ void test_fr256_to_bigint(aleo_u256_t *self);


#endif