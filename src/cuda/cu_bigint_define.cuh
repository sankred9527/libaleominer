#pragma once
#include <cstdint>

#define BIGINT256_IN_U64 ( sizeof(aleo_u256_t)/sizeof(uint64_t) )

#define BIGINT256_AS_BOOLEAN(p)  ( ((BigintUnionValue*)p)->boolean )
#define BIGINT256_AS_U8(p) ( ((BigintUnionValue*)p)->u8 )
#define BIGINT256_AS_U16(p) ( ((BigintUnionValue*)p)->u16 )
#define BIGINT256_AS_U32(p) ( ((BigintUnionValue*)p)->u32 )
#define BIGINT256_AS_U64(p) ( ((BigintUnionValue*)p)->u64 )
#define BIGINT256_AS_U128(p) ( ((BigintUnionValue*)p)->u128 )
#define BIGINT256_GET_BYTES(p, n)  ((p)->bytes[n])
#define BIGINT256_GET_U8(p, n)  ((p)->uint8[n])
#define BIGINT256_GET_U16(p, n)  ((p)->uint16[n])
#define BIGINT256_GET_U32(p, n)  ((p)->uint32[n])
#define BIGINT256_GET_U64(p, n)  ((p)->uint64[n])
#define BIGINT256_GET_U128(p, n)  ((p)->uint128[n])

typedef uint64_t aleo_u128_t[2];

typedef union
{
    uint8_t uint8[128 / (8*sizeof(uint8_t)) ];
    uint8_t bytes[128 / (8*sizeof(uint8_t)) ];
    uint32_t words[128 / (8*sizeof(uint32_t)) ];
    uint2 uint2s[128 / (8*sizeof(uint2))];
    int4 int4s[128 / (8*sizeof(uint4))];
    uint4 uint4s[128 / (8*sizeof(uint4))];
    uint64_t uint64[128/(8*sizeof(uint64_t))];
    __uint128_t uint128[128/(8*sizeof(__uint128_t))];
} bigint_u128_t;

typedef union
{
    int8_t bytes[128 / (8*sizeof(int8_t)) ];
    int32_t words[128 / (8*sizeof(int32_t)) ];
    int2 int2s[128 / (8*sizeof(int2))];
    int4 int4s[128 / (8*sizeof(int4))];
    int64_t int64[128/(8*sizeof(int64_t))];
} bigint_i128_t;

typedef union
{
    uint8_t uint8[256 / (8*sizeof(uint8_t)) ];
    uint16_t uint16[256 / (8*sizeof(uint16_t)) ];
    uint32_t uint32[256 / (8*sizeof(uint32_t)) ];
    uint8_t bytes[256 / (8*sizeof(uint8_t)) ];
    uint32_t words[256 / (8*sizeof(uint32_t)) ];
    uchar1 uchar1s[256 / (8*sizeof(uchar1)) ];
    ushort1 ushort1s[256 / (8*sizeof(ushort1)) ];
    uint1 uint1s[256 / (8*sizeof(uint1))];
    uint2 uint2s[256 / (8*sizeof(uint2))];
    uint4 uint4s[256 / (8*sizeof(uint4))];
    int4 int4s[256 / (8*sizeof(uint4))];
    uint64_t uint64[256/(8*sizeof(uint64_t))];
    __uint128_t uint128[256/(8*sizeof(__uint128_t))];
} aleo_u256_t;

typedef  aleo_u256_t bigint_u256_t;

typedef union 
{
    bool boolean;
    __uint8_t u8;
    __uint16_t u16;
    __uint32_t u32;
    __uint64_t u64;
    __uint128_t u128;
    bigint_u256_t u256;
} BigintUnionValue;

