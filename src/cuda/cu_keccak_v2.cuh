#pragma once
#include <cstdint>
#include "cu_bigint.cuh"
#include <cstdio>
#include "cu_aleo_globals.cuh"

typedef union 
{
    uint64_t stdint;
    uint2 nv_int;
} Sha3StateV2;


class KeccakV2{

public:
    static constexpr int KECCCAK256_HASH_LEN = 32;
    static constexpr int BLOCK_SIZE = 200 - KECCCAK256_HASH_LEN * 2; // Define BLOCK_SIZE here


    static ALEO_ADI uint2 ROL8(const uint2 a)
    {
        uint2 result;
        result.x = __byte_perm(a.y, a.x, 0x6543);
        result.y = __byte_perm(a.y, a.x, 0x2107);

        return result;
    }

    static ALEO_ADI  uint2 ROR8(const uint2 a)
    {
        uint2 result;
        result.x = __byte_perm(a.y, a.x, 0x0765);
        result.y = __byte_perm(a.y, a.x, 0x4321);

        return result;
    }

    static ALEO_ADI uint2 xor5(
    const uint2 a, const uint2 b, const uint2 c, const uint2 d, const uint2 e)
    {
    #if 1 //__CUDA_ARCH__ >= 500 && CUDA_VERSION >= 7050
        uint2 result;
        asm volatile (
            "// xor5\n\t"
            "lop3.b32 %0, %2, %3, %4, 0x96;\n\t"
            "lop3.b32 %0, %0, %5, %6, 0x96;\n\t"
            "lop3.b32 %1, %7, %8, %9, 0x96;\n\t"
            "lop3.b32 %1, %1, %10, %11, 0x96;"
            : "=r"(result.x), "=r"(result.y)
            : "r"(a.x), "r"(b.x), "r"(c.x),"r"(d.x),"r"(e.x),
            "r"(a.y), "r"(b.y), "r"(c.y),"r"(d.y),"r"(e.y));
        return result;
    #else
        return a ^ b ^ c ^ d ^ e;
    #endif
    }

    static ALEO_ADI uint64_t rotl64(const uint64_t value, const int offset)
    {
        uint2 result;
        if (offset >= 32)
        {
            asm("shf.l.wrap.b32 %0, %1, %2, %3;"
                : "=r"(result.x)
                : "r"(__double2loint(__longlong_as_double(value))),
                "r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
            asm("shf.l.wrap.b32 %0, %1, %2, %3;"
                : "=r"(result.y)
                : "r"(__double2hiint(__longlong_as_double(value))),
                "r"(__double2loint(__longlong_as_double(value))), "r"(offset));
        }
        else
        {
            asm("shf.l.wrap.b32 %0, %1, %2, %3;"
                : "=r"(result.x)
                : "r"(__double2hiint(__longlong_as_double(value))),
                "r"(__double2loint(__longlong_as_double(value))), "r"(offset));
            asm("shf.l.wrap.b32 %0, %1, %2, %3;"
                : "=r"(result.y)
                : "r"(__double2loint(__longlong_as_double(value))),
                "r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
        }
        return __double_as_longlong(__hiloint2double(result.y, result.x));
    }

    static ALEO_ADI uint2 chi(const uint2 a, const uint2 b, const uint2 c)
    {
        uint2 result;
        asm volatile (
            "// chi\n\t"
            "lop3.b32 %0, %2, %3, %4, 0xD2;\n\t"
            "lop3.b32 %1, %5, %6, %7, 0xD2;"
            : "=r"(result.x), "=r"(result.y)
            : "r"(a.x), "r"(b.x), "r"(c.x),  // 0xD2 = 0xF0 ^ ((~0xCC) & 0xAA)
            "r"(a.y), "r"(b.y), "r"(c.y)); // 0xD2 = 0xF0 ^ ((~0xCC) & 0xAA)
        return result;
    }

   
    static ALEO_ADI void fast_leaf_sha3_absorb_last_inline(Sha3StateV2 state[25]) 
    {

        #define SHA3ST (state)
        
        uint8_t r = 1;  // LFSR
        {
            // Theta step        
            {
                Sha3StateV2 c[5] = {};

                c[0].nv_int = SHA3ST[0].nv_int;
                c[1].stdint = SHA3ST[5].stdint ^ SHA3ST[8].stdint;
                c[2].nv_int = SHA3ST[10].nv_int;
                c[3].nv_int = SHA3ST[15].nv_int;
                c[4].nv_int = SHA3ST[20].nv_int;   

                #define _MK_T2(x) do {\
                    constexpr const int p1 = (x+4) %5;\
                    constexpr const int p2 = (x+1) %5;\
                    uint64_t d = c[p1].stdint ^ rotl64(c[p2].stdint, 1);\
                    SHA3ST[x*5+0].stdint ^= d;\
                    SHA3ST[x*5+1].stdint ^= d;\
                    SHA3ST[x*5+2].stdint ^= d;\
                    SHA3ST[x*5+3].stdint ^= d;\
                    SHA3ST[x*5+4].stdint ^= d;\
                } while(0)

                _MK_T2(0);
                _MK_T2(1);
                _MK_T2(2);
                _MK_T2(3);
                _MK_T2(4);

                #undef _MK_T2
            }
                    
            // Rho and pi steps
            {
                uint64_t u;

                u = SHA3ST[6].stdint;

                SHA3ST[6].stdint = rotl64(SHA3ST[21].stdint, 20);
                SHA3ST[21].stdint = rotl64(SHA3ST[14].stdint, 61);
                SHA3ST[14].stdint = rotl64(SHA3ST[22].stdint, 39);
                SHA3ST[22].stdint = rotl64(SHA3ST[4].stdint, 18);
                SHA3ST[4].stdint = rotl64(SHA3ST[10].stdint, 62);
                SHA3ST[10].stdint = rotl64(SHA3ST[12].stdint, 43);
                SHA3ST[12].stdint = rotl64(SHA3ST[17].stdint, 25);

                //SHA3ST[17].stdint = rotl64(SHA3ST[23].stdint, 8);
                SHA3ST[17].nv_int = ROL8(SHA3ST[23].nv_int);

                //SHA3ST[23].stdint = rotl64(SHA3ST[19].stdint, 56);
                SHA3ST[23].nv_int = ROR8(SHA3ST[19].nv_int);

                SHA3ST[19].stdint = rotl64(SHA3ST[3].stdint, 41);
                SHA3ST[3].stdint = rotl64(SHA3ST[20].stdint, 27);
                SHA3ST[20].stdint = rotl64(SHA3ST[24].stdint, 14);
                SHA3ST[24].stdint = rotl64(SHA3ST[9].stdint, 2);
                SHA3ST[9].stdint = rotl64(SHA3ST[16].stdint, 55);
                SHA3ST[16].stdint = rotl64(SHA3ST[8].stdint, 45);
                SHA3ST[8].stdint = rotl64(SHA3ST[1].stdint, 36);
                SHA3ST[1].stdint = rotl64(SHA3ST[15].stdint, 28);
                SHA3ST[15].stdint = rotl64(SHA3ST[18].stdint, 21);
                SHA3ST[18].stdint = rotl64(SHA3ST[13].stdint, 15);
                SHA3ST[13].stdint = rotl64(SHA3ST[7].stdint, 10);
                SHA3ST[7].stdint = rotl64(SHA3ST[11].stdint, 6);
                SHA3ST[11].stdint = rotl64(SHA3ST[2].stdint, 3);
                SHA3ST[2].stdint = rotl64(SHA3ST[5].stdint, 1);
                SHA3ST[5].stdint = rotl64(u, 44);
            }

            // Chi step
            {
                uint2 u,v;
                u = SHA3ST[0].nv_int;
                v = SHA3ST[5].nv_int;
                SHA3ST[0].nv_int = chi( SHA3ST[0].nv_int,  SHA3ST[5].nv_int,  SHA3ST[10].nv_int);
                SHA3ST[5].nv_int = chi( SHA3ST[5].nv_int,  SHA3ST[10].nv_int,  SHA3ST[15].nv_int);
                SHA3ST[10].nv_int = chi( SHA3ST[10].nv_int,  SHA3ST[15].nv_int,  SHA3ST[20].nv_int);
                SHA3ST[15].nv_int = chi( SHA3ST[15].nv_int,  SHA3ST[20].nv_int,  u);
                SHA3ST[20].nv_int = chi( SHA3ST[20].nv_int,  u,  v);
                u = SHA3ST[1].nv_int;
                v = SHA3ST[6].nv_int;
                SHA3ST[1].nv_int = chi( SHA3ST[1].nv_int,  SHA3ST[6].nv_int,  SHA3ST[11].nv_int);
                SHA3ST[6].nv_int = chi( SHA3ST[6].nv_int,  SHA3ST[11].nv_int,  SHA3ST[16].nv_int);
                SHA3ST[11].nv_int = chi( SHA3ST[11].nv_int,  SHA3ST[16].nv_int,  SHA3ST[21].nv_int);
                SHA3ST[16].nv_int = chi( SHA3ST[16].nv_int,  SHA3ST[21].nv_int,  u);
                SHA3ST[21].nv_int = chi( SHA3ST[21].nv_int,  u,  v);
                u = SHA3ST[2].nv_int;
                v = SHA3ST[7].nv_int;
                SHA3ST[2].nv_int = chi( SHA3ST[2].nv_int,  SHA3ST[7].nv_int,  SHA3ST[12].nv_int);
                SHA3ST[7].nv_int = chi( SHA3ST[7].nv_int,  SHA3ST[12].nv_int,  SHA3ST[17].nv_int);
                SHA3ST[12].nv_int = chi( SHA3ST[12].nv_int,  SHA3ST[17].nv_int,  SHA3ST[22].nv_int);
                SHA3ST[17].nv_int = chi( SHA3ST[17].nv_int,  SHA3ST[22].nv_int,  u);
                SHA3ST[22].nv_int = chi( SHA3ST[22].nv_int,  u,  v);
                u = SHA3ST[3].nv_int;
                v = SHA3ST[8].nv_int;
                SHA3ST[3].nv_int = chi( SHA3ST[3].nv_int,  SHA3ST[8].nv_int,  SHA3ST[13].nv_int);
                SHA3ST[8].nv_int = chi( SHA3ST[8].nv_int,  SHA3ST[13].nv_int,  SHA3ST[18].nv_int);
                SHA3ST[13].nv_int = chi( SHA3ST[13].nv_int,  SHA3ST[18].nv_int,  SHA3ST[23].nv_int);
                SHA3ST[18].nv_int = chi( SHA3ST[18].nv_int,  SHA3ST[23].nv_int,  u);
                SHA3ST[23].nv_int = chi( SHA3ST[23].nv_int,  u,  v);
                u = SHA3ST[4].nv_int;
                v = SHA3ST[9].nv_int;
                SHA3ST[4].nv_int = chi( SHA3ST[4].nv_int,  SHA3ST[9].nv_int,  SHA3ST[14].nv_int);
                SHA3ST[9].nv_int = chi( SHA3ST[9].nv_int,  SHA3ST[14].nv_int,  SHA3ST[19].nv_int);
                SHA3ST[14].nv_int = chi( SHA3ST[14].nv_int,  SHA3ST[19].nv_int,  SHA3ST[24].nv_int);
                SHA3ST[19].nv_int = chi( SHA3ST[19].nv_int,  SHA3ST[24].nv_int,  u);
                SHA3ST[24].nv_int = chi( SHA3ST[24].nv_int,  u,  v);
            }
            

            // Iota step
            SHA3ST[0].stdint ^= keccak_round_constants[0];
        }

        //stanard 24 rounds
        for (int i = 1; i < 23; i++) {
            // Theta step        
            {
                Sha3StateV2 c[5] = {};

                #define _MK_THETA(n)  c[n].nv_int = xor5(SHA3ST[n*5].nv_int, SHA3ST[n*5+1].nv_int, SHA3ST[n*5+2].nv_int, SHA3ST[n*5+3].nv_int, SHA3ST[n*5+4].nv_int)
                _MK_THETA(0);
                _MK_THETA(1);
                _MK_THETA(2);
                _MK_THETA(3);
                _MK_THETA(4);
                #undef _MK_THETA

                #define _MK_T2(x) do {\
                    constexpr const int p1 = (x+4) %5;\
                    constexpr const int p2 = (x+1) %5;\
                    uint64_t d = c[p1].stdint ^ rotl64(c[p2].stdint, 1);\
                    SHA3ST[x*5+0].stdint ^= d;\
                    SHA3ST[x*5+1].stdint ^= d;\
                    SHA3ST[x*5+2].stdint ^= d;\
                    SHA3ST[x*5+3].stdint ^= d;\
                    SHA3ST[x*5+4].stdint ^= d;\
                } while(0)

                _MK_T2(0);
                _MK_T2(1);
                _MK_T2(2);
                _MK_T2(3);
                _MK_T2(4);

                #undef _MK_T2
            }
                    
            // Rho and pi steps
            {
                uint64_t u;

                u = SHA3ST[6].stdint;

                SHA3ST[6].stdint = rotl64(SHA3ST[21].stdint, 20);
                SHA3ST[21].stdint = rotl64(SHA3ST[14].stdint, 61);
                SHA3ST[14].stdint = rotl64(SHA3ST[22].stdint, 39);
                SHA3ST[22].stdint = rotl64(SHA3ST[4].stdint, 18);
                SHA3ST[4].stdint = rotl64(SHA3ST[10].stdint, 62);
                SHA3ST[10].stdint = rotl64(SHA3ST[12].stdint, 43);
                SHA3ST[12].stdint = rotl64(SHA3ST[17].stdint, 25);

                //SHA3ST[17].stdint = rotl64(SHA3ST[23].stdint, 8);
                SHA3ST[17].nv_int = ROL8(SHA3ST[23].nv_int);

                //SHA3ST[23].stdint = rotl64(SHA3ST[19].stdint, 56);
                SHA3ST[23].nv_int = ROR8(SHA3ST[19].nv_int);

                SHA3ST[19].stdint = rotl64(SHA3ST[3].stdint, 41);
                SHA3ST[3].stdint = rotl64(SHA3ST[20].stdint, 27);
                SHA3ST[20].stdint = rotl64(SHA3ST[24].stdint, 14);
                SHA3ST[24].stdint = rotl64(SHA3ST[9].stdint, 2);
                SHA3ST[9].stdint = rotl64(SHA3ST[16].stdint, 55);
                SHA3ST[16].stdint = rotl64(SHA3ST[8].stdint, 45);
                SHA3ST[8].stdint = rotl64(SHA3ST[1].stdint, 36);
                SHA3ST[1].stdint = rotl64(SHA3ST[15].stdint, 28);
                SHA3ST[15].stdint = rotl64(SHA3ST[18].stdint, 21);
                SHA3ST[18].stdint = rotl64(SHA3ST[13].stdint, 15);
                SHA3ST[13].stdint = rotl64(SHA3ST[7].stdint, 10);
                SHA3ST[7].stdint = rotl64(SHA3ST[11].stdint, 6);
                SHA3ST[11].stdint = rotl64(SHA3ST[2].stdint, 3);
                SHA3ST[2].stdint = rotl64(SHA3ST[5].stdint, 1);
                SHA3ST[5].stdint = rotl64(u, 44);
            }

            // Chi step
            {
                uint2 u,v;
                u = SHA3ST[0].nv_int;
                v = SHA3ST[5].nv_int;
                SHA3ST[0].nv_int = chi( SHA3ST[0].nv_int,  SHA3ST[5].nv_int,  SHA3ST[10].nv_int);
                SHA3ST[5].nv_int = chi( SHA3ST[5].nv_int,  SHA3ST[10].nv_int,  SHA3ST[15].nv_int);
                SHA3ST[10].nv_int = chi( SHA3ST[10].nv_int,  SHA3ST[15].nv_int,  SHA3ST[20].nv_int);
                SHA3ST[15].nv_int = chi( SHA3ST[15].nv_int,  SHA3ST[20].nv_int,  u);
                SHA3ST[20].nv_int = chi( SHA3ST[20].nv_int,  u,  v);
                u = SHA3ST[1].nv_int;
                v = SHA3ST[6].nv_int;
                SHA3ST[1].nv_int = chi( SHA3ST[1].nv_int,  SHA3ST[6].nv_int,  SHA3ST[11].nv_int);
                SHA3ST[6].nv_int = chi( SHA3ST[6].nv_int,  SHA3ST[11].nv_int,  SHA3ST[16].nv_int);
                SHA3ST[11].nv_int = chi( SHA3ST[11].nv_int,  SHA3ST[16].nv_int,  SHA3ST[21].nv_int);
                SHA3ST[16].nv_int = chi( SHA3ST[16].nv_int,  SHA3ST[21].nv_int,  u);
                SHA3ST[21].nv_int = chi( SHA3ST[21].nv_int,  u,  v);
                u = SHA3ST[2].nv_int;
                v = SHA3ST[7].nv_int;
                SHA3ST[2].nv_int = chi( SHA3ST[2].nv_int,  SHA3ST[7].nv_int,  SHA3ST[12].nv_int);
                SHA3ST[7].nv_int = chi( SHA3ST[7].nv_int,  SHA3ST[12].nv_int,  SHA3ST[17].nv_int);
                SHA3ST[12].nv_int = chi( SHA3ST[12].nv_int,  SHA3ST[17].nv_int,  SHA3ST[22].nv_int);
                SHA3ST[17].nv_int = chi( SHA3ST[17].nv_int,  SHA3ST[22].nv_int,  u);
                SHA3ST[22].nv_int = chi( SHA3ST[22].nv_int,  u,  v);
                u = SHA3ST[3].nv_int;
                v = SHA3ST[8].nv_int;
                SHA3ST[3].nv_int = chi( SHA3ST[3].nv_int,  SHA3ST[8].nv_int,  SHA3ST[13].nv_int);
                SHA3ST[8].nv_int = chi( SHA3ST[8].nv_int,  SHA3ST[13].nv_int,  SHA3ST[18].nv_int);
                SHA3ST[13].nv_int = chi( SHA3ST[13].nv_int,  SHA3ST[18].nv_int,  SHA3ST[23].nv_int);
                SHA3ST[18].nv_int = chi( SHA3ST[18].nv_int,  SHA3ST[23].nv_int,  u);
                SHA3ST[23].nv_int = chi( SHA3ST[23].nv_int,  u,  v);
                u = SHA3ST[4].nv_int;
                v = SHA3ST[9].nv_int;
                SHA3ST[4].nv_int = chi( SHA3ST[4].nv_int,  SHA3ST[9].nv_int,  SHA3ST[14].nv_int);
                SHA3ST[9].nv_int = chi( SHA3ST[9].nv_int,  SHA3ST[14].nv_int,  SHA3ST[19].nv_int);
                SHA3ST[14].nv_int = chi( SHA3ST[14].nv_int,  SHA3ST[19].nv_int,  SHA3ST[24].nv_int);
                SHA3ST[19].nv_int = chi( SHA3ST[19].nv_int,  SHA3ST[24].nv_int,  u);
                SHA3ST[24].nv_int = chi( SHA3ST[24].nv_int,  u,  v);
            }
            

            // Iota step
            SHA3ST[0].stdint ^= keccak_round_constants[i];
        }

        //24 round 
        // Theta step        
        {
            Sha3StateV2 c[5] = {};

            #define _MK_THETA(n)  c[n].nv_int = xor5(SHA3ST[n*5].nv_int, SHA3ST[n*5+1].nv_int, SHA3ST[n*5+2].nv_int, SHA3ST[n*5+3].nv_int, SHA3ST[n*5+4].nv_int)
            _MK_THETA(0);
            _MK_THETA(1);
            _MK_THETA(2);
            _MK_THETA(3);
            _MK_THETA(4);
            #undef _MK_THETA

            {
                uint64_t d;

                d = c[4].stdint ^ rotl64(c[1].stdint, 1);
                SHA3ST[0].stdint ^= d;

                d = c[0].stdint ^ rotl64(c[2].stdint, 1);
                SHA3ST[5].stdint ^= d;
                SHA3ST[6].stdint ^= d;

                d = c[1].stdint ^ rotl64(c[3].stdint, 1);
                SHA3ST[10].stdint ^= d;
                SHA3ST[12].stdint ^= d;

                d = c[2].stdint ^ rotl64(c[4].stdint, 1);
                SHA3ST[15].stdint ^= d;
                SHA3ST[18].stdint ^= d;

                d = c[3].stdint ^ rotl64(c[0].stdint, 1);
                SHA3ST[20].stdint ^= d;
                SHA3ST[24].stdint ^= d;
            }
        }

        // Rho and pi steps
        {
            SHA3ST[10].stdint = rotl64(SHA3ST[12].stdint, 43);
            SHA3ST[20].stdint = rotl64(SHA3ST[24].stdint, 14);
            SHA3ST[15].stdint = rotl64(SHA3ST[18].stdint, 21);
            SHA3ST[5].stdint = rotl64(SHA3ST[6].stdint, 44);
        }

        // Chi step
        {
            uint2 u;
            u = SHA3ST[0].nv_int;            
            SHA3ST[0].nv_int = chi( SHA3ST[0].nv_int,  SHA3ST[5].nv_int,  SHA3ST[10].nv_int);
            SHA3ST[5].nv_int = chi( SHA3ST[5].nv_int,  SHA3ST[10].nv_int,  SHA3ST[15].nv_int);
            SHA3ST[10].nv_int = chi( SHA3ST[10].nv_int,  SHA3ST[15].nv_int,  SHA3ST[20].nv_int);
            SHA3ST[15].nv_int = chi( SHA3ST[15].nv_int,  SHA3ST[20].nv_int,  u);
        }
        

        // Iota step
        SHA3ST[0].stdint ^= keccak_round_constants[23];
        #undef SHA3ST
        
    }


    static ALEO_ADI void fast_sha3_absorb_last_inline(Sha3StateV2 state[25]) 
    {

        #define SHA3ST (state)
        
        uint8_t r = 1;  // LFSR

        //stanard 24 rounds
        for (int i = 0; i < 23; i++) {
            // Theta step        
            {
                Sha3StateV2 c[5] = {};

                #define _MK_THETA(n)  c[n].nv_int = xor5(SHA3ST[n*5].nv_int, SHA3ST[n*5+1].nv_int, SHA3ST[n*5+2].nv_int, SHA3ST[n*5+3].nv_int, SHA3ST[n*5+4].nv_int)
                _MK_THETA(0);
                _MK_THETA(1);
                _MK_THETA(2);
                _MK_THETA(3);
                _MK_THETA(4);
                #undef _MK_THETA

                #define _MK_T2(x) do {\
                    constexpr const int p1 = (x+4) %5;\
                    constexpr const int p2 = (x+1) %5;\
                    uint64_t d = c[p1].stdint ^ rotl64(c[p2].stdint, 1);\
                    SHA3ST[x*5+0].stdint ^= d;\
                    SHA3ST[x*5+1].stdint ^= d;\
                    SHA3ST[x*5+2].stdint ^= d;\
                    SHA3ST[x*5+3].stdint ^= d;\
                    SHA3ST[x*5+4].stdint ^= d;\
                } while(0)

                _MK_T2(0);
                _MK_T2(1);
                _MK_T2(2);
                _MK_T2(3);
                _MK_T2(4);

                #undef _MK_T2
            }
                    
            // Rho and pi steps
            {
                uint64_t u;

                u = SHA3ST[6].stdint;

                SHA3ST[6].stdint = rotl64(SHA3ST[21].stdint, 20);
                SHA3ST[21].stdint = rotl64(SHA3ST[14].stdint, 61);
                SHA3ST[14].stdint = rotl64(SHA3ST[22].stdint, 39);
                SHA3ST[22].stdint = rotl64(SHA3ST[4].stdint, 18);
                SHA3ST[4].stdint = rotl64(SHA3ST[10].stdint, 62);
                SHA3ST[10].stdint = rotl64(SHA3ST[12].stdint, 43);
                SHA3ST[12].stdint = rotl64(SHA3ST[17].stdint, 25);

                //SHA3ST[17].stdint = rotl64(SHA3ST[23].stdint, 8);
                SHA3ST[17].nv_int = ROL8(SHA3ST[23].nv_int);

                //SHA3ST[23].stdint = rotl64(SHA3ST[19].stdint, 56);
                SHA3ST[23].nv_int = ROR8(SHA3ST[19].nv_int);

                SHA3ST[19].stdint = rotl64(SHA3ST[3].stdint, 41);
                SHA3ST[3].stdint = rotl64(SHA3ST[20].stdint, 27);
                SHA3ST[20].stdint = rotl64(SHA3ST[24].stdint, 14);
                SHA3ST[24].stdint = rotl64(SHA3ST[9].stdint, 2);
                SHA3ST[9].stdint = rotl64(SHA3ST[16].stdint, 55);
                SHA3ST[16].stdint = rotl64(SHA3ST[8].stdint, 45);
                SHA3ST[8].stdint = rotl64(SHA3ST[1].stdint, 36);
                SHA3ST[1].stdint = rotl64(SHA3ST[15].stdint, 28);
                SHA3ST[15].stdint = rotl64(SHA3ST[18].stdint, 21);
                SHA3ST[18].stdint = rotl64(SHA3ST[13].stdint, 15);
                SHA3ST[13].stdint = rotl64(SHA3ST[7].stdint, 10);
                SHA3ST[7].stdint = rotl64(SHA3ST[11].stdint, 6);
                SHA3ST[11].stdint = rotl64(SHA3ST[2].stdint, 3);
                SHA3ST[2].stdint = rotl64(SHA3ST[5].stdint, 1);
                SHA3ST[5].stdint = rotl64(u, 44);
            }

            // Chi step
            {
                uint2 u,v;
                u = SHA3ST[0].nv_int;
                v = SHA3ST[5].nv_int;
                SHA3ST[0].nv_int = chi( SHA3ST[0].nv_int,  SHA3ST[5].nv_int,  SHA3ST[10].nv_int);
                SHA3ST[5].nv_int = chi( SHA3ST[5].nv_int,  SHA3ST[10].nv_int,  SHA3ST[15].nv_int);
                SHA3ST[10].nv_int = chi( SHA3ST[10].nv_int,  SHA3ST[15].nv_int,  SHA3ST[20].nv_int);
                SHA3ST[15].nv_int = chi( SHA3ST[15].nv_int,  SHA3ST[20].nv_int,  u);
                SHA3ST[20].nv_int = chi( SHA3ST[20].nv_int,  u,  v);
                u = SHA3ST[1].nv_int;
                v = SHA3ST[6].nv_int;
                SHA3ST[1].nv_int = chi( SHA3ST[1].nv_int,  SHA3ST[6].nv_int,  SHA3ST[11].nv_int);
                SHA3ST[6].nv_int = chi( SHA3ST[6].nv_int,  SHA3ST[11].nv_int,  SHA3ST[16].nv_int);
                SHA3ST[11].nv_int = chi( SHA3ST[11].nv_int,  SHA3ST[16].nv_int,  SHA3ST[21].nv_int);
                SHA3ST[16].nv_int = chi( SHA3ST[16].nv_int,  SHA3ST[21].nv_int,  u);
                SHA3ST[21].nv_int = chi( SHA3ST[21].nv_int,  u,  v);
                u = SHA3ST[2].nv_int;
                v = SHA3ST[7].nv_int;
                SHA3ST[2].nv_int = chi( SHA3ST[2].nv_int,  SHA3ST[7].nv_int,  SHA3ST[12].nv_int);
                SHA3ST[7].nv_int = chi( SHA3ST[7].nv_int,  SHA3ST[12].nv_int,  SHA3ST[17].nv_int);
                SHA3ST[12].nv_int = chi( SHA3ST[12].nv_int,  SHA3ST[17].nv_int,  SHA3ST[22].nv_int);
                SHA3ST[17].nv_int = chi( SHA3ST[17].nv_int,  SHA3ST[22].nv_int,  u);
                SHA3ST[22].nv_int = chi( SHA3ST[22].nv_int,  u,  v);
                u = SHA3ST[3].nv_int;
                v = SHA3ST[8].nv_int;
                SHA3ST[3].nv_int = chi( SHA3ST[3].nv_int,  SHA3ST[8].nv_int,  SHA3ST[13].nv_int);
                SHA3ST[8].nv_int = chi( SHA3ST[8].nv_int,  SHA3ST[13].nv_int,  SHA3ST[18].nv_int);
                SHA3ST[13].nv_int = chi( SHA3ST[13].nv_int,  SHA3ST[18].nv_int,  SHA3ST[23].nv_int);
                SHA3ST[18].nv_int = chi( SHA3ST[18].nv_int,  SHA3ST[23].nv_int,  u);
                SHA3ST[23].nv_int = chi( SHA3ST[23].nv_int,  u,  v);
                u = SHA3ST[4].nv_int;
                v = SHA3ST[9].nv_int;
                SHA3ST[4].nv_int = chi( SHA3ST[4].nv_int,  SHA3ST[9].nv_int,  SHA3ST[14].nv_int);
                SHA3ST[9].nv_int = chi( SHA3ST[9].nv_int,  SHA3ST[14].nv_int,  SHA3ST[19].nv_int);
                SHA3ST[14].nv_int = chi( SHA3ST[14].nv_int,  SHA3ST[19].nv_int,  SHA3ST[24].nv_int);
                SHA3ST[19].nv_int = chi( SHA3ST[19].nv_int,  SHA3ST[24].nv_int,  u);
                SHA3ST[24].nv_int = chi( SHA3ST[24].nv_int,  u,  v);
            }
            

            // Iota step
            SHA3ST[0].stdint ^= keccak_round_constants[i];
        }

        //24 round 
        // Theta step        
        {
            Sha3StateV2 c[5] = {};

            #define _MK_THETA(n)  c[n].nv_int = xor5(SHA3ST[n*5].nv_int, SHA3ST[n*5+1].nv_int, SHA3ST[n*5+2].nv_int, SHA3ST[n*5+3].nv_int, SHA3ST[n*5+4].nv_int)
            _MK_THETA(0);
            _MK_THETA(1);
            _MK_THETA(2);
            _MK_THETA(3);
            _MK_THETA(4);
            #undef _MK_THETA

            {
                uint64_t d;

                d = c[4].stdint ^ rotl64(c[1].stdint, 1);
                SHA3ST[0].stdint ^= d;

                d = c[0].stdint ^ rotl64(c[2].stdint, 1);
                SHA3ST[5].stdint ^= d;
                SHA3ST[6].stdint ^= d;

                d = c[1].stdint ^ rotl64(c[3].stdint, 1);
                SHA3ST[10].stdint ^= d;
                SHA3ST[12].stdint ^= d;

                d = c[2].stdint ^ rotl64(c[4].stdint, 1);
                SHA3ST[15].stdint ^= d;
                SHA3ST[18].stdint ^= d;

                d = c[3].stdint ^ rotl64(c[0].stdint, 1);
                SHA3ST[20].stdint ^= d;
                SHA3ST[24].stdint ^= d;
            }
        }

        // Rho and pi steps
        {
            SHA3ST[10].stdint = rotl64(SHA3ST[12].stdint, 43);
            SHA3ST[20].stdint = rotl64(SHA3ST[24].stdint, 14);
            SHA3ST[15].stdint = rotl64(SHA3ST[18].stdint, 21);
            SHA3ST[5].stdint = rotl64(SHA3ST[6].stdint, 44);
        }

        // Chi step
        {
            uint2 u;
            u = SHA3ST[0].nv_int;            
            SHA3ST[0].nv_int = chi( SHA3ST[0].nv_int,  SHA3ST[5].nv_int,  SHA3ST[10].nv_int);
            SHA3ST[5].nv_int = chi( SHA3ST[5].nv_int,  SHA3ST[10].nv_int,  SHA3ST[15].nv_int);
            SHA3ST[10].nv_int = chi( SHA3ST[10].nv_int,  SHA3ST[15].nv_int,  SHA3ST[20].nv_int);
            SHA3ST[15].nv_int = chi( SHA3ST[15].nv_int,  SHA3ST[20].nv_int,  u);
        }
        

        // Iota step
        SHA3ST[0].stdint ^= keccak_round_constants[23];
        #undef SHA3ST
        
    }

    static ALEO_ADI void fast_sha3_absorb_inline(Sha3StateV2 state[25]) 
    {

        #define SHA3ST (state)
        
        uint8_t r = 1;  // LFSR

        //stanard 24 rounds
        for (int i = 0; i < 24; i++) {
            // Theta step        
            {
                Sha3StateV2 c[5] = {};

                #define _MK_THETA(n)  c[n].nv_int = xor5(SHA3ST[n*5].nv_int, SHA3ST[n*5+1].nv_int, SHA3ST[n*5+2].nv_int, SHA3ST[n*5+3].nv_int, SHA3ST[n*5+4].nv_int)
                _MK_THETA(0);
                _MK_THETA(1);
                _MK_THETA(2);
                _MK_THETA(3);
                _MK_THETA(4);
                #undef _MK_THETA

                #define _MK_T2(x) do {\
                    constexpr const int p1 = (x+4) %5;\
                    constexpr const int p2 = (x+1) %5;\
                    uint64_t d = c[p1].stdint ^ rotl64(c[p2].stdint, 1);\
                    SHA3ST[x*5+0].stdint ^= d;\
                    SHA3ST[x*5+1].stdint ^= d;\
                    SHA3ST[x*5+2].stdint ^= d;\
                    SHA3ST[x*5+3].stdint ^= d;\
                    SHA3ST[x*5+4].stdint ^= d;\
                } while(0)

                _MK_T2(0);
                _MK_T2(1);
                _MK_T2(2);
                _MK_T2(3);
                _MK_T2(4);

                #undef _MK_T2
            }
                    
            // Rho and pi steps
            {
                uint64_t u;

                u = SHA3ST[6].stdint;

                SHA3ST[6].stdint = rotl64(SHA3ST[21].stdint, 20);
                SHA3ST[21].stdint = rotl64(SHA3ST[14].stdint, 61);
                SHA3ST[14].stdint = rotl64(SHA3ST[22].stdint, 39);
                SHA3ST[22].stdint = rotl64(SHA3ST[4].stdint, 18);
                SHA3ST[4].stdint = rotl64(SHA3ST[10].stdint, 62);
                SHA3ST[10].stdint = rotl64(SHA3ST[12].stdint, 43);
                SHA3ST[12].stdint = rotl64(SHA3ST[17].stdint, 25);

                //SHA3ST[17].stdint = rotl64(SHA3ST[23].stdint, 8);
                SHA3ST[17].nv_int = ROL8(SHA3ST[23].nv_int);

                //SHA3ST[23].stdint = rotl64(SHA3ST[19].stdint, 56);
                SHA3ST[23].nv_int = ROR8(SHA3ST[19].nv_int);

                SHA3ST[19].stdint = rotl64(SHA3ST[3].stdint, 41);
                SHA3ST[3].stdint = rotl64(SHA3ST[20].stdint, 27);
                SHA3ST[20].stdint = rotl64(SHA3ST[24].stdint, 14);
                SHA3ST[24].stdint = rotl64(SHA3ST[9].stdint, 2);
                SHA3ST[9].stdint = rotl64(SHA3ST[16].stdint, 55);
                SHA3ST[16].stdint = rotl64(SHA3ST[8].stdint, 45);
                SHA3ST[8].stdint = rotl64(SHA3ST[1].stdint, 36);
                SHA3ST[1].stdint = rotl64(SHA3ST[15].stdint, 28);
                SHA3ST[15].stdint = rotl64(SHA3ST[18].stdint, 21);
                SHA3ST[18].stdint = rotl64(SHA3ST[13].stdint, 15);
                SHA3ST[13].stdint = rotl64(SHA3ST[7].stdint, 10);
                SHA3ST[7].stdint = rotl64(SHA3ST[11].stdint, 6);
                SHA3ST[11].stdint = rotl64(SHA3ST[2].stdint, 3);
                SHA3ST[2].stdint = rotl64(SHA3ST[5].stdint, 1);
                SHA3ST[5].stdint = rotl64(u, 44);
            }

            // Chi step
            {
                uint2 u,v;
                u = SHA3ST[0].nv_int;
                v = SHA3ST[5].nv_int;
                SHA3ST[0].nv_int = chi( SHA3ST[0].nv_int,  SHA3ST[5].nv_int,  SHA3ST[10].nv_int);
                SHA3ST[5].nv_int = chi( SHA3ST[5].nv_int,  SHA3ST[10].nv_int,  SHA3ST[15].nv_int);
                SHA3ST[10].nv_int = chi( SHA3ST[10].nv_int,  SHA3ST[15].nv_int,  SHA3ST[20].nv_int);
                SHA3ST[15].nv_int = chi( SHA3ST[15].nv_int,  SHA3ST[20].nv_int,  u);
                SHA3ST[20].nv_int = chi( SHA3ST[20].nv_int,  u,  v);
                u = SHA3ST[1].nv_int;
                v = SHA3ST[6].nv_int;
                SHA3ST[1].nv_int = chi( SHA3ST[1].nv_int,  SHA3ST[6].nv_int,  SHA3ST[11].nv_int);
                SHA3ST[6].nv_int = chi( SHA3ST[6].nv_int,  SHA3ST[11].nv_int,  SHA3ST[16].nv_int);
                SHA3ST[11].nv_int = chi( SHA3ST[11].nv_int,  SHA3ST[16].nv_int,  SHA3ST[21].nv_int);
                SHA3ST[16].nv_int = chi( SHA3ST[16].nv_int,  SHA3ST[21].nv_int,  u);
                SHA3ST[21].nv_int = chi( SHA3ST[21].nv_int,  u,  v);
                u = SHA3ST[2].nv_int;
                v = SHA3ST[7].nv_int;
                SHA3ST[2].nv_int = chi( SHA3ST[2].nv_int,  SHA3ST[7].nv_int,  SHA3ST[12].nv_int);
                SHA3ST[7].nv_int = chi( SHA3ST[7].nv_int,  SHA3ST[12].nv_int,  SHA3ST[17].nv_int);
                SHA3ST[12].nv_int = chi( SHA3ST[12].nv_int,  SHA3ST[17].nv_int,  SHA3ST[22].nv_int);
                SHA3ST[17].nv_int = chi( SHA3ST[17].nv_int,  SHA3ST[22].nv_int,  u);
                SHA3ST[22].nv_int = chi( SHA3ST[22].nv_int,  u,  v);
                u = SHA3ST[3].nv_int;
                v = SHA3ST[8].nv_int;
                SHA3ST[3].nv_int = chi( SHA3ST[3].nv_int,  SHA3ST[8].nv_int,  SHA3ST[13].nv_int);
                SHA3ST[8].nv_int = chi( SHA3ST[8].nv_int,  SHA3ST[13].nv_int,  SHA3ST[18].nv_int);
                SHA3ST[13].nv_int = chi( SHA3ST[13].nv_int,  SHA3ST[18].nv_int,  SHA3ST[23].nv_int);
                SHA3ST[18].nv_int = chi( SHA3ST[18].nv_int,  SHA3ST[23].nv_int,  u);
                SHA3ST[23].nv_int = chi( SHA3ST[23].nv_int,  u,  v);
                u = SHA3ST[4].nv_int;
                v = SHA3ST[9].nv_int;
                SHA3ST[4].nv_int = chi( SHA3ST[4].nv_int,  SHA3ST[9].nv_int,  SHA3ST[14].nv_int);
                SHA3ST[9].nv_int = chi( SHA3ST[9].nv_int,  SHA3ST[14].nv_int,  SHA3ST[19].nv_int);
                SHA3ST[14].nv_int = chi( SHA3ST[14].nv_int,  SHA3ST[19].nv_int,  SHA3ST[24].nv_int);
                SHA3ST[19].nv_int = chi( SHA3ST[19].nv_int,  SHA3ST[24].nv_int,  u);
                SHA3ST[24].nv_int = chi( SHA3ST[24].nv_int,  u,  v);
            }
            

            // Iota step
            SHA3ST[0].stdint ^= keccak_round_constants[i];
        }
        #undef SHA3ST
        
    }

    template<int R>
    static ALEO_ADI void fast_path_hash_update_r0_to_r3(Sha3StateV2 state[25], uint8_t &last_bit , bigint_u256_t &input)
    {
        constexpr unsigned int blockOff_start = R*32;
        unsigned int blockOff = blockOff_start;

        constexpr unsigned int j_start = R*4;
        constexpr unsigned int j_end = R*4+4;

        //0 - 32
        #pragma unroll
        for(uint32_t j = j_start; j < j_end; j++)
        {
            for (uint32_t n = 0; n < 8; n++)
            {
                uint32_t i = (j-j_start)*8 + n;
                uint8_t input_msg = (input.uint8[i]<<1) | last_bit;
                last_bit = input.uint8[i] >> 7;
                state[5*(j%5)+j/5].stdint ^= static_cast<uint64_t>(input_msg) << ((blockOff & 7) << 3);
                blockOff++;            
            }
        }
    }

    /*
    
    R 取值 [0, 8), 表示 path hash 输入要8轮,每轮 256bit = 8个 u32
    U32_OFFSET 取值 [0, 8) , 表示每轮输入的 8个 u32 
    */
    template<int R, int U32_OFFSET>
    static ALEO_ADI void fast_path_hash_update_r0_to_r3_by_u32(Sha3StateV2 state[25], uint8_t &last_bit , uint32_t input)
    {
        constexpr unsigned int blockOff_start = R*32 + U32_OFFSET*sizeof(uint32_t);
        unsigned int blockOff = blockOff_start;

        constexpr unsigned int j_start = blockOff_start >> 3;
        constexpr unsigned int j_end = j_start + sizeof(uint32_t);

        //0 - 32
        #pragma unroll
        for(uint32_t j = j_start; j < j_end; j++)
        {                        
            uint32_t i = (j-j_start);
            uint8_t tmp = (input >> (i*8)) & 0xFF;
            uint8_t input_msg = (tmp<<1) | last_bit;
            last_bit = tmp >> 7;
            state[5*(j_start%5)+j_start/5].stdint ^= static_cast<uint64_t>(input_msg) << ((blockOff & 7) << 3);
            // printf("state[%u]=%0lx , msg=%02x\n", 5*(j % 5) + j / 5,  state[5*(j_start % 5) + j_start / 5].stdint , input_msg);
            // printf("blockoff=%u\n", blockOff);
            blockOff++;
        }
    }

    static __device__ void dump_state(Sha3StateV2 state[25])
    {
        for(int n = 0; n < 5; n++)
        {
            printf("state[%d][...]= ", n);
            for(int m = 0; m < 5; m++)
            {
                printf(" 0x%lx, ", state[n*5+m].stdint);
            }
            printf("\n");
        }
        
    }


    template<int U32_OFFSET>
    static ALEO_ADI void fast_path_hash_update_r4_by_u32(Sha3StateV2 state[25], uint8_t &last_bit , uint32_t input)
    {        
        //128-136,  >> 3 后: 16 - 17
        if constexpr ( U32_OFFSET < 2)
        {
            unsigned int blockOff = 128 + U32_OFFSET*sizeof(uint32_t);
            constexpr int j = 16;
            for (uint32_t i = 0; i < sizeof(uint32_t); i++)
            {
                uint8_t tmp = (input >> (i*8)) & 0xFF;
                uint8_t input_msg = (tmp<<1) | last_bit;
                last_bit = tmp >> 7;
                // 5*(j%5)+j/5 = 8
                state[8].stdint ^= static_cast<uint64_t>(input_msg) << ((blockOff & 7) << 3);
                blockOff++;            
            }
            if constexpr ( U32_OFFSET == 1 )
            {
                fast_sha3_absorb_inline(state);
            }
        }
        if constexpr ( U32_OFFSET >= 2 && U32_OFFSET < 8)
        {
            //block off 现在重置为0了
            constexpr unsigned int blockOff_start = (U32_OFFSET-2)*sizeof(uint32_t)  ;
            unsigned int blockOff = blockOff_start;

            constexpr unsigned int j_start = blockOff_start >> 3;
            constexpr unsigned int j_end = j_start + sizeof(uint32_t);

            #pragma unroll
            for(uint32_t j = j_start; j < j_end; j++)
            {                        
                uint32_t i = (j-j_start);
                uint8_t tmp = (input >> (i*8)) & 0xFF;
                uint8_t input_msg = (tmp<<1) | last_bit;
                last_bit = tmp >> 7;
                state[5*(j_start%5)+j_start/5].stdint ^= static_cast<uint64_t>(input_msg) << ((blockOff & 7) << 3);
                // printf("state[%u]=%0lx , msg=%02x\n", 5*(j_start % 5) + j_start / 5,  state[5*(j_start % 5) + j_start / 5].stdint , input_msg);
                // printf("blockoff=%u, j=%u\n", blockOff, j);
                blockOff++;
            }
        }

    }

    template<int R, int U32_OFFSET>
    static ALEO_ADI void fast_path_hash_update_r5_to_r7_by_u32(Sha3StateV2 state[25], uint8_t &last_bit , uint32_t input)
    {
        constexpr unsigned int blockOff_start = (R-4)*32 - 8 + (U32_OFFSET)*sizeof(uint32_t);
        unsigned int blockOff = blockOff_start;

        constexpr unsigned int j_start = blockOff_start >> 3;
        constexpr unsigned int j_end = j_start + sizeof(uint32_t);

        /*
            对 R5 :
            U32_OFFSET = 0 ,  blockOff_start = 24
            U32_OFFSET = 1 ,  blockOff_start = 28
            U32_OFFSET = 2 ,  blockOff_start = 32
            U32_OFFSET = 3 ,  blockOff_start = 36
        
        */
        //0 - 32
        #pragma unroll
        for(uint32_t j = j_start; j < j_end; j++)
        {                        
            uint32_t i = (j-j_start);
            uint8_t tmp = (input >> (i*8)) & 0xFF;
            uint8_t input_msg = (tmp<<1) | last_bit;
            last_bit = tmp >> 7;
            if constexpr (R == 5 ) {
                // printf("state[%u]=%0lx , msg=%02x\n", 5*(j % 5) + j / 5,  state[5*(j_start % 5) + j_start / 5].stdint , input_msg);
                // printf("blockoff=%u, j=%u\n", blockOff, j);
            }
            state[5*(j_start%5)+j_start/5].stdint ^= static_cast<uint64_t>(input_msg) << ((blockOff & 7) << 3);
            if constexpr (R == 5 ) {
                //printf("state[%u]=%0lx , msg=%02x\n", 5*(j % 5) + j / 5,  state[5*(j_start % 5) + j_start / 5].stdint , input_msg);                
            }
            
            blockOff++;
        }        
    }

    template<int R>
    static ALEO_ADI void fast_path_hash_update_r5_to_r7(Sha3StateV2 state[25], uint8_t &last_bit , bigint_u256_t &input)
    {
        
        /*
        block off = 24-56
        int j = blockOff >> 3; 
        j in 3-7
        */
        constexpr unsigned int blockOff_start = (R-5)*32 + 24;
        unsigned int blockOff = blockOff_start;

        constexpr unsigned int j_start = (R-5)*4 + 3;
        constexpr unsigned int j_end = (R-5)*4 + 3 + 4;
        
        #pragma unroll
        for ( int j = j_start; j < j_end; j++)
        {
            for (uint32_t n = 0; n < 8; n++)
            {
                uint32_t i = ((j-j_start)*8 ) + n;
                uint8_t input_msg = (input.uint8[i]<<1) | last_bit;
                last_bit = input.uint8[i] >> 7;
                if constexpr (R == 5 ) {
                    // printf("state[%u]=%0lx , msg=%02x\n", 5*(j % 5) + j / 5,  state[5*(j_start % 5) + j_start / 5].stdint , input_msg);
                    // printf("blockoff=%u, j=%u\n", blockOff, j);
                }
                state[5*(j%5)+j/5].stdint ^= static_cast<uint64_t>(input_msg) << ((blockOff & 7) << 3);
                if constexpr (R == 5 ) {
                    //printf("state[%u]=%0lx , msg=%02x\n", 5*(j % 5) + j / 5,  state[5*(j_start % 5) + j_start / 5].stdint , input_msg);
                }
                blockOff++;   
            }
        }        
        //printf("r=%u , block_off=%u\n", R, blockOff);
    }

    //offset 取值从  [0-7]
    template <unsigned int OFFSET>
    static ALEO_ADI uint32_t fast_path_hash_final_by_u32(Sha3StateV2 state[25])
    {
        
        uint32_t data = 0;
        

        constexpr int i_0 = 4*OFFSET;
        data |= (static_cast<uint8_t>(state[5*((i_0>>3) % 5)].stdint >> ((i_0 & 7) << 3)) ) ;

        constexpr int i_1 = 4*OFFSET + 1;
        data |= (static_cast<uint8_t>(state[5*((i_1>>3) % 5)].stdint >> ((i_1 & 7) << 3)) ) << 8;

        constexpr int i_2 = 4*OFFSET + 2;
        data |= (static_cast<uint8_t>(state[5*((i_2>>3) % 5)].stdint >> ((i_2 & 7) << 3)) ) << 16;

        constexpr int i_3 = 4*OFFSET + 3;
        data |= (static_cast<uint8_t>(state[5*((i_3>>3) % 5)].stdint >> ((i_3 & 7) << 3)) ) << 24;

        return data;

        // union 
        // {
        //     uint8_t bytes[4];
        //     uint32_t data;    
        // };
                
        // #pragma unroll
        // for (int i = 4*OFFSET; i < 4*(OFFSET+1); i++)
        // {                            
        //     bytes[i - 4*OFFSET] = (static_cast<uint8_t>(state[5*((i>>3) % 5)].stdint >> ((i & 7) << 3)) ); 
        // }
        // return data;
    }

    static ALEO_ADI void fast_path_hash_final_state(Sha3StateV2 state[25],  uint8_t &last_bit)
    {        

        constexpr unsigned int blockOff = 120;
        {
            constexpr int j = 15; // blockOff >> 3;
            state[3].stdint ^= static_cast<uint64_t>(last_bit) << ((blockOff & 7) << 3);
            
        }

        
        // Final block and padding
        {
            constexpr unsigned int blockOff_2 = 121;
            constexpr int j_2 = blockOff_2 >> 3;
            constexpr int index_2 = 5*(j_2%5) + j_2/5;
            state[index_2].stdint ^= UINT64_C(0x06) << ((blockOff_2 & 7) << 3);
            
            constexpr int blockOff_3 = BLOCK_SIZE - 1;
            constexpr int j_3 = blockOff_3 >> 3;
            constexpr int index_3 = 5*(j_3%5) + j_3/5;

            state[index_3].stdint ^= UINT64_C(0x80) << ((blockOff_3 & 7) << 3);
            //fast_sha3_absorb_inline(state);
            fast_sha3_absorb_last_inline(state);
        }

    }

    static __device__ __forceinline__ void fast_path_hash_final(Sha3StateV2 state[25],  uint8_t &last_bit, bigint_u256_t &output)
    {        

        constexpr unsigned int blockOff = 120;
        {
            constexpr int j = 15; // blockOff >> 3;
            state[3].stdint ^= static_cast<uint64_t>(last_bit) << ((blockOff & 7) << 3);
            
        }

        
        // Final block and padding
        {
            constexpr unsigned int blockOff_2 = 121;
            constexpr int j_2 = blockOff_2 >> 3;
            constexpr int index_2 = 5*(j_2%5) + j_2/5;
            state[index_2].stdint ^= UINT64_C(0x06) << ((blockOff_2 & 7) << 3);
            
            constexpr int blockOff_3 = BLOCK_SIZE - 1;
            constexpr int j_3 = blockOff_3 >> 3;
            constexpr int index_3 = 5*(j_3%5) + j_3/5;

            state[index_3].stdint ^= UINT64_C(0x80) << ((blockOff_3 & 7) << 3);
            fast_sha3_absorb_inline(state);
        }

        // Uint64 array to bytes in little endian         
        for (int i = 0; i < KECCCAK256_HASH_LEN; i++) {
            output.bytes[i] = static_cast<uint8_t>(state[5*((i>>3) % 5)].stdint >> ((i & 7) << 3));
        }
    }
    

    // N 取值  0,1,3,4,5,6,7
    template <int N>
    static ALEO_ADI void fast_leaf_hash_update(Sha3StateV2 state[25], uint8_t &last_bit, uint32_t input)
    {
        constexpr int BLOCK_OFF_START = N*4;
        constexpr int j = N/2 ;
        //0 , 4, 8 , 12, 16 , 20
        #pragma unroll
        for(uint32_t i = 0; i < 4; i++)
        {
            uint8_t input_msg = ( input >> (i*8) )  & 0xFF;
            uint8_t tmp = input_msg;
            input_msg = (input_msg<<1) | last_bit;
            last_bit = tmp >> 7;

            uint32_t blockOff = BLOCK_OFF_START + i;            
            state[5*(j%5)+j/5].stdint ^= static_cast<uint64_t>(input_msg) << ((blockOff & 7) << 3);
        } 
    }

    static ALEO_ADI void fast_leaf_hash_final_state(Sha3StateV2 state[25])
    {
        constexpr unsigned  int blockOff_1 = 32;
        // Final block and padding
        {
            constexpr int i = blockOff_1 >> 3;
            constexpr int offset1_help = 5*(i % 5) +i / 5;
            state[20].stdint ^= UINT64_C(0x06) << ((blockOff_1 & 7) << 3);
            constexpr unsigned  int blockOff_2 = BLOCK_SIZE - 1;
            constexpr unsigned int j = blockOff_2 >> 3;
            constexpr int offset2_help = 5*(j % 5) + j / 5;
            state[8].stdint ^= UINT64_C(0x80) << ((blockOff_2 & 7) << 3);  
            fast_leaf_sha3_absorb_last_inline(state);
            //fast_sha3_absorb_last_inline(state);
        }       
        
    }
};