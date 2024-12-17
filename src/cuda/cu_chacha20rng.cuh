#pragma once

#include <stdio.h>
#include <stdint.h>
#include "cu_common.h"
#include "cu_utils.cuh"
#include "libdevcore/ChaChaCommon.h"


static __device__ void dump_chacha_state(chacha_state_t *state)
{
    printf("dump chacha state:\n");
    for(int n =0;n<2;n++)
    {
        for(int m=0;m<4;m++)
        {
            printf("%u\n", state->key[n][m] );
        }
    }
    printf("\n");
}

class CuChaCha20Rng
{
private:    


    static 
    #ifdef __CUDA_ARCH__
    __host__ __device__
    #endif 
    inline uint64_t uint32x4_pos64(uint32x4 *data)
    {
        uint64_t low = (*data)[0] ;
        uint64_t high = (*data)[1];
        
        return low | (high<<32);
    }

    static ALEO_ADI void quarter_round(
                uint32_t &a, uint32_t &b,
                uint32_t &c, uint32_t &d
            )
    {
        a += b;
        d ^= a;
        d = (d << 16) | (d >> 16);
        c += d;
        b ^= c;
        b = (b << 12) | (b >> 20);
        a += b;
        d ^= a;
        d = (d << 8) | (d >> 24);
        c += d;
        b ^= c;
        b = (b << 7) | (b >> 25);
    }

    static __device__ void refill_wide_iter_once(uint32x4 *key, uint32x4 *nounce, uint32_t state[16]) 
    {
        uint32_t init_state[16] = {
                    CHACHA_MAGIC,
                    key[0][0], key[0][1], key[0][2], key[0][3],
                    key[1][0], key[1][1], key[1][2], key[1][3],
                    nounce[0][0],  nounce[0][1],  nounce[0][2],  nounce[0][3]
                };

        memcpy(state, init_state, sizeof(init_state));

        #pragma unroll
        for (int i = 0; i < 10; ++i) {
            quarter_round(state[0], state[4], state[8], state[12]);
            quarter_round(state[1], state[5], state[9], state[13]);
            quarter_round(state[2], state[6], state[10], state[14]);
            quarter_round(state[3], state[7], state[11], state[15]);

            quarter_round(state[0], state[5], state[10], state[15]);
            quarter_round(state[1], state[6], state[11], state[12]);
            quarter_round(state[2], state[7], state[8], state[13]);
            quarter_round(state[3], state[4], state[9], state[14]);
        }
        return;
    }

    static ALEO_ADI void sum_uint32x4_and_output(uint32x4* d1, uint32x4* d2, uint8_t out[16])
    {
        #pragma unroll
        for(int n=0;n<4;n++)
        {
            uint32_t *p = NULL;        
            p = (uint32_t*)(out + n*4);
            //printf("sum %08x ,%08x \n", ((uint32_t*)d1)[n] , ((uint32_t*)d2)[n] );
            *p = ((uint32_t*)d1)[n] + ((uint32_t*)d2)[n];
        }
    }


    static __device__ void refill_wide_impl_simple(chacha_state_t *state) 
    {
        refill_wide_impl(state, NULL);
    }

    /*
    state.key[0] : 对应 rust rand_chacha里的 state.b
    state.key[1] : 对应 rust rand_chacha里的 state.c
    state.nounce[0] : 对应 rust rand_chacha里的 state.d
    */
    static __device__ void refill_wide_impl(chacha_state_t *state, aleo_u256_t *out256) 
    {    
        #define CC_NOUNCE (state->nounce[0])
        #define CC_KEY (state->key)

        uint64_t pos = uint32x4_pos64(state->nounce);

        uint32x4 d0 = { CC_NOUNCE[0],  CC_NOUNCE[1],  CC_NOUNCE[2],  CC_NOUNCE[3] };
        pos += 1;
        uint32x4 d1 = { (uint32_t)pos, (uint32_t)(pos>>32) , CC_NOUNCE[2],  CC_NOUNCE[3] };
        pos += 1;
        uint32x4 d2 = { (uint32_t)pos, (uint32_t)(pos>>32) , CC_NOUNCE[2],  CC_NOUNCE[3] };
        pos += 1;
        uint32x4 d3 = { (uint32_t)pos, (uint32_t)(pos>>32) , CC_NOUNCE[2],  CC_NOUNCE[3] };    
        pos += 1;
        uint32x4 d4 = { (uint32_t)pos, (uint32_t)(pos>>32) , CC_NOUNCE[2],  CC_NOUNCE[3] };    

        uint32x4 b; 
        memcpy(b, CC_KEY[0], sizeof(uint32x4));
        uint32x4 c; 
        memcpy(c, CC_KEY[1], sizeof(uint32x4));    

        /*
        out_state[0] 对应 rand_chacha里 refill_wide_impl 里的 ( x.a[0], x.b[0], x.c[0], x.d[0] )

        out_state[3] 对应 rand_chacha里 refill_wide_impl 里的 ( x.a[3], x.b[3], x.c[3], x.d[3] )
        */
        uint32_t out_state[4][16];
        refill_wide_iter_once(CC_KEY, &d0, out_state[0]);
        refill_wide_iter_once(CC_KEY, &d1, out_state[1]);
        refill_wide_iter_once(CC_KEY, &d2, out_state[2]);
        refill_wide_iter_once(CC_KEY, &d3, out_state[3]);
        
        uint32x4 *sb = CC_KEY + 0;
        uint32x4 *sc = CC_KEY + 1;
        uint32x4 sd[4];
        memcpy(sd+0, CC_NOUNCE, sizeof(uint32x4));
        memcpy(sd+1, d1, sizeof(uint32x4));
        memcpy(sd+2, d2, sizeof(uint32x4));
        memcpy(sd+3, d3, sizeof(uint32x4));

        memcpy(CC_NOUNCE, d4, sizeof(uint32x4));

        #pragma unroll
        for (int n = 0; n < 4 ; n++)
        {
            uint32x4 k = { CHACHA_MAGIC };
            uint32x4 *a = ((uint32x4*)out_state[n])+0;
            uint32x4 *b = ((uint32x4*)out_state[n])+1;
            uint32x4 *c = ((uint32x4*)out_state[n])+2;
            uint32x4 *d = ((uint32x4*)out_state[n])+3;

            if ( out256 == NULL ) {
                //输出到 256个字节内
                sum_uint32x4_and_output(a, &k, (state->out) + 64*n + 0);
                sum_uint32x4_and_output(b, sb, (state->out) + 64*n + 16);
                sum_uint32x4_and_output(c, sc, (state->out) + 64*n + 32);
                sum_uint32x4_and_output(d, sd+n, (state->out) + 64*n + 48);
            } else {
                //每个aleo_u256_t 是32个字节, 这里需要填充 8个 aleo_u256_t
                sum_uint32x4_and_output(a, &k, (out256[n*2].bytes) + 0);
                sum_uint32x4_and_output(b, sb, (out256[n*2].bytes) + 16);

                sum_uint32x4_and_output(c, sc, (out256[n*2+1].bytes) + 0);
                sum_uint32x4_and_output(d, sd+n, (out256[n*2+1].bytes) + 16);
            }
            
        }    
    }

public:

    static __device__ uint32_t pcg32(uint64_t *state)
    {
        const uint64_t MUL = 6364136223846793005ULL;
        const uint64_t INC = 11634580027462260723ULL;
        
        *state = (*state)*MUL + INC;        
        uint64_t new_state = *state;
        uint32_t xorshifted = (((new_state >> 18) ^ new_state) >> 27);        
        uint32_t rot = (new_state >> 59);        
        uint32_t v1 = xorshifted >> rot;
        uint32_t v2 = xorshifted << (32-rot);
        
        return  v1 | v2;        
    }

    static __device__ void init_from_u64(chacha_state_t *state,  uint64_t u64_state);

    static __device__ void init_with_key(chacha_state_t *state, uint32_t key[CHACHA_KEY_SIZE_IN_INT32])
    {
        memcpy(state->key, key, sizeof(uint32_t)*CHACHA_KEY_SIZE_IN_INT32);
        memset(state->nounce, 0, sizeof(state->nounce[0]));
        memset(state->out, 0, CHACHA_OUT_BUF_SIZE);
        state->cache_index = CHACHA_OUT_BUF_SIZE;
    }

    static __device__ uint32_t chacha_rand_in_range_100000_200000(chacha_state_t *state);    
    static __device__ uint32_t chacha_rand_next_u32(chacha_state_t *state);
    static __device__ uint64_t chacha_rand_next_u64(chacha_state_t *state);
    static __device__ bigint_u128_t chacha_rand_next_u128(chacha_state_t *state);

};

extern "C" __device__ void chacha_init_from_u64(chacha_state_t *state,  uint64_t u64_state);

#ifdef ENABLE_TEST


//默认的 nounce 全 0 
static __global__ void test_for_chacha_rng_u64(
        uint32_t key[CHACHA_KEY_SIZE_IN_INT32],        
        uint64_t *out, uint32_t out_len)
{
    chacha_state_t *state = new chacha_state_t();
	CuChaCha20Rng::init_with_key(state, key);
    for(int n=0; n < out_len; n++)
    {
        out[n] = CuChaCha20Rng::chacha_rand_next_u64(state);
    }
    free(state);
}

static __global__ void test_for_chacha_range_random_from_u64(uint64_t origin_seed, uint32_t *out, uint32_t out_len)
{   
    chacha_state_t *state = new chacha_state_t();	
    chacha_init_from_u64(state, origin_seed);
    for(int n=0; n < out_len; n++)
    {
        out[n] = CuChaCha20Rng::chacha_rand_in_range_100000_200000(state);
    }
    free(state);
}

static __global__ void test_for_chacha_range_random(
        uint32_t key[CHACHA_KEY_SIZE_IN_INT32],        
        uint32_t *out, uint32_t out_len)
{
    chacha_state_t *state = new chacha_state_t();
	CuChaCha20Rng::init_with_key(state, key);
    for(int n=0; n < out_len; n++)
    {
        out[n] = CuChaCha20Rng::chacha_rand_in_range_100000_200000(state);
    }
    free(state);
}

#endif 