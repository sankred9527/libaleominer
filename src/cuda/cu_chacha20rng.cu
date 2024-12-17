
#include "cu_chacha20rng.cuh"

__device__ uint32_t CuChaCha20Rng::chacha_rand_in_range_100000_200000(chacha_state_t *state)
{
    /// 从 rand-0.8.5/src/distributions/uniform.rs 复制来的算法
    uint64_t zone = 14073748835532799999;
    uint64_t low_range = 100000;
    uint64_t range = 100000 ;
    uint32_t ret ;
    while (1)
    {
        uint64_t v = chacha_rand_next_u64(state);            
        //wide multi            
        aleo_u128_t ret128;
        multiply_u64(v, range, ret128);

        uint64_t high = ret128[1];
        uint64_t low = ret128[0];
        if ( low < zone) {                
            ret = low_range + high;                
            break;
        }
    }
    return ret;
}

__device__ void CuChaCha20Rng::init_from_u64(chacha_state_t *state,  uint64_t u64_state)
{        
    uint32_t seed[8];
    for(int n = 0;n < 8; n++ )
    {
        uint32_t v = pcg32(&u64_state);
        seed[n] = v;
    }
    CuChaCha20Rng::init_with_key(state, seed);
}


__device__ uint32_t CuChaCha20Rng::chacha_rand_next_u32(chacha_state_t *state)
{
    const uint32_t len = CHACHA_OUT_BUF_SIZE;
    const uint32_t u32_size = 4;

    //每次从out buf 里读取 4 个字节
    if ( state->cache_index < (len - u32_size + 1) ) 
    {    
        uint32_t val = *(uint32_t*)(state->out + state->cache_index);
        state->cache_index += u32_size;
        return val;
    } else if ( state->cache_index >= len )
    {
        refill_wide_impl_simple(state);
        uint32_t val = *(uint32_t*)(state->out);
        state->cache_index = u32_size;
        return val;
    } else {
        uint32_t val;
        uint8_t *p = (uint8_t*)&val;
        uint32_t prev_size = len - 1 - state->cache_index + 1;
        for(int n = 0; state->cache_index <= (len - 1); state->cache_index++, n++ )
        {
            p[n] = state->out[state->cache_index];
        }
        uint32_t left = u32_size - prev_size;
        refill_wide_impl_simple(state);
        state->cache_index = 0;
        for (int n=0; n < left; n++ ) 
        {
            p[n+prev_size] = state->out[state->cache_index];
            state->cache_index += 1;
        }
        return val;
    }
}

__device__ bigint_u128_t CuChaCha20Rng::chacha_rand_next_u128(chacha_state_t *state)
{
    bigint_u128_t v;

    v.uint64[0] = CuChaCha20Rng::chacha_rand_next_u64(state);
    v.uint64[1] = CuChaCha20Rng::chacha_rand_next_u64(state);

    return v;
}

__device__ uint64_t CuChaCha20Rng::chacha_rand_next_u64(chacha_state_t *state)
{

    const uint32_t len = CHACHA_OUT_BUF_SIZE;
    const uint32_t u64_size = 8;    

    //每次从out buf 里读取 8 个字节
    if ( state->cache_index < (len-7) ) 
    {    
        uint64_t val = *(uint64_t*)(state->out + state->cache_index);
        state->cache_index += u64_size;
        return val;
    } else if ( state->cache_index >= len )
    {
        uint64_t val;        
        refill_wide_impl_simple(state);
        //FIXME: 注意对齐访问
        val = *(uint64_t*)(state->out);                
        state->cache_index = u64_size;        
        return val;
    } else {
        uint64_t val;
        uint8_t *p = (uint8_t*)&val;
        uint32_t prev_size = len - 1 - state->cache_index + 1;
        for(int n = 0; state->cache_index <= (len - 1); state->cache_index++, n++ )
        {
            p[n] = state->out[state->cache_index];
        }
        uint32_t left = u64_size - prev_size;
        refill_wide_impl_simple(state);
        state->cache_index = 0;
        for (int n=0; n < left; n++ ) 
        {
            p[n+prev_size] = state->out[state->cache_index];
            state->cache_index += 1;
        }
        return val;
    }
}


extern "C" __device__  void chacha_init_from_u64(chacha_state_t *state,  uint64_t u64_state)
{
    CuChaCha20Rng::init_from_u64(state,  u64_state);
}