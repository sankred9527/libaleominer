#pragma once
#include "cu_common.h"
#include "cu_keccak_v2.cuh"
#include <cstdint>


#define SHA3_update_state_with_uin32x8(round, state, last_bit, data) \
    path_hash_update_state_with_u32<round, 0>(state, last_bit, data[0]);\
    path_hash_update_state_with_u32<round, 1>(state, last_bit, data[1]);\
    path_hash_update_state_with_u32<round, 2>(state, last_bit, data[2]);\
    path_hash_update_state_with_u32<round, 3>(state, last_bit, data[3]);\
    path_hash_update_state_with_u32<round, 4>(state, last_bit, data[4]);\
    path_hash_update_state_with_u32<round, 5>(state, last_bit, data[5]);\
    path_hash_update_state_with_u32<round, 6>(state, last_bit, data[6]);\
    path_hash_update_state_with_u32<round, 7>(state, last_bit, data[7]);

#define SHA3_update_state_with_macro(round, state, last_bit, data_macro) \
    path_hash_update_state_with_u32<round, 0>(state, last_bit, data_macro##_0);\
    path_hash_update_state_with_u32<round, 1>(state, last_bit, data_macro##_1);\
    path_hash_update_state_with_u32<round, 2>(state, last_bit, data_macro##_2);\
    path_hash_update_state_with_u32<round, 3>(state, last_bit, data_macro##_3);\
    path_hash_update_state_with_u32<round, 4>(state, last_bit, data_macro##_4);\
    path_hash_update_state_with_u32<round, 5>(state, last_bit, data_macro##_5);\
    path_hash_update_state_with_u32<round, 6>(state, last_bit, data_macro##_6);\
    path_hash_update_state_with_u32<round, 7>(state, last_bit, data_macro##_7);
    


#define SHA3_update_state_with_uin32x8_dynamic(var_round, state, last_bit, data) do {\
    switch (var_round)\
    {\
    case 0:\
        SHA3_update_state_with_uin32x8(0, state, last_bit, data)\
        break;\
    case 1:\
        SHA3_update_state_with_uin32x8(1, state, last_bit, data)\
        break;\
    case 2:\
        SHA3_update_state_with_uin32x8(2, state, last_bit, data)\
        break;\
    case 3:\
        SHA3_update_state_with_uin32x8(3, state, last_bit, data)\
        break;\
    case 4:\
        SHA3_update_state_with_uin32x8(4, state, last_bit, data)\
        break;\
    case 5:\
        SHA3_update_state_with_uin32x8(5, state, last_bit, data)\
        break;\
    case 6:\
        SHA3_update_state_with_uin32x8(6, state, last_bit, data)\
        break;\
    case 7:\
        SHA3_update_state_with_uin32x8(7, state, last_bit, data)\
        break;\
    default:\
        break;\
    }\
} while(0)

template <unsigned int ROUND, unsigned int OFFSET>
static ALEO_ADI void path_hash_update_state_with_u32(Sha3StateV2  sha3_state[25], uint8_t &last_bit, uint32_t input)
{
    if constexpr ( ROUND <= 3 ) {
        KeccakV2::fast_path_hash_update_r0_to_r3_by_u32<ROUND, OFFSET>(sha3_state, last_bit, input);
    }
    
    if constexpr ( ROUND == 4 ) {
        KeccakV2::fast_path_hash_update_r4_by_u32<OFFSET>(sha3_state, last_bit, input);
    }

    if constexpr ( ROUND >= 5 && ROUND <= 7 ) {
        KeccakV2::fast_path_hash_update_r5_to_r7_by_u32<ROUND, OFFSET>(sha3_state, last_bit, input);
    }

}    
