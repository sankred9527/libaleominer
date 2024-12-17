
#pragma once
#include <cstdint>
#include "aleo_macro.h"
#include "cu_bigint_define.cuh"

#define MAX_PROGRAM_BIN_SIZE (10240)

extern __constant__ uint64_t keccak_round_constants[24];

extern __constant__ uint8_t d_epoch_hash[ALEO_FIELD_SIZE_IN_BYTES];
extern __constant__ uint8_t d_address[ALEO_FIELD_SIZE_IN_BYTES];
extern __constant__ uint64_t d_target;


extern __constant__ uint8_t d_program_bin[MAX_PROGRAM_BIN_SIZE];
extern __constant__ uint64_t d_program_size;


extern __constant__ uint32_t d_max_thread;



extern __constant__ bigint_u256_t CONST_LEAF_HASH_ZERO;
extern __constant__ bigint_u256_t CONST_LEAF_HASH_ONE;

extern __constant__ bigint_u256_t CONST_empty_hash_r0;
extern __constant__ bigint_u256_t CONST_empty_hash_r1;
extern __constant__ bigint_u256_t CONST_empty_hash_r2;
extern __constant__ bigint_u256_t CONST_empty_hash_r3;
extern __constant__ bigint_u256_t CONST_empty_hash_r4;
extern __constant__ bigint_u256_t CONST_empty_hash_r5;

/// @brief ///////////////////////////////////

extern __constant__ uint32_t CONST_LEAF_HASH_ZERO_ONE_U32X8[2][8];
extern __constant__ uint32_t CONST_LEAF_HASH_ZERO_U32X8[8];
extern __constant__ uint32_t CONST_LEAF_HASH_ONE_U32X8[8];
extern __constant__ uint32_t CONST_empty_hash_r0_U32X8[8];
extern __constant__ uint32_t CONST_empty_hash_r1_U32X8[8];
extern __constant__ uint32_t CONST_empty_hash_r2_U32X8[8];
extern __constant__ uint32_t CONST_empty_hash_r3_U32X8[8];
extern __constant__ uint32_t CONST_empty_hash_r4_U32X8[8];
extern __constant__ uint32_t CONST_empty_hash_r5_U32X8[8];

extern __constant__ uint32_t CONST_first_round_path_hash_U32X8[256*8];

extern __constant__ uint32_t CONST_leaf_hash_for_u8[256*8];

