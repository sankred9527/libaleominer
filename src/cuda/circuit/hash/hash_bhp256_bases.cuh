#pragma once
#include "cuda/cu_bigint_define.cuh"

extern __constant__ bigint_u256_t hash_bhp256_bases[592];

extern __constant__  bigint_u256_t hash_bhp256_domain;

extern __constant__ uint64_t bhp256_pre_caculate_sum[10][8];