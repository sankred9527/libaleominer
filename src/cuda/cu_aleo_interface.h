

#pragma once 
#include <cstdint>
#include "cu_aleo_globals.cuh"
#include "libaleocore/AleoProve.h"

void cu_aleo_search(int grid_size, int block_size, cudaStream_t stream,
            uint8_t *leaves_memory,
            uint64_t start_counter, volatile Search_results *result);
