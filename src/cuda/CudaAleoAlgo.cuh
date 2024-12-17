#pragma once

#include <cstdint>
#include "libaleocore/AleoProve.h"

__global__ void cu_aleo_kernel_search(void *dev_mem, 
            uint64_t start_counter, 
            volatile Search_results *results);


