#include "cu_solution.cuh"
#include "cu_sha256.cuh"
#include "cu_aleo_globals.cuh"

__device__ uint64_t Solution::solution_new(uint64_t counter)
{
    uint8_t buff[48];
   
    memcpy(buff, d_epoch_hash, 8);
    memcpy(buff+8, d_address, 32);
    memcpy(buff+40, &counter, 8);

    return cu_sha256d_to_u64(buff, 48);            
}