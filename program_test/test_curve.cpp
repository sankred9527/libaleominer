
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "cu_macro.h"
#include "const_inverse_u16.h"

extern "C" void test_curve();
extern "C" void test_inverse();

int main()
{    

    cudaDeviceProp device_prop{};
    int current_device{0};
    CUDA_SAFE_CALL(cudaGetDevice(&current_device));
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&device_prop, current_device));
    std::cout << "GPU: " << device_prop.name << std::endl;
    std::cout << "L2 Cache Size: " << device_prop.l2CacheSize / 1024 / 1024
              << " MB" << std::endl;
    std::cout << "Max Persistent L2 Cache Size: "
              << device_prop.persistingL2CacheMaxSize / 1024 / 1024 << " MB"
              << std::endl;    

    printf("s=%lu\n", sizeof(const_u16_inverse_hash));
    //test_curve();

    
    //test_curve();
    test_inverse();

}