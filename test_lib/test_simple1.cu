
#include <functional>
#include <sstream>  
#include <map>
#include <iomanip>
#include <cuda_runtime.h>
#include <unistd.h>
#include "Miner.h"
#include "cu_macro.h"
#include "CUDAMiner.h"
#include "test_data.h"
#include "cuda/cu_aleo_interface.h"

using namespace dev;
using namespace std;



extern "C" void testlib_simple1(uint8_t *epoch_bin, void *program , size_t program_size)
{
    std::chrono::_V2::system_clock::time_point start;
    std::chrono::_V2::system_clock::time_point end;
    std::chrono::microseconds duration;

    CUSettings m_CUSettings;
    std::map<std::string, DeviceDescriptor> m_DevicesCollection = {};

    CUDAMiner::enumDevices(m_DevicesCollection);

    DeviceDescriptor dev_desc;
    for (auto it = m_DevicesCollection.begin(); it != m_DevicesCollection.end(); it++)
    {
        dev_desc = it->second;
        break;
    }
    CUDAMiner *cuminer = new CUDAMiner(0, m_CUSettings, dev_desc);
    cuminer->initDevice();

    uint8_t epoch[32] = TEST_EPOCH_HASH;

    if ( epoch_bin != nullptr )
    {
        memcpy(epoch, epoch_bin, 32);
    }

    //uint8_t epoch[] = {222, 58, 142, 83, 213, 156, 20, 87, 165, 216, 222, 194, 251, 222, 150, 65, 144, 138, 84, 164, 216, 109, 74, 166, 183, 88, 99, 220, 222, 4, 106, 18};

    uint8_t addr[] = TEST_ADDRESS;
    uint64_t target = 10000;
    printf("program size=%ld\n", program_size);
    
    CUSettings &m_settings = cuminer->m_settings;       
    
    volatile Search_results* buffer;
    CUDA_SAFE_CALL(cudaMallocHost(&buffer, sizeof(Search_results)));
    uint64_t start_counter = 0x1234567887654321ULL;
    cuminer->initEpoch_internal(epoch, addr, target, program, program_size);
    printf("run search counter=%lx grid size=%d , blocksize=%d\n", start_counter, m_settings.gridSize, m_settings.blockSize);    

    cudaStream_t stream;
    CUDA_SAFE_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for(int n =0;n<3;n++)
    {
        start = std::chrono::high_resolution_clock::now();
        // Run the batch for this stream
        cu_aleo_search(m_settings.gridSize, 
                m_settings.blockSize, stream, 
                (uint8_t*)m_settings.device_mem_addr,
                start_counter,
                buffer);
        cudaDeviceSynchronize();    
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Execution time: " << duration.count() / 1000.0 << " microseconds" << std::endl;    
    }

#if 1
    int64_t total_us = 0;
    constexpr uint32_t loop_cnt = 100;
    for(int n =0;n<loop_cnt;n++)
    {
        buffer->count = 0;
        start = std::chrono::high_resolution_clock::now();
        // Run the batch for this stream
        cu_aleo_search(m_settings.gridSize, 
                m_settings.blockSize, stream,
                (uint8_t*)m_settings.device_mem_addr,
                start_counter,
                buffer);
        cudaDeviceSynchronize();    
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        total_us += duration.count();
        std::cout << "Execution time: " << duration.count() / 1000.0 << " microseconds" << std::endl;              
    }

    std::cout << "AVG Execution time: " << total_us / (loop_cnt*1000.0) << " microseconds" << std::endl;         
        
#endif     

    printf("found %u solution\n", buffer->count);
    if ( buffer->count > 0 ) 
    {
        
        for(int n = 0; n < buffer->count; n++)
        {
            printf("find solution, counter=0x%lx, target=%lx\n", buffer->result[n].counter, buffer->result[n].solution_target);
        }
    }
    
}