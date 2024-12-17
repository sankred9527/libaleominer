
#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include "Miner.h"

namespace dev
{


class CUDAMiner : public Miner
{
public :
    CUDAMiner(unsigned _index, CUSettings _settings, DeviceDescriptor& _device);
    ~CUDAMiner() override;

    CUSettings m_settings;

    static int getNumDevices();
    static void enumDevices(std::map<string, DeviceDescriptor>& _DevicesCollection, uint8_t miner_mask = 0xff );

    bool initDevice() override;

    void initEpoch_internal(uint8_t *epoch_hash, uint8_t *address, uint64_t target, void *epoch_program, size_t program_size) override;

    void kick_miner() override;

    void search(const WorkPackage& w);

private:
    std::atomic<bool> m_new_work = {false};

    void workLoop() override;    

    std::vector<volatile Search_results*> m_search_buf;
    std::vector<cudaStream_t> m_streams;

    uint32_t m_batch_size;
    uint32_t m_streams_batch_size;

};

}