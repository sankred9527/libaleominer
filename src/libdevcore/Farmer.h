
#pragma once

#include <atomic>
#include <map>
#include <string>
#include "Miner.h"


namespace dev
{

/**
 * @brief Class for hosting one or more Miners.
 * @warning Must be implemented in a threadsafe manner since it will be called from multiple
 * miner threads.
 */
class Farm 
{
public:
    
    Farm(std::map<std::string, DeviceDescriptor>& _DevicesCollection,
        CUSettings _CUSettings
       );

    CUSettings m_CUSettings;  // Cuda settings passed to CUDA Miner instantiator

    /**
     * @brief Get information on the progress of mining this work package.
     * @return The progress with mining so far.
     */
    TelemetryType& Telemetry() { return m_telemetry; }

    /**
     * @brief Sets the current mining mission.
     * @param _wp The work package we wish to be mining.
     */
    void setWork(uint8_t *_epoch, uint8_t *addr, uint64_t target, uint64_t start_counter, void *program, size_t program_size);

    /**
     * @brief Start a number of miners.
     */
    bool start();

    /**
     * @brief All mining activities to a full stop.
     * Implies all mining threads are stopped.
     */
    void stop();

    /**
     * @brief Returns whether or not the farm has been started
     */
    bool isMining() const { return m_isMining.load(std::memory_order_relaxed); }

    // Collects data about hashing and hardware status
    void collectData(void);

private :
    mutable Mutex x_minerWork;
    std::vector<std::shared_ptr<Miner>> m_miners;  // Collection of miners

    WorkPackage m_currentWp;

    std::atomic<bool> m_isMining = {false};

    std::map<std::string, DeviceDescriptor>& m_DevicesCollection;

    TelemetryType m_telemetry;  // Holds progress and status info for farm and miners

};


}