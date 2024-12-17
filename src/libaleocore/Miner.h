

#pragma once

#include <bitset>
#include <list>
#include <numeric>
#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>

//#include <libdevcore/Common.h>
#include <libdevcore/Log.h>
#include <libdevcore/Worker.h>
#include <condition_variable>

#include "Guards.h"
#include "Worker.h"
#include "AleoProve.h"


using namespace std;

namespace dev
{

enum class DeviceTypeEnum
{
    Unknown,
    Cpu,
    Gpu,
    Accelerator
};


enum class DeviceSubscriptionTypeEnum
{
    None,
    OpenCL,
    Cuda,
    Cpu

};

enum class MinerType
{
    Mixed,
    CL,
    CUDA,
    CPU
};

enum class HwMonitorInfoType
{
    UNKNOWN,
    NVIDIA,
    AMD,
    CPU
};

enum class ClPlatformTypeEnum
{
    Unknown,
    Amd,
    Clover,
    Nvidia,
    Intel
};

enum class SolutionAccountingEnum
{
    Accepted,
    Rejected,
    Wasted,
    Failed
};


struct MinerSettings
{
    vector<unsigned> devices;
};

// Holds settings for CUDA Miner
struct CUSettings : public MinerSettings
{
    unsigned streams = 1;
    unsigned schedule = 4;
    unsigned gridSize = 8192;
    unsigned blockSize = 32;

    size_t chunk_size = 10ULL*1024*1024;
    size_t mem_size;
    size_t chunk_count = 950;

    void *device_mem_addr;
};

// Holds settings for OpenCL Miner
struct CLSettings : public MinerSettings
{
    bool noBinary = false;
    bool noExit = false;
    unsigned globalWorkSize = 0;
    unsigned globalWorkSizeMultiplier = 65536;
    unsigned localWorkSize = 128;
};

// Holds settings for CPU Miner
struct CPSettings : public MinerSettings
{
};

struct SolutionAccountType
{
    unsigned accepted = 0;
    unsigned rejected = 0;
    unsigned wasted = 0;
    unsigned failed = 0;
    std::chrono::steady_clock::time_point tstamp = std::chrono::steady_clock::now();
    string str()
    {
        string _ret = "A" + to_string(accepted);
        if (wasted)
            _ret.append(":W" + to_string(wasted));
        if (rejected)
            _ret.append(":R" + to_string(rejected));
        if (failed)
            _ret.append(":F" + to_string(failed));
        return _ret;
    };
};

struct HwSensorsType
{
    int tempC = 0;
    int fanP = 0;
    double powerW = 0.0;
    string str()
    {
        string _ret = to_string(tempC) + "C " + to_string(fanP) + "%";
        if (powerW) {
            std::stringstream ss;
            ss << " " << std::fixed << std::setprecision(2) << powerW << "W";
            _ret.append(ss.str());
        }
        return _ret;
    };
};

struct TelemetryAccountType
{
    string prefix = "";
    float hashrate = 0.0f;
    bool paused = false;
    HwSensorsType sensors; //温度，风扇
    SolutionAccountType solutions; // accept, rejct solution...
};

struct TelemetryType
{
    bool hwmon = false;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    TelemetryAccountType farm;
    std::vector<TelemetryAccountType> miners;
};


struct DeviceDescriptor
{
    DeviceTypeEnum type = DeviceTypeEnum::Unknown;
    DeviceSubscriptionTypeEnum subscriptionType = DeviceSubscriptionTypeEnum::None;

    string uniqueId;     // For GPUs this is the PCI ID
    size_t totalMemory;  // Total memory available by device
    string name;         // Device Name

    //TODO: ADD opencl supported

    bool cuDetected;  // For CUDA detected devices
    string cuName;
    unsigned int cuDeviceOrdinal;
    unsigned int cuDeviceIndex;
    string cuCompute;
    unsigned int cuComputeMajor;
    unsigned int cuComputeMinor;
    unsigned int cuMultiProcessorCount;
    
};

struct HwMonitorInfo
{
    HwMonitorInfoType deviceType = HwMonitorInfoType::UNKNOWN;
    string devicePciId;
    int deviceIndex = -1;
};

/// Pause mining
enum MinerPauseEnum
{
    PauseDueToOverHeating,
    PauseDueToAPIRequest,
    PauseDueToFarmPaused,
    PauseDueToInsufficientMemory,
    PauseDueToInitEpochError,
    Pause_MAX  // Must always be last as a placeholder of max count
};


/**
 * @brief A miner - a member and adoptee of the Farm.
 * @warning Not threadsafe. It is assumed Farm will synchronise calls to/from this class.
 */

class Miner : public Worker
{
public:
    Miner(std::string const& _name, unsigned _index)
      : Worker(_name + std::to_string(_index)), m_index(_index)
    {}

    ~Miner() override = default;

    /**
     * @brief Gets the device descriptor assigned to this instance
     */
    DeviceDescriptor getDescriptor();

    /**
     * @brief Assigns hashing work to this instance
     */
    void setWork(WorkPackage const& _work);


    unsigned Index() { return m_index; };

    // HwMonitorInfo hwmonInfo() { return m_hwmoninfo; }

    // void setHwmonDeviceIndex(int i) { m_hwmoninfo.deviceIndex = i; }

    /**
     * @brief Kick an asleep miner.
     */
    virtual void kick_miner() = 0;

    /**
     * @brief Retrieves currrently collected hashrate
     */
    float RetrieveHashRate() noexcept;

    void TriggerHashRateUpdate() noexcept;


    //算法相关的内容
    

protected:
    /**
     * @brief Initializes miner's device.
     */
    virtual bool initDevice() = 0;


    /**
     * @brief Miner's specific initialization to current (or changed) epoch.
     */
    virtual void initEpoch_internal(uint8_t *epoch_hash, uint8_t *address, uint64_t targe, void *epoch_program, size_t program_size) = 0;

    /**
     * @brief Returns current workpackage this miner is working on
     */
    WorkPackage work() const;

    void updateHashRate(uint32_t _groupSize, uint32_t _increment) noexcept;

    const unsigned m_index = 0;           // Ordinal index of the Instance (not the device)
    DeviceDescriptor m_deviceDescriptor;  // Info about the device


#ifdef DEV_BUILD
    std::chrono::steady_clock::time_point m_workSwitchStart;
#endif

    //HwMonitorInfo m_hwmoninfo;
    mutable Mutex x_work;
    mutable Mutex x_pause;
    std::condition_variable m_new_work_signal;
    

private:
    
    uint64_t m_counter;
    
    WorkPackage m_work;

    std::chrono::steady_clock::time_point m_hashTime = std::chrono::steady_clock::now();
    std::atomic<float> m_hashRate = {0.0};
    uint64_t m_groupCount = 0;
    atomic<int> m_hashRateUpdate = {0};
};

}  // namespace dev
