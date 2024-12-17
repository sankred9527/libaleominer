
#include <memory>
#include <random>
#include "Farmer.h"
#include "Log.h"
#include "CUDAMiner.h"
#include "Miner.h"


using namespace dev;


Farm::Farm(std::map<std::string, DeviceDescriptor>& _DevicesCollection,
        CUSettings _CUSettings)
    :   m_DevicesCollection(_DevicesCollection),
        m_CUSettings(_CUSettings)
{
    
}

bool Farm::start()
{
    if (m_isMining.load(std::memory_order_relaxed))
        return true;

    DEV_BUILD_LOG_PROGRAMFLOW(cnote, "Farm::start() begin");

    Guard l(x_minerWork);

    if (!m_miners.size())
    {        
        for (auto it = m_DevicesCollection.begin(); it != m_DevicesCollection.end(); it++)
        {            
            TelemetryAccountType minerTelemetry;
            if (it->second.subscriptionType == DeviceSubscriptionTypeEnum::Cuda)
            {
                minerTelemetry.prefix = "cu";                
                m_miners.push_back(std::make_shared<CUDAMiner>(m_miners.size(), m_CUSettings, it->second));
                m_miners.back()->startWorking();
            }
            if (minerTelemetry.prefix.empty())
                continue;            
            m_telemetry.miners.push_back(minerTelemetry);
        }
        
        m_isMining.store(true, std::memory_order_relaxed);
    } else {
        for (auto const& miner : m_miners)
            miner->startWorking();
        m_isMining.store(true, std::memory_order_relaxed);
    }

    return m_isMining.load(std::memory_order_relaxed);
    
}

void Farm::stop()
{
    DEV_BUILD_LOG_PROGRAMFLOW(cnote, "Farm::stop() begin");
    // Avoid re-entering if not actually mining.
    // This, in fact, is also called by destructor
    if (isMining())
    {
        {
            Guard l(x_minerWork);
            for (auto const& miner : m_miners)
            {
                miner->triggerStopWorking();
                miner->kick_miner();
            }

            m_miners.clear();
            m_isMining.store(false, std::memory_order_relaxed);
        }
    }
    DEV_BUILD_LOG_PROGRAMFLOW(cnote, "Farm::stop() end");
}


void Farm::setWork(uint8_t *epoch, uint8_t *addr, uint64_t target, uint64_t start_counter, void *program, size_t program_size)
{

    // Set work to each miner giving it's own starting nonce
    Guard l(x_minerWork);

    m_currentWp.init_from(epoch, addr, target, program, program_size);
    m_currentWp.counter = start_counter;    

    for (unsigned int i = 0; i < m_miners.size(); i++)
    {
        uint64_t offset = 1ULL<<32;
        m_currentWp.counter += static_cast<uint64_t>(i)*offset;
        m_miners.at(i)->setWork(m_currentWp);        
    }

}

void Farm::collectData()
{
    uint32_t index = 0;
    
    for (auto const& miner : m_miners)
    {
        int minerIdx = miner->Index();                
        m_telemetry.miners.at(minerIdx).hashrate = miner->RetrieveHashRate();
        miner->TriggerHashRateUpdate();
    }    
    
}