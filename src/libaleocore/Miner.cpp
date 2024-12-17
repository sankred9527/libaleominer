
#include "Miner.h"


namespace dev
{


DeviceDescriptor Miner::getDescriptor()
{
    return m_deviceDescriptor;
}

void Miner::setWork(WorkPackage const& _work)
{
    {
        Guard l(x_work);
        m_work = _work;        
#ifdef DEV_BUILD
        m_workSwitchStart = std::chrono::steady_clock::now();
#endif
    }

    kick_miner();    
}



float Miner::RetrieveHashRate() noexcept
{
    //return m_hashRate.load(std::memory_order_relaxed);
    int b = 3;
    if (m_hashRateUpdate.compare_exchange_strong(b,0))
    {
        auto ret = m_hashRate.load(std::memory_order_relaxed);
        return ret;
    } else {  
        return 0.0f;
    }    
}

void Miner::TriggerHashRateUpdate() noexcept
{
    int b = 0;
    m_hashRateUpdate.compare_exchange_strong(b, 1); 
}

WorkPackage Miner::work() const
{
    Guard l(x_work);
    return m_work;
}

void Miner::updateHashRate(uint32_t _groupSize, uint32_t _increment) noexcept
{

    m_groupCount += _increment;
    int b = 1;
    if (!m_hashRateUpdate.compare_exchange_strong(b, 2))
        return;

    m_hashRate.store( float(m_groupCount * _groupSize) );   
    m_hashRateUpdate.store(3);
    m_groupCount = 0;
}




}  // namespace dev
