
#include "MinerContext.h"
#include <map>
#include "fmt/core.h"
#include <random>
#include "CUDAMiner.h"
#include "Farmer.h"

using namespace dev;
using namespace std;

class MinerContext {
public:
    std::map<std::string, DeviceDescriptor> m_DevicesCollection = {};
    CUSettings m_CUSettings;
    Farm *m_farm;
    submit_callback_t *m_submit_cb;

    uint8_t m_miner_mask;

    MinerContext () {
        m_farm = NULL;
        m_miner_mask = 0xff;
    }

    void choose_cuda_device(unsigned int *dev_seq , unsigned int len)
    {
        for(unsigned int n = 0; n< len;n++)
        {
            m_CUSettings.devices.push_back(dev_seq[n]);
        }
    }

    void prepare()
    {
        CUDAMiner::enumDevices(m_DevicesCollection, m_miner_mask);


        if (!m_DevicesCollection.size())
            throw std::runtime_error("No usable mining devices found");
    
        clog << "version: v20241110-hb" << std::endl;
        clog << "List Found GPU:" << std::endl;
        for (auto const& p : m_DevicesCollection)
        {
            auto msg = fmt::format("id={}, name={}, totalMemory={} MB bytes, compute={}.{}", 
                    p.first, 
                    p.second.name,                    
                    p.second.totalMemory/(1024*1024),
                    p.second.cuComputeMajor, p.second.cuComputeMinor
                );

            clog << msg << std::endl;
        }

        if ( m_CUSettings.devices.size() )
        {
            for (auto index : m_CUSettings.devices)
            {
                if (index < m_DevicesCollection.size())
                {
                    auto it = m_DevicesCollection.begin();
                    std::advance(it, index);
                    if (!it->second.cuDetected)
                        throw std::runtime_error("Can't CUDA subscribe a non-CUDA device.");
                    it->second.subscriptionType = DeviceSubscriptionTypeEnum::Cuda;                    
                }
            }
        } else {
            for (auto it = m_DevicesCollection.begin(); it != m_DevicesCollection.end(); it++)
            {
                if (!it->second.cuDetected ||
                    it->second.subscriptionType != DeviceSubscriptionTypeEnum::None)
                    continue;
                it->second.subscriptionType = DeviceSubscriptionTypeEnum::Cuda;
            }
        }
        

        m_farm = new Farm(m_DevicesCollection, m_CUSettings);
        
    }

    bool start()
    {
        return m_farm->start();
    }

    void stop()
    {
        m_farm->stop();
    }

    void set_new_work(uint8_t *epoch, uint8_t *address, uint64_t target, void *program, size_t program_size)
    {
        set_new_work(epoch, address, target, 0, program, program_size);
    }
    //must be 32 bytes length
    void set_new_work(uint8_t *epoch, uint8_t *address, uint64_t target, uint64_t counter, void *program, size_t program_size)
    {
        if ( counter == 0 )
        {
            std::random_device rd;    
            std::mt19937_64 gen(rd());            
            std::uniform_int_distribution<uint64_t> dis(0, std::numeric_limits<uint64_t>::max());            
            counter = dis(gen);
        }
        m_farm->setWork(epoch, address, target, counter, program, program_size);
    }
};

extern "C" {

void *aleo_miner_create(simple_log_callback_t cb_log, submit_callback_t cb_submit, uint8_t mask)
{

    MinerContext *ctx = new MinerContext();

    g_log_callback = cb_log;
    g_submit_callback = cb_submit;
    ctx->m_miner_mask = mask;
    ctx->prepare();
    
    return ctx;
}

bool aleo_miner_start(void *ctx)
{
    MinerContext *miner_ctx = (MinerContext *)ctx;
    return miner_ctx->start();
}

void aleo_miner_set_new_work(void *ctx, uint8_t *epoch, uint8_t *address, uint64_t target, uint64_t start_counter, void *program, size_t program_size)
{
    MinerContext *miner_ctx = (MinerContext *)ctx;
    miner_ctx->set_new_work(epoch, address, target, start_counter, program, program_size);
}

void aleo_miner_query_hash(void *ctx, uint32_t *sol , uint32_t *sol_len)
{
    MinerContext *miner_ctx = (MinerContext *)ctx;

    miner_ctx->m_farm->collectData();

    TelemetryType telemetry = miner_ctx->m_farm->Telemetry();

    for(int n = 0; n < telemetry.miners.size() && n < *sol_len; n++ )    
    {
        sol[n] = telemetry.miners.at(n).hashrate;
    }
    *sol_len = telemetry.miners.size();
}

}