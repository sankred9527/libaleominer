
#include <functional>
#include <sstream>  
#include <map>
#include <iomanip>
#include <cuda_runtime.h>
#include <unistd.h>
#include "fmt/core.h"
#include "Miner.h"
#include "CUDAMiner.h"
#include "Log.h"
#include "CommonData.h"
#include "AleoProve.h"
#include "AleoCallBack.h"
#include "cu_macro.h"
#include "cuda/cu_aleo_interface.h"
#include "const_leaf_u16.h"
#include "const_inverse_u16.h"

using namespace std;
using namespace dev;


struct CUDAChannel : public LogChannel
{
    static const char* name() { return "cu"; }
    static const int verbosity = 2;
};
#define cudalog clog(CUDAChannel)


int CUDAMiner::getNumDevices()
{
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err == cudaSuccess)
        return deviceCount;

    if (err == cudaErrorInsufficientDriver)
    {
        int driverVersion = 0;
        cudaDriverGetVersion(&driverVersion);
        if (driverVersion == 0)
            std::cerr << "CUDA Error : No CUDA driver found" << std::endl;
        else
            std::cerr << "CUDA Error : Insufficient CUDA driver " << std::to_string(driverVersion)
                      << std::endl;
    }
    else
    {
        std::cerr << "CUDA Error : " << cudaGetErrorString(err) << std::endl;
    }

    return 0;
}


void CUDAMiner::enumDevices(std::map<string, DeviceDescriptor>& _DevicesCollection, uint8_t miner_mask)
{

    int numDevices = getNumDevices();

    if ( numDevices > 8 ) numDevices = 8;

    for (uint8_t i = 0; i < numDevices; i++)
    {
        string uniqueId;
        ostringstream s;
        DeviceDescriptor deviceDescriptor;
        cudaDeviceProp props;

        if ( 0 == ( (miner_mask>>i) & 1 ) )
            continue;

        try
        {
            size_t freeMem, totalMem;
            CUDA_SAFE_CALL(cudaGetDeviceProperties(&props, i));
            CUDA_SAFE_CALL(cudaMemGetInfo(&freeMem, &totalMem));
            s << std::setw(2) << setfill('0') << hex << props.pciBusID << ":" << setw(2)
              << props.pciDeviceID << ".0";
            uniqueId = s.str();

            if (_DevicesCollection.find(uniqueId) != _DevicesCollection.end())
                deviceDescriptor = _DevicesCollection[uniqueId];
            else
                deviceDescriptor = DeviceDescriptor();

            deviceDescriptor.name = string(props.name);            
            deviceDescriptor.cuDetected = true;
            deviceDescriptor.uniqueId = uniqueId;
            deviceDescriptor.type = DeviceTypeEnum::Gpu;
            deviceDescriptor.cuDeviceIndex = i;
            deviceDescriptor.cuDeviceOrdinal = i;
            deviceDescriptor.cuName = string(props.name);
            deviceDescriptor.totalMemory = freeMem;
            deviceDescriptor.cuCompute =
                (to_string(props.major) + "." + to_string(props.minor));
            deviceDescriptor.cuComputeMajor = props.major;
            deviceDescriptor.cuComputeMinor = props.minor;
            deviceDescriptor.cuMultiProcessorCount = props.multiProcessorCount;

            _DevicesCollection[uniqueId] = deviceDescriptor;

            // printf("sharedMemPerBlock=%lu sharedMemPerMultiprocessor=%lu\n", 
            //     props.sharedMemPerBlock , props.sharedMemPerMultiprocessor );
            // std::cout<< "name=" << deviceDescriptor.name << "cuname=" << deviceDescriptor.cuName  \
            //     << ",memory=" << deviceDescriptor.totalMemory  << std::endl;
            // std::cout<< "compute major=" << props.major << "," << props.minor << std::endl;
        }
        catch (const cuda_runtime_error& _e)
        {
            std::cerr << _e.what() << std::endl;
        }
    }
}


CUDAMiner::CUDAMiner(unsigned _index, CUSettings _settings, DeviceDescriptor& _device)
  : Miner("cuda-", _index),
    m_settings(_settings)    
{
    m_deviceDescriptor = _device;
}

CUDAMiner::~CUDAMiner()
{
    DEV_BUILD_LOG_PROGRAMFLOW(cudalog, "cuda-" << m_index << " CUDAMiner::~CUDAMiner() begin");
    stopWorking();
    kick_miner();
    DEV_BUILD_LOG_PROGRAMFLOW(cudalog, "cuda-" << m_index << " CUDAMiner::~CUDAMiner() end");
}

static int getSPcores(DeviceDescriptor &devProp)
{  
    int cores = 0;
    auto mp = devProp.cuMultiProcessorCount;
    switch (devProp.cuComputeMajor){    
        case 7: // Volta and Turing
            if ((devProp.cuComputeMinor == 0) || (devProp.cuComputeMinor == 5)) 
                cores = mp * 64;        
            break;
        case 8: // Ampere
            if (devProp.cuComputeMinor == 0) 
                cores = mp * 64;
            else if (devProp.cuComputeMinor == 6) 
                cores = mp * 128;
            else if (devProp.cuComputeMinor == 9) 
                cores = mp * 128; // ada lovelace        
            break;
        case 9: // Hopper
            if (devProp.cuComputeMinor == 0) 
                cores = mp * 128;
            break;
        default:
            break;
    }
    return cores;
}

bool CUDAMiner::initDevice()
{
    cudalog << "Using Pci Id : " << m_deviceDescriptor.uniqueId << " " << m_deviceDescriptor.cuName
            << " (Compute " + m_deviceDescriptor.cuCompute + ") Memory : "
            << dev::getFormattedMemory((double)m_deviceDescriptor.totalMemory);

    try
    {
        CUDA_SAFE_CALL(cudaSetDevice(m_deviceDescriptor.cuDeviceIndex));
        CUDA_SAFE_CALL(cudaDeviceReset());        

        /*
        1253376 : 1M + 200k
        1458176 : 1M + 400K
        1048576 : 1M
        */
        m_settings.chunk_size = 153600; //150k

        m_settings.blockSize = 256;
                
        // 就是 线程数量， 暂定为 gpu 核心数量 *4 
        {
            uint32_t core_cnt = getSPcores(m_deviceDescriptor) * 4;
            if ( core_cnt == 0 )
            {
                cudalog<<"not support cuda device";
                return false;
            }
            m_settings.chunk_count = core_cnt;
            CUDA_SAFE_CALL(cudaMemcpyToSymbol((void*)&d_max_thread, (void*)&core_cnt, sizeof(d_max_thread)));
        }

        if ( m_settings.chunk_count % m_settings.blockSize  != 0 ) 
        {
            m_settings.chunk_count -= m_settings.chunk_count % m_settings.blockSize ;
        }

        m_settings.mem_size = m_settings.chunk_size*m_settings.chunk_count;

        //多的这2M 字节， 给 u16 leaf hash 查表
        m_settings.mem_size += 2097152ULL;

        // 这4M 字节，给 u16 inverse 查表
        m_settings.mem_size += 4194304ULL;
        
        m_settings.gridSize = m_settings.chunk_count / m_settings.blockSize ;
    
        m_batch_size = m_settings.gridSize * m_settings.blockSize;
        m_streams_batch_size = m_batch_size * m_settings.streams;

        CUDA_SAFE_CALL(cudaMalloc((void **)&m_settings.device_mem_addr, m_settings.mem_size));

        CUDA_SAFE_CALL(cudaMemcpy(m_settings.device_mem_addr, const_u16_leaf_hash, sizeof(const_u16_leaf_hash), cudaMemcpyHostToDevice));

        CUDA_SAFE_CALL(cudaMemcpy( 
                static_cast<uint8_t*>(m_settings.device_mem_addr) + 2*1024*1024,
                const_u16_inverse_hash, sizeof(const_u16_inverse_hash), cudaMemcpyHostToDevice));

        m_search_buf.resize(m_settings.streams);
        m_streams.resize(m_settings.streams);
            
        for (unsigned i = 0; i != m_settings.streams; ++i)
        {
            CUDA_SAFE_CALL(cudaMallocHost(&m_search_buf[i], sizeof(Search_results)));
            CUDA_SAFE_CALL(cudaStreamCreateWithFlags(&m_streams[i], cudaStreamNonBlocking));
        }
        // cudalog<<"cuda grid size = " << m_settings.gridSize << ",  block size = " << m_settings.blockSize;
        // cudalog<<"cuda mem chunk count = " << m_settings.chunk_count << ", chunk size=" << m_settings.chunk_size;
        // cudalog<<"cuda dev mem size = " << m_settings.mem_size;
                
    }
    catch (const cuda_runtime_error& ec)
    {
        cudalog << "Could not set CUDA device on Pci Id " << m_deviceDescriptor.uniqueId
                << " Error : " << ec.what();
        cudalog << "Mining aborted on this device.";
        return false;
    }
    return true;    
}

void CUDAMiner::initEpoch_internal(uint8_t *epoch_hash, uint8_t *address, uint64_t target, void *epoch_program, size_t program_size)
{   
    //CUDA_SAFE_CALL(cudaMemset(m_settings.device_mem_addr, 0 , m_settings.mem_size));        

    CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_epoch_hash, epoch_hash, sizeof(d_epoch_hash)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_address, address, sizeof(d_address)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol((void*)&d_target, (void*)&target, sizeof(target)));

    if ( program_size <= EPOCH_PROGRAM_SIZE ) {        
        CUDA_SAFE_CALL(cudaMemcpyToSymbol((void*)&d_program_size, (void*)&program_size, sizeof(program_size)));
        CUDA_SAFE_CALL(cudaMemcpyToSymbol((void*)&d_program_bin, (void*)epoch_program, program_size));
    }
    else {
        printf("init program error\n") ;
    }

    // if ( program_size <= EPOCH_PROGRAM_SIZE ) {
    //     CUDA_SAFE_CALL(cudaMemcpy(m_epoch_program_ptr, epoch_program, program_size, cudaMemcpyHostToDevice));
    //     CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_program_size, &program_size, sizeof(program_size)));

    //     m_epoch_program_size = program_size;
    // }
    // else {
    //     cwarn << "init program error\n" ;
    // }
}

void CUDAMiner::kick_miner() 
{
    m_new_work.store(true, std::memory_order_relaxed);
    m_new_work_signal.notify_one();    
}

void CUDAMiner::workLoop()
{
    WorkPackage current;

    if ( !initDevice() )
        return;

    try{

        while (!shouldStop())
        {
            WorkPackage w = work();
            if ( w.is_empty() )
            {
                auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(3);
                std::unique_lock l(x_work);
                m_new_work_signal.wait_until(l, timeout);
                cudalog<< "waiting new work ...";
                continue;
            }
            m_new_work.store(false, std::memory_order_relaxed);

            if ( !current.is_equal(w) )
            {
                cudalog<< fmt::format("find new work,epoch={:x},{:x},{:x},{:x}", w.epoch_hash[0], w.epoch_hash[1], w.epoch_hash[2], w.epoch_hash[3]);
            }

            current.init_from(w.epoch_hash, w.address, w.target, w.epoch_program, w.epoch_program_size);
            current.counter = w.counter;
            // cudalog<< "working, epoch="<< w.epoch_hash[0];
            search(current);
        }
        // Reset miner and stop working
        CUDA_SAFE_CALL(cudaDeviceReset());

    }
    catch (cuda_runtime_error const& _e)
    {
        string _what = "GPU error: ";
        _what.append(_e.what());
        throw std::runtime_error(_what);
    }    
}

void CUDAMiner::search(const WorkPackage& w)
{
    uint64_t start_counter = w.counter;
    uint32_t current_index;    

    //TODO: set program
    this->initEpoch_internal((uint8_t*)w.epoch_hash, (uint8_t*)w.address, w.target, (void*)w.epoch_program, w.epoch_program_size);

    for (current_index = 0; current_index < m_settings.streams; 
        current_index++,  start_counter += m_batch_size)
    {
        cudaStream_t stream = m_streams[current_index];
        volatile Search_results& buffer(*m_search_buf[current_index]);
        buffer.count = 0;

        //cudalog<< fmt::format("run search[{}] counter={:x} , batch_size={} ", current_index, start_counter, m_batch_size);
        // Run the batch for this stream
        cu_aleo_search(m_settings.gridSize, 
                m_settings.blockSize, stream, 
                (uint8_t*)m_settings.device_mem_addr,
                start_counter,
                &buffer);
    }

    bool done = false;
    while (!done)
    {
        // Exit next time around if there's new work awaiting
        bool t = true;
        done = m_new_work.compare_exchange_strong(t, false);
        
        for (current_index = 0; current_index < m_settings.streams; current_index++, start_counter += m_batch_size)
        {
            //cudalog <<"I'm, searching";

            // Each pass of this loop will wait for a stream to exit,
            // save any found solutions, then restart the stream
            // on the next group of nonces.
            cudaStream_t stream = m_streams[current_index];
            // Wait for the stream complete
            CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
            
            if (shouldStop())
            {
                cudalog<< "should stop!!!";
                m_new_work.store(false, std::memory_order_relaxed);
                done = true;
            }

            // Detect solutions in current stream's solution buffer
            volatile Search_results& buffer(*m_search_buf[current_index]);
            uint32_t found_count = std::min((unsigned)buffer.count, MAX_SEARCH_RESULTS);
            Search_Result to_send_result[MAX_SEARCH_RESULTS];
            if (found_count)
            {
                cudalog << fmt::format("find {} solution", found_count);
                buffer.count = 0;
                for (uint32_t i = 0; i < found_count; i++)
                {                    
                    memcpy(&to_send_result[i], (void *)&buffer.result[i], sizeof(Search_Result));
                }
            }

            //cudalog<< fmt::format("rerun search[{}] counter={:x} , batch_size={}", current_index, start_counter, m_batch_size);
            // Run the batch for this stream
            cu_aleo_search(m_settings.gridSize, 
                    m_settings.blockSize, stream,
                    (uint8_t*)m_settings.device_mem_addr,
                    start_counter,
                    &buffer);

            if ( found_count )
            {
                for(int i = 0; i < found_count; i++)
                {
                    if ( g_submit_callback != nullptr )
                        g_submit_callback(to_send_result[i].solution_id, to_send_result[i].counter);
                    else {
                        //submit to uplevel
                        cudalog<< fmt::format("sid={:x}, target={:x}, counter={:x}", 
                            to_send_result[i].solution_id,
                            to_send_result[i].solution_target,
                            to_send_result[i].counter
                        );

                    }
                    
                }
            }
        }

        updateHashRate(m_batch_size, m_settings.streams);
        
        if (done) {
            //cudalog<< "search quit for new work";
            //TODO: optimize it
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
            //cudalog<< "search quit for new work done";
        }

    } //end: while (!done)
}

extern "C" int cuminer_getNumDevices() {

    return CUDAMiner::getNumDevices();
}

extern "C" void cuminer_enumDevices() {

    //CUDAMiner::enumDevices();    
}



