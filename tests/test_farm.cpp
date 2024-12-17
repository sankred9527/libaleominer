
#include <map>
#include "MinerContext.h"
#include "Farmer.h"
#include "CUDAMiner.h"
#include "Log.h"
#include <stdio.h>
using namespace std;
using namespace dev;
#include "test_data.h"
#include "program_bin.h"

void thread_task(void *miner_ctx) {
    
    uint32_t sol_len = 8;
    uint32_t *sol = new uint32_t[sol_len];

    uint32_t *sol_total = new uint32_t[sol_len];
    for(int n = 0; n < sol_len; n++)
    {
        sol_total[n] = 0;
    }
    while ( true )
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(20*1000));
     
        sol_len = 8;
        aleo_miner_query_hash(miner_ctx, sol, &sol_len);
        for(int n = 0; n < sol_len; n++)
        {
            sol_total[n] += sol[n];
            printf("time=%ld hash[%d]=%u avg/s=%d\n", time(NULL), n, sol_total[n], sol[n]/20);
        }
    }
    
}


void mylog(const char* msg)
{
    printf("%s", msg);
}

void mysubmit(uint64_t solution_id, uint64_t counter)
{
    printf("found new solution=0x%lx counter=0x%lx\n", solution_id, counter);
}

int main()
{
    //uint64_t start_counter = 0x1234567887654321ULL;
    void *miner_ctx = aleo_miner_create(mylog, nullptr, 0xff);

    if ( aleo_miner_start(miner_ctx) )
    {
        std::cout<<"start mine ok\n";
    }

    std::thread hash_thread  = std::thread(thread_task, miner_ctx);
    hash_thread.detach();

    //模拟等待2秒从服务器拿到 epoch 和 address        
    sleep(2);

    uint8_t epoch[] = TEST_EPOCH_HASH ;
    uint8_t address[] = TEST_ADDRESS ;
    uint64_t target = 10000*1000;
    uint64_t counter = 0x1234567887654321ULL;
    aleo_miner_set_new_work(miner_ctx, epoch, address, target, counter, program_bin, sizeof(program_bin));

    //模拟10秒后切换 epoch 
    sleep(10);
    //epoch[0] += 1;
    //aleo_miner_set_new_work(miner_ctx, epoch, address, target, counter, program_bin, sizeof(program_bin));

    sleep(1000);
    return 0;
}
