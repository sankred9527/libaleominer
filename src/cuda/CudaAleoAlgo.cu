
#include "CudaAleoAlgo.cuh"
#include "cu_aleo_globals.cuh"
#include "cu_synthesis.cuh"
#include "cu_smem.cuh"
#include "cu_program.cuh"
#include "circuit/circuit.cuh"
#include "cu_utils.cuh"
#include "cu_merkle_tree.cuh"
#include "cu_keccak_v2.cuh"
#include "cu_mtree_public_hash.cuh"
#include "ChaChaCommon.h"


//#define MY_DEBUG(...) do { if ( idx == 0 ) printf(__VA_ARGS__);} while(0)


#define MERKLE_TREE_DEPTH  (9)
#define MERKLE_TREE_ARITY  (8)

//#define ALEO_DEBUG 1

#ifdef ALEO_DEBUG
#define cudbg(...) do { printf(__VA_ARGS__); } while(0)
#define cuerr(...) do { printf(__VA_ARGS__); } while(0)
#else
#define cudbg(...) do {}while(0)
#define cuerr(...) do {}while(0)
#endif


static ALEO_ADI  uint32_t checked_next_power_of_root(uint32_t base)
{
    if ( base <= 32768)
        return 32768;
    else 
        return 262144;
    
}

static ALEO_ADI  int8_t calculate_tree_depth(uint32_t tree_size)
{    
    double tree_depth_float = log((double)tree_size)/log((double)MERKLE_TREE_ARITY);
    uint8_t tree_depth = ((uint64_t)floor(tree_depth_float)) & 0xff;
    if ( tree_depth <= MERKLE_TREE_DEPTH) 
    {
        return tree_depth;
    } else {
        return -1;
    }
}


static ALEO_ADI bool merkle_tree_parent(uint32_t index, uint32_t &result)
{

    if ( index > 0 ) {
        result = (index-1)/MERKLE_TREE_ARITY;
        return true;
    } else {
        return false;
    }
}

__global__  void cu_aleo_kernel_search(void *ctx, uint64_t start_counter, volatile Search_results *results)
{
    
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t r0_bits_cache = 0;
    uint32_t r0_bits_cache_len = 0;

    Sha3StateV2 gl_r0_st[25];
    uint8_t gl_r0_last_bit = 1;
    uint32_t gl_r0_round = 0;

    Sha3StateV2 gl_r1_st[25];
    uint8_t gl_r1_last_bit = 1;
    uint32_t gl_r1_round = 0;

    
    
    memset(gl_r0_st, 0, sizeof(gl_r0_st));
    memset(gl_r1_st, 0, sizeof(gl_r1_st));

#if 1
    //每个block 的 0 线程， 初始化 share memory
    if ( idx % blockDim.x == 0 ) {
        memcpy(mtree_round_0_cache, CONST_first_round_path_hash_U32X8, sizeof(uint32_t)*256*8);
    }    
    __syncthreads();
#endif

    svm_set_program_index(SVM_PARAM, 0);    
    svm_set_reg_index(SVM_PARAM, 0);
    svm_set_reg_stack_index(SVM_PARAM, 0);
    svm_set_hash_psd2_count(SVM_PARAM, 0);
    svm_set_soa_leaves_length(SVM_PARAM, 0);
            
    smem_set_soa_meta(0);    
    smem_set_circuit_meta2(0);
    smem_set_circuit_meta(0);
    smem_set_circuit_is_u8_leaf_hash(0);
        
    
    cudbg("gl_static_shared_memory= %lu mtree_round_0_cache=%lu warp_meta_t=%lu\n", 
            sizeof(gl_static_shared_memory),  sizeof(mtree_round_0_cache),
            sizeof(warp_meta_t)
            );       

    uint64_t solution_id = 0;

    //这一步 生成  14 个 input 寄存器和 501 个 public 变量
    solution_id = svm_generate_constant_from_rand(SVM_PARAM, start_counter+idx);    
        
    cudbg("solution_id == %lx counter=%lx\n", solution_id, start_counter+idx);
        

#if 1    
    {
        VmInstruction *instruction = nullptr;
        while ( (instruction = program_fetch_one_instruction_from_const(SVM_PARAM)) != nullptr )
        {
            #ifdef ALEO_DEBUG
            {
                char buf[64] = {0};
                program_instruction_to_str(instruction, buf, 64);
                cudbg("fetch instruction=%s\n",buf);
            }
            #endif

            CirCuit_Exec_Step exec_step = CEXEC_STEP_INIT;
            bool is_instruction_finish = true;
            do {
                /* 
                大部分指令都不存 full leaves
                所以每次执行之前，设置 set_max_allow_full_leaves_in_smem 为0  
                默认所有的 bits array 都存在 share memory

                需要定制该参数 的 指令， 自己设置
                */
                RoundZeroMem::set_max_allow_full_leaves_in_smem(0);

                is_instruction_finish = circuit_execute_instruction(SVM_PARAM, instruction, exec_step);                
                mtree_do_sha3(SVM_PARAM, MTREE_PARAM);
                smem_set_circuit_is_u8_leaf_hash(0);
                smem_set_soa_leaves_len(0, 0);
                cudbg("total reg len=%u r0_bits_cache=%08x, meta=%08x\n", svm_get_reg_index(SVM_PARAM) ,  r0_bits_cache, r0_bits_cache_len);
            } while (!is_instruction_finish);
            //cudbg("total reg len=%u r0_bits_cache=%08x, meta=%08x\n", svm_get_reg_index(SVM_PARAM) ,  r0_bits_cache, r0_bits_cache_len);
        }
        //mtree_do_sha3(SVM_PARAM, MTREE_PARAM);         
        cudbg("program end r0_bits_cache=%08x, len= %u\n", r0_bits_cache, r0_bits_cache_len);
    }
    
#endif

    merkle_tree_run(SVM_PARAM, MTREE_PARAM);
    merkle_tree_run_round_23(SVM_PARAM, MTREE_PARAM);
    bool is_pow_8_5 = merkle_tree_run_round_45(SVM_PARAM, MTREE_PARAM);    

#if 1
    {
        uint32_t full_leaves_len; //= checked_next_power_of_n(leaf_len, 8);  // 是 8的 power
        if ( is_pow_8_5 )
            full_leaves_len = 32768; // pow(8,5)
        else 
            full_leaves_len = 262144; // pow(8,6)

        unsigned int  num_nodes = (checked_next_power_of_root(full_leaves_len)-1)/(MERKLE_TREE_ARITY-1);
        uint32_t minimum_tree_size = full_leaves_len + num_nodes;
        uint32_t tree_depth = calculate_tree_depth(minimum_tree_size);
        uint32_t padding_depth = MERKLE_TREE_DEPTH - tree_depth;
        //printf("padding depth=%u\n", padding_depth);
                
        uint64_t target = svm_merkle_tree_final_step(SVM_PARAM, MTREE_PARAM, padding_depth);
        if ( target > d_target )
        {
            uint32_t index = atomicInc((uint32_t*)&results->count, 0xffffffff);
            if (index >= MAX_SEARCH_RESULTS)
                return;
            results->result[index].gid = blockIdx.x;
            results->result[index].solution_id = solution_id;
            results->result[index].counter = start_counter + idx;
            results->result[index].solution_target = target;
            //printf("found index=%d\n", index);
        }
                
    }
#endif

}

