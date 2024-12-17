#pragma once
#include "cuda/cu_synthesis.cuh"
#include "cuda/cu_bigint_define.cuh"
#include "cuda/cu_smem.cuh"
#include "circuit_hash_help.cuh"
#include "cuda/circuit/field/circuit_field_base.cuh"
#include "poseidon_data.cuh"

//参考 ： snarkVM-miner/circuit/algorithms/src/poseidon/hash.rs 里的 fn hash_many() 函数 


#define PSD_PREIMAG_SIZE (4)
#define PSD_RATE (2) //对应 hash.psd2
#define PSD_CAPACITY (1)
#define PSD2_partial_rounds (31)
#define PSD2_full_rounds (8)
#define PSD2_alpha (17)
#define PSD_STATE_SIZE (3)

class HashPoseidon
{
public:


static ALEO_ADI bigint_u256_t* get_mdf(uint8_t x, uint8_t y)
{
    if ( x <= 2 && y<= 2) {
        return &PSD2_MDF[ x*3+y ];
    } else {
        return nullptr;
    }
}

static ALEO_ADI bigint_u256_t* get_ark(uint8_t x, uint8_t y)
{
    if ( x <= 38 && y<= 2 ) {
        return &PSD2_ARK[ x*3 + y ];
    } else {
        return nullptr;
    }
}

static ALEO_ADI void psd2_apply_ark(uint64_t state[PSD_STATE_SIZE][QBigInt::BIGINT_LEN], int round)
{    
    for ( int n = 0; n < PSD_STATE_SIZE; n++ ) {
        // auto tmp = get_ark(round,n);
        // CirCuit_Fields::field_add_assign(state[n], *tmp);
        CirCuit_Fields_Base::field_add_assign(state[n], PSD2_ARK[ round*3 + n ].uint64);
    }
}

static ALEO_ADI void pow_for_hash_psd2(SVM_PARAM_CDEF,
                uint64_t operand[QBigInt::BIGINT_LEN]
                )
{
    // exponent must be const
    //bool self_is_const = is_const & 1 ;

    // bigint_u256_t exponent_raw = {
    //     .uint64 = {0x0000000000000011ULL, 0, 0, 0}
    // } ;
    //bigint_u256_t &output = dest; // initial value is CirCuit_Fields::get_field_one();

    //exponent_raw: from big endian to little ==> 0001 0001 

    // 4th bit is true    
    //output = operand;
    bool output_const_flag = true;
    if ( (is_const & 1) == false )
        output_const_flag = false;

    #if 0
    // 3th bit
    CirCuit_Fields::operator_mul(SVM_PARAM, output_const_flag, output, output, output);
    // 2th bit
    CirCuit_Fields::operator_mul(SVM_PARAM, output_const_flag, output, output, output);
    // 1th bit
    CirCuit_Fields::operator_mul(SVM_PARAM, output_const_flag, output, output, output);
    // 0 bit is true
    CirCuit_Fields::operator_mul(SVM_PARAM, output_const_flag, output, output, output);
    #endif

    uint64_t output[QBigInt::BIGINT_LEN];    

    QBigInt::copy(output, operand);
    for ( int n = 0; n < 4 ; n++){
        CirCuit_Fields::operator_mul(SVM_PARAM, output_const_flag, output, output, output);        
    }
    CirCuit_Fields::operator_mul(SVM_PARAM, output_const_flag, output, operand, operand);

#if 0
    uint64_t output_u64[QBigInt::BIGINT_LEN];
    uint64_t dest_u64[QBigInt::BIGINT_LEN];    

    QBigInt::copy(output_u64, operand);
    
    for ( int n = 0; n < 4 ; n++){
        QBigInt::bigint_mont_mul_assign_no_unroll(output_u64, output_u64, dest_u64);
        QBigInt::copy(output_u64, dest_u64);
        if ( !output_const_flag ) {
            svm_push_leaf_full_with_convert(SVM_PARAM, dest_u64);
        }
    }

    QBigInt::bigint_mont_mul_assign_no_unroll(output_u64, operand, dest_u64);      
    QBigInt::copy(operand, dest_u64);    
    if ( !output_const_flag ) {
        svm_push_leaf_full_with_convert(SVM_PARAM, dest_u64);
    }    
#endif   
    
}

static ALEO_ADI void psd2_apply_s_box_with_full_is_true(SVM_PARAM_DEF,             
            bool is_const, 
            uint64_t state[QBigInt::BIGINT_LEN])
{    
    pow_for_hash_psd2(SVM_PARAM, is_const, state);
}  

static ALEO_ADI void psd2_apply_s_box_with_full_is_false(SVM_PARAM_DEF,    
            bool state0_is_const, 
            uint64_t state[QBigInt::BIGINT_LEN])
{     
    pow_for_hash_psd2(SVM_PARAM, state0_is_const, state );
}  



static ALEO_ADI void psd2_apply_mds(bool state_is_const[PSD_STATE_SIZE], uint64_t state[PSD_STATE_SIZE][QBigInt::BIGINT_LEN])
{
    uint64_t new_state[PSD_STATE_SIZE][QBigInt::BIGINT_LEN];
    for ( int n = 0; n < PSD_STATE_SIZE; n++ )
    {
        uint64_t accumulator[QBigInt::BIGINT_LEN] = { 0 };

        bool has_const = true;        
        for ( int m = 0; m < PSD_STATE_SIZE; m++ )
        {
            uint64_t tmp[QBigInt::BIGINT_LEN];
            QBigInt::bigint_mont_mul_assign_no_unroll(state[m], (*get_mdf(n,m)).uint64, tmp);
            CirCuit_Fields_Base::field_add_assign(accumulator, tmp);
            has_const = has_const & state_is_const[m];
        }        
        QBigInt::copy(new_state[n], accumulator);
        state_is_const[n] = has_const;
    }
    
    for ( int n = 0; n < PSD_STATE_SIZE; n++ )
    {
        QBigInt::copy(state[n], new_state[n]);
    }    
}

    static ALEO_ADI void hash_psd2_unpack_meta(SVM_PARAM_DEF, uint32_t &permute_round,  uint8_t &state_offset, bool state_is_const[PSD_STATE_SIZE])
    {
        uint32_t meta = smem_get_circuit_meta();

        uint8_t const_bits = meta;
        state_is_const[0] = meta & 1;
        state_is_const[1] = (meta>>1) & 1;
        state_is_const[2] = (meta >>2) & 1;

        state_offset = (meta >> 8) & 0xff;
        permute_round = meta >> 16;
    }

    static ALEO_ADI void hash_psd2_pack_meta(SVM_PARAM_DEF, uint32_t &permute_round, uint8_t state_offset, bool state_is_const[PSD_STATE_SIZE])
    {
        uint32_t meta = 0;     

        uint8_t const_bits = 0;    
        if ( state_is_const[0] )
            const_bits = const_bits | 1;
        if ( state_is_const[1] )
            const_bits = const_bits | 0b10;
        if ( state_is_const[2] )
            const_bits = const_bits | 0b100;
        
        meta = permute_round; 
        meta = (meta << 8 ) | state_offset;
        meta = (meta << 8 ) | const_bits;    

        smem_set_circuit_meta(meta);
    }

    //static __device__ void prepare_context(SVM_PARAM_DEF, bigint_u256_t src);
static ALEO_ADI void prepare_context(SVM_PARAM_DEF, uint64_t src[QBigInt::BIGINT_LEN])
{
    uint64_t state[PSD_STATE_SIZE][QBigInt::BIGINT_LEN]; //save to share memory
    bool state_is_const[3] = {true, true, true};
    uint32_t permute_round = 0;

    state[0][0] = 0xe2134873ea683b53ULL;
    state[0][1] = 0x6d97170102a1aad5ULL;
    state[0][2] = 0xeb94e683ef7aa75aULL;
    state[0][3] = 0x00fdd395e0e34f38ULL;

    state[1][0] = 0xa893dc034ae58ac9ULL;
    state[1][1] = 0x64d26f3f04b51024ULL;
    state[1][2] = 0x74fab6eddbf64f83ULL;
    state[1][3] = 0x0df820eb14802e05ULL;

    state[2][0] = 0x8034174940775aadULL;
    state[2][1] = 0xbb36b501c56d3a85ULL;
    state[2][2] = 0x5ec40964cfcef5a0ULL;
    state[2][3] = 0x104843389eb35830ULL;

    //absorb 

    uint64_t input1_reg[QBigInt::BIGINT_LEN] = { 0 };
    uint64_t input2_reg[QBigInt::BIGINT_LEN] = { 0 };
    circuit_plaintext_to_field(SVM_PARAM, VARIANT_Field, src, input1_reg, input2_reg);
    CirCuit_Fields_Base::field_add_assign(state[1], input1_reg);    
    state_is_const[1] = false;
    CirCuit_Fields_Base::field_add_assign(state[2], input2_reg);
    state_is_const[2] = false;


    hash_psd2_pack_meta(SVM_PARAM, permute_round, 0, state_is_const);    

    //hash_psd2_save_state(SVM_PARAM, state);
    smem_save_circuit_params(0, state[0]);
    smem_save_circuit_params(1, state[1]);
    smem_save_circuit_params(2, state[2]);
}    

    //static __device__ bool exec_hash(SVM_PARAM_DEF);


static ALEO_ADI bool exec_hash(SVM_PARAM_DEF)
{
    // do permute , 总共39次循环 ，每次循环最多 15个 hash 
    bool state_is_const[PSD_STATE_SIZE];
    uint32_t permute_round;
    uint64_t state[PSD_STATE_SIZE][QBigInt::BIGINT_LEN];
    uint8_t state_offset = 0;
    
    smem_load_circuit_params(0, state[0]);
    smem_load_circuit_params(1, state[1]);
    smem_load_circuit_params(2, state[2]);    

    hash_psd2_unpack_meta(SVM_PARAM, permute_round, state_offset, state_is_const);

    constexpr int full_rounds_over_2 = PSD2_full_rounds / 2;
    constexpr int partial_round_range_start = full_rounds_over_2;
    constexpr int partial_round_range_end = full_rounds_over_2 + PSD2_partial_rounds;

    bool is_full_round;
    is_full_round = !( permute_round >= partial_round_range_start && permute_round < partial_round_range_end);    

    if ( is_full_round ) {
        if ( state_offset == 0 ) {            
            psd2_apply_ark(state, permute_round);            
        }
        if ( state_offset < PSD_STATE_SIZE ) {
            psd2_apply_s_box_with_full_is_true(SVM_PARAM, state_is_const[state_offset], state[state_offset]);
        }
        state_offset++;

        if ( state_offset == PSD_STATE_SIZE ) {
            //done 
            psd2_apply_mds(state_is_const, state);
            state_offset = 0;
            permute_round++;
        }
 
    } else {
        psd2_apply_ark(state, permute_round);
        psd2_apply_s_box_with_full_is_false(SVM_PARAM, state_is_const[0], state[0]);
        psd2_apply_mds(state_is_const, state);
        permute_round++;
    }          

    hash_psd2_pack_meta(SVM_PARAM, permute_round, state_offset, state_is_const);    
    smem_save_circuit_params(0, state[0]);
    smem_save_circuit_params(1, state[1]);
    smem_save_circuit_params(2, state[2]);

    if (permute_round == (PSD2_full_rounds+PSD2_partial_rounds) )
    {
        return true;
    } else {
        return false;
    }
}    
   
};