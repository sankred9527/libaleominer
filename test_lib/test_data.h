#pragma once


#define TEST_EPOCH_HASH { 52, 195, 52, 121, 238, 211, 36, 92, 201, 3, 88, 90, 35, 85, 78, 143, 237, 11, 207, 94, 196, 106, 101, 243, 119, 205, 17, 124, 133, 7, 20, 11}
#define TEST_EPOCH_HASH_HEX { 0x5c24d3ee7934c333, 0x8f4e55235a5803c9, 0xf3656ac45ecf0bed,0xb1407857c11cd77 }
#define TEST_ADDRESS {76, 64, 99, 46, 103, 207, 243, 204, 83, 127, 160, 168, 217, 1, 175, 68, 100, 181, 49, 225, 155, 172, 149, 85, 110, 217, 46, 178, 245, 249, 87, 10}
#define TEST_COUNTER 0x1234567887654321ULL

static void init_nonce()
{
    uint8_t epoch_hash[32] = TEST_EPOCH_HASH;    
    
    uint8_t address[32] = TEST_ADDRESS;
    
    uint64_t target = 100;    

    // void *epoch_program = nullptr;
    // size_t program_size = 1024*10;
    // epoch_program = malloc(program_size);
                
    //TODO: set program
    //cu_aleo_init_epoch_internal(epoch_hash, address, target, nullptr, 0);
}