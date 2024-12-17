#pragma once

#include <string>
#include <cstring> 
#include <cstdint>
#include "aleo_macro.h"

#define MAX_SEARCH_RESULTS 32U

struct Search_Result
{
    // One word for gid and 8 for mix hash
    uint32_t gid;
    uint64_t solution_id;
    uint64_t solution_target;
    uint64_t counter;
    uint32_t pad[1];  // pad to size power of 2
};

typedef struct 
{
    Search_Result result[MAX_SEARCH_RESULTS];
    uint32_t count = 0;
} Search_results;

namespace dev
{

class WorkPackage
{
public:
    WorkPackage() = default;

    void init_from(void *_epoch, void* _address, uint64_t _target, void *program, size_t program_size) {
        std::memcpy(epoch_hash, _epoch, ALEO_FIELD_SIZE_IN_BYTES);
        std::memcpy(address, _address, ALEO_FIELD_SIZE_IN_BYTES);
        if( program != nullptr ) {
            std::memcpy(epoch_program, program, program_size);
            this->epoch_program_size = program_size;
        }
        target = _target;
    }

    uint64_t epoch_hash[4] = {0};
    uint64_t address[4] = {0};
    uint64_t target;
    uint8_t epoch_program[EPOCH_PROGRAM_SIZE];
    size_t epoch_program_size ;

    uint64_t counter;

    bool is_empty() {
        for (int n = 0; n < 4; n++)
        {
            if ( epoch_hash[n] != 0 )
                return false;
        }
        return true;
    }

    bool is_equal(WorkPackage &other)
    {
        return is_equal((void*)other.epoch_hash, (void*)other.address);
    }

    bool is_equal(void *_epoch, void* _address)
    {
        return (std::memcmp(_epoch, epoch_hash, ALEO_FIELD_SIZE_IN_BYTES) == 0) 
            && (std::memcmp(_address, address, ALEO_FIELD_SIZE_IN_BYTES) == 0);
    }
};

struct Solution {
    int foo;
};


}