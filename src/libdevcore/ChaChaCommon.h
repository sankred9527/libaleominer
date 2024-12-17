#pragma once

#include <stdint.h>
#include "Common.h"

#define CHACHA_MAGIC 0x61707865, 0x3320646e, 0x79622d32, 0x6b206574

#define CHACHA_KEY_SIZE_IN_INT32  (8)
#define CHACHA_NOUNCE_SIZE_IN_INT32  (4)
#define CHACHA_OUT_BUF_SIZE (256)

typedef uint32_t uint32x4[4] ;



typedef struct  ALEO_ALIGN(8)  {
    uint32x4 key[2];    
    uint32x4 nounce[1];    
    uint8_t out[CHACHA_OUT_BUF_SIZE];
    
    uint32_t cache_index;
} chacha_state_t;