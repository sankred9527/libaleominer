#pragma once
#include "cuda/cu_synthesis.cuh"
#include "cuda/circuit/circuit_type.cuh"
#include "cuda/bls12_377.cuh"
#include "cuda/circuit/field/circuit_field.cuh"
//#include "cuda/circuit/integer/circuit_integer.cuh"

// __device__ void circuit_plaintext_to_field(SVM_PARAM_DEF, LiteralVariantAsU8 variant, 
//         bigint_u256_t* src, bigint_u256_t *dest1, bigint_u256_t *dest2 );


static ALEO_ADI int append_to_byte(uint8_t &data, int *offset, uint8_t bit_value, int *__bit_offset, int *__bit_len)
{
    if ( *offset >= 8 ) {
        //printf("append offset wrong \n");
        return -1;
    }
    if ( *__bit_offset + *__bit_len > 8 ) {
        //printf("bit total error\n");
        return -1;
    }    

    int bit_len = *__bit_len;
    if ( *offset + bit_len > 8 )
        bit_len = 8 - *offset;

        // 2 6 0 2
    //printf("test0 data=%u %u %d %u\n", bit_value, *__bit_offset, *offset, bit_len);
    uint8_t v =  bit_value >> *__bit_offset;
    
    v = v & ((1 << bit_len) - 1);

    v = v << *offset;

    //printf("test1, data=%u v=%u\n", *data, v);
    data = data | v ;
    //printf("test2, data=%u\n", *data);

    *offset += bit_len;
    *__bit_len -= bit_len;
    *__bit_offset = *__bit_offset + bit_len ;

    return 0;
}

static ALEO_ADI int append_to_field(bigint_u256_t *dest, int dest_bit_offset ,uint8_t variant, int data_bit_size , int data_bit_offset )
{

    while ( data_bit_size > 0 ) {
        int nbytes_offset = dest_bit_offset / 8;
        int nbit_offset = dest_bit_offset % 8;

        int ret = append_to_byte(dest->uint8[nbytes_offset], &nbit_offset, variant, &data_bit_offset, &data_bit_size );
        //printf("before nbit offset=%d , data_bit_offset=%d data_bit_size=%d\n", nbit_offset, data_bit_offset, data_bit_size);
        if ( ret < 0 ) {
            //printf("ret <0\n");
            return -1;
        }
                
        //printf("after nbit offset=%d , data_bit_offset=%d, data_bit_size=%d\n", nbit_offset, data_bit_offset, data_bit_size);
        
        dest_bit_offset = nbytes_offset*8 + nbit_offset;
        //printf("after all_bit_offset=%d, nbit offset=%d , data_bit_size=%d\n", dest_bit_offset, nbit_offset, data_bit_size);        
    }
    return dest_bit_offset;
}

static ALEO_ADI int append_to_field_for_qbigint(uint64_t dest[QBigInt::BIGINT_LEN], 
        int dest_bit_offset ,uint8_t variant, int data_bit_size , int data_bit_offset )
{

    while ( data_bit_size > 0 ) {
        int nbytes_offset = dest_bit_offset / 8;
        int nbit_offset = dest_bit_offset % 8;

        uint8_t v8 = QBigInt::get_u8_by_offset(dest, nbytes_offset);

        //printf("before nbit offset=%d , data_bit_offset=%d data_bit_size=%d\n", nbit_offset, data_bit_offset, data_bit_size);
        int ret = append_to_byte(v8, &nbit_offset, variant, &data_bit_offset, &data_bit_size );
        if ( ret < 0 ) {
            //printf("ret <0\n");
            return -1;
        }

        QBigInt::set_by_u8(dest, v8, nbytes_offset);
                
        //printf("after nbit offset=%d , data_bit_offset=%d, data_bit_size=%d\n", nbit_offset, data_bit_offset, data_bit_size);
        
        dest_bit_offset = nbytes_offset*8 + nbit_offset;
        //printf("after all_bit_offset=%d, nbit offset=%d , data_bit_size=%d\n", dest_bit_offset, nbit_offset, data_bit_size);        
    }
    return dest_bit_offset;
}



// dest应该是一个长度未定的 vec, 但是目前实现的时候，只需要2个就够了，所以写死 
static ALEO_ADI void circuit_plaintext_to_field(SVM_PARAM_DEF, LiteralVariantAsU8 variant, 
            uint64_t src[QBigInt::BIGINT_LEN], uint64_t dest1[QBigInt::BIGINT_LEN], 
            uint64_t dest2[QBigInt::BIGINT_LEN] )
{
    /*
    参考代码： 
    snarkVM-miner/circuit/program/src/data/plaintext/to_bits.rs ： write_bits_le
    snarkVM-miner/circuit/program/src/data/plaintext/to_fields.rs : to_fields
    */

#if 1
    
    uint64_t tmp_src[QBigInt::BIGINT_LEN];
    CirCuit_Fields::field_to_bigint_with_witness(SVM_PARAM, 0, src, tmp_src);

    //bigint_u256_t *dest = dest1;
    int bit_offset = 0;
    // 设置2 bit false
    bit_offset += 2 ;

    bit_offset = append_to_field_for_qbigint(dest1, bit_offset, variant, 8, 0);

    uint16_t field_size_in_bit = BLS12_377_FR_PARAMETER_MODULUS_BITS;
    uint8_t *p = (uint8_t*)&field_size_in_bit;
    bit_offset = append_to_field_for_qbigint(dest1, bit_offset, p[0], 8, 0);
    bit_offset = append_to_field_for_qbigint(dest1, bit_offset, p[1], 8, 0);
    

    uint64_t *dest = dest1;    
    for (int n = 0 ; n < sizeof(bigint_u256_t); n++ ) 
    {
        uint8_t data =  QBigInt::get_u8_by_offset(tmp_src, n);
        uint8_t max_bit_count , max_bit_contain ;
        if ( n == (sizeof(bigint_u256_t)-1) )
        {
            max_bit_contain = max_bit_count = 8 - BLS12_377_FR_PARAMETER_REPR_SHAVE_BITS;
        } else {
            //bit_offset = append_to_field(dest1, bit_offset, data, 8, 0);
            max_bit_contain = max_bit_count = 8;
        }
        
        while ( max_bit_count > 0 )
        {
            int to_save_bit_count;
            
            if ( max_bit_count + bit_offset < (BLS12_377_FR_PARAMETER_MODULUS_BITS-1)) 
            {
                to_save_bit_count = max_bit_count;
            } else {
                to_save_bit_count = BLS12_377_FR_PARAMETER_MODULUS_BITS - 1 - bit_offset;
            }

            bit_offset = append_to_field_for_qbigint(dest, bit_offset, data, to_save_bit_count, max_bit_contain-max_bit_count);                    
            if ( bit_offset < 0 )
                return;            
            if ( bit_offset == (BLS12_377_FR_PARAMETER_MODULUS_BITS-1) )
            {
                // printf("chunk is %d, break. to_save_bit_count=%d\n",  
                //         BLS12_377_FR_PARAMETER_MODULUS_BITS-1,
                //         to_save_bit_count
                //         );
                //补上 0 为最高1 bit 
                bit_offset = append_to_field_for_qbigint(dest, bit_offset, 0, 1, 0);
                
                //切换 
                bit_offset = 0;
                dest = dest2;
            }
            max_bit_count -= to_save_bit_count;
        }                
    }

    // Adds one final bit to the data, to serve as a terminus indicator.
    // During decryption, this final bit ensures we've reached the end.
    //bits_le.push(Boolean::constant(true));
    bit_offset = append_to_field_for_qbigint(dest, bit_offset, 1, 1, 0);

    QBigInt::bigint_to_field(dest1);
    QBigInt::bigint_to_field(dest2);
#endif    
}        