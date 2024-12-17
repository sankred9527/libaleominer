#pragma once

class CirCuit_Fields_Base
{
public:            

    static ALEO_ADI void field_sub_assign(bigint_u256_t &self, bigint_u256_t other)
    {
        field_neg(other);
        bigint_add_carry_ptx(self, other, self);
    }

    static ALEO_ADI void field_sub_assign(uint64_t self[QBigInt::BIGINT_LEN], uint64_t other[QBigInt::BIGINT_LEN])
    {
        uint64_t tmp[QBigInt::BIGINT_LEN];
        QBigInt::copy(tmp, other);
        field_neg(tmp);
        QBigInt::add_ptx(self, tmp, self);
    }

    static ALEO_ADI void field_sub(uint64_t self[QBigInt::BIGINT_LEN], uint64_t other[QBigInt::BIGINT_LEN])
    {
        if ( QBigInt::compare(self, other) < 0 )
        {
            QBigInt::add_with_modulus(self);
        }
        QBigInt::sub_ptx(self, other);
    }

    static ALEO_ADI void field_sub(bigint_u256_t &self, bigint_u256_t other)
    {
        if ( aleo_u256_compare_less_than(&self, &other)) 
        {
            bigint_add_carry_ptx(self, BLS12_377_FR_PARAMETER_MODULUS, self);
        }
        bigint256_sub_noborrow_ptx(self, other);
    }

    static ALEO_ADI void field_div(uint64_t self[QBigInt::BIGINT_LEN], uint64_t other[QBigInt::BIGINT_LEN])
    {
        if ( bigint_inverse_for_quick_bigint(other) ) {
            QBigInt::bigint_mont_mul_assign_no_unroll(self, other, self);
        }
    }

    static ALEO_ADI void field_div(bigint_u256_t &self, bigint_u256_t other)
    {
        if ( bigint_inverse(other) ) {
            bigint_mont_mul_assign(self, other, self);
        }
    }

    static ALEO_ADI void field_neg(bigint_u256_t &self)
    {
        if ( bigint256_is_zero(&self) ) {
            return;
        }
        
        bigint_u256_t dest  = BLS12_377_FR_PARAMETER_MODULUS;
        bigint256_sub_noborrow_ptx(dest, self);
        self = dest;
    }

    static ALEO_ADI void field_neg(uint64_t self[QBigInt::BIGINT_LEN])
    {
        if ( QBigInt::is_zero(self) ) {
            return;
        }
                
        uint64_t dest[QBigInt::BIGINT_LEN] = {
            BLS12_377_FR_PARAMETER_MODULUS.uint64[0],
            BLS12_377_FR_PARAMETER_MODULUS.uint64[1],
            BLS12_377_FR_PARAMETER_MODULUS.uint64[2],
            BLS12_377_FR_PARAMETER_MODULUS.uint64[3]
        };
        QBigInt::sub_ptx(dest, self);
        QBigInt::copy(self, dest);
    }

    static ALEO_ADI void field_add_assign(bigint_u256_t &self, bigint_u256_t &other)
    {
        bigint_add_carry_ptx(self, other, self);
        field_reduce(self);
    }

    static ALEO_ADI void field_reduce(bigint_u256_t &src)
    {
        if ( !aleo_u256_compare_less_than(&src, &BLS12_377_FR_PARAMETER_MODULUS)) 
        {        
            //printf("not valid , should reduce\n");
            //bigint256_sub_noborrow(&src, &BLS12_377_FR_PARAMETER_MODULUS );
            src.uint64[0] = ptx::u64::sub_cc(src.uint64[0], 725501752471715841ULL );
            src.uint64[1] = ptx::u64::subc_cc(src.uint64[1], 6461107452199829505ULL );
            src.uint64[2] = ptx::u64::subc_cc(src.uint64[2], 6968279316240510977ULL );
            src.uint64[3] = ptx::u64::subc(src.uint64[3], 1345280370688173398ULL );
        } 
    }

    static ALEO_ADI void field_add_assign(uint64_t self[QBigInt::BIGINT_LEN], uint64_t other[QBigInt::BIGINT_LEN])
    {        
        QBigInt::add_ptx(self, other, self);
        field_reduce(self);
    }

    static ALEO_ADI void field_reduce(uint64_t src[QBigInt::BIGINT_LEN])
    {
        if ( QBigInt::compare(src, BLS12_377_FR_PARAMETER_MODULUS.uint64) > 0 )
        {
            src[0] = ptx::u64::sub_cc(src[0], BLS12_377_FR_PARAMETER_MODULUS_V0 );
            src[1] = ptx::u64::subc_cc(src[1], BLS12_377_FR_PARAMETER_MODULUS_V1 );
            src[2] = ptx::u64::subc_cc(src[2], BLS12_377_FR_PARAMETER_MODULUS_V2 );
            src[3] = ptx::u64::subc(src[3], BLS12_377_FR_PARAMETER_MODULUS_V3 );
        }
    }    

    static ALEO_ADI bigint_u256_t get_field_one(void)
    {
        bigint_u256_t tmp = {
            .uint64 = {
                9015221291577245683ULL,  8239323489949974514ULL,  1646089257421115374ULL,   958099254763297437ULL
            } 
        };
        return tmp;
    }

    static ALEO_ADI void set_field_one(uint64_t data[QBigInt::BIGINT_LEN])
    {
        data[0] = 9015221291577245683ULL;
        data[1] = 8239323489949974514ULL;
        data[2] = 1646089257421115374ULL;
        data[3] = 958099254763297437ULL;        
    }

    static ALEO_ADI bigint_u256_t get_field_neg_one(void)
    {
        bigint_u256_t tmp = { 
            .uint64 = {
                10157024534604021774ULL, 16668528035959406606ULL, 5322190058819395602ULL, 387181115924875961ULL
            }
        };
        return tmp;
    }

    static ALEO_ADI void get_field_neg_one(uint64_t data[QBigInt::BIGINT_LEN])
    {
        data[0] = 10157024534604021774ULL;
        data[1] = 16668528035959406606ULL;
        data[2] = 5322190058819395602ULL;
        data[3] = 387181115924875961ULL;
    }

};