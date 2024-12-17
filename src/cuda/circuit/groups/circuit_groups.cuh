#pragma once
#include "cuda/cu_common.h"
#include "cuda/cu_bigint.cuh"
#include "cuda/circuit/field/circuit_field_base.cuh"
#include "cuda/cu_mtree_leaf.cuh"

class Circuit_Groups
{
public:

    static constexpr uint64_t cu_group_const_a_v0 = 0x8cf500000000000eULL;
    static constexpr uint64_t cu_group_const_a_v1 = 0xe75281ef6000000eULL;
    static constexpr uint64_t cu_group_const_a_v2 = 0x49dc37a90b0ba012ULL;
    static constexpr uint64_t cu_group_const_a_v3 = 0x55f8b2c6e710ab9ULL;
     

    static void ALEO_ADI add(SVM_PARAM_CDEF, 
                uint64_t self_x[QBigInt::BIGINT_LEN], uint64_t self_y[QBigInt::BIGINT_LEN], 
                uint64_t other_x[QBigInt::BIGINT_LEN], uint64_t other_y[QBigInt::BIGINT_LEN]
            )
    {
        // __constant__ uint64_t a[QBigInt::BIGINT_LEN] = {
        //     0x8cf500000000000eULL, 0xe75281ef6000000eULL, 0x49dc37a90b0ba012ULL, 0x55f8b2c6e710ab9ULL
        // };

        // __constant__ uint64_t d[QBigInt::BIGINT_LEN] = {
        //     0xd047ffffffff5e30ULL, 0xf0a91026ffff57d2ULL, 0x9013f560d102582ULL, 0x9fd242ca7be5700ULL
        // };


        bool self_is_const = is_const & 1;
        bool other_is_const = (is_const>>1) & 1;

        if ( self_is_const && QBigInt::is_zero(self_x) && QBigInt::is_zero(self_y)) 
        {
            QBigInt::copy(self_x, other_x);
            QBigInt::copy(self_y, other_y);
            return;
        } else if ( other_is_const && QBigInt::is_zero(other_x) && QBigInt::is_zero(other_x)) 
        {
            return;
        }        
        
        uint64_t *this_x;
        uint64_t *this_y;
        uint64_t *that_x;
        uint64_t *that_y;
        bool that_is_const;
        bool this_is_const;
        if ( other_is_const ) 
        {            
            this_x = self_x;
            this_y = self_y;
            that_x = other_x;
            that_y = other_y;

            this_is_const = self_is_const;
            that_is_const = other_is_const;
        } else {
            this_x = other_x;
            this_y = other_y;
            that_x = self_x;
            that_y = self_y;
            that_is_const = self_is_const;
            this_is_const = other_is_const;
        }

        // bigint256_dump_with_prefix("group add, this_x=", "", &this_x);
        // bigint256_dump_with_prefix("group add, this_y=", "", &this_y);
        

        // Compute U = (-A * x1 + y1) * (x2 + y2)
        uint64_t u1[QBigInt::BIGINT_LEN] = {
            cu_group_const_a_v0, cu_group_const_a_v1, cu_group_const_a_v2, cu_group_const_a_v3
        };
        uint64_t u2[QBigInt::BIGINT_LEN];

        {                        
            CirCuit_Fields_Base::field_neg(u1);
            QBigInt::bigint_mont_mul_assign_no_unroll(this_x, u1, u1);
            QBigInt::add_ptx(u1, this_y, u1);                        
            CirCuit_Fields_Base::field_reduce(u1);
            QBigInt::add_ptx(that_x, that_y, u2);
            CirCuit_Fields_Base::field_reduce(u2);
        }

        uint64_t u[QBigInt::BIGINT_LEN];
        QBigInt::bigint_mont_mul_assign_no_unroll(u1, u2, u);        
        if ( !this_is_const && !that_is_const ) {
            svm_push_leaf_full_with_convert(SVM_PARAM, u);
        }
        // bigint256_dump_with_prefix("group add, u1=", "", &u1);
        // bigint256_dump_with_prefix("group add, u2=", "", &u2);
        // bigint256_dump_with_prefix("group add, u=", "", &u);

        //bigint_u256_t & v0 = u1; 
        //bigint_u256_t & v1 = u2; 
        uint64_t v2[QBigInt::BIGINT_LEN];
        
        // Compute v0 = x1 * y2
        QBigInt::bigint_mont_mul_assign_no_unroll(this_x, that_y, u1);
        if ( !this_is_const && !that_is_const ) {
            svm_push_leaf_full_with_convert(SVM_PARAM, u1);
        }

        // Compute v1 = x2 * y1
        QBigInt::bigint_mont_mul_assign_no_unroll(that_x, this_y, u2);
        if ( !this_is_const && !that_is_const ) {
            svm_push_leaf_full_with_convert(SVM_PARAM, u2);
        }        

        // Compute v2 = d * v0 * v1
        QBigInt::bigint_mont_mul_assign_no_unroll(u1, u2, v2);
        if ( !this_is_const || !that_is_const ) {            
            svm_push_leaf_full_with_convert(SVM_PARAM, v2);
        }

        uint64_t tmp[QBigInt::BIGINT_LEN] = {
            0xd047ffffffff5e30ULL, 0xf0a91026ffff57d2ULL, 0x9013f560d102582ULL, 0x9fd242ca7be5700ULL
        };
        QBigInt::bigint_mont_mul_assign_no_unroll(v2, tmp, v2);

        // Compute x3 and y3.
        uint64_t xy3[QBigInt::BIGINT_LEN];
        //uint64_t y3[QBigInt::BIGINT_LEN];
        {                        
            CirCuit_Fields_Base::set_field_one(tmp);

            QBigInt::add_ptx(u1, u2, xy3);
            QBigInt::add_ptx(tmp, v2, tmp);

            CirCuit_Fields_Base::field_reduce(xy3);
            CirCuit_Fields_Base::field_reduce(tmp);
            CirCuit_Fields_Base::field_div(xy3, tmp);

            if ( !self_is_const || !other_is_const)
            {
                svm_push_leaf_full_with_convert(SVM_PARAM, xy3);
            }
            QBigInt::copy(self_x, xy3);

            tmp[0] = cu_group_const_a_v0;
            tmp[1] = cu_group_const_a_v1;
            tmp[2] = cu_group_const_a_v2;
            tmp[3] = cu_group_const_a_v3;
            QBigInt::bigint_mont_mul_assign_no_unroll(u1, tmp, xy3);
            QBigInt::add_ptx(xy3, u, xy3);
                        
            CirCuit_Fields_Base::field_reduce(xy3);
            CirCuit_Fields_Base::field_sub(xy3, u2);
                        
            CirCuit_Fields_Base::set_field_one(tmp);
            CirCuit_Fields_Base::field_sub(tmp, v2);
            CirCuit_Fields_Base::field_div(xy3, tmp);
        }

        if ( !self_is_const || !other_is_const)
        {
            svm_push_leaf_full_with_convert(SVM_PARAM, xy3);            
        }

        QBigInt::copy(self_y, xy3);        
    }

    static void ALEO_ADI add(SVM_PARAM_CDEF, bigint_u256_t self_x, bigint_u256_t self_y, 
                bigint_u256_t other_x, bigint_u256_t other_y,
                bigint_u256_t &dest_x, bigint_u256_t &dest_y)
    {
        constexpr bigint_u256_t a = {
            .uint64 = { 0x8cf500000000000eULL, 0xe75281ef6000000eULL, 0x49dc37a90b0ba012ULL, 0x55f8b2c6e710ab9ULL }
        };

        constexpr bigint_u256_t d = {
            .uint64 = { 0xd047ffffffff5e30ULL, 0xf0a91026ffff57d2ULL, 0x9013f560d102582ULL, 0x9fd242ca7be5700ULL }
        };


        bool self_is_const = is_const & 1;
        bool other_is_const = (is_const>>1) & 1;

        if ( self_is_const && bigint256_is_zero(&self_x) && bigint256_is_zero(&self_y)) 
        {
            dest_x = other_x;
            dest_y = other_y;            
            return;
        } else if ( other_is_const && bigint256_is_zero(&other_x) && bigint256_is_zero(&other_x)) 
        {
            dest_x = self_x;
            dest_y = self_y;
            return;
        }        
        
        bigint_u256_t *this_x;
        bigint_u256_t *this_y;
        bigint_u256_t *that_x;
        bigint_u256_t *that_y;
        bool that_is_const;
        bool this_is_const;
        if ( other_is_const ) 
        {            
            this_x = &self_x;
            this_y = &self_y;
            that_x = &other_x;
            that_y = &other_y;

            this_is_const = self_is_const;
            that_is_const = other_is_const;
        } else {
            this_x = &other_x;
            this_y = &other_y;
            that_x = &self_x;
            that_y = &self_y;
            that_is_const = self_is_const;
            this_is_const = other_is_const;
        }

        // bigint256_dump_with_prefix("group add, this_x=", "", &this_x);
        // bigint256_dump_with_prefix("group add, this_y=", "", &this_y);
        

        // Compute U = (-A * x1 + y1) * (x2 + y2)
        bigint_u256_t u1, u2;
        {
            u1 = a;
            CirCuit_Fields_Base::field_neg(u1);
            bigint_mont_mul_assign(*this_x, u1, u1);
            bigint_add_carry_ptx(u1, *this_y, u1);
            CirCuit_Fields_Base::field_reduce(u1);
            bigint_add_carry_ptx(*that_x, *that_y, u2);
            CirCuit_Fields_Base::field_reduce(u2);
        }

        bigint_u256_t u;
        bigint_mont_mul_assign(u1, u2, u);
        if ( !this_is_const && !that_is_const ) {
            svm_push_leaf_full_with_convert(SVM_PARAM, u);
        }
        // bigint256_dump_with_prefix("group add, u1=", "", &u1);
        // bigint256_dump_with_prefix("group add, u2=", "", &u2);
        // bigint256_dump_with_prefix("group add, u=", "", &u);

        bigint_u256_t & v0 = u1; 
        bigint_u256_t & v1 = u2; 
        bigint_u256_t v2;
        
        // Compute v0 = x1 * y2
        bigint_mont_mul_assign(*this_x, *that_y, v0);
        if ( !this_is_const && !that_is_const ) {
            svm_push_leaf_full_with_convert(SVM_PARAM, v0);
        }

        // Compute v1 = x2 * y1
        bigint_mont_mul_assign(*that_x, *this_y, v1);
        if ( !this_is_const && !that_is_const ) {
            svm_push_leaf_full_with_convert(SVM_PARAM, v1);
        }        

        // Compute v2 = d * v0 * v1
        bigint_mont_mul_assign(v0, v1, v2);
        if ( !this_is_const || !that_is_const ) {            
            svm_push_leaf_full_with_convert(SVM_PARAM, v2);
        }

        bigint_mont_mul_assign(v2, d, v2);        

        // Compute x3 and y3.
        bigint_u256_t x3;
        bigint_u256_t y3;
        {
            bigint_u256_t one = CirCuit_Fields_Base::get_field_one();
            bigint_u256_t tmp;
            bigint_add_carry_ptx(v0, v1, x3);
            bigint_add_carry_ptx(one, v2, tmp);
            CirCuit_Fields_Base::field_reduce(x3);
            CirCuit_Fields_Base::field_reduce(tmp);            
            CirCuit_Fields_Base::field_div(x3, tmp);

            bigint_mont_mul_assign(v0, a, y3);
            bigint_add_carry_ptx(y3, u, y3);
            CirCuit_Fields_Base::field_reduce(y3);
            CirCuit_Fields_Base::field_sub(y3, v1);
            
            tmp = one;            
            CirCuit_Fields_Base::field_sub(tmp, v2);            
            CirCuit_Fields_Base::field_div(y3, tmp);
        }

        if ( !self_is_const || !other_is_const)
        {
            svm_push_leaf_full_with_convert(SVM_PARAM, x3);
            svm_push_leaf_full_with_convert(SVM_PARAM, y3);
        }
        

        dest_x = x3;
        dest_y = y3;
    }

};
