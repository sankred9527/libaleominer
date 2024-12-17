#include <stdio.h>
#include "cu_bigint.cuh"
#include "cu_utils.cuh"
#include "circuit/integer/circuit_integer.cuh"


__device__ void bigint256_dump_with_prefix(const char *prefix, const char *suffix, bigint_u256_t *self)
{
    printf("%s", prefix);
    for(int n = 0; n < BIGINT256_IN_U64; n++)
    {
        printf("0x%016lx, ", self->uint64[n]);
    }
    printf("%s\n", suffix);
}


#ifdef ENABLE_TEST

__global__ void test_fr256_to_bigint(aleo_u256_t *self)
{
    aleo_u256_t v = {
        //.uint64 = { 5600085399146854981ULL, 3181821509559646731ULL, 15477203983971543107ULL, 275312087371647491ULL}
        //.uint64 = { 5244301028584586661ULL, 14661949949927751053ULL, 3985221112749649105ULL, 452675718251711882ULL}
        
        //.uint64 = { 725501752471715818ULL, 6461107452199829505ULL , 6968279316240510977ULL, 1345280370688173398ULL } 
        //.uint64 = {  7688348251362763991ULL,  427302382221189071ULL,  8534466366982032917ULL,   188369333827820598ULL }
        //.uint8 = {223, 244, 166, 55, 189, 233, 118, 182, 22, 178, 144, 101, 77, 185, 40, 139, 167, 87, 35, 129, 124, 200, 111, 75, 15, 172, 41, 78, 141, 255, 29, 8}
        .uint64 = { 4280610255603328046ULL, 1530878810334725522ULL , 6290772886244765876ULL, 1186642979551667127ULL } 
    };
    
    
    
uint64_t a = 0xf59010b007e04f7a;
uint64_t b = 0x3c039d67ce7214fd;
uint64_t c = 0xd47f7fc0eefe7866;

    uint64_t carry = 0x82abb39c828e261;
    uint64_t ret = u64_mac_with_carry_ptx(a,b,c, carry);
    printf("low = 0x%lx  high=0x%lx\n", ret, carry);

    // bigint_from_field(&v);
    // bigint256_dump(&v);

}

#if 0

__global__ void test_bigint256_div2(aleo_u256_t *v)
{    
    bigint256_div2(v);
    bigint256_dump(v);
}

__global__ void test_bigint256_sub_noborrow(aleo_u256_t *self, aleo_u256_t *other)
{    
    bigint256_sub_noborrow(self, other);
    bigint256_dump(self);    
}

__global__ void test_fr256_to_bigint(aleo_u256_t *self)
{
    aleo_u256_t v = {
        //.uint64 = { 5600085399146854981ULL, 3181821509559646731ULL, 15477203983971543107ULL, 275312087371647491ULL}
        //.uint64 = { 5244301028584586661ULL, 14661949949927751053ULL, 3985221112749649105ULL, 452675718251711882ULL}
        
        //.uint64 = { 10157024534604021774ULL,  16668528035959406606ULL, 5322190058819395602ULL, 387181115924875961ULL } 
        .uint64 = {  7688348251362763991ULL,  427302382221189071ULL,  8534466366982032917ULL,   188369333827820598ULL }
    };
    //v.bytes[0] = 84;
    bigint_from_field(&v);
    bigint256_dump(&v);

    bigint_u256_t dest;
    CirCuit_Integer::field_from_bigint(&v, &dest);
    bigint256_dump(&dest);


    memset(&dest, 0 , sizeof(dest));
    BIGINT256_GET_U16(&dest, 0) = (uint16_t)(-32768);
    bigint256_dump(&dest);
}

__global__ void test_bigint256_from_bigint(aleo_u256_t *self, aleo_u256_t *dest)
{
        
    CirCuit_Integer::field_from_bigint(self, dest);

    bigint_from_field(dest);
    bigint256_dump(dest);
}

#endif 
#endif 