
#include "cu_utils.cuh"


__constant__ aleo_u256_t BLS12_377_FR_PARAMETER_MODULUS = {
    .uint64 = {
        725501752471715841ULL,
        6461107452199829505ULL,
        6968279316240510977ULL,
        1345280370688173398ULL
    }
};

__constant__ aleo_u256_t BLS12_377_FR_PARAMETER_R2 = {
    .uint64 = {    
        2726216793283724667ULL,
        14712177743343147295ULL,
        12091039717619697043ULL,
        81024008013859129ULL
    }
};

static __device__ void aleo_u256_to_str(aleo_u256_t *val, char *buf, size_t buf_size)
{
    size_t n = 0;
    uint32_t cnt = 0;
    uint8_t *p = val->bytes;
    for( ; n < buf_size && cnt < sizeof(aleo_u256_t) ; cnt++ )
    {
        int start = n;

        int v = p[cnt] ;
        int v_left = v;

        for(int m = 100; m >=1; m =  m/10)
        {
            v = v_left / m;
            v_left = v_left % m;
            if ( v > 0 ) {
                buf[n++] = '0' + v;
            } else if ( n>start || m == 1) {
                buf[n++] = '0';
            }
        }
        buf[n++] = ',';
    }

    buf[n] = '\0';
}

__device__ uint64_t u64_mac_with_carry(uint64_t a, uint64_t b, uint64_t c, uint64_t &carry)
{    
    aleo_u128_t result;
    
    multiply_u64(b,c, result);

    //printf("t0=%016lx%016lx\n", result[1], result[0]);

    uint64_t sum = result[0] + a;
    //TODO: USE PTX optimize         
    result[1] += (uint64_t)(sum < result[0]);
    result[0] = sum;

    //printf("a+b*c=%016lx,%016lx\n", result[1], result[0]);

    sum = result[0] + carry;
    result[1] += (uint64_t)(sum < result[0]);
    result[0] = sum;

    //printf("t2=%016lx,%016lx\n", result[1], result[0]);

    carry = result[1];

    return result[0];
}



__device__ size_t cu_strlen(const char *src)
{
    size_t i = 0;
    while ( src[i] != 0 ) {
        i++;
    }
    return i;
}

__device__ void cu_strncat(char *dest, const char *src, size_t n)
{
    char *p = dest + cu_strlen(dest);
    int i = 0;
    for ( i = 0; i< n && i<cu_strlen(src); i++)
    {
        p[i] = src[i];
    }
    p[i] = 0;
}

__device__ static void	reverse(char *, size_t);

__device__ size_t cu_sltoa(char *s, long int n) {
	size_t i = 0;
	long int sign_mask;
	unsigned long int nn;

	sign_mask = n >> sizeof(long int) * 8 - 1;
	nn = (n + sign_mask) ^ sign_mask;
	do {
		s[i++] = nn % 10 + '0';
	} while (nn /= 10);

	s[i] = '-';
	i += sign_mask & 1;
	s[i] = '\0';

	reverse(s, i);
	return (i);
}

__device__ size_t cu_ultoa(char *s, unsigned long int n) {
	size_t i = 0;

	do {
		s[i++] = n % 10 + '0';
	} while (n /= 10);
	s[i] = '\0';

	reverse(s, i);
	return (i);
}

__device__ static void reverse(char *s, size_t s_len) {
	size_t i, j;
	char swap;

	for (i = 0, j = s_len - 1; i < j; ++i, --j) {
		swap = s[i];
		s[i] = s[j];
		s[j] = swap;
	}
}


__device__  uint32_t checked_next_power_of_n(uint32_t base, uint32_t n)
{
    if ( n <= 1 ) 
        return 0;
    
    uint32_t value = 1;
    while (value < base) {
        value *=n;
    }

    return value;
}



#ifdef ENABLE_TEST

__global__ void test_fp256_to_bigint(void)
{
    aleo_u256_t val = {
        .uint64 = {
                    863389543018879815ULL, 
                    17958235059814174874ULL, 
                    7314732469303877470ULL, 
                    703563799569133862ULL
                    }};
    
    bigint_from_field(&val);
    
}

static  __global__ void test_bigint_mac_carry(void)
{
    uint64_t a = 0x1234567887654321ULL;
    a = 0xfffffffffffffff1ULL;
    uint64_t  b = 0x8765432112345678ULL;
    uint64_t  c = 0xA5559999BBCCCCCCULL;
    uint64_t carry = 0x120127812ULL;
    
    u64_mac_with_carry(a,b,c, carry);    
}

#endif