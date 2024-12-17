

#include <functional>
#include <sstream>  
#include <map>
#include <iomanip>
#include <cuda_runtime.h>
#include <unistd.h>


using namespace std;

static void dump_program(void *program , size_t program_size)
{
    uint8_t *p = (uint8_t*)program ;

    char buf[102400] = {0};
    int len = sizeof(buf);
    int ret  = 0;

    ret += snprintf(buf+ret, len-ret, "#include <stdint.h>\nuint8_t program_bin[] = ");

    ret += snprintf(buf+ret, len-ret, "\n{\n");
    for ( int n = 0; n < program_size; n++)
    {
        if ( n == (program_size -1 ) )
            ret += snprintf(buf+ret, len-ret, "0x%02x", p[n]);
        else 
            ret += snprintf(buf+ret, len-ret, "0x%02x ,", p[n]);
        if ( (n + 1) % 16 == 0 )
            ret += snprintf(buf+ret, len-ret, "\n");
    }
    ret += snprintf(buf+ret, len-ret, "\n};\n");

    FILE *fp = fopen("/tmp/program.h", "w+");
    fwrite(buf, strlen(buf),1, fp);
    fclose(fp);
}


extern "C" void testlib_save_program_as_bin(void *program , size_t program_size)
{
    dump_program(program, program_size);
    return;
}
