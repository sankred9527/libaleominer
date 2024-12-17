#include <functional>
#include <sstream>  
#include <map>
#include <iomanip>
#include <cuda_runtime.h>
#include <unistd.h>
#include "program_bin.h"

//using namespace dev;
using namespace std;


extern "C" void testlib_simple1(uint8_t *epoch_bin, void *program , size_t program_size);

int main()
{
    testlib_simple1(nullptr, program_bin, sizeof(program_bin));
}