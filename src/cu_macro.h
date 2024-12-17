#pragma once
#include <stdexcept>
#include <sstream>

struct cuda_runtime_error : public virtual std::runtime_error
{
    cuda_runtime_error(const std::string& msg) : std::runtime_error(msg) {}
};

#define CUDA_SAFE_CALL(call)                                                              \
    do                                                                                    \
    {                                                                                     \
        cudaError_t err = call;                                                           \
        if (cudaSuccess != err)                                                           \
        {                                                                                 \
            std::stringstream ss;                                                         \
            ss << "CUDA error in func " << __FUNCTION__ << " at line " << __LINE__ << ' ' \
               << cudaGetErrorString(err);                                                \
            throw cuda_runtime_error(ss.str());                                           \
        }                                                                                 \
    } while (0)

