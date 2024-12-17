#pragma once
#include "circuit_integer.cuh"

// template <typename T>
// class FOO
// {

// };

/* 
根据 CircuitTypes 转为对应的 Circuit_Integer<T> 类型
*/
template<CircuitTypes CT>
struct TYPE_CONVERT_CT_CINT;

template<>
struct TYPE_CONVERT_CT_CINT<CT_I8> {
    using type = char1;
};

template<>
struct TYPE_CONVERT_CT_CINT<CT_U8> {
    using type = uchar1;
};



template<typename T>
struct type_to_string {
    __device__ static const char* get() { return "Unknown"; }
};
template<>
struct type_to_string<uint32_t> {
    __device__ static const char* get() { return "uint32_t"; }
};
template<>
struct type_to_string<uint8_t> {
    __device__ static const char* get() { return "uint8_t"; }
};
template<>
struct type_to_string<uchar1> {
    __device__ static const char* get() { return "uchar1"; }
};
template<>
struct type_to_string<uint16_t> {
    __device__ static const char* get() { return "uint16_t"; }
};
template<>
struct type_to_string<ushort1> {
    __device__ static const char* get() { return "ushort1"; }
};
template<>
struct type_to_string<uint64_t> {
    __device__ static const char* get() { return "uint64_t"; }
};
template<>
struct type_to_string<int64_t> {
    __device__ static const char* get() { return "int64_t"; }
};
template<>
struct type_to_string<__uint128_t> {
    __device__ static const char* get() { return "__uint128_t"; }
};
template<>
struct type_to_string<uint4> {
    __device__ static const char* get() { return "uint4"; }
};
template<>
struct type_to_string<int4> {
    __device__ static const char* get() { return "int4"; }
};
template<>
struct type_to_string<uint2> {
    __device__ static const char* get() { return "uint2"; }
};
template<>
struct type_to_string<int2> {
    __device__ static const char* get() { return "int2"; }
};
template<>
struct type_to_string<int1> {
    __device__ static const char* get() { return "int1"; }
};
template<>
struct type_to_string<uint1> {
    __device__ static const char* get() { return "uint1"; }
};

// Function to print type
template<typename T>
__device__ void print_type() {
    printf( "Type: %s\n", type_to_string<T>::get() );
}

