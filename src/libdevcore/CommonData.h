#pragma once


#include <algorithm>
#include <cstring>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include "Common.h"

namespace dev
{

enum class ScaleSuffix
{
    DontAdd = 0,
    Add = 1
};

/// Generic function to scale a value
std::string getScaledSize(double _value, double _divisor, int _precision, std::string _sizes[],
    size_t _numsizes, ScaleSuffix _suffix = ScaleSuffix::Add);

/// Formats memory
std::string getFormattedMemory(
    double _mem, ScaleSuffix _suffix = ScaleSuffix::Add, int _precision = 2);

}