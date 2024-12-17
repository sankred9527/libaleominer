

#include <sstream>
#include <iomanip>
#include "CommonData.h"

using namespace std;
using namespace dev;

// Global vars
bool g_running = false;
bool g_exitOnError = false;  // Whether or not ethminer should exit on mining threads errors

std::string dev::getScaledSize(double _value, double _divisor, int _precision, string _sizes[],
    size_t _numsizes, ScaleSuffix _suffix)
{
    double _newvalue = _value;
    size_t i = 0;
    while (_newvalue > _divisor && i <= (_numsizes - 1))
    {
        _newvalue /= _divisor;
        i++;
    }

    std::stringstream _ret;
    _ret << fixed << setprecision(_precision) << _newvalue;
    if (_suffix == ScaleSuffix::Add)
        _ret << " " << _sizes[i];
    return _ret.str();
}

std::string dev::getFormattedMemory(double _mem, ScaleSuffix _suffix, int _precision)
{
    static std::string suffixes[] = {"B", "KB", "MB", "GB"};
    return dev::getScaledSize(_mem, 1024.0, _precision, suffixes, 4, _suffix);
}
