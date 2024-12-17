

#pragma once

#include <atomic>
#include <mutex>

namespace dev
{
using Mutex = std::mutex;
using Guard = std::lock_guard<std::mutex>;
using UniqueGuard = std::unique_lock<std::mutex>;

template <class GuardType, class MutexType>
struct GenericGuardBool : GuardType
{
    GenericGuardBool(MutexType& _m) : GuardType(_m) {}
    bool b = true;
};

/** @brief Simple block guard.
 * The expression/block following is guarded though the given mutex.
 * Usage:
 * @code
 * Mutex m;
 * unsigned d;
 * ...
 * ETH_(m) d = 1;
 * ...
 * ETH_(m) { for (auto d = 10; d > 0; --d) foo(d); d = 0; }
 * @endcode
 *
 * There are several variants of this basic mechanism for different Mutex types and Guards.
 *
 * There is also the UNGUARD variant which allows an unguarded expression/block to exist within a
 * guarded expression. eg:
 *
 * @code
 * Mutex m;
 * int d;
 * ...
 * ETH_GUARDED(m)
 * {
 *   for (auto d = 50; d > 25; --d)
 *     foo(d);
 *   ETH_UNGUARDED(m)
 *     bar();
 *   for (; d > 0; --d)
 *     foo(d);
 * }
 * @endcode
 */

#define DEV_GUARDED(MUTEX) \
    for (GenericGuardBool<Guard, Mutex> __eth_l(MUTEX); __eth_l.b; __eth_l.b = false)

}  // namespace dev
