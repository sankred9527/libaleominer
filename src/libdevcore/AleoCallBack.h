

#pragma
#include <cstdint>

typedef void (*submit_callback_t)(uint64_t solution_id, uint64_t counter);
typedef void (*simple_log_callback_t)(const char *msg);

extern simple_log_callback_t g_log_callback;
extern submit_callback_t g_submit_callback;