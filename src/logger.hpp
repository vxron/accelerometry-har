#include <chrono>
#include <csignal>
#include <iostream>
#include <cstdint>
#include <string_view>
#include <cstdlib>   // getenv
#include <iomanip>
#include <string>

namespace {
using clock_t = std::chrono::steady_clock;
const auto g_t0 = clock_t::now();

inline uint64_t ms_since_start() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(clock_t::now() - g_t0).count();
}
inline bool verbose() {
    const char* v = std::getenv("VERBOSE");
    return v && *v && std::string_view(v) != "0";
}

thread_local const char* tlabel = "main";  // set per-thread

#define LOG_ALWAYS(msg) do { \
    std::cout << "[" << std::setw(6) << ms_since_start() << " ms] " \
              << tlabel << ": " << msg << std::endl; \
} while(0)

#define LOG_DBG(msg) do { if (verbose()) LOG_ALWAYS(msg); } while(0)
} 
