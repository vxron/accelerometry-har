// Needs three threads:
// 1- Consumer: main processing pipeline (pulls from ring buffer, constructs data table, labels data)
// 2 - Producer: main data acquisition pipeline (talks to sensor over i2c, pushes to ring buffer)
// 3 - Joystick state machine: only in calibration mode to track whether were in active or rest activity block

#include <thread>
#include <chrono>
#include <iostream>
#include "ringBuffer.hpp"
#include "types.hpp"
#include <atomic>
#include "lsm9ds1.h"
#include <csignal>
#include <iostream>
#include <vector>
#include <cstdint>
#include <iomanip>
#include <string_view>
#include <cstdlib>   // getenv

// ===================== logger setup =====================
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
// ===================== logger end =======================

// Global "please stop" flag set by Ctrl+C (SIGINT) to shut down cleanly
static std::atomic<bool> g_stop{false};

#ifdef CALIBRATION_MODE
static std::atomic<bool> g_record{false}; // toggled by joystick presses
static std::atomic<classes_e> g_label{CLASS_NO_LABEL}; // current label
#endif

// Interrupt signal sent when ctrl+c is pressed
void handle_sigint(int) {
    g_stop.store(true,std::memory_order_relaxed);
}



void producer_thread_fn(ringBuffer_C<accel_burst_t>& rb){
    // simulated sample stream 
    uint8_t i = 0;
    using namespace std::chrono_literals;
    LOG_ALWAYS("producer start");

    // ctrl+c check
    while(!g_stop.load(std::memory_order_relaxed)){
        accel_burst_t accel_burst_sample;
        // fill it with fake data for now
        for(i=0;i<LSM9DS1_NUM_BYTES_PER_BURST;i++){
            accel_burst_sample.accel_burst[i] = i;
        }
        
        // blocking push function... will always (eventually) run and return true, 
        // unless queue is closed, in which case we want out
        if(!rb.push(accel_burst_sample)){
            break;
        }
        
        // mimick 100 Hz operation to match accelerometer sampling rate
        // 100Hz = 0.01s priod
        std::this_thread::sleep_for(10ms);
    }

    // on thread shutdown, close queue to call release n unblock any consumer waiting for acquire
    rb.close();
}

// note ctrl+c exit handled on producer side so only one thread reads g_stop
// when we close the ring buffer, [rb.close() above], pop will return false when consumer tries to pop -> clean exit; otherwise, blocking
void consumer_thread_fn(ringBuffer_C<accel_burst_t>& rb){
    using namespace std::chrono_literals;
    LOG_ALWAYS("consumer start");
    std::vector<accel_sample_s> input_datastream; // want to save as acc 16 bit int values for x,y,z... struct type ACCEL_SAMPLE_S
    accel_burst_t raw_sample; 
    int16_t x = 0x0;
    int16_t y = 0x0;
    int16_t z = 0x0;


    while(1){
        if(!rb.pop(&raw_sample)){
            break;
        }
        // little endian [xl xh yl yh zl zh]
        // is there an issue with this logic given ints are signed???
        // need to do raw shift/manipulation cast to unsigned and then reinterpret as signed at the end
        x=int16_t((uint16_t(raw_sample.accel_burst[1])<<8) | uint16_t(raw_sample.accel_burst[0]));
        y=int16_t((uint16_t(raw_sample.accel_burst[3])<<8) | uint16_t(raw_sample.accel_burst[2]));
        z=int16_t((uint16_t(raw_sample.accel_burst[5])<<8) | uint16_t(raw_sample.accel_burst[4]));
        input_datastream.push_back(accel_sample_s{x,y,z});

        std::this_thread::sleep_for(80ms); // slightly below 100Hz freq (slower than producer!)
    }
}

#ifdef CALIBRATION_MODE
void joystick_thread_fn() {
    using namespace std::chrono_literals;
    bool last = false;
    auto last_change_time = clock_t::now();
    while(!g_stop.load(std::memory_order_relaxed)){
        bool pressed = /* non-blocking read from Sense HAT joystick */ false;
        
        if(pressed != last) { // check for transition
            auto now = clock_t::now();
            // debounce
            if(std::chrono::duration_cast<std::chrono::milliseconds>(now - last_change).count() > 50){
                // IMPLEMENT BETTER DEBOUNCE
            }
        }
    }
}
#endif

int main() {
    LOG_ALWAYS("start (VERBOSE=" << verbose() << ")");

    ringBuffer_C<accel_burst_t> ringBuf;

    // interrupt caused by SIGINT -> 'handle_singint' acts like ISR (callback handle)
    std::signal(SIGINT, handle_sigint);

    // START THREADS. We pass the ring buffer by reference (std::ref) becauase each thread needs the actual shared 'ringBuf' instance, not just a copy...
    // This builds a new thread that starts executing immediately, running producer_thread_rn in parallel with the main thread (same for cons)
    std::thread prod(producer_thread_fn,std::ref(ringBuf));
    std::thread cons(consumer_thread_fn,std::ref(ringBuf));

    // CONTROL TIMEOUT HERE (TODO: MAKE IT A VARIABLE AT THE TOP)
    // Let the demo run for ~10 seconds (or hit Ctrl+C to stop early).
    using namespace std::chrono_literals;
    for (int i = 0; i < 500 && !g_stop.load(); ++i) {
        std::this_thread::sleep_for(200ms);
    }

    // on system shutdown:
    // ctrl+c called...
    g_stop.store(true,std::memory_order_relaxed);
    ringBuf.close();
    prod.join();
    cons.join(); // close "join" individual threads

    return 0;
}
