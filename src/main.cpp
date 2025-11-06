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
#include "lsm9ds1.hpp"
#include <csignal>
#include <iostream>
#include <vector>
#include <cstdint>
#include <iomanip>
#include <string_view>
#include <cstdlib>   // getenv

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

#ifndef I2C_MOCK
    // only producer talks to hw :)
    lsm9ds1_driver lsm9ds1("/dev/i2c-1", ADDR_XG);
    lsm9ds1.lsm9ds1_init();
    // init joystick
#endif

    using namespace std::chrono_literals;
    logger::tlabel = "producer";
    LOG_ALWAYS("producer start");

    // ctrl+c check
    while(!g_stop.load(std::memory_order_relaxed)){
        accel_burst_t accel_burst_sample;

#ifdef I2C_MOCK
        // simulated sample stream 
        uint8_t i = 0;
        for(i=0;i<LSM9DS1_NUM_BYTES_PER_BURST;i++){
            accel_burst_sample.accel_burst[i] = i;
            // mimick 119 Hz operation to match accelerometer sampling rate
            // 119Hz = 0.008s period -> PER BURST
            std::this_thread::sleep_for(8ms);
        }
#else
        lsm9ds1.lsm9ds1_read_burst(OUT_X_L_XL, &accel_burst_sample);
        // error handle?
#endif
        
        // blocking push function... will always (eventually) run and return true, 
        // unless queue is closed, in which case we want out
        if(!rb.push(accel_burst_sample)){
            break;
        }

        
    }

    // on thread shutdown, close queue to call release n unblock any consumer waiting for acquire
    rb.close();
}

// note ctrl+c exit handled on producer side so only one thread reads g_stop
// when we close the ring buffer, [rb.close() above], pop will return false when consumer tries to pop -> clean exit; otherwise, blocking
void consumer_thread_fn(ringBuffer_C<accel_burst_t>& rb){
    using namespace std::chrono_literals;
    logger::tlabel = "consumer";
    LOG_ALWAYS("consumer start");
    std::vector<accel_sample_t> input_datastream; // want to save as acc 16 bit int values for x,y,z... struct type ACCEL_SAMPLE_S
    accel_burst_t raw_sample;
    int16_t x = 0x0;
    int16_t y = 0x0;
    int16_t z = 0x0;

    while(1){
        if(!rb.pop(&raw_sample)){
            break;
        }
#ifdef I2C_MOCK
        // little endian [xl xh yl yh zl zh]
        // is there an issue with this logic given ints are signed???
        // need to do raw shift/manipulation cast to unsigned and then reinterpret as signed at the end
        x=int16_t((uint16_t(raw_sample.accel_burst[1])<<8) | uint16_t(raw_sample.accel_burst[0]));
        y=int16_t((uint16_t(raw_sample.accel_burst[3])<<8) | uint16_t(raw_sample.accel_burst[2]));
        z=int16_t((uint16_t(raw_sample.accel_burst[5])<<8) | uint16_t(raw_sample.accel_burst[4]));
        // need to add label and idx
        input_datastream.push_back(accel_sample_t{x,y,z});
#endif
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
    LOG_ALWAYS("start (VERBOSE=" << logger::verbose << ")");

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
