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
        lsm9ds1.lsm9ds1_read_burst(OUT_X_L_XL, &accel_burst_sample); // error handle?
        // can this cause problems, is atomic sufficient or do i need to consider semaphore in addition?
#ifdef CALIBRATION_MODE
// does this need a mutex guard or is atomic sufficient?
        accel_burst_sample.active_label = g_record.load(memory_order_acquire); // reader = acquire
#endif

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

    sliding_window_t window; // should acquire the data for 1 window with that many pops n then increment by hop... 
    accel_burst_t temp; // placeholder for accel burst storage 
    
    // wait for first signal that we've reached the WINDOW_SAMPLES length in the buffer
    while(rb.get_count() < window.winLen){
        // don't have enough data
        continue;
    }
    // BUILD FIRST WINDOW - do we need a mutex guard here?  i kinda dont think so because no one else is gonna be popping? and consumer/producer push/pop is handled intrinsically by semaphores in ringbuf class
    for(int i=0;i<window.winLen;i++){
        if(!rb.pop(&temp)){
            break; // throw error
        }
        else {
            // pop successful -> push into sliding window
            window.sliding_window.push(temp);
        }
    }

    while(1){
        // emit window to feature extractor (DEEEP COPY)
        featureExtractor.readin(window);
        // pop out half of window for 50% hop
        for(int k=0;k<window.winHop;k++){
            window.sliding_window.pop();
        }
        // after the first time, we increment by hop size rather than window size (as long as we have hop size available in array, we can pull a window)
        while(rb.get_count() < window.winHop){
            continue;
        }
        // we have enough to make a window from head by adding the hop amount 
        // keep the tail of the sliding window, overwrite the head (older) 
        for(int j=0;j<window.winHop;j++){
            if(!rb.pop(&temp)){
                break; // throw error
            }
            else {
                // pop successful -> push into sliding window
                window.sliding_window.push(temp);
            }
        }
    }
        
#ifdef I2C_MOCK
        accel_burst_t raw_sample;
        int16_t x = 0x0;
        int16_t y = 0x0;
        int16_t z = 0x0;
        if(!rb.pop(&raw_sample)){
            break;
        }
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
void joystick_thread_fn(const char* devnode = "/dev/input/event5", ringBuffer_C<accel_burst_t>& rb) {
    logger::tlabel = "joystick";
    int fd = ::open(devnode, O_RDONLY | O_CLOEXEC | O_NONBLOCK);
    if (fd < 0) {
      LOG_ALWAYS("open(" << devnode << ") failed: " << std::strerror(errno));
      return;
    }
    LOG_ALWAYS("opened " << devnode);

    // should wait (poll for) button press event 
    // then process that event & run state machine
    using namespace std::chrono_literals;
    using clock = std::chrono::steady_clock;
    auto last_toggle = clock::now();

    while(!g_stop.load(std::memory_order_relaxed)){
        input_event joystick_ev;
        ssize_t n = ::read(fd,&ev, sizeof(ev));
        if (ev.type != EV_KEY) continue;
        // value: 1 for press, 0 for release
        if(ev.value == 1 && is_center(ev.code)){
            auto now = clock::now();
            if(now - last_toggle > std::chrono(150ms)) {
                // toggle g_record atomic on
                bool new_state = !g_record.load(std::memory_order_relaxed);
                g_record.store(new_state, std::memory_order_relaxed);
                last_toggle = now;
                LOG_ALWAYS(std::string("record = ") + (new_state ? "ON" : "OFF"));
            }
        }
        
    }
    ::close(fd);
    LOG_ALWAYS("closed");
}
#endif

int main() {
    LOG_ALWAYS("start (VERBOSE=" << logger::verbose << ")");

    ringBuffer_C<accel_burst_t> ringBuf(RING_BUFFER_CAPACITY);

    // interrupt caused by SIGINT -> 'handle_singint' acts like ISR (callback handle)
    std::signal(SIGINT, handle_sigint);

    // START THREADS. We pass the ring buffer by reference (std::ref) becauase each thread needs the actual shared 'ringBuf' instance, not just a copy...
    // This builds a new thread that starts executing immediately, running producer_thread_rn in parallel with the main thread (same for cons)
    std::thread prod(producer_thread_fn,std::ref(ringBuf));
    std::thread cons(consumer_thread_fn,std::ref(ringBuf));
#ifdef CALIBRATION_MODE
    std::thread joys(joystick_thread_fn,"/dev/input/event5",std::ref(ringBuf));
#endif

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
#ifdef CALIBRATION_MODE
    joys.join();
#endif

    return 0;
}
