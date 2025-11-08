// Needs three threads:
// 1- Consumer: main processing pipeline (pulls from ring buffer, constructs data table, labels data)
// 2 - Producer: main data acquisition pipeline (talks to sensor over i2c, pushes to ring buffer)
// 3 - Joystick state machine: only in calibration mode to track whether were in active or rest activity block

#include <thread>
#include <chrono>
#include <iostream>
#include "ringBuffer.hpp"
#include "types.hpp"
#include "window_configs.hpp"
#include <atomic>
#include "lsm9ds1.hpp"
#include "logger.hpp"
#include <csignal>
#include <vector>
#include <cstdint>
#include <iomanip>
#include <string_view>
#include <cstdlib>
#include <linux/input.h> // Linux evdev: struct input_event, EV_* types, KEY_* codes
#include <fcntl.h> // open(), O_RDONLY, O_NONBLOCK, O_CLOEXEC
#include <unistd.h> // read(), close()
#include <poll.h> // poll(), struct pollfd, POLLIN, POLLERR, POLLHUP
#include <cerrno>
#include <cstring>
#if CALIBRATION_MODE
#include <fstream>
#endif

// Global "please stop" flag set by Ctrl+C (SIGINT) to shut down cleanly
static std::atomic<bool> g_stop{false};

#if CALIBRATION_MODE
static std::atomic<bool> g_record{false}; // toggled by joystick presses
#endif

// Interrupt signal sent when ctrl+c is pressed
void handle_sigint(int) {
    g_stop.store(true,std::memory_order_relaxed);
}

void producer_thread_fn(ringBuffer_C<accel_burst_t>& rb){

    using namespace std::chrono_literals;
    logger::tlabel = "producer";
    LOG_ALWAYS("producer start");
#if I2C_MOCK
    LOG_ALWAYS("PATH=MOCK");
#else
    LOG_ALWAYS("PATH=HARDWARE");
#endif

#if !I2C_MOCK
    // only producer talks to hw :)
    lsm9ds1_driver lsm9ds1("/dev/i2c-1", ADDR_XG);
    if (lsm9ds1.lsm9ds1_init() != 0){
        LOG_ALWAYS("lsm9ds1_init failed; exiting producer");
        rb.close();
        return;
    };
    // init joystick
#endif
    size_t tick_count = 0;
    // ctrl+c check
    while(!g_stop.load(std::memory_order_relaxed)){
        accel_burst_t accel_burst_sample {};
        
#if I2C_MOCK
        // simulated sample stream 
        static int16_t i = 0;
        accel_burst_sample.x = i;
        accel_burst_sample.y = i;
        accel_burst_sample.z = i; 
        i++;
        tick_count++;
        accel_burst_sample.tick = tick_count;
        // mimick 119 Hz operation to match accelerometer sampling rate
        // 119Hz = 0.008s period -> PER BURST
        std::this_thread::sleep_for(8ms);
#else
        lsm9ds1.lsm9ds1_read_burst(OUT_X_L_XL, &accel_burst_sample); // error handle?
        // can this cause problems, is atomic sufficient or do i need to consider semaphore in addition?
        tick_count++;
        accel_burst_sample.tick = tick_count;
#if CALIBRATION_MODE
// does this need a mutex guard or is atomic sufficient?
        accel_burst_sample.active_label = g_record.load(std::memory_order_acquire); // reader = acquire

#endif // CALIBRATION_MODE
#endif // I2C_MOCK
        
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
    size_t tick_count = 0;

    sliding_window_t window; // should acquire the data for 1 window with that many pops n then increment by hop... 
    accel_burst_t temp; // placeholder for accel burst storage 
    /*
    // wait for first signal that we've reached the WINDOW_SAMPLES length in the buffer
    while(rb.get_count() < window.winLen){
        // don't have enough data
        continue;
    }
    */

#if CALIBRATION_MODE
    std::ofstream csv("accel_calib_data.csv");
    csv << "tick,x,y,z,active\n";
    size_t rows_written = 0;
#endif

    // BUILD FIRST WINDOW - do we need a mutex guard here?  i kinda dont think so because no one else is gonna be popping? and consumer/producer push/pop is handled intrinsically by semaphores in ringbuf class
    for(int i=0;i<window.winLen;i++){
        if(!rb.pop(&temp)){ // internally wait here (pop cmd is blocking)
            break; // throw error
        }
        else {
            // pop successful -> push into sliding window
            window.sliding_window.push(temp);
        }
    }

    while(!g_stop.load(std::memory_order_relaxed)){
        // emit window to feature extractor (DEEEP COPY)
        //featureExtractor.readin(window);
        // pop out half of window for 50% hop
        accel_burst_t discard{};
        for(size_t k=0;k<window.winHop;k++){
            window.sliding_window.pop(&discard); 
        }
        // dont think i need get count cuz pressure is handled intrinsically in ring buff
        /*
        // after the first time, we increment by hop size rather than window size (as long as we have hop size available in array, we can pull a window)
        while(rb.get_count() < window.winHop){
            continue;
        }
        */
        // we have enough to make a window from head by adding the hop amount 
        // keep the tail of the sliding window, overwrite the head (older) 
        for(size_t j=0;j<window.winHop;j++){
            if(!rb.pop(&temp)){
                break; // throw error
            }
            else {
                // pop successful -> push into sliding window
                window.sliding_window.push(temp);
                // each successful pop is something we've acquired from rb
#if CALIBRATION_MODE
                if((tick_count%120)==0){
                    LOG_ALWAYS(std::to_string(temp.tick) + " " + std::to_string(temp.x) + " " + std::to_string(temp.y) + " " + std::to_string(temp.z) + " " + std::to_string(temp.active_label) + "\n");
                }
                
                csv << temp.tick << ',' << temp.x << ',' << temp.y << ',' << temp.z << ','
                    << (temp.active_label ? 1 : 0) << '\n';
                rows_written++;
                if(rows_written % 500 == 0) { csv.flush(); } // flush every 500 rows for speed 
#endif
            }
        }
    }
    // exiting due to producer exiting means we need to close window rb
    window.sliding_window.close();
    rb.close();
#if CALIBRATION_MODE
    csv.flush();
    csv.close();
#endif
}

#if CALIBRATION_MODE
static inline bool is_center(uint16_t code){
    return code == KEY_ENTER || code == KEY_SPACE; // some images map center to SPACE instead of ENTER
}

void joystick_thread_fn(const char* devnode = "/dev/input/event4") {
    logger::tlabel = "joystick";
    // 1) Open joystick in non-blocking mod
    const int fd = ::open(devnode, O_RDONLY | O_CLOEXEC | O_NONBLOCK);
    if (fd < 0) {
      LOG_ALWAYS("open(" << devnode << ") failed: " << std::strerror(errno));
      return;
    }
    LOG_ALWAYS("opened " << devnode);

    // 2) Prepare a pollfd struct telling poll() what fd to watch and for what events -> fills revents on rtn
    // poll(2) can wait on *multiple* fds at once, so its API takes an ARRAY of pollfd.
    // We’re only watching ONE fd here, so we create an array of length 1.
    //   struct pollfd  (from <poll.h>):
    //     int   fd;       // which file descriptor to watch
    //     short events;   // what we want (e.g., POLLIN = “readable”)
    //     short revents;  // what actually happened (filled by kernel)
    struct pollfd fds[1];
    fds[0].fd = fd;
    fds[0].events = POLLIN;
    fds[0].revents = 0;

    // 3) For simple press debounce
    using clock = std::chrono::steady_clock;
    auto last_toggle = clock::now();
    const auto debounce = std::chrono::milliseconds(50);

    // poll()’s 3rd arg is a timeout in ms. With 500 ms we periodically wake up
    // to check g_stop; if we set it to -1, poll() would block “forever”.
    const int timeout_ms = 500;

    while(!g_stop.load(std::memory_order_relaxed)){

        // poll(fds, nfds, timeout_ms):
        //  - returns >0 if at least one fd is ready
        //  - returns  0 on timeout
        //  - returns <0 on error (often EINTR)
        const int num_ready = ::poll(/*fds=*/fds, /*nfds=*/1, /*timeout=*/timeout_ms);
        LOG_ALWAYS(std::to_string(num_ready) + " value of num_ready");
        if (num_ready == 0) {
            // Timeout: nothing to read; loop to re-check g_stop
            LOG_ALWAYS("if statement 1");
            continue;
        }
        if (num_ready < 0) {
            LOG_ALWAYS("if statement 2");
            if (errno == EINTR) continue; // interrupted by signal: try again
            std::cerr << "poll() failed: " << std::strerror(errno) << "\n";
            break;
        }

        // The kernel sets fds[0].revents to describe what happened on that fd.
        if (fds[0].revents & POLLIN) {
            LOG_ALWAYS("if statement 3");
          // fd is readable: drain all pending input_event structs.
          // struct input_event (from <linux/input.h>) describes one event:
          //   .type  (e.g., EV_KEY for key/button events)
          //   .code  (which key: KEY_LEFT, KEY_RIGHT, etc.)
          //   .value (0=release, 1=press, 2=auto-repeat)
          while(true) {
            input_event ev{}; // input_event struct is from linux/imput.h (evdev event format .type (EV_KEY), .code (KEY_), .value)
            LOG_ALWAYS(std::string("KEY code=") + std::to_string(ev.code) + " value=" + std::to_string(ev.value));
            ssize_t n = ::read(fd, &ev, sizeof(ev));

            if (n == -1) {
              // Non-blocking fd: EAGAIN/EWOULDBLOCK = no more events right now.
              if (errno == EAGAIN || errno == EWOULDBLOCK) break;
              std::cerr << "read() failed: " << std::strerror(errno) << "\n";
              break;
            }

            //if (ev.type != EV_KEY) continue;
            // value: 1 for press, 0 for release
            if(ev.value == 1 || ev.value == 0){
                auto now = clock::now();
                if(now - last_toggle >= debounce) { // guard
                    // toggle g_record atomic on
                    bool new_state = !g_record.load(std::memory_order_relaxed);
                    g_record.store(new_state, std::memory_order_release); // release is for writer; acquire is for reader
                    last_toggle = now;
                    LOG_ALWAYS(std::string("record = ") + (new_state ? "ON" : "OFF"));
                }
            } 
          }  
        }
    }
    ::close(fd);
    LOG_ALWAYS("closed");
}
#endif // CALIBRATION_MODE

int main() {
    LOG_ALWAYS("start (VERBOSE=" << logger::verbose() << ")");

    ringBuffer_C<accel_burst_t> ringBuf(RING_BUFFER_CAPACITY);

    // interrupt caused by SIGINT -> 'handle_singint' acts like ISR (callback handle)
    std::signal(SIGINT, handle_sigint);

    // START THREADS. We pass the ring buffer by reference (std::ref) becauase each thread needs the actual shared 'ringBuf' instance, not just a copy...
    // This builds a new thread that starts executing immediately, running producer_thread_rn in parallel with the main thread (same for cons)
    std::thread prod(producer_thread_fn,std::ref(ringBuf));
    std::thread cons(consumer_thread_fn,std::ref(ringBuf));
#if CALIBRATION_MODE
    std::thread joys(joystick_thread_fn,"/dev/input/event4");
#endif

    // Poll the atomic flag g_stop; keep sleep tiny so Ctrl-C feels instant
    while(g_stop.load(std::memory_order_acquire) == 0){
        std::this_thread::sleep_for(std::chrono::milliseconds{30});
    }

    // on system shutdown:
    // ctrl+c called...
    ringBuf.close();
    prod.join();
    cons.join(); // close "join" individual threads
#if CALIBRATION_MODE
    joys.join();
#endif
    return 0;
}

