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
#include <cerrno>
#include <cstring>
#include "logger.hpp"
#include <optional>
#include <fstream>
#include "ledmatrix.hpp"
#include "SWTimer.hpp"

#if !CALIBRATION_MODE
#include "FeatureExtractor.hpp"
#include <onnxruntime_cxx_api.h>
#endif

#if !I2C_MOCK && defined(__linux__)
// raspberry pi path
#include <linux/input.h> // Linux evdev: struct input_event, EV_* types, KEY_* codes
#include <fcntl.h> // open(), O_RDONLY, O_NONBLOCK, O_CLOEXEC
#include <unistd.h> // read(), close()
#include <poll.h> // poll(), struct pollfd, POLLIN, POLLERR, POLLHUP
#endif

#ifdef close
#undef close
#endif

// Global "please stop" flag set by Ctrl+C (SIGINT) to shut down cleanly
static std::atomic<bool> g_stop{false};

// Activity order (cyclical for labelling)
static constexpr std::array<classes_e, 4> ACTIVITY_ORDER = {CLASS_STANDING, CLASS_WALKING, CLASS_SITTING, CLASS_TURNING_ON_SPOT};

#if CALIBRATION_MODE
static std::atomic<classes_e> g_record{CLASS_UNKNOWN}; // moves through ACTIVITY_ORDER by joystick presses
#else
static std::atomic<int> g_last_class_id{-1}; 
static std::atomic<int> g_last_enum{-1}; // store enum as int
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

#if CALIBRATION_MODE
void consumer_thread_fn(ringBuffer_C<accel_burst_t>& rb){
#else 
void consumer_thread_fn(ringBuffer_C<accel_burst_t>& rb, OnnxConfigs_S& cfgs, OnnxClassifier_C& classifier){
#endif
    using namespace std::chrono_literals;
    logger::tlabel = "consumer";
    LOG_ALWAYS("consumer start");

#if !CALIBRATION_MODE
    // lambda helper func for int id to class (takes ref to cfgs)
    auto intToClass = [&cfgs](int decision) -> classes_e {
        if (decision == cfgs.sit_id) {
            return CLASS_SITTING;
        } else if (decision == cfgs.stand_id) {
            return CLASS_STANDING;
        } else if (decision == cfgs.walk_id) {
            return CLASS_WALKING;
        } else if (decision == cfgs.turn_id) {
            return CLASS_TURNING_ON_SPOT;
        } else {
            return CLASS_UNKNOWN;
        }
    };
#endif
    // lambda helper func for int id to class (takes ref to cfgs)
    auto classToString = [](classes_e cls) -> std::string {
        if (cls == CLASS_SITTING) {
            return "sitting";
        } else if (cls == CLASS_STANDING) {
            return "standing";
        } else if (cls == CLASS_WALKING) {
            return "walking";
        } else if (cls == CLASS_TURNING_ON_SPOT) {
            return "turning_on_spot";
        } else if (cls == CLASS_UNKNOWN) {
            return "unknown";
        } else {
            // just in case new enums get added and this isn't updated
            return "unmapped_class";
        }
    };

    size_t tick_count = 0;
    sliding_window_t window; // should acquire the data for 1 window with that many pops n then increment by hop... 
    accel_burst_t temp; // placeholder for accel burst storage
    LedMatrixDriver ledMatrix;
    ledMatrix.display_welcome_message();

#if !CALIBRATION_MODE 
    FeatureVector_C ftrVec(cfgs);
    
    // (A) CSV for runtime decisions
    std::ofstream decisions_csv("rt_decisions.csv");
    decisions_csv << "window_idx,last_tick,raw_id,raw_label,stable_id,stable_label,"
              << "raw_changed,stable_changed,suppressed_change\n";

    
    // (B) CSV for features + raw window (for Python verification / debug)
    std::ofstream ftr_raw_csv("rt_features_with_raw.csv");
    ftr_raw_csv << "window_idx";
    for (const auto& name : cfgs.feat_names) {
        ftr_raw_csv << ',' << name;
    }
    // add x0,y0,z0,...,xN-1,yN-1,zN-1 (major-interleaved format)
    for (size_t i = 0; i < window.winLen; ++i) {
        ftr_raw_csv << ",x" << i << ",y" << i << ",z" << i;
    }
    ftr_raw_csv << '\n';

    size_t window_idx = 0;
    // Temporal smoothing against noisy transitions (timer guard)
    int prev_raw_id    = -1; // last raw classifier output
    int prev_stable_id = -1; // last committed stable state
    int pending_id     = -1; // candidate new state
    bool pending_active = false;
    SW_Timer_C debounceTimer;
    constexpr auto GUARD_DURATION = std::chrono::milliseconds{1000}; // 1s guard
#endif

#if CALIBRATION_MODE
    
    // (C) Raw (unsegmented) data log
    std::ofstream csv("accel_calib_data.csv");
    csv << "tick,x,y,z,label_id,label_name\n";
    size_t rows_written = 0;

    // (D) per-window training set (raw windows + activity label from joystick)
    std::ofstream train_csv("training_windows.csv");
    train_csv << "window_idx"
              << ",label_id"
              << ",label_name";
    for (size_t i = 0; i < window.winLen; ++i) {
        train_csv << ",x" << i << ",y" << i << ",z" << i;
    }
    train_csv << '\n';

    size_t window_idx = 0;

#endif

    // BUILD FIRST WINDOW - consumer/producer push/pop is handled intrinsically by semaphores in ringbuf class
    for(int i=0;i<window.winLen;i++){
        if(!rb.pop(&temp)){ // internally wait here (pop cmd is blocking)
            // exit due to error
            g_stop.store(true, std::memory_order_relaxed);
            break;
        }
        else {
            // pop successful -> push into sliding window
            window.sliding_window.push(temp);
        }
    }

    while(!g_stop.load(std::memory_order_relaxed)){

#if !CALIBRATION_MODE
        // (1) emit window to feature extractor
        window.feature_vector = ftrVec.writeFeatureVector(window);

        // (2) take non-destructive snapshot of the raw samples in this window
        std::vector<accel_burst_t> windowSnapshot;
        window.getWindowSnapshot(&windowSnapshot);

        // (3) classify
        int decision = classifier.classify(window.feature_vector);

        // Flags for logging
        bool raw_changed        = false;
        bool stable_changed     = false;
        bool suppressed_change  = false;

        // Keep copy of last STABLE (i.e. CONFIRMED decision state)
        int stable_id     = prev_stable_id;

        // accept first-ever window immediately
        if(prev_stable_id < 0){
            stable_id = decision;
            prev_stable_id = decision;
            pending_id = -1;
            pending_active = false;
            raw_changed = true;
            stable_changed = true;
        }

        // (4) debounce guard for FP transitions
        else {
            if(decision != prev_raw_id){
                raw_changed = true;
            }
            else {
                raw_changed = false;
            }
            
            if(decision == prev_stable_id){
                // went back to stable state (cancel any pending transition) (i.e. "failed" to pass debounce)
                stable_id = prev_stable_id;
                pending_id = -1;
                pending_active = false;
                debounceTimer.stop_timer();
            }

            else {
                // new raw state different from current stable... 
                if(!pending_active || decision != pending_id){
                    // new candidate transition (either we're not alr pending, or we're alr pending but a diff decision was pending)
                    pending_id = decision;
                    pending_active = true;
                    suppressed_change = true;
                    stable_id = prev_stable_id;
                    if(debounceTimer.is_started()){
                        debounceTimer.stop_timer();
                    }
                    debounceTimer.start_timer(GUARD_DURATION);
                }

                else{
                    // pending is active with sustained pending_id...
                    // check timer expired
                    if(debounceTimer.check_timer_expired()){
                        debounceTimer.stop_timer();
                        // successful debounce, can transition
                        stable_id = decision;
                        prev_stable_id = decision;
                        pending_id = -1;
                        pending_active = false;
                        stable_changed = true;
                    }
                    else {
                        // still waiting
                        stable_id = prev_stable_id;
                        suppressed_change = true;
                    }
                }

            }
            
        } // else

        // (5) assign decision and raw decision to window
        window.rawDecision = intToClass(decision);
        window.decision = intToClass(stable_id);
        std::string stable_label  = classToString(window.decision);
        std::string raw_label   = classToString(window.rawDecision);
        window.stuckInDebounce = pending_active || (stable_id != decision);

        prev_raw_id = decision; // to detect next transition

        // visualizations in real time to show whats happening at all svm levels (preclassifier, 1 and 2)
        // use STABLE (confirmed) states for this 
        g_last_class_id.store(stable_id, std::memory_order_release);
        g_last_enum.store(static_cast<int>(window.decision), std::memory_order_release);
        
        // (6) increment window index for this iter
        window_idx++;

        // (7a) Log features + raw window (rt_features_with_raw.csv) 
        ftr_raw_csv << window_idx;
        for (float v : window.feature_vector) {
            ftr_raw_csv << ',' << v;
        }
        for (const auto& s : windowSnapshot) {
            ftr_raw_csv << ',' << s.x << ',' << s.y << ',' << s.z;
        }
        ftr_raw_csv << '\n';

        if (window_idx % 50 == 0) {
            ftr_raw_csv.flush();
        }

        // (7b) DECISIONS CSV LOGGING -> suppressed changes vs true based on debounce guard
        // `temp` always holds the most recent sample pushed into the window
        // so we can use its tick to indicate num accel_burst_t in the window now
        decisions_csv << window_idx << ','
                      << temp.tick << ','           // last sample tick in window
                      << decision << ','            // raw_id
                      << raw_label << ','           // raw_label
                      << stable_id << ','           // stable_id
                      << stable_label << ','        // stable_label
                      << (raw_changed       ? 1 : 0) << ','  // raw_changed
                      << (stable_changed    ? 1 : 0) << ','  // stable_changed
                      << (suppressed_change ? 1 : 0)         // suppressed_change
                      << '\n';

        if (window_idx % 100 == 0) {
            decisions_csv.flush();
        }
        
        // (8) REAL TIME DISPLAY ON MATRIX
        ledMatrix.display_class_on_matrix(window.decision);
#else 
        // (1) snapshot current window
        std::vector<accel_burst_t> windowSnapshot;
        window.getWindowSnapshot(windowSnapshot);

        // (2) decide window label from sample-level active_label
        classes_e winLabel = CLASS_UNKNOWN;
        bool first = true;
        bool consistent = true;

        for (const auto& s : windowSnapshot) {
            if (first) {
                winLabel = s.active_label;
                first = false;
            } else {
                if (s.active_label != winLabel) {
                    consistent = false;
                    break;
                }
            }
        }

        // (3) only log training windows if label is consistent and not UNKNOWN
        if (consistent && winLabel != CLASS_UNKNOWN) {
            window_idx++;
            std::string label_name = classToString(winLabel);

            train_csv << window_idx
                      << ',' << static_cast<int>(winLabel)
                      << ',' << label_name;

            for (const auto& s : windowSnapshot) {
                train_csv << ',' << s.x << ',' << s.y << ',' << s.z;
            }
            train_csv << '\n';

            if (window_idx % 50 == 0) {
                train_csv.flush();
            }
        }

        // (4) calib mode display for this cycle
        ledMatrix.display_calibration_on_matrix(g_record.load(std::memory_order_acquire));
#endif

        // (9) pop out half of window for 50% hop
        accel_burst_t discard{};
        for(size_t k=0;k<window.winHop;k++){
            window.sliding_window.pop(&discard); 
        }
        // we have enough to make a window from head by adding the hop amount 
        // keep the tail of the sliding window, overwrite the head (older) 
        for(size_t j=0;j<window.winHop;j++){
            if(!rb.pop(&temp)){
                // exit due to error
                g_stop.store(true, std::memory_order_relaxed);
                break;
            }
            else {
                // (10) pop successful -> push into sliding window
                window.sliding_window.push(temp);
                // each successful pop is something we've acquired from rb
#if CALIBRATION_MODE
                // (7C)
                tick_count++;
                if((tick_count%2000)==0){
                    LOG_ALWAYS(std::to_string(temp.tick) + " " 
                               + std::to_string(temp.x) + " " 
                               + std::to_string(temp.y) + " " 
                               + std::to_string(temp.z) + " " 
                               + classToString(temp.active_label) + "\n");
                }

                csv << temp.tick << ',' 
                    << temp.x << ',' << temp.y << ',' << temp.z << ','
                    << static_cast<int>(temp.active_label) << ','
                    << classToString(temp.active_label) << '\n';
            
                rows_written++;
                if(rows_written % 500 == 0) { csv.flush(); }
            
                // (7D)


#endif
            }
        }
    }
    // (11) exiting due to producer exiting means we need to close window rb
    ledMatrix.close_and_reset();
    window.sliding_window.close();
    rb.close();
#if CALIBRATION_MODE
    csv.flush();
    csv.close();
    train_csv.flush();
#else
    decisions_csv.flush();
    ftr_raw_csv.flush();
#endif
}

#if CALIBRATION_MODE && !I2C_MOCK && defined(__linux__)
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
    int activity_idx = -1;
    size_t num_activities = ACTIVITY_ORDER.size();

    while(!g_stop.load(std::memory_order_relaxed)){
        
        // poll(fds, nfds, timeout_ms):
        //  - returns >0 if at least one fd is ready
        //  - returns  0 on timeout
        //  - returns <0 on error (often EINTR)
        const int num_ready = ::poll(/*fds=*/fds, /*nfds=*/1, /*timeout=*/timeout_ms);
        LOG_ALWAYS(std::to_string(num_ready) + " value of num_ready");
        if (num_ready == 0) {
            // Timeout: nothing to read; loop to re-check g_stop
            continue;
        }
        if (num_ready < 0) {
            if (errno == EINTR) continue; // interrupted by signal: try again
            std::cerr << "poll() failed: " << std::strerror(errno) << "\n";
            break;
        }

        // The kernel sets fds[0].revents to describe what happened on that fd.
        if (fds[0].revents & POLLIN) {
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

            if (ev.type != EV_KEY) continue;
            // value: 1 for press, 0 for release -> toggle every release
            // move thru cyclical activity orders
            if(ev.value == 1){
                auto now = clock::now();
                if(now - last_toggle >= debounce) { // guard
                    // move through activity cycle
                    activity_idx++;
                    
                    // wraparound if necessary
                    if(activity_idx >= num_activities){
                        activity_idx = 0;
                    }

                    // record new state
                    classes_e new_state = ACTIVITY_ORDER[activity_idx];
                    g_record.store(new_state, std::memory_order_release); // release is for writer; acquire is for reader
                    last_toggle = now;
                    LOG_ALWAYS(std::string("record = ") + (enumToString(new_state)));
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
    try {
    LOG_ALWAYS("start (VERBOSE=" << logger::verbose() << ")");
    
    ringBuffer_C<accel_burst_t> ringBuf(RING_BUFFER_CAPACITY);

#if !CALIBRATION_MODE
#ifdef MODEL_META_JSON_PATH
    std::string jsonPath = MODEL_META_JSON_PATH;
#else
    std::string jsonPath = "src/models/veron/har_hierarchy_meta.json";
#endif

    OnnxClassifier_C classifier(jsonPath);
    OnnxConfigs_S cfgs = classifier.getConfigs();
#endif // !CALIBRATION_MODE

    // interrupt caused by SIGINT -> 'handle_singint' acts like ISR (callback handle)
    std::signal(SIGINT, handle_sigint);

    // START THREADS. We pass the ring buffer by reference (std::ref) becauase each thread needs the actual shared 'ringBuf' instance, not just a copy...
    // This builds a new thread that starts executing immediately, running producer_thread_rn in parallel with the main thread (same for cons)
    std::thread prod(producer_thread_fn,std::ref(ringBuf));
#if !CALIBRATION_MODE
    std::thread cons(consumer_thread_fn,std::ref(ringBuf), std::ref(cfgs), std::ref(classifier));
#else
    std::thread cons(consumer_thread_fn,std::ref(ringBuf));
#endif
#if CALIBRATION_MODE && !I2C_MOCK && defined(__linux__)
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
    } catch (const std::exception& ex) {
        LOG_ALWAYS("FATAL: unhandled exception in main: " << ex.what());
        return 1;
    } catch (...) {
        LOG_ALWAYS("FATAL: unknown exception in main");
        return 1;
    }
}


