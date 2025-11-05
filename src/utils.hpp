// for use as 'stopwatch' countning timer without limit -> set very high timer_dur
// for use as timeout/alarm -> set specific timeout time in 'timer_dur'

#pragma once
#include <chrono>

class sw_timer_t {
public:
	using clock_t = std::chrono::steady_clock;
	using dur_t = clock_t::duration;
	using timepoint_t = clock_t::time_point;
	
	// start timer
	bool start_timer(dur_t timer_dur) {
		if (is_started == true) {
			return false;
		}
		timer_start_time = clock_t::now();
		until = timer_start_time + timer_dur;
		is_started = true;
	}
	
	// stop (reset) timer and get elapsed time (for counting timers, 'stopwatch' application)
	std::chrono::milliseconds stop_timer() {
		auto ended_at = get_timer_value_ms();
		is_started = false;
		return ended_at;
	}
	
	// get current value of timer (elapsed time in ms)
	std::chrono::milliseconds get_timer_value_ms() {
		if(is_started==false) {
			return std::chrono::milliseconds{0};
		}
		return std::chrono::duration_cast<std::chrono::milliseconds>(clock_t::now() - timer_start_time);
	}
	
	// check if the timer has expired
	bool check_timer_expired() const {
		if(is_started && clock_t::now() >= until) {
			return true; // expired
		}
		else {
			return false; 
		}
	}
	
private:
	bool is_started = false;
	timepoint_t until {};
	timepoint_t timer_start_time {}; 
};
