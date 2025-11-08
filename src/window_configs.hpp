#pragma once
#include <cstddef>
#include "ringBuffer.hpp"

// central place for sizes used across headers
inline constexpr std::size_t RING_BUFFER_CAPACITY       = 600;  // ring buffer capacity
inline constexpr std::size_t WINDOW_SAMPLES    = RING_BUFFER_CAPACITY/3;     // e.g. 1.7 s @ ~119 Hz
inline constexpr std::size_t WINDOW_HOP        = RING_BUFFER_CAPACITY/6; // 50% overlap

struct sliding_window_t {
    // should take a number of accel_burst_t i believe
    size_t const winLen = WINDOW_SAMPLES; // period of about 200*8ms = 1.6s
    size_t const winHop = WINDOW_HOP; // amount to jump for next window
    ringBuffer_C<accel_burst_t> sliding_window{WINDOW_SAMPLES};
};