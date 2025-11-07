#pragma once
#include <cstddef>

// central place for sizes used across headers
inline constexpr std::size_t RING_BUFFER_CAPACITY       = 600;  // ring buffer capacity
inline constexpr std::size_t WINDOW_SAMPLES    = RING_BUFFER_CAPACITY/3;     // e.g. 1.7 s @ ~119 Hz
inline constexpr std::size_t WINDOW_HOP        = RING_BUFFER_CAPACITY/6; // 50% overlap
