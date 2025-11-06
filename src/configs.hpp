#pragma once
#include <cstddef>

// central place for sizes used across headers
inline constexpr std::size_t RING_BUFFER_CAPACITY       = 250;  // ring buffer capacity
inline constexpr std::size_t WINDOW_SAMPLES    = RING_BUFFER_CAPACITY;     // e.g. 1.6 s @ ~119 Hz
inline constexpr std::size_t WINDOW_HOP        = RING_BUFFER_CAPACITY / 2; // 50% overlap
