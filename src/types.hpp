#pragma once
#include <array>
#include <csignal>
#include <iostream>
#include <vector>
#include <cstdint>
#include "utils.hpp"
#include "configs.hpp"
#include "ringBuffer.hpp"

constexpr uint8_t LSM9DS1_SLAVE_I2C_ADDRESS = 0x6A; // 0x6B on some systems
constexpr uint8_t LSM9DS1_ACCEL_OUT_BASE_REG = 0x28; // starts on accel XL (little Endian)
constexpr size_t LSM9DS1_NUM_BYTES_PER_BURST = 6; // [xl xh yl yh zl zh]
constexpr size_t LSM9DS1_WORD_LENGTH_BYTES = 1; // 1 byte per register

enum classes_e {
    CLASS_WALKING,
    CLASS_SITTING,
    CLASS_STANDING,
    CLASS_TURNING_ON_SPOT,
};

struct accel_burst_t {
    int16_t x;
    int16_t y;
    int16_t z;
    uint32_t tick; // monotonic sample index
#ifdef CALIBRATION_MODE
    bool active_label; // should obtain from joystick state; 1 means we're in active block
#endif
};

struct sliding_window_t {
    // should take a number of accel_burst_t i believe
    size_t const winLen = WINDOW_SAMPLES; // period of about 200*8ms = 1.6s
    size_t const winHop = WINDOW_HOP; // amount to jump for next window
    ringBuffer_C<accel_burst_t> sliding_window{WINDOW_SAMPLES};
};
