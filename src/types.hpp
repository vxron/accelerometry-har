#pragma once
#include <array>
#include <csignal>
#include <iostream>
#include <vector>
#include <cstdint>
#include "utils.hpp"

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
