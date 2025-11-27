#pragma once
#include <array>
#include <csignal>
#include <iostream>
#include <vector>
#include <cstdint>

constexpr uint8_t LSM9DS1_SLAVE_I2C_ADDRESS = 0x6A; // 0x6B on some systems
constexpr uint8_t LSM9DS1_ACCEL_OUT_BASE_REG = 0x28; // starts on accel XL (little Endian)
constexpr size_t LSM9DS1_NUM_BYTES_PER_BURST = 6; // [xl xh yl yh zl zh]
constexpr size_t LSM9DS1_WORD_LENGTH_BYTES = 1; // 1 byte per register
constexpr size_t FS_HZ = 119.0; // sampling frequency


enum classes_e {
    CLASS_WALKING,
    CLASS_SITTING,
    CLASS_STANDING,
    CLASS_TURNING_ON_SPOT,
    CLASS_UNKNOWN
};

std::string enumToString(classes_e enumVal){
    switch(enumVal){
        case(CLASS_WALKING):
            return "walk";
        case(CLASS_SITTING):
            return "sit";
        case(CLASS_TURNING_ON_SPOT):
            return "turn";
        case(CLASS_STANDING):
            return "stand";
        default:
            return "unknown";
    }
}

struct accel_burst_t {
    int16_t x;
    int16_t y;
    int16_t z;
    uint32_t tick; // monotonic sample index
#ifdef CALIBRATION_MODE
    classes_e active_label; // should obtain from joystick state; 1 means we're in active block
#endif
};

#if !CALIBRATION_MODE
struct RTDecisionSnapShot {
    int raw_id = -1;
    classes_e decision = CLASS_UNKNOWN;
};
#endif