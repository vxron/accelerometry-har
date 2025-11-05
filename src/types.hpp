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
    CLASS_NO_LABEL,
};

enum joystick_state_e {
    JOYSTICK_STATE_PRESSED_CONFIRMED,
    JOYSTICK_STATE_RELEASED_CONFIRMED,
    JOYSTICK_STATE_UNQUALIFIED,
    JOYSTICK_STATE_FAULT,
    JOYSTICK_STATE_PRESSED_WAITING_DEBOUNCE,
    JOYSTICK_STATE_RELEASED_WAITING_DEBOUNCE,
};

enum joystick_event_e {
    JOYSTICK_EVENT_PRESS,
    JOYSTICK_EVENT_RELEASE,
    JOYSTICK_EVENT_DEBOUNCE_TIMEOUT
};

struct accel_burst_t {
    size_t burstLen_bytes = LSM9DS1_NUM_BYTES_PER_BURST;
    std::array<uint8_t, LSM9DS1_NUM_BYTES_PER_BURST> accel_burst{};
    classes_e burst_label; // 0 in RUN mode
};

struct labeled_sliding_window_t {
    // nothn yet
};

// dk if i need this :
struct accel_sample_s {
    int16_t x;
    int16_t y;
    int16_t z;
};

struct joystick_s {
    joystick_state_e state;
    joystick_state_e prevState;
    sw_timer_t debounceTimer;
};