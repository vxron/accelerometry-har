/*
* DRIVER FOR LSM9DS1 ACC SENSOR ON PI HAT
methods required:
* lsm9ds1_init()
* lsm9ds1_read_burst()
* lsm9ds1_read_single_register()
* lsm9ds1_close()
* lsm9ds1_
dependencies:
* i2c_hal.h
*/

# pragma once
#include "i2c_hal.hpp"
#include <array>

// LSM9DS1 XG (accel) reg addresses 
static constexpr uint8_t ADDR_XG        = 0x6A;   // slave address of LSM9DS1 chip on sense hat
static constexpr uint8_t WHO_AM_I_XG    = 0x0F;   // expect 0x68
static constexpr uint8_t CTRL_REG5_XL   = 0x1F;   // decimation bits & x,y,z enable
static constexpr uint8_t CTRL_REG6_XL   = 0x20;   // toggles between accel/powerdown modes; ODR/FS/BW for accel; AA LPF
static constexpr uint8_t CTRL_REG8      = 0x22;   // IF_ADD_INC bit (so readBurst6() walks through XL -> ZH)
static constexpr uint8_t OUT_X_L_XL     = 0x28;   // accel burst start


class lsm9ds1_driver {
public:
    // Constructor (dev is shorthand for 'device file path' on Linux - i2c bus 'dev/i2c-1')
    lsm9ds1_driver(const char* dev_file_path, const uint8_t slave_addr); // should bind bus to address
     
    // Init: performs health checks; enters accel state; initializes all register settings on the sensor; returns 0 if successful
    int lsm9ds1_init();
    // Close: closes the sensor; powers down; returns 0 if successful (stops measurements)
    int lsm9ds1_close();

    // Operations: Can only be done if sensor is open
    // (1) read 6-byte burst from start_reg then store in ptr to dest (type accel_burst_t); returns 0 on success
    int lsm9ds1_read_burst(uint8_t start_reg, accel_burst_t* dest);
    // (2) read single reg, write to dest ptr (uint8_t)
    int lsm9ds1_read_single_register(uint8_t reg, uint8_t* dest);

private:
    I2CBus bus_;
    uint8_t i2c_addr_;
    bool is_open_;
};