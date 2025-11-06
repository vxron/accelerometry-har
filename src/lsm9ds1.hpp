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

// LSM9DS1 XG (accel) addresses + key regs
static constexpr uint8_t ADDR_XG        = 0x6A;   // slave address of LSM9DS1 chip on sense hat
static constexpr uint8_t WHO_AM_I_XG    = 0x0F;   // expect 0x68
static constexpr uint8_t CTRL_REG5_XL   = 0x1F;   // x,y,z axis enables
static constexpr uint8_t CTRL_REG6_XL   = 0x20;   // ODR/FS/BW for accel (sets ODR = 100Hz and FS = +-2G)
static constexpr uint8_t CTRL_REG8      = 0x22;   // IF_ADD_INC bit (so readBurst6() walks through XL -> ZH)
static constexpr uint8_t OUT_X_L_XL     = 0x28;   // accel burst start

class lsm9ds1_driver {
public:
    // Constructor (dev is shorthand for 'device file path' on Linux - i2c bus 'dev/i2c-1')
    lsm9ds1_driver(const char* dev_file_path, const uint8_t slave_addr); // should bind bus to address
     
    // Init: performs health checks; initializes all register settings on the sensor; returns 0 if successful
    int lsm9ds1_init();
    // Close: closes the sensor; returns 0 if successful (stops measurements)
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

lsm9ds1_driver(const char* dev_file_path, const uint8_t slave_addr) {
  bus_ = I2CBus(dev_file_path);
  i2c_addr_ = slave_addr;
  is_open_ = true;
}

int lsm9ds1_driver::lsm9ds1_init() {
  if(is_open_ == false){
    return 1; 
  }
  if (!bus_.I2Cok()) {
    std::cerr << "open /dev/i2c-1 failed\n";
    return 1;
  }

  // Set slave to the accel/gyro die
  if (bus_.setSlave(ADDR_XG) != 0) {
    std::cerr << "setSlave failed (check address with i2cdetect)\n";
    return 1;
  }

  // 1) WHO_AM_I check on reg 0x0F -> make sure we're on accel (not gyro) -> expect 0x68
  uint8_t who = 0;
  if (bus_.readReg8(WHO_AM_I_XG, &who) != 0) {
    std::cerr << "WHO_AM_I read failed\n";
    return 1;
  }
  std::cout << "WHO_AM_I_XG = 0x" << std::hex << int(who) << std::dec << "\n";
  if (who != 0x68) {
    std::cerr << "Unexpected WHO_AM_I (expected 0x68)\n";
  }

  // 2) Enable register auto-increment so bursts can step XL,XH,YL... (CTRL_REG8.IF_ADD_INC=1)
  if (bus.writeReg8(CTRL_REG8, 0x04) != 0) {  // IF_ADD_INC=1
    std::cerr << "CTRL_REG8 write failed\n";
    return 1;

  }
  // 3) Set ODR=100 Hz, FS=±2g (CTRL_REG6_XL = 0b011 << 5 = 0x60)
  if (bus.writeReg8(CTRL_REG6_XL, 0x60) != 0) { // ODR=100Hz, FS=±2g, default BW
    std::cerr << "CTRL_REG6_XL write failed\n";
    return 1;
  }

  // 4) Enable XYZ axes in CTRL_REG5_XL (bits 5:3 = 111 -> 0x38)
  if (bus.writeReg8(CTRL_REG5_XL, 0x38) != 0) { // Xen/Yen/Zen enable
    std::cerr << "CTRL_REG5_XL write failed\n";
    return 1;
  }
  return 0;
}

int lsm9ds1_driver::lsm9ds1_close() {
  is_open_ = false;
  return 0;
}

int lsm9ds1_driver::lsm9ds1_read_burst(uint8_t start_reg, accel_burst_t* dest){
  if(is_open_ == false) {
    return 1;
  }
  // Read to array  [xl xh yl yh zl zh]
  bus_.readBurst6(start_reg, dest->accel_burst);
  // when we want to add label, we can just check state in main upon creating this burst
  return 0;
}

int lsm9ds1_driver::lsm9ds1_read_single_register(uint8_t reg, uint8_t* dest){
  if(is_open_ == false) {
    return 1;
  }
  bus_.readReg8(reg, dest);
  // when we want to add label, we can just check state in main upon creating this burst
  return 0;
}