#include "lsm9ds1.hpp"

lsm9ds1_driver::lsm9ds1_driver(const char* dev_file_path, const uint8_t slave_addr)
: bus_(dev_file_path), i2c_addr_(slave_addr), is_open_(false) {}

int lsm9ds1_driver::lsm9ds1_init() {
  is_open_ = true;
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
  // All other values should be default (0)
  if (bus_.writeReg8(CTRL_REG8, 0x04) != 0) {  // IF_ADD_INC=1
    std::cerr << "CTRL_REG8 write failed\n";
    return 1;
  }

// 3) What we want to write to CTRL_REG6_XL: 0b01110011
    // ODR_XL[2:0], FS_XL[4:3], BW
    // output data rate (ODR)  SAMPLING RATE stays at 119Hz thus 011
    // FS_XL: (00: ±2g; 01: ±16 g; 10: ±4 g; 11: ±8 g) -> bits 3,4 should be 10 (choose +/- 4g for best op to detect jumping)
    // will do basic anti-aliasing LPF bandwidth = 50 Hz (remove above 50 Hz) -> last 2 bits 11
  if (bus_.writeReg8(CTRL_REG6_XL, 0b01110011) != 0) {
    std::cerr << "CTRL_REG6_XL write failed\n";
    return 1;
  }

  // 4) What we want to write to CTRL_REG5_XL: 0b00111000
    // x,y,z axis enables bits [2:4] amd decimation bits [0:1]
    // keep decimation bits at 0 (no downsampling, keep ODR 119Hz as sampling freq)
  if (bus_.writeReg8(CTRL_REG5_XL, 0b00111000) != 0) {
    std::cerr << "CTRL_REG5_XL write failed\n";
    return 1;
  }

  return 0;
}

int lsm9ds1_driver::lsm9ds1_close() {
  is_open_ = false;
  // TODO: implement power down on CTRL_REG6_XL
  return 0;
}

int lsm9ds1_driver::lsm9ds1_read_burst(uint8_t start_reg, accel_burst_t* dest){
  if(is_open_ == false) {
    return 1;
  }
  // Read to array  [xl xh yl yh zl zh]
  std::array<uint8_t, 6> temp_arr;
  bus_.readBurst6(start_reg, temp_arr.data()); // .data() turns std array to ptr
  // create accel_burst_t obj
  // first make them 16 bit uints so our shifting workings, then convert to signed
  dest->x = int16_t(uint16_t(temp_arr[0]) | uint16_t(temp_arr[1]<<8));
  dest->y = int16_t(uint16_t(temp_arr[2]) | uint16_t(temp_arr[3]<<8));
  dest->z = int16_t(uint16_t(temp_arr[4]) | uint16_t(temp_arr[5]<<8));
  dest->tick = dest->tick + 1;
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