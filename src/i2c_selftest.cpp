#include <array>
#include <cstdint>
#include <iostream>
#include <thread>
#include <chrono>

// Include your HAL header (or just forward declare if your class lives in a .cpp)
#include "i2c_hal.hpp"

// LSM9DS1 XG (accel) addresses + key regs
static constexpr uint8_t ADDR_XG        = 0x6A;    // i2cdetect shows that
static constexpr uint8_t WHO_AM_I_XG    = 0x0F;   // expect 0x68
static constexpr uint8_t CTRL_REG5_XL   = 0x1F;   // x,y,z axis enables
static constexpr uint8_t CTRL_REG6_XL   = 0x20;   // ODR/FS/BW for accel (sets ODR = 100Hz and FS = +-2G)
static constexpr uint8_t CTRL_REG8      = 0x22;   // IF_ADD_INC bit (so readBurst6() walks through XL -> ZH)
static constexpr uint8_t OUT_X_L_XL     = 0x28;   // accel burst start

int main() {
  I2CBus bus("/dev/i2c-1");
  if (!bus.I2Cok()) {
    std::cerr << "open /dev/i2c-1 failed\n";
    return 1;
  }

  // Set slave to the accel/gyro die
  if (bus.setSlave(ADDR_XG) != 0) {
    std::cerr << "setSlave failed (check address with i2cdetect)\n";
    return 1;
  }

  // 1) WHO_AM_I check
  uint8_t who = 0;
  if (bus.readReg8(WHO_AM_I_XG, &who) != 0) {
    std::cerr << "WHO_AM_I read failed\n";
    return 1;
  }
  std::cout << "WHO_AM_I_XG = 0x" << std::hex << int(who) << std::dec << "\n";
  if (who != 0x68) {
    std::cerr << "Unexpected WHO_AM_I (expected 0x68)\n";
  }

  // 2) Minimal accel init:
  //    - Enable register auto-increment so bursts walk XL→XH→YL→... (CTRL_REG8.IF_ADD_INC=1)
  //    - Set ODR=100 Hz, FS=±2g (CTRL_REG6_XL = 0b011 << 5 = 0x60)
  //    - Enable XYZ axes in CTRL_REG5_XL (bits 5:3 = 111 -> 0x38)
  if (bus.writeReg8(CTRL_REG8, 0x04) != 0) {  // IF_ADD_INC=1
    std::cerr << "CTRL_REG8 write failed\n";
    return 1;
  }
  if (bus.writeReg8(CTRL_REG6_XL, 0x60) != 0) { // ODR=100Hz, FS=±2g, default BW
    std::cerr << "CTRL_REG6_XL write failed\n";
    return 1;
  }
  if (bus.writeReg8(CTRL_REG5_XL, 0x38) != 0) { // Xen/Yen/Zen enable
    std::cerr << "CTRL_REG5_XL write failed\n";
    return 1;
  }

  // 3) Read a few bursts and print signed counts + approx g
  for (int k = 0; k < 50; ++k) {
    uint8_t raw[6] = {0};
    if (bus.readBurst6(OUT_X_L_XL, raw) != 0) {
      std::cerr << "burst read failed\n";
      return 1;
    }

    auto u16 = [](uint8_t lo, uint8_t hi){ return uint16_t( (uint16_t(hi)<<8) | lo ); };
    int16_t x = int16_t(u16(raw[0], raw[1]));
    int16_t y = int16_t(u16(raw[2], raw[3]));
    int16_t z = int16_t(u16(raw[4], raw[5]));

    // For ±2g, sensitivity ≈ 0.000061 g/LSB
    float gx = x * 0.000061f;
    float gy = y * 0.000061f;
    float gz = z * 0.000061f;

    std::cout << "counts: x=" << x << " y=" << y << " z=" << z
              << "   g: " << gx << ", " << gy << ", " << gz << "\n";

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  return 0;
}
