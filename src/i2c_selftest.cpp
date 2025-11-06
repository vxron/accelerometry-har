// Self-test using HAL + driver:
// - Constructs lsm9ds1_driver on /dev/i2c-1 @ 0x6A
// - Calls lsm9ds1_init()
// - Reads a few 6-byte accel bursts and prints signed counts (+ approx g)

#include <cstdint>
#include <iostream>
#include <iomanip>
#include <thread>
#include <chrono>

#include "i2c_hal.hpp"      // HAL
#include "lsm9ds1.hpp"      // driver
#include "types.hpp"        // for accel_burst_t (if defined there)

// If lsm9ds1_init() sets FS = ±4 g (CTRL_REG6_XL FS=10), use 0.000122f.
// If it's ±2 g, use 0.000061f. Keep this in sync with init config.
static constexpr float G_PER_LSB = 0.000122f; // adjust if FS changes

int main() {
    // Construct the driver (binds to /dev/i2c-1 and stores slave addr)
    lsm9ds1_driver imu("/dev/i2c-1", ADDR_XG);

    // Initialize accel: auto-increment, ODR/FS/BW, axes enable, etc.
    if (int rc = imu.lsm9ds1_init(); rc != 0) {
        std::cerr << "lsm9ds1_init failed (rc=" << rc << ")\n";
        return 1;
    }

    // Read/print a few bursts
    for (int k = 0; k < 50; ++k) {
        accel_burst_t burst{};
        if (int rc = imu.lsm9ds1_read_burst(OUT_X_L_XL, &burst); rc != 0) {
            std::cerr << "burst read failed (rc=" << rc << ")\n";
            return 1;
        }

        auto u16 = [](uint8_t lo, uint8_t hi) {
            return uint16_t((uint16_t(hi) << 8) | lo);
        };
        const int16_t x = int16_t(u16(burst.accel_burst[0], burst.accel_burst[1]));
        const int16_t y = int16_t(u16(burst.accel_burst[2], burst.accel_burst[3]));
        const int16_t z = int16_t(u16(burst.accel_burst[4], burst.accel_burst[5]));

        const float gx = x * G_PER_LSB;
        const float gy = y * G_PER_LSB;
        const float gz = z * G_PER_LSB;

        std::cout << "counts: x=" << std::setw(6) << x
                  << " y=" << std::setw(6) << y
                  << " z=" << std::setw(6) << z
                  << "   g: " << std::fixed << std::setprecision(6)
                  << gx << ", " << gy << ", " << gz << "\n";

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    (void)imu.lsm9ds1_close();
    return 0;
}
