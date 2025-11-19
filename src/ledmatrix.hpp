#pragma once
#include "types.hpp"
#include <cstdint>

/* SenseHat's LED Matrix Architecture
- The 8x8 RGB matrix is NOT on the I2C bus directly; 
- The matrix is driven by LED2472G
- And this is driven by an ATTINY88 microcontroller which is on the I2C bus at 0x46
- Raspberry Pi ships kernel driver that exposes the matrix as a 64-pixel RGB565 framebuffer at /dev/fb1
*/

class LedMatrixDriver {
public:
    LedMatrixDriver();
    ~LedMatrixDriver();
    // Draw a colour + letter corresponding to the classified activity
    void display_class_on_matrix(classes_e class_label);
    void display_calibration_on_matrix(bool active_recording); // true if active block

private:
    int fb_fd_;  // file descriptor for /dev/fb1
    void open_dev_kernel();
    void close_dev_kernel();

    static uint16_t rgb888_to_rgb565(uint8_t r, uint8_t g, uint8_t b);
};