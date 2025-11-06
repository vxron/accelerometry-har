/*
* I2C HARDWARE ABSTRACTION LAYER FOR RASPBERRY PI COMMS
methods required:
* bool i2c_init(i2c buss, slave addy, i2c_obj_t* dest)
* void i2c_close(i2c_obj_t* obj)
* bool i2c_read_single_register (i2c bus, slave addy, register addy, dest*) calls i2cget -y <bus> <slave> <reg>
* bool i2c_read_burst (i2c bus, slave addy, register start addy, len bytes, dest*) calls i2ctransfer -y 1 w1@<slave> <start_reg> r<len>
notes:
- for Sense Hat, I2C address is 0x6a
- for accelerometer on hat, first read register is 0x28 (XL)
- for a chunk from the sensor, 6 bytes are returned [XL, XH, YL, YH, ZL, ZH] -> this is little Endian (lower bits 'L' first)
- ^^currently written, methods are command line calls (wrapper-based); inflexible; brittle; should be replaced with robust back-end eventually (or compile time modes)
*/

#pragma once
#if defined(_WIN32)
// If we ever compile this on Windows (mock build), map to _open/_close.
// But ideally, we don't compile the Linux HAL on Windows at all.
  #include <io.h>
  #include <fcntl.h>
  #define open  _open
  #define close _close
  #ifndef O_CLOEXEC
    #define O_CLOEXEC _O_NOINHERIT
  #endif
#else
  #include <fcntl.h>        // ::open, O_RDWR, O_CLOEXEC
  #include <unistd.h>       // ::close, ::read, ::write
  #include <sys/ioctl.h>    // ::ioctl
  #include <linux/i2c-dev.h>// I2C_SLAVE, I2C_RDWR, etc.
  #include <sys/types.h> // ssize_t
#endif

#include <stdbool.h>
#include <cstdint>
#include "logger.hpp"
#include "types.hpp"
#include <cerrno>      // errno
#include <cstring>     // strerror
#include <string_view>
#include <chrono>
#include <thread>

// Can hold separate I2CBus instances for joystick & accel, set the 2 diff slaves
// Public API

class I2CBus {
public:
    explicit I2CBus(const char* dev); // open; set fd_ or throw error
    ~I2CBus(); // close if open
    bool I2Cok() const noexcept; // if fd_ >= 0

    int setSlave(uint8_t addr) noexcept;
    int writeReg8(uint8_t reg, uint8_t val) noexcept;
    int readReg8(uint8_t reg, uint8_t* dest) noexcept;
    int readBurst6(uint8_t startReg, uint8_t* dest) noexcept; // dest should be array of bytes (e.g. sliding window, ringBuf, etc)

private:
    int fd_{-1};
};










