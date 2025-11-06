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


I2CBus::I2CBus(const char* dev) {
    if(!dev || std::string_view(dev).empty()){
        LOG_ALWAYS("I2CBus: empty device path");
        fd_ = -1;
        return;
    }
    
    // try to open fd_ (handle for I2C device file on linux, most likely /dev/i2c-1)
    fd_ = ::open(dev, O_RDWR | O_CLOEXEC);
    if (fd_ < 0) {
        LOG_ALWAYS(std::string("I2CBus: open(") + dev + ") failed: " + std::strerror(errno));
    } else {
        LOG_ALWAYS(std::string("I2CBus: opened ") + dev + " fd=" + std::to_string(fd_));
    }
}

I2CBus::~I2CBus() noexcept {
    if(fd_ >= 0){
        ::close(fd_);
        LOG_ALWAYS("I2CBus: closed fd");
        fd_ = -1;
    }
}

bool I2CBus::I2Cok() const{
    if(fd_ >= 0) {
        return true;
    }
    else {
        return false;
    }
}

int I2CBus::setSlave(uint8_t addr) noexcept {
    if(!I2Cok()) {
        return -ENODEV;
    }

     // Call the kernel to set the active 7-bit I2C address for this fd
    int rc = ::ioctl(fd_, I2C_SLAVE, addr);
    
    if (rc == 0) {
        return 0;
    }

    return -errno;
}

int I2CBus::writeReg8(uint8_t reg, uint8_t val) noexcept {
    if(!I2Cok()){
        return -errno;
    }
    // payload = 2 bytes [reg, val]
    uint8_t buf[2] = {reg, val};
    for (int tries = 0; tries < 3; ++tries) {
        ssize_t n = ::write(fd_, buf, 2);
        if (n == 2) return 0;               // success
        if (n < 0 && (errno == EINTR || errno == EAGAIN)) continue; // retry
        // Short write or other error
        return (n < 0) ? -errno : -EIO;
    }
    return -EIO; // too many retries
}

// convert to signed later -> just raw bytes for now
int I2CBus::readReg8(uint8_t reg, uint8_t* dest) noexcept {
    if (!I2Cok() || dest == nullptr) {
        return -ENODEV;
    }
    
    // Step 1: write the register address (pointer set)
    for (int tries = 0; tries < 3; ++tries) {
        ssize_t n = ::write(fd_, &reg, 1);
        if (n == 1) break; // success
        if (n < 0 && (errno == EINTR || errno == EAGAIN)) continue;
        return (n < 0) ? -errno : -EIO;
    }

    // Step 2: read one byte from that register
    for (int tries = 0; tries < 3; ++tries) {
        ssize_t n = ::read(fd_, dest, 1);
        if (n == 1) return 0;
        if (n < 0 && (errno == EINTR || errno == EAGAIN)) continue;
        return (n < 0) ? -errno : -EIO;
    }
    return -EIO;
   
}

int I2CBus::readBurst6(uint8_t startReg, uint8_t* dest) noexcept {
    if (!I2Cok() || dest == nullptr) {
        return -ENODEV;
    }
    
    // Step 1: write start register
    for (int tries = 0; tries < 3; ++tries) {
        ssize_t n = ::write(fd_, &startReg, 1);
        if (n == 1) break;
        if (n < 0 && (errno == EINTR || errno == EAGAIN)) continue;
        return (n < 0) ? -errno : -EIO;
    }

    // Step 2: read 6 bytes in one go (since accel_burst is 6 bytes)
    for (int tries = 0; tries < 3; ++tries) {
        ssize_t n = ::read(fd_, dest, LSM9DS1_NUM_BYTES_PER_BURST);
        if (n == 6) return 0;
        if (n < 0 && (errno == EINTR || errno == EAGAIN)) continue;
        return (n < 0) ? -errno : -EIO;
    }

    return -EIO;
    // requires IF_ADD_INC=1 in ctrl_reg8 so internal address auto-increments across x,y,z
}










