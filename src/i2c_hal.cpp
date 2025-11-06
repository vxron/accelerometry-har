#include "i2c_hal.hpp"

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

bool I2CBus::I2Cok() const noexcept{
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

     // ioctl tells linux kernel which SLAVE ADDRESS (addr) on that bus you want to talk to
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
        // ssize_t allows negative numbers to indicate failures
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