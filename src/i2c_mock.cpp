// i2c_mock.cpp
// Mock implementation for Windows / non-Linux builds

#include "i2c_hal.hpp"
#include <cstring>      // for std::memset

#ifdef _WIN32  // only use this mock on Windows

I2CBus::I2CBus(const char* dev)
{
    // On Windows we don't have a real /dev/i2c-* device.
    // Just pretend this "opened" fine.
    (void)dev;
    fd_ = -1;
}

I2CBus::~I2CBus()
{
    // Nothing to close in the mock
}

bool I2CBus::I2Cok() const noexcept
{
    // Always "OK" in the mock
    return true;
}

int I2CBus::setSlave(uint8_t addr) noexcept
{
    (void)addr;
    return 0;  // success
}

int I2CBus::writeReg8(uint8_t reg, uint8_t val) noexcept
{
    (void)reg;
    (void)val;
    return 0;  // success
}

int I2CBus::readReg8(uint8_t reg, uint8_t* dest) noexcept
{
    (void)reg;
    if (dest) {
        *dest = 0;  // dummy byte
    }
    return 0;  // success
}

int I2CBus::readBurst6(uint8_t startReg, uint8_t* dest) noexcept
{
    (void)startReg;
    if (dest) {
        std::memset(dest, 0, 6);  // 6 dummy bytes
    }
    return 0;  // success
}

#endif // _WIN32
