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

#include <stdbool.h>
#include <stdint.h>