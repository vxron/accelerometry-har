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
notes:
*/

# pragma once
#include "i2c_hal.h"