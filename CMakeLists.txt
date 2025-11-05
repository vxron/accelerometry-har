cmake_minimum_required(VERSION 3.20)
project(sensehat_fsm_demo LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Toggle mock vs linux HAL (default: mock on Windows)
option(USE_MOCK_HAL "Use mock I2C HAL" ON)

add_executable(sensehat_fsm_demo
  src/main.cpp
  src/lsm9ds1.c
)

target_include_directories(sensehat_fsm_demo PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if (WIN32 OR USE_MOCK_HAL)
  target_sources(sensehat_fsm_demo PRIVATE src/i2c_hal.c)
  target_compile_definitions(sensehat_fsm_demo PRIVATE I2C_MOCK=1)
else()
  target_sources(sensehat_fsm_demo PRIVATE src/i2c_hal_linux.c)
  target_compile_definitions(sensehat_fsm_demo PRIVATE I2C_LINUX=1)
endif()

option(CALIBRATION_MODE "Enable calibration-mode joystick gating" OFF)
if (CALIBRATION_MODE)
  target_compile_definitions(sensehat_fsm_demo PRIVATE CALIBRATION_MODE=1)
endif()

find_package(Threads REQUIRED)
target_link_libraries(sensehat_fsm_demo PRIVATE Threads::Threads)

# Helpful warnings in debug
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  if (MSYS OR MINGW)
    target_compile_options(sensehat_fsm_demo PRIVATE -Wall -Wextra -pedantic)
  endif()
endif()