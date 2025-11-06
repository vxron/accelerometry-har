cmake_minimum_required(VERSION 3.20)
project(sensehat_fsm_demo LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Toggle mock vs linux HAL (default: mock on Windows)
option(USE_MOCK_HAL "Use mock I2C HAL" ON)
option(CALIBRATION_MODE "Enable calibration-mode joystick gating" OFF)

# --- Main app
add_executable(sensehat_fsm_demo
  src/main.cpp
  src/lsm9ds1.cpp
  src/logger.cpp        # <-- always include logger
)

target_include_directories(sensehat_fsm_demo PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# I2C_MOCK flag: means we're on WIN32; otherwise (real) has I2CMOCK = 0
if (WIN32 OR USE_MOCK_HAL)
  target_compile_definitions(sensehat_fsm_demo PRIVATE I2C_MOCK=1)
else()
  target_sources(sensehat_fsm_demo PRIVATE src/i2c_hal.cpp)
  target_compile_definitions(sensehat_fsm_demo PRIVATE I2C_MOCK=0)

  # --- Self-test utility (Linux/RPi only)
  add_executable(i2c_selftest
    src/i2c_selftest.cpp
    src/i2c_hal.cpp
    src/lsm9ds1.cpp
    src/logger.cpp
  )
  target_include_directories(i2c_selftest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_compile_definitions(i2c_selftest PRIVATE I2C_LINUX=1)
  target_link_libraries(i2c_selftest PRIVATE Threads::Threads)
endif()

if (CALIBRATION_MODE)
  target_compile_definitions(sensehat_fsm_demo PRIVATE CALIBRATION_MODE=1)
endif()

find_package(Threads REQUIRED)
target_link_libraries(sensehat_fsm_demo PRIVATE Threads::Threads)

# Helpful warnings in debug (GNU/Clang)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(sensehat_fsm_demo PRIVATE -Wall -Wextra -Wpedantic)
    if (TARGET i2c_selftest)
      target_compile_options(i2c_selftest PRIVATE -Wall -Wextra -Wpedantic)
    endif()
  endif()
endif()
