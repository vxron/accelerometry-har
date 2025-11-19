cmake_minimum_required(VERSION 3.20)
project(sensehat_fsm_demo LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Toggle mock vs linux HAL (default: mock)
option(USE_MOCK_HAL "Use mock I2C HAL" ON)
option(CALIBRATION_MODE "Enable calibration-mode joystick gating" OFF)

# Toggle ONNX Runtime usage (default on)
option(USE_ONNXRUNTIME "Enable ONNX Runtime inference" ON)

# Main app
if (CALIBRATION_MODE)
set(USE_ONNXRUNTIME OFF CACHE BOOL "Enable ONNX Runtime inference" FORCE)
add_executable(sensehat_fsm_demo
  src/main.cpp
  src/lsm9ds1.cpp
  src/logger.cpp 
  src/ledmatrix.cpp
)
target_compile_definitions(sensehat_fsm_demo PRIVATE CALIBRATION_MODE=1)
else()
add_executable(sensehat_fsm_demo
  src/main.cpp
  src/lsm9ds1.cpp
  src/logger.cpp 
  src/FeatureExtractor.cpp
  src/ONXXClassifier.cpp
  src/ledmatrix.cpp  
)
endif()

target_include_directories(sensehat_fsm_demo PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# I2C_MOCK flag: means we're on WIN32; otherwise (real) has I2CMOCK = 0
# Default: mock on Windows and on Pi UNLESS OVERWRITTEN
if (WIN32 OR USE_MOCK_HAL)
  target_compile_definitions(sensehat_fsm_demo PRIVATE I2C_MOCK=1)
  target_sources(sensehat_fsm_demo PRIVATE src/i2c_mock.cpp)
else()
  target_sources(sensehat_fsm_demo PRIVATE src/i2c_hal.cpp)
  target_compile_definitions(sensehat_fsm_demo PRIVATE I2C_MOCK=0)

  # --- Self-test utility (Linux/RPi only)
  #add_executable(i2c_selftest
    #src/i2c_selftest.cpp
    #src/i2c_hal.cpp
    #src/lsm9ds1.cpp
    #src/logger.cpp
  #)
  #target_include_directories(i2c_selftest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  #target_compile_definitions(i2c_selftest PRIVATE I2C_LINUX=1)
  #target_link_libraries(i2c_selftest PRIVATE Threads::Threads)
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

# ===================== ONNX Runtime wiring (diff builds for Windows + RasperryPi Linux) =====================
if (USE_ONNXRUNTIME)
  # Choose the right root dir based on platform
  if (WIN32)
    set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/external/ONNXRuntime/win-x64") # windows
  elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    # 64-bit Raspberry Pi OS
    set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/external/ONNXRuntime/linux-aarch64")
  else()
    message(WARNING
      "USE_ONNXRUNTIME=ON, but no ONNXRuntime root configured for this platform. "
      "ONNX Runtime will be disabled.")
    set(USE_ONNXRUNTIME OFF)
  endif()
endif()

if (USE_ONNXRUNTIME)
  message(STATUS "Using ONNX Runtime from: ${ONNXRUNTIME_ROOT}")

  target_compile_definitions(sensehat_fsm_demo PRIVATE USE_ONNXRUNTIME=1)

  # Include headers
  target_include_directories(sensehat_fsm_demo PRIVATE
    "${ONNXRUNTIME_ROOT}/include"
  )

  # Link libs
  if (WIN32)
    # Windows: link onnxruntime.lib
    target_link_directories(sensehat_fsm_demo PRIVATE "${ONNXRUNTIME_ROOT}/lib")
    target_link_libraries(sensehat_fsm_demo PRIVATE onnxruntime)
  else()
    # Linux / Raspberry Pi: onnxruntime + dl + m (and you already link Threads)
    target_link_directories(sensehat_fsm_demo PRIVATE "${ONNXRUNTIME_ROOT}/lib")
    target_link_libraries(sensehat_fsm_demo PRIVATE onnxruntime dl m)
  endif()
endif()