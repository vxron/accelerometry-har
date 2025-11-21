cmake_minimum_required(VERSION 3.20)
project(sensehat_fsm_demo LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# BUILD FLAGS
# Toggle mock vs linux HAL (default: mock)
option(USE_MOCK_HAL "Use mock I2C HAL" ON)
# Toggle calibration vs run mode (default: run mode)
option(CALIBRATION_MODE "Enable calibration-mode joystick gating" OFF)
# Toggle ONNX Runtime usage (default on)
option(USE_ONNXRUNTIME "Enable ONNX Runtime inference" ON)

# MAIN APP
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
# need onnx classifier in run mode
set(USE_ONNXRUNTIME ON CACHE BOOL "Enable ONNX Runtime inference" FORCE)
endif()

target_include_directories(sensehat_fsm_demo PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# I2C_MOCK flag: means we're on WIN32; otherwise (real) has I2CMOCK = 0
# Default: mock on Windows and on Pi UNLESS OVERWRITTEN
if (WIN32)
  target_compile_definitions(sensehat_fsm_demo PRIVATE I2C_MOCK=1)
  target_sources(sensehat_fsm_demo PRIVATE src/i2c_mock.cpp)
else()
  target_sources(sensehat_fsm_demo PRIVATE src/i2c_hal.cpp)
  target_compile_definitions(sensehat_fsm_demo PRIVATE I2C_MOCK=0)
  option(USE_MOCK_HAL "Use mock I2C HAL" OFF)
  target_compile_definitions(sensehat_fsm_demo PRIVATE
    MODEL_META_JSON_PATH="${CMAKE_SOURCE_DIR}/src/models/veron/har_hierarchy_meta.json"
  )

  # Self-test utility (Linux/RPi only)
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

# ===================== ONNX Runtime wiring =====================
if (USE_ONNXRUNTIME)
  if (WIN32)
    # Windows build
    set(ONNXRUNTIME_ROOT
      "${CMAKE_SOURCE_DIR}/external/ONNXRuntime/win-x64")
  else()
    # Pi / Linux build
    set(ONNXRUNTIME_ROOT
      "${CMAKE_SOURCE_DIR}/external/ONNXRuntime/linux-aarch64/onnxruntime")
  endif()

  message(STATUS "Using ONNX Runtime from: ${ONNXRUNTIME_ROOT}")

  target_compile_definitions(sensehat_fsm_demo PRIVATE USE_ONNXRUNTIME=1)

  # Add ONNX headers
  target_include_directories(sensehat_fsm_demo PRIVATE
    "${ONNXRUNTIME_ROOT}/include"
  )

  # Add ONNX libs
  if (WIN32)
    target_link_directories(sensehat_fsm_demo PRIVATE "${ONNXRUNTIME_ROOT}/lib")
    target_link_libraries(sensehat_fsm_demo PRIVATE onnxruntime)
  else()
    target_link_directories(sensehat_fsm_demo PRIVATE "${ONNXRUNTIME_ROOT}/lib")
    target_link_libraries(sensehat_fsm_demo PRIVATE onnxruntime dl m)
  endif()
endif()
