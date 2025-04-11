cmake_minimum_required(VERSION 3.12)
project(fatigue_detection)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 查找 pkg-config
find_package(PkgConfig REQUIRED)

# 使用 pkg-config 查找 libcamera
pkg_check_modules(LIBCAMERA REQUIRED libcamera)

# 配置头文件包含路径
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${LIBCAMERA_INCLUDE_DIRS}
)

# 添加源文件（main.cpp 为单文件）
add_executable(fatigue_detection main.cpp)

# 链接库文件
target_link_libraries(fatigue_detection
    ${OpenCV_LIBS}
    ${LIBCAMERA_LIBRARIES}
    pthread
    dlib::dlib
)

# 树莓派平台特定优化（如果为 ARMv7）
if (CMAKE_SYSTEM_PROCESSOR MATCHES "armv7l")
    message(STATUS "ARMv7 detected: applying Raspberry Pi optimization flags")
    add_compile_options(-march=armv8-a+crc -mfpu=neon-fp-armv8 -mfloat-abi=hard)
endif()
