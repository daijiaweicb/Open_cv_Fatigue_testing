cmake_minimum_required(VERSION 3.12)
project(fatigue_detection)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 查找libcamera
find_library(LIBCAMERA libcamera)
find_path(LIBCAMERA_INCLUDE_DIR libcamera/camera.h)

# 配置头文件包含路径
include_directories(
    ${LIBCAMERA_INCLUDE_DIR}
    ${LIBCAMERA_INCLUDE_DIR}/libcamera
)

# 添加可执行文件
add_executable(fatigue_detection
    main.cpp
)

# 链接库文件
target_link_libraries(fatigue_detection
    ${OpenCV_LIBS}
    ${LIBCAMERA}
    pthread
)

# 树莓派特定优化
if (CMAKE_SYSTEM_PROCESSOR MATCHES "armv7l")
    add_compile_options(-march=armv8-a+crc -mfpu=neon-fp-armv8 -mfloat-abi=hard)
endif()