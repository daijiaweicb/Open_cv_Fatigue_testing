cmake_minimum_required(VERSION 3.12)
project(fatigue_detection)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 查找pkg-config
find_package(PkgConfig REQUIRED)

# 使用pkg-config查找libcamera
pkg_check_modules(LIBCAMERA REQUIRED libcamera)

# 配置头文件包含路径
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${LIBCAMERA_INCLUDE_DIRS}  # 从 pkg-config 提供的路径
)

# 添加可执行文件
add_executable(fatigue_detection main.cpp)

# 链接库文件
target_link_libraries(fatigue_detection
    ${OpenCV_LIBS}
    ${LIBCAMERA_LIBRARIES}  # 使用 pkg-config 提供的 libcamera 库
    pthread
)

# 树莓派特定优化
if (CMAKE_SYSTEM_PROCESSOR MATCHES "armv7l")
    add_compile_options(-march=armv8-a+crc -mfpu=neon-fp-armv8 -mfloat-abi=hard)
endif()
