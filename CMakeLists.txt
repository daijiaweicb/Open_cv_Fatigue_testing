cmake_minimum_required(VERSION 3.12)
project(fatigue_detection)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 查找 dlib（必须使用 find_package，否则 target_link_libraries 无效）
find_package(dlib REQUIRED)

# 查找 pkg-config
find_package(PkgConfig REQUIRED)

# 使用 pkg-config 查找 libcamera
pkg_check_modules(LIBCAMERA REQUIRED libcamera)

# 包含头文件路径
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${LIBCAMERA_INCLUDE_DIRS}
    ${dlib_INCLUDE_DIRS}
    .
)

# 添加源文件
add_executable(fatigue_detection
    main.cpp
    libcam2opencv.cpp
)

# 链接库
target_link_libraries(fatigue_detection
    ${OpenCV_LIBS}
    ${LIBCAMERA_LIBRARIES}
    dlib::dlib
    pthread
)

# 树莓派特定优化
if (CMAKE_SYSTEM_PROCESSOR MATCHES "armv7l|aarch64")
    add_compile_options(-march=armv8-a+crc -mfpu=neon-fp-armv8 -mfloat-abi=hard)
endif()
