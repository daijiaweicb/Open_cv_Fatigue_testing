cmake_minimum_required(VERSION 3.10)
project(FatigueDetection)

set(CMAKE_CXX_STANDARD 17)

# 找 OpenCV
find_package(OpenCV REQUIRED)

# 找 dlib
find_package(dlib REQUIRED)

# 添加 libcamera
find_library(LIBCAMERA_LIB camera REQUIRED)

# 头文件路径
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${dlib_INCLUDE_DIRS}
    .
)

# 源文件
add_executable(fatigue_detect
    main.cpp
    libcam2opencv.cpp
)

# 链接库
target_link_libraries(fatigue_detect
    ${OpenCV_LIBS}
    dlib::dlib
    ${LIBCAMERA_LIB}
    pthread
)
