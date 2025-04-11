cmake_minimum_required(VERSION 3.10)
project(FatigueDetection)

set(CMAKE_CXX_STANDARD 17)

# 查找 OpenCV 和 dlib
find_package(OpenCV REQUIRED)
find_package(dlib REQUIRED)

# 添加 libcamera 手动查找（或你也可以写死路径）
find_path(LIBCAMERA_INCLUDE_DIR NAMES libcamera/libcamera.h)
find_library(LIBCAMERA_LIB NAMES camera REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${LIBCAMERA_INCLUDE_DIR}
    .
)

add_executable(fatigue_detect
    main.cpp
    libcam2opencv.cpp
)

target_link_libraries(fatigue_detect
    ${OpenCV_LIBS}
    dlib::dlib
    ${LIBCAMERA_LIB}
    pthread
)
