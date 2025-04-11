cmake_minimum_required(VERSION 3.10)
project(FatigueDetection)

set(CMAKE_CXX_STANDARD 17)

# OpenCV
find_package(OpenCV REQUIRED)

# dlib
find_package(dlib REQUIRED)

# 添加可执行文件
add_executable(fatigue_detection
    fatigue_detection.cpp
    libcam2opencv.cpp
)

# 链接库
target_link_libraries(fatigue_detection
    ${OpenCV_LIBS}
    dlib::dlib
    pthread
)

