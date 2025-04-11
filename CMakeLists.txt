cmake_minimum_required(VERSION 3.10)
project(FatigueDetection)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 OpenCV
find_package(OpenCV REQUIRED)

# 查找 dlib（假设你已经安装了）
find_package(dlib REQUIRED)

# 添加包含目录
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${dlib_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# 编译目标
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
