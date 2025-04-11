cmake_minimum_required(VERSION 3.10)
project(FatigueDetection)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 OpenCV
find_package(OpenCV REQUIRED)

# 查找 dlib（确保你已安装 libdlib-dev 或通过源码安装）
find_package(dlib REQUIRED)

# 添加头文件目录（可选）
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${dlib_INCLUDE_DIRS}
)

# 编译目标
add_executable(fatigue_detection main.cpp)

# 链接库
target_link_libraries(fatigue_detection
    ${OpenCV_LIBS}
    dlib::dlib
    pthread
)
