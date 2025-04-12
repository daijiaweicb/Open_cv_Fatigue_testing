cmake_minimum_required(VERSION 3.10)
project(FatigueNoVCam)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV
find_package(OpenCV REQUIRED)

# dlib
find_package(dlib REQUIRED)

# libcamera
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBCAMERA REQUIRED libcamera)

# 添加 libcamera 头文件目录
include_directories(${LIBCAMERA_INCLUDE_DIRS})

# 添加可执行文件
add_executable(FatigueNoVCam main.cpp)

# 链接库
target_link_libraries(FatigueNoVCam
    ${OpenCV_LIBS}
    dlib::dlib
    pthread
    ${LIBCAMERA_LIBRARIES}
)
