cmake_minimum_required(VERSION 3.12)
project(fatigue_detection)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# libcamera
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBCAMERA REQUIRED libcamera)

# dlib (非模块安装，用 find_library)
find_library(DLIB_LIB dlib)

# 找到 BLAS（为了解决 cblas_dgemm 链接问题）
find_package(BLAS REQUIRED)

# 包含目录
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${LIBCAMERA_INCLUDE_DIRS}
)

# 可执行文件
add_executable(fatigue_detection main.cpp)

# 链接库
target_link_libraries(fatigue_detection
    ${OpenCV_LIBS}
    ${LIBCAMERA_LIBRARIES}
    ${DLIB_LIB}
    ${BLAS_LIBRARIES}
    pthread
)
