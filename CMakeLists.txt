cmake_minimum_required(VERSION 3.12)
project(fatigue_detection)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV
find_package(OpenCV REQUIRED)

# libcamera (通过 pkg-config)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBCAMERA REQUIRED libcamera)

# dlib (通过 apt 安装的)
find_library(DLIB_LIB dlib)

# BLAS + LAPACK （dlib 内部依赖）
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# 头文件路径
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
    ${LAPACK_LIBRARIES}
    pthread
)
