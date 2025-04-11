cmake_minimum_required(VERSION 3.12)
project(fatigue_detection)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBCAMERA REQUIRED libcamera)

find_library(DLIB_LIB dlib)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${LIBCAMERA_INCLUDE_DIRS}
)

add_executable(fatigue_detection main.cpp)

target_link_libraries(fatigue_detection
    ${OpenCV_LIBS}
    ${LIBCAMERA_LIBRARIES}
    ${DLIB_LIB}
    pthread
)
