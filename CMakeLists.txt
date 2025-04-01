cmake_minimum_required(VERSION 3.10)
project(CameraTest)


find_package(OpenCV REQUIRED)


message(STATUS "OpenCV Include Dir: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV Libraries: ${OpenCV_LIBS}")

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(camera_test main.cpp)

target_link_libraries(camera_test ${OpenCV_LIBS})
