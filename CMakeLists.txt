cmake_minimum_required(VERSION 3.10)
project(fatigue_detection)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
find_package(dlib REQUIRED)

add_executable(fatigue_detect main.cpp)
target_link_libraries(fatigue_detect ${OpenCV_LIBS} dlib::dlib)
