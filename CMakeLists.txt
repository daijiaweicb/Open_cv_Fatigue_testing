cmake_minimum_required(VERSION 3.10)
project(FatigueNoVCam)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(dlib REQUIRED)

add_executable(FatigueNoVCam main.cpp)

target_link_libraries(FatigueNoVCam
    ${OpenCV_LIBS}
    dlib::dlib
)

