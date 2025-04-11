cmake_minimum_required(VERSION 3.10)
project(MediaPipeFatigueNoVCam)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)

add_executable(MediaPipeFatigueNoVCam main.cpp)
target_link_libraries(MediaPipeFatigueNoVCam ${OpenCV_LIBS})

