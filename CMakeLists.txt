cmake_minimum_required(VERSION 3.10)
project(csi_camera_viewer)

set(CMAKE_CXX_STANDARD 17)


find_package(OpenCV REQUIRED)


add_executable(csi_cam main.cpp)


target_link_libraries(csi_cam ${OpenCV_LIBS})
