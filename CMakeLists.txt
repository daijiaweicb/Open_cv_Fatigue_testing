cmake_minimum_required(VERSION 3.10)


project(FatigueDetection VERSION 1.0)


find_package(OpenCV REQUIRED)


find_package(Dlib REQUIRED)


find_library(LIBCAME_LIB cam REQUIRED)


set(CMAKE_CXX_STANDARD 11)

add_executable(fatigue_detect main.cpp)

target_link_libraries(fatigue_detect
    ${OpenCV_LIBS}
    ${Dlib_LIBS}
    ${LIBCAME_LIB}
    pthread
)

include_directories(${OpenCV_INCLUDE_DIRS})


include_directories(${Dlib_INCLUDE_DIRS})



include_directories(/usr/include/libcamera)
