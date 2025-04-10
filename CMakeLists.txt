cmake_minimum_required(VERSION 3.16)
project(FatigueDetection)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找必需依赖包
find_package(OpenCV REQUIRED
  COMPONENTS core imgproc highgui
)

find_package(dlib REQUIRED)
find_package(libcamera REQUIRED)

# 自定义查找libcamera (适用于树莓派环境)
if(NOT libcamera_FOUND)
    find_library(LIBCAMERA_LIB libcamera 
        PATHS /usr/lib /usr/local/lib
    )
    find_path(LIBCAMERA_INCLUDE libcamera/libcamera.h
        PATHS /usr/include /usr/local/include
    )
    
    if(LIBCAMERA_LIB AND LIBCAMERA_INCLUDE)
        set(libcamera_LIBRARIES ${LIBCAMERA_LIB})
        set(libcamera_INCLUDE_DIRS ${LIBCAMERA_INCLUDE})
        message(STATUS "Found libcamera manually")
    else()
        message(FATAL_ERROR "libcamera not found!")
    endif()
endif()

# 设置可执行文件
add_executable(fatigue_detect
    src/main.cpp
)

# 包含目录
target_include_directories(fatigue_detect PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${dlib_INCLUDE_DIRS}
    ${libcamera_INCLUDE_DIRS}
)

# 链接库
target_link_libraries(fatigue_detect
    PRIVATE
    ${OpenCV_LIBRARIES}
    dlib::dlib
    ${libcamera_LIBRARIES}
    pthread
    stdc++fs  # 需要filesystem库
)

# 设置编译优化
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(fatigue_detect PRIVATE -O3)
endif()