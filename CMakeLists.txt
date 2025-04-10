cmake_minimum_required(VERSION 3.16)
project(FatigueDetection LANGUAGES CXX)

# 设置编译选项
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 默认Release构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 查找OpenCV组件
find_package(OpenCV REQUIRED COMPONENTS
    core
    imgproc
    highgui
)

# 查找dlib
find_package(dlib REQUIRED)

# 高级查找libcamera
find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(LIBCAMERA QUIET libcamera)
endif()

if(NOT LIBCAMERA_FOUND)
    message(STATUS "正在执行手动libcamera查找...")
    
    # 搜索库文件路径
    find_library(LIBCAMERA_LIBRARY
        NAMES libcamera.so camera
        PATHS /usr/lib
              /usr/lib/aarch64-linux-gnu
              /usr/lib/arm-linux-gnueabihf
              /usr/local/lib
    )
    
    # 搜索头文件路径
    find_path(LIBCAMERA_INCLUDE_DIR
        NAMES libcamera/libcamera.h
        PATHS /usr/include
              /usr/local/include
    )
    
    if(LIBCAMERA_LIBRARY AND LIBCAMERA_INCLUDE_DIR)
        set(LIBCAMERA_LIBRARIES ${LIBCAMERA_LIBRARY})
        set(LIBCAMERA_INCLUDE_DIRS ${LIBCAMERA_INCLUDE_DIR})
        set(LIBCAMERA_FOUND TRUE)
        message(STATUS "找到libcamera库: ${LIBCAMERA_LIBRARIES}")
        message(STATUS "找到libcamera头文件: ${LIBCAMERA_INCLUDE_DIRS}")
    else()
        message(FATAL_ERROR "libcamera未找到！\n"
                "请确认：\n"
                "1. 已安装libcamera-dev: sudo apt install libcamera-dev\n"
                "2. 库文件位于标准路径")
    endif()
endif()

# 配置可执行文件
add_executable(fatigue_detect
    src/main.cpp
)

# 包含目录
target_include_directories(fatigue_detect PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${dlib_INCLUDE_DIRS}
    ${LIBCAMERA_INCLUDE_DIRS}
)

# 链接库（注意顺序）
target_link_libraries(fatigue_detect PRIVATE
    ${LIBCAMERA_LIBRARIES}  # 必须首先链接
    dlib::dlib
    ${OpenCV_LIBRARIES}
    pthread
    stdc++fs
)

# 编译器优化
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(fatigue_detect PRIVATE
        -O3
        -Wall
        -Wextra
    )
    
    # ARM平台NEON优化
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
        target_compile_options(fatigue_detect PRIVATE
            -mcpu=native
            -mfpu=neon
        )
        message(STATUS "启用ARM NEON优化")
    endif()
endif()

# 安装规则（可选）
install(TARGETS fatigue_detect DESTINATION bin)
install(FILES models/shape_predictor_68_face_landmarks.dat
        DESTINATION share/fatigue_detect)

# 调试信息
message(STATUS "构建类型: ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenCV版本: ${OpenCV_VERSION}")
message(STATUS "dlib版本: ${dlib_VERSION}")