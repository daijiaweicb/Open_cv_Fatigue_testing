cmake_minimum_required(VERSION 3.10)

# 设置项目名称和版本
project(FatigueDetection VERSION 1.0)

# 查找 OpenCV 包
find_package(OpenCV REQUIRED)

# 查找 Dlib 包
find_package(dlib REQUIRED)

# 查找 libcamera
find_library(LIBCAME_LIB NAMES camera libcamera REQUIRED)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)

# 设置可执行文件
add_executable(fatigue_detect main.cpp)

# 链接 OpenCV、Dlib 和 libcamera 库
target_link_libraries(fatigue_detect
    ${OpenCV_LIBS}
    ${Dlib_LIBS}
    ${LIBCAME_LIB}
    pthread
)

# 设置 OpenCV 库的包含路径
include_directories(${OpenCV_INCLUDE_DIRS})

# 设置 Dlib 库的包含路径
include_directories(${Dlib_INCLUDE_DIRS})

# 设置 libcamera 库的包含路径
include_directories(/usr/include/libcamera)
