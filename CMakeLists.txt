cmake_minimum_required(VERSION 3.10)
project(FatigueDetection)

# 使用 C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 查找 dlib（确保你已安装 dlib）
find_package(dlib REQUIRED)
include_directories(${dlib_INCLUDE_DIRS})

# 添加可执行文件
add_executable(FatigueDetection main.cpp)

# 链接 OpenCV 和 dlib 库
target_link_libraries(FatigueDetection ${OpenCV_LIBS} dlib::dlib)
