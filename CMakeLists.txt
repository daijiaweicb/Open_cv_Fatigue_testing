cmake_minimum_required(VERSION 3.10)
project(FatigueDetector)

# 使用 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 OpenCV
find_package(OpenCV REQUIRED)

# 查找 dlib
find_package(dlib REQUIRED)

# 添加可执行文件
add_executable(FatigueDetector main.cpp)

# 链接 OpenCV、dlib、pthread（用于 std::thread）
target_link_libraries(FatigueDetector
    ${OpenCV_LIBS}
    dlib::dlib
    pthread
)
