cmake_minimum_required(VERSION 3.10)
project(FatigueNoVCam)

# 使用 C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 OpenCV
find_package(OpenCV REQUIRED)

# 查找 dlib
find_package(dlib REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBCAMERA REQUIRED libcamera)

# 添加源文件
add_executable(FatigueNoVCam main.cpp)

# 链接 OpenCV、dlib 和 pthread（用于多线程，保险起见）
target_link_libraries(FatigueNoVCam
    ${OpenCV_LIBS}
    dlib::dlib
    pthread
    ${LIBCAMERA_LIBRARIES}  # 添加 libcamera 库
)
