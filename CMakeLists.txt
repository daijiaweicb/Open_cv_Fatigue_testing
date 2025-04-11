cmake_minimum_required(VERSION 3.10)
project(FatigueDetection)

set(CMAKE_CXX_STANDARD 17)

# OpenCV
find_package(OpenCV REQUIRED)

# dlib
find_package(dlib REQUIRED)

# 查找 libcamera
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBCAMERA REQUIRED libcamera)

# 设置编译器优化标志（可选）
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")

# 添加可执行文件
add_executable(fatigue_detection
    main.cpp
    libcam2opencv.cpp
)

# 链接库
target_link_libraries(fatigue_detection
    ${OpenCV_LIBS}
    dlib::dlib
    pthread
    ${LIBCAMERA_LIBRARIES}  # 添加 libcamera 库
)

# 添加 libcamera 的头文件路径
target_include_directories(fatigue_detection
    PRIVATE
    ${LIBCAMERA_INCLUDE_DIRS}  # 添加 libcamera 的头文件路径
)

# 生成构建目标时确保 libcamera 库的正确设置
if (NOT LIBCAMERA_FOUND)
    message(FATAL_ERROR "libcamera not found, please install libcamera and set PKG_CONFIG_PATH")
endif()


