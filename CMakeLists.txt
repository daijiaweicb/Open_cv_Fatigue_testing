cmake_minimum_required(VERSION 3.10)

project(FatigueDetection)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)

# 找到 OpenCV 和 dlib 库
find_package(OpenCV REQUIRED)
find_package(dlib REQUIRED)

# 设置 libcamera 的路径
# 如果 libcamera 库没有安装到系统默认路径，使用以下语句设置路径
set(LIBCAMERA_INCLUDE_DIR "/usr/include/libcamera")   # 请根据实际路径修改
set(LIBCAMERA_LIB_DIR "/usr/lib/aarch64-linux-gnu")   # 请根据实际路径修改

# 包含目录
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${dlib_INCLUDE_DIRS}
    ${LIBCAMERA_INCLUDE_DIR}   # 添加 libcamera 头文件路径
)

# 设置源文件
set(SOURCES
    main.cpp  # 如果 main.cpp 不在 src 文件夹中，请修改路径
)

# 添加可执行文件
add_executable(fatigue_detect ${SOURCES})

# 链接 OpenCV、dlib 和 libcamera 库
target_link_libraries(fatigue_detect
    ${OpenCV_LIBS}
    dlib::dlib
    ${LIBCAMERA_LIB_DIR}/libcamera.so  # 根据实际路径修改
)

# 设置调试和发布配置
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(fatigue_detect PRIVATE -O3 -DNDEBUG)
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(fatigue_detect PRIVATE -g)
endif()
