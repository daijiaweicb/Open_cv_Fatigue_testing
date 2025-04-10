#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main() {
    FILE* pipe = popen("libcamera-vid -t 0 --codec yuv420 --width 640 --height 480 -n -o -", "r");
    if (!pipe) {
        std::cerr << "无法启动 libcamera-vid，请检查摄像头连接状态" << std::endl;
        return -1;
    }

    int width = 640;
    int height = 480;
    int frame_size = width * height * 3 / 2;

    std::vector<uchar> buffer(frame_size);
    cv::Mat yuvImg(height + height / 2, width, CV_8UC1);
    cv::Mat bgrImg;

    while (true) {
        size_t read_bytes = fread(buffer.data(), 1, frame_size, pipe);
        if (read_bytes != frame_size) {
            std::cerr << "读取失败，可能摄像头断开或输出结束" << std::endl;
            break;
        }

        memcpy(yuvImg.data, buffer.data(), frame_size);
        cv::cvtColor(yuvImg, bgrImg, cv::COLOR_YUV2BGR_I420);
        cv::imshow("CSI Camera with OpenCV", bgrImg);

        if (cv::waitKey(1) == 'q') break;
    }

    pclose(pipe);
    return 0;
}
