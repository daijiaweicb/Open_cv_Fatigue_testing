#include <iostream>
#include <vector>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

// 左右眼索引（来自 MediaPipe）
const std::vector<int> LEFT_EYE_IDX  = { 33, 160, 158, 133, 153, 144 };
const std::vector<int> RIGHT_EYE_IDX = { 263, 387, 385, 362, 380, 373 };

float eye_aspect_ratio(const std::vector<cv::Point2f>& eye) {
    float A = cv::norm(eye[1] - eye[5]);
    float B = cv::norm(eye[2] - eye[4]);
    float C = cv::norm(eye[0] - eye[3]);
    return (A + B) / (2.0f * C);
}

int main() {
    const int width = 640, height = 480;
    const int frame_size = width * height * 3 / 2;
    std::vector<uchar> buffer(frame_size);
    cv::Mat yuvImg(height + height / 2, width, CV_8UC1), bgrImg;

    FILE* pipe = popen("libcamera-vid --width 640 --height 480 --framerate 15 --codec yuv420 --nopreview --timeout 0 -o -", "r");
    if (!pipe) {
        std::cerr << "无法启动 libcamera-vid" << std::endl;
        return -1;
    }

    // 加载 ONNX 模型
    cv::dnn::Net net = cv::dnn::readNetFromONNX("face_mesh.onnx");
    const float EAR_THRESHOLD = 0.25f;
    const int EYES_CLOSED_FRAMES = 15;
    int counter = 0;

    while (true) {
        size_t bytes = fread(buffer.data(), 1, frame_size, pipe);
        if (bytes != frame_size) {
            std::cerr << "读取失败或中断" << std::endl;
            break;
        }

        memcpy(yuvImg.data, buffer.data(), frame_size);
        cv::cvtColor(yuvImg, bgrImg, cv::COLOR_YUV2BGR_I420);

        // 缩放为 192x192，准备输入
        cv::Mat input;
        cv::resize(bgrImg, input, cv::Size(192, 192));
        input.convertTo(input, CV_32F, 1.0 / 255.0);
        cv::Mat blob = cv::dnn::blobFromImage(input);
        net.setInput(blob);
        cv::Mat output = net.forward();  // [1,468,3]

        output = output.reshape(1, 468); // [468,3]
        std::vector<cv::Point2f> left_eye, right_eye;

        for (int idx : LEFT_EYE_IDX) {
            float x = output.at<float>(idx, 0) * width;
            float y = output.at<float>(idx, 1) * height;
            left_eye.emplace_back(x, y);
        }
        for (int idx : RIGHT_EYE_IDX) {
            float x = output.at<float>(idx, 0) * width;
            float y = output.at<float>(idx, 1) * height;
            right_eye.emplace_back(x, y);
        }

        float ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0f;

        if (ear < EAR_THRESHOLD) {
            counter++;
            if (counter >= EYES_CLOSED_FRAMES) {
                cv::putText(bgrImg, "DROWSINESS ALERT!", cv::Point(50, 50),
                            cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
            }
        } else {
            counter = 0;
        }

        for (auto& pt : left_eye)
            cv::circle(bgrImg, pt, 2, cv::Scalar(0, 255, 0), -1);
        for (auto& pt : right_eye)
            cv::circle(bgrImg, pt, 2, cv::Scalar(0, 255, 0), -1);

        cv::imshow("MediaPipe Fatigue Detection", bgrImg);
        if (cv::waitKey(1) == 'q') break;
    }

    pclose(pipe);
    return 0;
}
