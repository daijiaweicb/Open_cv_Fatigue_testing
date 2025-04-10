#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <iostream>
#include <vector>
#include <chrono>

// EAR 计算函数
double computeEAR(const std::vector<cv::Point>& eye) {
    double A = cv::norm(eye[1] - eye[5]);
    double B = cv::norm(eye[2] - eye[4]);
    double C = cv::norm(eye[0] - eye[3]);
    return (A + B) / (2.0 * C);
}

// 提取眼睛关键点
std::vector<cv::Point> getEyePoints(const dlib::full_object_detection& shape, bool left) {
    std::vector<cv::Point> points;
    int start = left ? 36 : 42;
    for (int i = 0; i < 6; ++i) {
        auto pt = shape.part(start + i);
        points.emplace_back(cv::Point(pt.x(), pt.y()));
    }
    return points;
}

int main() {
    // 初始化 Dlib 检测器和模型
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor predictor;
    dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> predictor;

    // OpenCV 打开摄像头（树莓派上建议设置分辨率）
    cv::VideoCapture cap(0);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    if (!cap.isOpened()) {
        std::cerr << "摄像头打开失败！" << std::endl;
        return -1;
    }

    const double EAR_THRESHOLD = 0.21;
    const int EAR_CONSEC_FRAMES = 15;

    int frame_counter = 0;
    bool fatigued = false;

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        dlib::cv_image<dlib::bgr_pixel> dlib_img(frame);
        std::vector<dlib::rectangle> faces = detector(dlib_img);

        for (auto face : faces) {
            auto shape = predictor(dlib_img, face);

            auto leftEye = getEyePoints(shape, true);
            auto rightEye = getEyePoints(shape, false);

            double leftEAR = computeEAR(leftEye);
            double rightEAR = computeEAR(rightEye);
            double ear = (leftEAR + rightEAR) / 2.0;

            // 可视化眼睛轮廓
            for (const auto& pt : leftEye)
                cv::circle(frame, pt, 2, cv::Scalar(0, 255, 0), -1);
            for (const auto& pt : rightEye)
                cv::circle(frame, pt, 2, cv::Scalar(0, 255, 0), -1);

            if (ear < EAR_THRESHOLD) {
                frame_counter++;
                if (frame_counter >= EAR_CONSEC_FRAMES) {
                    fatigued = true;
                    cv::putText(frame, "FATIGUE DETECTED!", cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                                cv::Scalar(0, 0, 255), 2);
                }
            } else {
                frame_counter = 0;
                fatigued = false;
            }

            // 显示 EAR 值
            char text[50];
            sprintf(text, "EAR: %.2f", ear);
            cv::putText(frame, text, cv::Point(10, frame.rows - 10), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        fatigued ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0), 2);
        }

        cv::imshow("Fatigue Detection", frame);
        if (cv::waitKey(1) == 'q') break;
    }

    return 0;
}

