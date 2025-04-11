#include <iostream>
#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include "libcam2opencv.h"  // 你的头文件

float eye_aspect_ratio(const std::vector<cv::Point2f>& eye) {
    float A = cv::norm(eye[1] - eye[5]);
    float B = cv::norm(eye[2] - eye[4]);
    float C = cv::norm(eye[0] - eye[3]);
    return (A + B) / (2.0f * C);
}

std::vector<cv::Point2f> extract_eye(const dlib::full_object_detection& shape, bool left) {
    std::vector<cv::Point2f> eye;
    int start = left ? 36 : 42;
    for (int i = 0; i < 6; ++i)
        eye.emplace_back(shape.part(start + i).x(), shape.part(start + i).y());
    return eye;
}

class FatigueCallback : public Libcam2OpenCV::Callback {
public:
    FatigueCallback() {
        // 加载 dlib 关键点模型
        try {
            dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> predictor;
        } catch (std::exception& e) {
            std::cerr << "无法加载关键点模型：" << e.what() << std::endl;
            std::exit(1);
        }

        // 加载 Haar 人脸检测器
        if (!face_cascade.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")) {
            std::cerr << "无法加载 Haar 模型" << std::endl;
            std::exit(1);
        }
    }

    void hasFrame(const cv::Mat &frame, const libcamera::ControlList &) override {
        const float EAR_THRESHOLD = 0.25f;
        const int EYES_CLOSED_FRAMES = 15;
        static int counter = 0;

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(80, 80));

        cv::Mat draw = frame.clone();

        for (const auto& face : faces) {
            cv::rectangle(draw, face, cv::Scalar(255, 0, 0), 2);

            dlib::cv_image<dlib::bgr_pixel> cimg(draw);
            dlib::rectangle dlib_rect(face.x, face.y, face.x + face.width, face.y + face.height);
            dlib::full_object_detection shape = predictor(cimg, dlib_rect);

            auto left_eye = extract_eye(shape, true);
            auto right_eye = extract_eye(shape, false);
            float ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0f;

            if (ear < EAR_THRESHOLD) {
                counter++;
                if (counter >= EYES_CLOSED_FRAMES) {
                    cv::putText(draw, "DROWSINESS ALERT!", cv::Point(50, 50),
                                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
                }
            } else {
                counter = 0;
            }

            for (const auto& pt : left_eye) cv::circle(draw, pt, 2, cv::Scalar(0, 255, 0), -1);
            for (const auto& pt : right_eye) cv::circle(draw, pt, 2, cv::Scalar(0, 255, 0), -1);
        }

        cv::imshow("Fatigue Detection (libcam2opencv)", draw);
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) std::exit(0);
    }

private:
    dlib::shape_predictor predictor;
    cv::CascadeClassifier face_cascade;
};

int main() {
    Libcam2OpenCV cam;
    FatigueCallback callback;
    cam.registerCallback(&callback);

    Libcam2OpenCVSettings settings;
    settings.width = 1280;  // 推荐用 1280x720 或 1640x1232
    settings.height = 720;
    settings.framerate = 15;
    cam.start(settings);

    while (true) std::this_thread::sleep_for(std::chrono::seconds(1));
}
