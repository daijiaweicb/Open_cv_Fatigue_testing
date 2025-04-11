#include "fatigue_callback.h"
#include "window.h"  // 包含 Qt 的界面类

#include <iostream>

FatigueCallback::FatigueCallback() : frame_counter(0), total_frames(0), fps(0), last_log_time(0), dlib_loaded(false), haar_loaded(false) {
    start_time = std::chrono::steady_clock::now();

    try {
        dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> predictor;
        std::cout << "✅ 加载 dlib 模型成功\n";
        dlib_loaded = true;
    } catch (std::exception& e) {
        std::cerr << "❌ 无法加载 dlib 模型：" << e.what() << "\n";
    }

    if (face_cascade.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")) {
        std::cout << "✅ 加载 Haar 模型成功\n";
        haar_loaded = true;
    } else {
        std::cerr << "❌ 无法加载 Haar 模型\n";
    }
}

FatigueCallback::~FatigueCallback() {}

float FatigueCallback::eye_aspect_ratio(const std::vector<cv::Point2f>& eye) {
    float A = cv::norm(eye[1] - eye[5]);
    float B = cv::norm(eye[2] - eye[4]);
    float C = cv::norm(eye[0] - eye[3]);
    return (A + B) / (2.0f * C);
}

std::vector<cv::Point2f> FatigueCallback::extract_eye(const dlib::full_object_detection& shape, bool left) {
    std::vector<cv::Point2f> eye;
    int start = left ? 36 : 42;
    for (int i = 0; i < 6; ++i)
        eye.emplace_back(shape.part(start + i).x(), shape.part(start + i).y());
    return eye;
}

void FatigueCallback::hasFrame(const cv::Mat &frame, const libcamera::ControlList &) {
    total_frames++;
    frame_counter++;

    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
    if (elapsed >= 1000) {
        fps = frame_counter * 1000.0f / elapsed;
        frame_counter = 0;
        start_time = now;
    }

    cv::Mat display_frame = frame.clone();
    float ear = 0.0f;
    bool alarm = false;

    if (dlib_loaded && haar_loaded) {
        try {
            cv::Mat gray;
            cv::cvtColor(display_frame, gray, cv::COLOR_BGR2GRAY);
            std::vector<cv::Rect> faces;
            face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(80, 80));

            for (const auto& face : faces) {
                dlib::cv_image<dlib::bgr_pixel> cimg(display_frame);
                dlib::rectangle dlib_rect(face.x, face.y, face.x + face.width, face.y + face.height);
                dlib::full_object_detection shape = predictor(cimg, dlib_rect);

                auto left_eye = extract_eye(shape, true);
                auto right_eye = extract_eye(shape, false);
                float left_ear = eye_aspect_ratio(left_eye);
                float right_ear = eye_aspect_ratio(right_eye);
                ear = (left_ear + right_ear) / 2.0f;

                const float EAR_THRESHOLD = 0.25f;
                const int EYES_CLOSED_FRAMES = 15;

                if (ear < EAR_THRESHOLD) {
                    drowsiness_counter++;
                    if (drowsiness_counter >= EYES_CLOSED_FRAMES)
                        alarm = true;
                } else {
                    drowsiness_counter = 0;
                }
            }
        } catch (...) {}
    }

    // 更新 Qt 界面
    if (window) {
        window->updateImage(display_frame, fps, ear, alarm);
    }
}

int FatigueCallback::getTotalFrames() const {
    return total_frames;
}
