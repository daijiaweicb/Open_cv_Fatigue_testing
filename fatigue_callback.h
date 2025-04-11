#ifndef FATIGUE_CALLBACK_H
#define FATIGUE_CALLBACK_H

#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <mutex>
#include <chrono>
#include "libcam2opencv.h"

class Window;  // 前向声明

class FatigueCallback : public Libcam2OpenCV::Callback {
public:
    FatigueCallback();
    ~FatigueCallback();

    void hasFrame(const cv::Mat &frame, const libcamera::ControlList &) override;
    int getTotalFrames() const;

    Window* window = nullptr;  // Qt 主窗口指针，供界面更新用

private:
    dlib::shape_predictor predictor;
    cv::CascadeClassifier face_cascade;
    cv::Mat latest_frame;

    std::chrono::time_point<std::chrono::steady_clock> start_time;
    int frame_counter;
    int total_frames;
    float fps;
    time_t last_log_time;

    bool dlib_loaded;
    bool haar_loaded;
    int drowsiness_counter = 0;

    float eye_aspect_ratio(const std::vector<cv::Point2f>& eye);
    std::vector<cv::Point2f> extract_eye(const dlib::full_object_detection& shape, bool left);
};

#endif // FATIGUE_CALLBACK_H
