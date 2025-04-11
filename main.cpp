#include <iostream>
#include <vector>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>

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

int main() {
    const int width = 640;
    const int height = 480;
    const int frame_size = width * height * 3 / 2;  // YUV420
    std::vector<uchar> buffer(frame_size);
    cv::Mat yuvImg(height + height / 2, width, CV_8UC1);
    cv::Mat bgrImg, gray;

    // 启动 libcamera-vid
    FILE* pipe = popen("libcamera-vid --width 640 --height 480 --framerate 15 --codec yuv420 --nopreview --timeout 0 -o -", "r");
    if (!pipe) {
        std::cerr << "无法启动 libcamera-vid" << std::endl;
        return -1;
    }

    // 加载 dlib 模型
    dlib::shape_predictor predictor;
    try {
        dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> predictor;
    } catch (std::exception& e) {
        std::cerr << "无法加载模型：" << e.what() << std::endl;
        return -1;
    }

    // 加载 OpenCV 眼睛检测器
    cv::CascadeClassifier eyes_cascade;
    if (!eyes_cascade.load("/usr/share/opencv4/haarcascades/haarcascade_eye_tree_eyeglasses.xml")) {
        std::cerr << "无法加载眼睛 Haar 模型" << std::endl;
        return -1;
    }

    const float EAR_THRESHOLD = 0.25f;
    const int EYES_CLOSED_FRAMES = 15;
    int counter = 0;

    while (true) {
        size_t read_bytes = fread(buffer.data(), 1, frame_size, pipe);
        if (read_bytes != frame_size) {
            std::cerr << "读取失败或结束" << std::endl;
            break;
        }

        memcpy(yuvImg.data, buffer.data(), frame_size);
        cv::cvtColor(yuvImg, bgrImg, cv::COLOR_YUV2BGR_I420);
        cv::cvtColor(bgrImg, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Rect> eyes;
        eyes_cascade.detectMultiScale(gray, eyes, 1.1, 3, 0, cv::Size(30, 30));

        if (eyes.size() >= 2) {
            dlib::cv_image<dlib::bgr_pixel> cimg(bgrImg);

            // 分别扩展眼睛区域（保证包含周围面部结构）
            cv::Rect left_roi = eyes[0] + cv::Size(40, 40);
            cv::Rect right_roi = eyes[1] + cv::Size(40, 40);
            left_roi &= cv::Rect(0, 0, bgrImg.cols, bgrImg.rows);
            right_roi &= cv::Rect(0, 0, bgrImg.cols, bgrImg.rows);

            dlib::rectangle left_region(left_roi.x, left_roi.y, left_roi.x + left_roi.width, left_roi.y + left_roi.height);
            dlib::rectangle right_region(right_roi.x, right_roi.y, right_roi.x + right_roi.width, right_roi.y + right_roi.height);

            dlib::full_object_detection shape_left = predictor(cimg, left_region);
            dlib::full_object_detection shape_right = predictor(cimg, right_region);

            if (shape_left.num_parts() == 68 && shape_right.num_parts() == 68) {
                auto left_eye = extract_eye(shape_left, true);
                auto right_eye = extract_eye(shape_right, false);

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

                for (const auto& pt : left_eye) cv::circle(bgrImg, pt, 2, cv::Scalar(0, 255, 0), -1);
                for (const auto& pt : right_eye) cv::circle(bgrImg, pt, 2, cv::Scalar(0, 255, 0), -1);
            }
        }

        cv::imshow("Fatigue Detection (Eye-Only, dlib-stable)", bgrImg);
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;
    }

    pclose(pipe);
    return 0;
}
