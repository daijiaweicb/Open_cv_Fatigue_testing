#include <iostream>
#include <vector>
#include <cstdio>

#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>

// 计算 EAR
float eye_aspect_ratio(const std::vector<cv::Point2f>& eye) {
    float A = cv::norm(eye[1] - eye[5]);
    float B = cv::norm(eye[2] - eye[4]);
    float C = cv::norm(eye[0] - eye[3]);
    return (A + B) / (2.0 * C);
}

// 提取左/右眼特征点
std::vector<cv::Point2f> extract_eye(const dlib::full_object_detection& shape, bool left) {
    std::vector<cv::Point2f> eye;
    int start = left ? 36 : 42;
    for (int i = 0; i < 6; ++i) {
        eye.emplace_back(
            static_cast<float>(shape.part(start + i).x()),
            static_cast<float>(shape.part(start + i).y())
        );
    }
    return eye;
}

int main() {
    // 启动 libcamera-vid
    FILE* pipe = popen("libcamera-vid -t 0 --codec yuv420 --width 640 --height 480 -n -o -", "r");
    if (!pipe) {
        std::cerr << "无法启动 libcamera-vid，请检查摄像头连接状态" << std::endl;
        return -1;
    }

    // 初始化 dlib 人脸检测器和特征点预测器
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor predictor;
    try {
        dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> predictor;
    } catch (std::exception& e) {
        std::cerr << "无法加载 landmark 模型文件：" << e.what() << std::endl;
        return -1;
    }

    int width = 640, height = 480;
    int frame_size = width * height * 3 / 2;
    std::vector<unsigned char> buffer(frame_size);
    cv::Mat yuvImg(height + height / 2, width, CV_8UC1);
    cv::Mat bgrImg;

    const float EAR_THRESHOLD = 0.25f;
    const int EYES_CLOSED_FRAMES = 15;
    int counter = 0;

    while (true) {
        size_t read_bytes = fread(buffer.data(), 1, frame_size, pipe);
        if (read_bytes != static_cast<size_t>(frame_size)) {
            std::cerr << "读取失败，可能摄像头断开或输出结束" << std::endl;
            break;
        }

        memcpy(yuvImg.data, buffer.data(), frame_size);
        cv::cvtColor(yuvImg, bgrImg, cv::COLOR_YUV2BGR_I420);

        dlib::cv_image<dlib::bgr_pixel> cimg(bgrImg);
        std::vector<dlib::rectangle> faces = detector(cimg);

        for (const auto& face : faces) {
            dlib::full_object_detection shape = predictor(cimg, face);
            if (shape.num_parts() != 68) continue;

            auto left_eye = extract_eye(shape, true);
            auto right_eye = extract_eye(shape, false);
            float ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0f;

            if (ear < EAR_THRESHOLD) {
                counter++;
                if (counter >= EYES_CLOSED_FRAMES) {
                    cv::putText(bgrImg, "DROWSINESS ALERT! " + std::to_string(counter),
                                cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX,
                                1.0, cv::Scalar(0, 0, 255), 2);
                }
            } else {
                counter = 0;
            }

            // 画出眼睛关键点和轮廓线
            for (size_t i = 0; i < 6; ++i) {
                cv::circle(bgrImg, left_eye[i], 2, cv::Scalar(0, 255, 0), -1);
                cv::circle(bgrImg, right_eye[i], 2, cv::Scalar(0, 255, 0), -1);
                cv::line(bgrImg, left_eye[i], left_eye[(i + 1) % 6], cv::Scalar(255, 255, 0), 1);
                cv::line(bgrImg, right_eye[i], right_eye[(i + 1) % 6], cv::Scalar(255, 255, 0), 1);
            }
        }

        cv::imshow("Fatigue Detection", bgrImg);
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;
    }

    pclose(pipe);
    return 0;
}
