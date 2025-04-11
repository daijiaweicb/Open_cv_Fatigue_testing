#include <iostream>
#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <atomic>
#include "libcam2opencv.h"

// 全局变量用于线程间通信
std::atomic<bool> g_running(true);

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
            std::cout << "成功加载面部关键点模型" << std::endl;
        } catch (std::exception& e) {
            std::cerr << "无法加载关键点模型：" << e.what() << std::endl;
            std::exit(1);
        }

        // 加载 Haar 人脸检测器
        if (!face_cascade.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")) {
            std::cerr << "无法加载 Haar 模型" << std::endl;
            std::exit(1);
        } else {
            std::cout << "成功加载人脸检测器" << std::endl;
        }
        
        // 创建窗口
        cv::namedWindow("Fatigue Detection (libcam2opencv)", cv::WINDOW_AUTOSIZE);
    }
    
    ~FatigueCallback() {
        // 确保窗口被正确销毁
        cv::destroyWindow("Fatigue Detection (libcam2opencv)");
    }

    void hasFrame(const cv::Mat &frame, const libcamera::ControlList &) override {
        const float EAR_THRESHOLD = 0.25f;
        const int EYES_CLOSED_FRAMES = 15;
        
        // 显示原始帧率信息
        static int frame_count = 0;
        static auto last_time = std::chrono::high_resolution_clock::now();
        frame_count++;
        
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - last_time).count();
        if (elapsed >= 1) {
            fps = frame_count / elapsed;
            frame_count = 0;
            last_time = current_time;
        }

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(80, 80));

        cv::Mat draw = frame.clone();

        // 显示帧率
        cv::putText(draw, "FPS: " + std::to_string(fps), cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        for (const auto& face : faces) {
            cv::rectangle(draw, face, cv::Scalar(255, 0, 0), 2);

            dlib::cv_image<dlib::bgr_pixel> cimg(draw);
            dlib::rectangle dlib_rect(face.x, face.y, face.x + face.width, face.y + face.height);
            dlib::full_object_detection shape = predictor(cimg, dlib_rect);

            auto left_eye = extract_eye(shape, true);
            auto right_eye = extract_eye(shape, false);
            float left_ear = eye_aspect_ratio(left_eye);
            float right_ear = eye_aspect_ratio(right_eye);
            float ear = (left_ear + right_ear) / 2.0f;

            // 显示 EAR 值
            cv::putText(draw, "EAR: " + std::to_string(ear).substr(0, 5), 
                        cv::Point(face.x, face.y - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

            if (ear < EAR_THRESHOLD) {
                counter++;
                if (counter >= EYES_CLOSED_FRAMES) {
                    cv::putText(draw, "DROWSINESS ALERT!", cv::Point(50, 70),
                                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
                    // 可以在这里添加警报声音或其他通知
                }
            } else {
                counter = 0;
            }

            // 绘制眼睛关键点
            for (const auto& pt : left_eye) cv::circle(draw, pt, 2, cv::Scalar(0, 255, 0), -1);
            for (const auto& pt : right_eye) cv::circle(draw, pt, 2, cv::Scalar(0, 255, 0), -1);
            
            // 绘制眼睛轮廓
            for (int i = 0; i < 5; i++) {
                cv::line(draw, left_eye[i], left_eye[i+1], cv::Scalar(255, 0, 0), 1);
                cv::line(draw, right_eye[i], right_eye[i+1], cv::Scalar(255, 0, 0), 1);
            }
            cv::line(draw, left_eye[5], left_eye[0], cv::Scalar(255, 0, 0), 1);
            cv::line(draw, right_eye[5], right_eye[0], cv::Scalar(255, 0, 0), 1);
        }

        // 显示处理后的图像
        latest_frame = draw.clone();
        cv::imshow("Fatigue Detection (libcam2opencv)", draw);
    }
    
    // 获取最新的处理后的帧
    cv::Mat getLatestFrame() {
        return latest_frame;
    }

private:
    dlib::shape_predictor predictor;
    cv::CascadeClassifier face_cascade;
    int counter = 0;
    cv::Mat latest_frame;
    int fps = 0;
};

int main() {
    try {
        std::cout << "初始化摄像头..." << std::endl;
        Libcam2OpenCV cam;
        FatigueCallback callback;
        cam.registerCallback(&callback);

        Libcam2OpenCVSettings settings;
        settings.width = 1280;
        settings.height = 720;
        settings.framerate = 15;
        
        std::cout << "启动摄像头..." << std::endl;
        cam.start(settings);
        std::cout << "摄像头已启动，开始检测..." << std::endl;

        // 主事件循环
        std::cout << "按 'q' 或 ESC 键退出程序" << std::endl;
        while (g_running) {
            int key = cv::waitKey(10); // 处理窗口事件并等待10毫秒
            if (key == 'q' || key == 27) { // 'q' 或 ESC 退出
                g_running = false;
                break;
            }
        }
        
        std::cout << "停止摄像头..." << std::endl;
        cam.stop();
        std::cout << "程序已安全退出" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "发生错误: " << e.what() << std::endl;
        return 1;
    }
}