#include <iostream>
#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <atomic>
#include <mutex>
#include "libcam2opencv.h"

// 全局变量用于线程间通信
std::atomic<bool> g_running(true);
std::mutex frame_mutex;

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
    FatigueCallback() : frame_counter(0), fps(0), last_log_time(0) {
        // 初始计时
        start_time = std::chrono::steady_clock::now();
        
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
        cv::namedWindow("Fatigue Detection", cv::WINDOW_NORMAL);
        std::cout << "初始化完成，等待摄像头帧..." << std::endl;
    }
    
    ~FatigueCallback() {
        // 确保窗口被正确销毁
        cv::destroyAllWindows();
    }

    void hasFrame(const cv::Mat &frame, const libcamera::ControlList &) override {
        // 帧计数器增加
        frame_counter++;
        
        // 计算当前时间和经过的时间
        auto current_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            current_time - start_time).count();
        
        // 计算FPS (每秒更新一次)
        if (elapsed >= 1000) {  // 每秒更新一次
            fps = static_cast<float>(frame_counter) * 1000.0f / elapsed;
            frame_counter = 0;
            start_time = current_time;
            
            // 记录到控制台 (每5秒)
            auto now = std::time(nullptr);
            if (now - last_log_time >= 5) {
                std::cout << "当前FPS: " << fps << std::endl;
                last_log_time = now;
            }
        }

        // 处理帧
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::Mat display_frame = frame.clone();

        // 显示FPS
        std::string fps_text = "FPS: " + std::to_string(fps).substr(0, 5);
        cv::putText(display_frame, fps_text, cv::Point(10, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
                   
        // 显示帧计数
        cv::putText(display_frame, "Frame: " + std::to_string(frame_counter), 
                   cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, 
                   cv::Scalar(0, 255, 0), 2);

        try {
            // 人脸检测
            std::vector<cv::Rect> faces;
            face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(80, 80));
            
            if (!faces.empty()) {
                cv::putText(display_frame, "Faces: " + std::to_string(faces.size()),
                       cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.7, 
                       cv::Scalar(0, 255, 0), 2);
            }

            for (const auto& face : faces) {
                cv::rectangle(display_frame, face, cv::Scalar(255, 0, 0), 2);

                try {
                    dlib::cv_image<dlib::bgr_pixel> cimg(display_frame);
                    dlib::rectangle dlib_rect(face.x, face.y, face.x + face.width, face.y + face.height);
                    dlib::full_object_detection shape = predictor(cimg, dlib_rect);

                    auto left_eye = extract_eye(shape, true);
                    auto right_eye = extract_eye(shape, false);
                    float left_ear = eye_aspect_ratio(left_eye);
                    float right_ear = eye_aspect_ratio(right_eye);
                    float ear = (left_ear + right_ear) / 2.0f;

                    // 显示 EAR 值
                    cv::putText(display_frame, "EAR: " + std::to_string(ear).substr(0, 5), 
                                cv::Point(face.x, face.y - 10),
                                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

                    const float EAR_THRESHOLD = 0.25f;
                    const int EYES_CLOSED_FRAMES = 15;

                    if (ear < EAR_THRESHOLD) {
                        drowsiness_counter++;
                        if (drowsiness_counter >= EYES_CLOSED_FRAMES) {
                            cv::putText(display_frame, "DROWSINESS ALERT!", cv::Point(50, 120),
                                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
                        }
                    } else {
                        drowsiness_counter = 0;
                    }

                    // 绘制眼睛关键点和轮廓
                    for (const auto& pt : left_eye) cv::circle(display_frame, pt, 2, cv::Scalar(0, 255, 0), -1);
                    for (const auto& pt : right_eye) cv::circle(display_frame, pt, 2, cv::Scalar(0, 255, 0), -1);
                    
                    for (int i = 0; i < 5; i++) {
                        cv::line(display_frame, left_eye[i], left_eye[i+1], cv::Scalar(255, 0, 0), 1);
                        cv::line(display_frame, right_eye[i], right_eye[i+1], cv::Scalar(255, 0, 0), 1);
                    }
                    cv::line(display_frame, left_eye[5], left_eye[0], cv::Scalar(255, 0, 0), 1);
                    cv::line(display_frame, right_eye[5], right_eye[0], cv::Scalar(255, 0, 0), 1);
                } catch (std::exception& e) {
                    std::cerr << "处理人脸特征点时出错: " << e.what() << std::endl;
                }
            }
        } catch (std::exception& e) {
            std::cerr << "处理帧时出错: " << e.what() << std::endl;
        }

        // 更新最新帧
        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            latest_frame = display_frame.clone();
        }
        
        // 显示图像
        cv::imshow("Fatigue Detection", display_frame);
    }
    
    // 获取最新处理过的帧
    cv::Mat getLatestFrame() {
        std::lock_guard<std::mutex> lock(frame_mutex);
        return latest_frame.clone();
    }

private:
    dlib::shape_predictor predictor;
    cv::CascadeClassifier face_cascade;
    cv::Mat latest_frame;
    
    // FPS计算
    std::chrono::time_point<std::chrono::steady_clock> start_time;
    int frame_counter;
    float fps;
    time_t last_log_time;
    
    // 疲劳检测
    int drowsiness_counter = 0;
};

int main() {
    try {
        std::cout << "初始化摄像头..." << std::endl;
        
        // 创建相机和回调对象
        Libcam2OpenCV cam;
        FatigueCallback callback;
        
        // 注册回调
        cam.registerCallback(&callback);

        // 配置相机
        Libcam2OpenCVSettings settings;
        settings.width = 640;    // 降低分辨率以提高性能
        settings.height = 480;
        settings.framerate = 15;
        settings.brightness = 0.1;  // 稍微调整亮度
        settings.contrast = 1.1;    // 稍微提高对比度
        
        // 启动相机
        std::cout << "启动摄像头..." << std::endl;
        cam.start(settings);
        std::cout << "摄像头已启动，开始检测..." << std::endl;

        // 主事件循环
        std::cout << "按 'q' 或 ESC 键退出程序" << std::endl;
        
        while (g_running) {
            int key = cv::waitKey(30); // 30ms延迟，适应大约30fps
            
            if (key == 'q' || key == 27) { // 'q' 或 ESC 退出
                g_running = false;
                break;
            }
        }
        
        std::cout << "正在停止摄像头..." << std::endl;
        cam.stop();
        std::cout << "程序已安全退出" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "程序发生错误: " << e.what() << std::endl;
        return 1;
    }
}