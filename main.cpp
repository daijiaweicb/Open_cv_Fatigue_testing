#include <iostream>
#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <atomic>
#include <mutex>
#include <signal.h>
#include <thread>
#include "libcam2opencv.h"

// 全局变量
std::atomic<bool> g_running(true);
std::mutex frame_mutex;

// 信号处理函数
void signal_handler(int signal) {
    std::cout << "接收到信号: " << signal << "，准备优雅退出" << std::endl;
    g_running = false;
}

// 眼睛纵横比计算
float eye_aspect_ratio(const std::vector<cv::Point2f>& eye) {
    float A = cv::norm(eye[1] - eye[5]);
    float B = cv::norm(eye[2] - eye[4]);
    float C = cv::norm(eye[0] - eye[3]);
    return (A + B) / (2.0f * C);
}

// 提取眼睛关键点
std::vector<cv::Point2f> extract_eye(const dlib::full_object_detection& shape, bool left) {
    std::vector<cv::Point2f> eye;
    int start = left ? 36 : 42;
    for (int i = 0; i < 6; ++i)
        eye.emplace_back(shape.part(start + i).x(), shape.part(start + i).y());
    return eye;
}

// 回调类实现
class FatigueCallback : public Libcam2OpenCV::Callback {
public:
    FatigueCallback() : 
        frame_counter(0), 
        total_frames(0),
        fps(0),
        last_log_time(0),
        dlib_loaded(false),
        haar_loaded(false) {
        
        // 初始计时
        start_time = std::chrono::steady_clock::now();
        
        // 尝试加载 dlib 关键点模型
        try {
            dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> predictor;
            std::cout << "✅ 成功加载面部关键点模型" << std::endl;
            dlib_loaded = true;
        } catch (std::exception& e) {
            std::cerr << "❌ 无法加载关键点模型：" << e.what() << std::endl;
            std::cerr << "将继续运行但不进行疲劳检测" << std::endl;
        }

        // 尝试加载 Haar 人脸检测器
        if (face_cascade.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")) {
            std::cout << "✅ 成功加载人脸检测器" << std::endl;
            haar_loaded = true;
        } else {
            std::cerr << "❌ 无法加载 Haar 模型" << std::endl;
            std::cerr << "将继续运行但不进行人脸检测" << std::endl;
        }
        
        // 创建窗口
        cv::namedWindow("摄像头测试", cv::WINDOW_NORMAL);
        cv::resizeWindow("摄像头测试", 800, 600);
        std::cout << "✅ 初始化完成，等待摄像头帧..." << std::endl;
    }
    
    ~FatigueCallback() {
        std::cout << "销毁回调对象，清理资源..." << std::endl;
        cv::destroyAllWindows();
    }

    void hasFrame(const cv::Mat &frame, const libcamera::ControlList &metadata) override {
        // 每次收到帧时记录日志
        total_frames++;
        if (total_frames <= 5 || total_frames % 30 == 0) {
            std::cout << "📷 收到帧 #" << total_frames << " 大小: " << 
                frame.cols << "x" << frame.rows << " 类型: " << frame.type() << std::endl;
        }
        
        // 帧计数器增加
        frame_counter++;
        
        // 计算当前时间和经过的时间
        auto current_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            current_time - start_time).count();
        
        // 计算FPS (每秒更新一次)
        if (elapsed >= 1000) {  // 每秒更新一次
            fps = static_cast<float>(frame_counter) * 1000.0f / elapsed;
            std::cout << "⏱️ FPS计算: " << frame_counter << " 帧在 " << 
                elapsed << " 毫秒 = " << fps << " FPS" << std::endl;
            
            frame_counter = 0;
            start_time = current_time;
            
            // 记录到控制台 (每5秒)
            auto now = std::time(nullptr);
            if (now - last_log_time >= 5) {
                std::cout << "📊 当前FPS: " << fps << " 总帧数: " << total_frames << std::endl;
                last_log_time = now;
            }
        }

        // 创建一个工作副本
        cv::Mat display_frame = frame.clone();
        
        // 确保帧有效
        if (display_frame.empty()) {
            std::cerr << "❌ 收到空帧！" << std::endl;
            return;
        }

        // 显示基本信息
        std::string fps_text = "FPS: " + std::to_string(fps).substr(0, fps > 0 ? 5 : 1);
        cv::putText(display_frame, fps_text, cv::Point(10, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
                   
        // 显示帧计数
        cv::putText(display_frame, "Frame: " + std::to_string(total_frames), 
                   cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, 
                   cv::Scalar(0, 255, 0), 2);

        // 只有当两个模型都加载成功时才进行疲劳检测
        if (dlib_loaded && haar_loaded) {
            try {
                // 转灰度图进行人脸检测
                cv::Mat gray;
                cv::cvtColor(display_frame, gray, cv::COLOR_BGR2GRAY);

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
                        std::cerr << "❌ 处理人脸特征点时出错: " << e.what() << std::endl;
                    }
                }
            } catch (std::exception& e) {
                std::cerr << "❌ 处理帧时出错: " << e.what() << std::endl;
            }
        } else {
            // 如果模型未能加载，显示简单警告
            cv::putText(display_frame, "模型未加载完全，仅显示摄像头画面", 
                       cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.7, 
                       cv::Scalar(0, 0, 255), 2);
        }

        // 更新最新帧并显示
        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            latest_frame = display_frame.clone();
        }
        
        // 显示图像
        cv::imshow("摄像头测试", display_frame);
    }
    
    // 获取最新处理过的帧
    cv::Mat getLatestFrame() {
        std::lock_guard<std::mutex> lock(frame_mutex);
        return latest_frame.clone();
    }

    // 获取总帧数
    int getTotalFrames() const {
        return total_frames;
    }

private:
    dlib::shape_predictor predictor;
    cv::CascadeClassifier face_cascade;
    cv::Mat latest_frame;
    
    // FPS计算
    std::chrono::time_point<std::chrono::steady_clock> start_time;
    int frame_counter;
    int total_frames;
    float fps;
    time_t last_log_time;
    
    // 模型状态
    bool dlib_loaded;
    bool haar_loaded;
    
    // 疲劳检测
    int drowsiness_counter = 0;
};

// 修改libcam2opencv.cpp的实现
// 这部分需要单独编译链接，这里只是伪代码说明修改点
/*
void Libcam2OpenCV::requestComplete(libcamera::Request *request) {
    static int request_counter = 0;
    request_counter++;
    
    std::cout << "===== 请求完成 #" << request_counter << " =====" << std::endl;
    
    if (nullptr == request) {
        std::cerr << "请求指针为空!" << std::endl;
        return;
    }
    
    if (request->status() == libcamera::Request::RequestCancelled) {
        std::cerr << "请求被取消!" << std::endl;
        return;
    }

    // 处理缓冲区
    const libcamera::Request::BufferMap &buffers = request->buffers();
    std::cout << "缓冲区数量: " << buffers.size() << std::endl;
    
    for (auto bufferPair : buffers) {
        libcamera::FrameBuffer *buffer = bufferPair.second;
        libcamera::StreamConfiguration &streamConfig = config->at(0);
        unsigned int vw = streamConfig.size.width;
        unsigned int vh = streamConfig.size.height;
        unsigned int vstr = streamConfig.stride;
        
        std::cout << "处理帧: " << vw << "x" << vh << ", stride=" << vstr << std::endl;
        
        auto mem = Mmap(buffer);
        if (mem.empty()) {
            std::cerr << "内存映射为空!" << std::endl;
            continue;
        }
        
        frame.create(vh, vw, CV_8UC3);
        uint ls = vw*3;
        uint8_t *ptr = mem[0].data();
        for (unsigned int i = 0; i < vh; i++, ptr += vstr) {
            memcpy(frame.ptr(i), ptr, ls);
        }
        
        if (nullptr != callback) {
            std::cout << "调用回调函数..." << std::endl;
            callback->hasFrame(frame, request->metadata());
        } else {
            std::cerr << "回调为空!" << std::endl;
        }
    }

    // 检查请求状态
    std::cout << "重新入队请求..." << std::endl;
    try {
        if (request->status() == libcamera::Request::RequestCancelled) {
            std::cerr << "请求在处理过程中被取消!" << std::endl;
            return;
        }
        
        request->reuse(libcamera::Request::ReuseBuffers);
        int ret = camera->queueRequest(request);
        if (ret < 0) {
            std::cerr << "重新入队请求失败: " << ret << std::endl;
        } else {
            std::cout << "请求成功重新入队" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "重新入队请求时发生异常: " << e.what() << std::endl;
    }
}
*/

// 主函数
int main() {
    // 注册信号处理
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    try {
        std::cout << "=====================================" << std::endl;
        std::cout << "   疲劳检测系统 - 深度调试版" << std::endl;
        std::cout << "=====================================" << std::endl;
        
        std::cout << "📷 初始化摄像头..." << std::endl;
        
        // 创建相机和回调对象
        Libcam2OpenCV cam;
        FatigueCallback callback;
        
        // 注册回调
        cam.registerCallback(&callback);

        // 配置相机 - 使用最低分辨率开始测试
        Libcam2OpenCVSettings settings;
        settings.width = 320;     // 最低分辨率便于调试
        settings.height = 240;
        settings.framerate = 15;  // 低帧率以减少处理负担
        settings.brightness = 0.0;
        settings.contrast = 1.0;
        
        // 启动相机
        std::cout << "🚀 启动摄像头..." << std::endl;
        cam.start(settings);
        std::cout << "✅ 摄像头已启动" << std::endl;
        
        // 创建监视线程，每秒检查一次帧更新情况
        int last_frame_count = 0;
        std::thread monitor_thread([&]() {
            while (g_running) {
                std::this_thread::sleep_for(std::chrono::seconds(3));
                int current_frames = callback.getTotalFrames();
                if (current_frames == last_frame_count) {
                    std::cerr << "⚠️ 警告: 3秒内没有新的帧!" << std::endl;
                } else {
                    std::cout << "✓ 3秒内收到 " << (current_frames - last_frame_count) << " 个新帧" << std::endl;
                }
                last_frame_count = current_frames;
            }
        });

        // 主事件循环
        std::cout << "👁️ 开始检测 (按 'q' 或 ESC 键退出)" << std::endl;
        
        while (g_running) {
            int key = cv::waitKey(30); // 30ms延迟
            
            if (key == 'q' || key == 27) { // 'q' 或 ESC 退出
                std::cout << "⏹️ 用户请求退出..." << std::endl;
                g_running = false;
                break;
            }
        }
        
        std::cout << "🛑 正在停止摄像头..." << std::endl;
        cam.stop();
        
        // 等待监视线程结束
        if (monitor_thread.joinable()) {
            monitor_thread.join();
        }
        
        std::cout << "✅ 程序已安全退出" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 程序发生致命错误: " << e.what() << std::endl;
        return 1;
    }
}