#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <libcamera/libcamera.h>
#include <sys/mman.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <atomic>
#include <mutex>
#include <thread>

// 参数配置
constexpr double EAR_THRESHOLD = 0.21;   // 可配置化
constexpr int EAR_CONSEC_FRAMES = 15;    // 连续帧阈值
constexpr int DETECTION_INTERVAL = 3;    // 检测间隔帧数
constexpr int RESIZE_SCALE = 0.5;        // 图像缩小比例

// 资源管理RAII类
class CameraRAII {
public:
    CameraRAII(libcamera::CameraManager& cm) : cm_(cm) { cm_.start(); }
    ~CameraRAII() { cm_.stop(); }
private:
    libcamera::CameraManager& cm_;
};

// 疲劳检测器类
class FatigueDetector {
public:
    FatigueDetector(const std::string& modelPath) {
        try {
            dlib::deserialize(modelPath) >> predictor_;
        } catch (dlib::serialization_error& e) {
            throw std::runtime_error("模型加载失败: " + std::string(e.what()));
        }
        detector_ = dlib::get_frontal_face_detector();
    }

    double computeEAR(const std::vector<cv::Point>& eye) {
        double A = cv::norm(eye[1] - eye[5]);
        double B = cv::norm(eye[2] - eye[4]);
        double C = cv::norm(eye[0] - eye[3]);
        return (A + B) / (2.0 * C);
    }

    bool detect(const cv::Mat& frame) {
        cv::Mat smallFrame;
        cv::resize(frame, smallFrame, cv::Size(), RESIZE_SCALE, RESIZE_SCALE);
        
        dlib::cv_image<dlib::bgr_pixel> dlibImg(smallFrame);
        auto faces = detector_(dlibImg);
        
        if (faces.empty()) return false;

        auto& face = faces[0]; // 只处理第一张人脸
        auto shape = predictor_(dlibImg, face);

        auto leftEye = getEyePoints(shape, true);
        auto rightEye = getEyePoints(shape, false);

        double ear = (computeEAR(leftEye) + computeEAR(rightEye)) / 2.0;
        return ear < EAR_THRESHOLD;
    }

private:
    std::vector<cv::Point> getEyePoints(const dlib::full_object_detection& shape, bool left) {
        std::vector<cv::Point> points;
        int start = left ? 36 : 42;
        for (int i = 0; i < 6; ++i) {
            auto pt = shape.part(start + i);
            points.emplace_back(pt.x() / RESIZE_SCALE, pt.y() / RESIZE_SCALE);
        }
        return points;
    }

    dlib::frontal_face_detector detector_;
    dlib::shape_predictor predictor_;
};

int main() {
    try {
        // 初始化摄像头
        libcamera::CameraManager cm;
        CameraRAII cameraManagerRAII(cm); // RAII管理
        
        auto cameras = cm.cameras();
        if (cameras.empty()) throw std::runtime_error("没有可用摄像头");
        
        auto camera = cameras.front();
        auto config = camera->generateConfiguration({libcamera::StreamRole::VideoRecording});
        if (!config || config->validate() != libcamera::CameraConfiguration::Valid) {
            throw std::runtime_error("摄像头配置失败");
        }

        config->at(0).size = {640, 480};  // 设置分辨率
        config->at(0).bufferCount = 4;
        if (camera->configure(config.get()) < 0) {
            throw std::runtime_error("配置应用失败");
        }

        // 映射内存
        std::vector<void*> mappedBuffers;
        for (const auto& buffer : config->at(0).buffers) {
            const auto& plane = buffer->planes()[0];
            void* data = mmap(nullptr, plane.length, PROT_READ, MAP_SHARED, plane.fd.get(), 0);
            if (data == MAP_FAILED) throw std::runtime_error("内存映射失败");
            mappedBuffers.push_back(data);
        }

        // 创建请求池
        std::vector<std::unique_ptr<libcamera::Request>> requests;
        for (unsigned int i = 0; i < config->at(0).bufferCount; ++i) {
            auto req = camera->createRequest();
            if (!req) throw std::runtime_error("创建请求失败");
            
            if (req->addBuffer(config->at(0).stream(), config->at(0).buffers[i].get()) < 0) {
                throw std::runtime_error("添加缓冲区失败");
            }
            requests.push_back(std::move(req));
        }

        if (camera->start() < 0) throw std::runtime_error("摄像头启动失败");

        // 初始化检测器
        FatigueDetector detector("shape_predictor_68_face_landmarks.dat");
        
        // 状态变量
        int frameCounter = 0;
        bool fatigued = false;
        int frameCount = 0;
        cv::Mat displayFrame;

        // 主循环
        for (auto& req : requests) camera->queueRequest(req.get());
        
        while (true) {
            auto req = camera->waitForCompletedRequest();
            auto* buffer = req->buffers()[config->at(0).stream()];
            const auto& plane = buffer->planes()[0];
            
            // YUV420转BGR
            cv::Mat yuv(480 * 3/2, 640, CV_8UC1, mappedBuffers[buffer->index()]);
            cv::cvtColor(yuv, displayFrame, cv::COLOR_YUV2BGR_I420);
            
            // 每DETECTION_INTERVAL帧检测一次
            bool currentState = false;
            if (++frameCount % DETECTION_INTERVAL == 0) {
                currentState = detector.detect(displayFrame);
            }

            // 疲劳状态判断
            if (currentState) {
                if (++frameCounter >= EAR_CONSEC_FRAMES) {
                    fatigued = true;
                }
            } else {
                frameCounter = 0;
                fatigued = false;
            }

            // 绘制界面
            cv::putText(displayFrame, 
                       fatigued ? "FATIGUE ALERT!" : "NORMAL", 
                       {10, 30}, 
                       cv::FONT_HERSHEY_SIMPLEX, 
                       0.8, 
                       fatigued ? cv::Scalar(0,0,255) : cv::Scalar(0,255,0), 
                       2);
            
            cv::imshow("Fatigue Detection", displayFrame);
            if (cv::waitKey(1) == 'q') break;

            camera->queueRequest(req.get()); // 重新入队
        }

        // 清理资源
        for (auto ptr : mappedBuffers) munmap(ptr, config->at(0).size.width * config->at(0).size.height * 3/2);
        camera->stop();

    } catch (const std::exception& e) {
        std::cerr << "发生错误: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}