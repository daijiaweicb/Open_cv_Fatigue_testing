#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <iostream>
#include <libcamera/libcamera.h>
#include <libcamera/camera_manager.h>

double compute_ear(const std::vector<dlib::point>& eye) {
    double A = dlib::length(eye[1] - eye[5]);
    double B = dlib::length(eye[2] - eye[4]);
    double C = dlib::length(eye[0] - eye[3]);
    return (A + B) / (2.0 * C);
}

int main() {
    // 创建 libcamera 的 CameraManager 和获取摄像头设备
    libcamera::CameraManager* cm = libcamera::CameraManager::instance();
    cm->start();

    // 获取第一个摄像头
    libcamera::Camera* camera = cm->get(0);
    if (!camera) {
        std::cerr << "未找到摄像头！" << std::endl;
        return -1;
    }

    // 配置摄像头
    camera->acquire();
    libcamera::StreamConfiguration cfg = camera->generateConfiguration({libcamera::StreamRole::VideoRecording});
    cfg.size = libcamera::Size(640, 480); // 设置分辨率
    camera->configure(cfg);

    // 创建 OpenCV 窗口
    cv::namedWindow("Fatigue Detection", cv::WINDOW_NORMAL);

    // Dlib 人脸检测模型
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor pose_model;
    dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

    const double EAR_THRESHOLD = 0.25;
    const int EAR_CONSEC_FRAMES = 20;

    int frame_counter = 0;
    bool alarm_on = false;

    // 捕获图像流
    while (true) {
        // 请求一个图像帧
        libcamera::Request* request = camera->createRequest();
        libcamera::FrameBuffer* buffer = camera->requestBuffer(request);

        if (!request || !buffer) {
            std::cerr << "无法获取图像帧！" << std::endl;
            break;
        }

        // 处理请求并获取帧数据
        camera->queueRequest(request);

        // 将 libcamera 图像数据转换为 OpenCV Mat 格式
        cv::Mat frame(cfg.size.height, cfg.size.width, CV_8UC3, buffer->data());

        if (frame.empty()) {
            std::cerr << "获取图像数据失败！" << std::endl;
            break;
        }

        dlib::cv_image<dlib::bgr_pixel> cimg(frame);
        std::vector<dlib::rectangle> faces = detector(cimg);

        for (auto face : faces) {
            dlib::full_object_detection shape = pose_model(cimg, face);

            std::vector<dlib::point> left_eye{
                shape.part(36), shape.part(37), shape.part(38),
                shape.part(39), shape.part(40), shape.part(41)
            };
            std::vector<dlib::point> right_eye{
                shape.part(42), shape.part(43), shape.part(44),
                shape.part(45), shape.part(46), shape.part(47)
            };

            double leftEAR = compute_ear(left_eye);
            double rightEAR = compute_ear(right_eye);
            double ear = (leftEAR + rightEAR) / 2.0;

            if (ear < EAR_THRESHOLD) {
                frame_counter++;
                if (frame_counter >= EAR_CONSEC_FRAMES) {
                    if (!alarm_on) {
                        alarm_on = true;
                        std::cout << "⚠ 疲劳检测：眨眼时间过长！" << std::endl;
                    }
                    cv::putText(frame, "DROWSINESS ALERT!", cv::Point(50, 100),
                                cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 0, 255), 4);
                }
            } else {
                frame_counter = 0;
                alarm_on = false;
            }
        }

        cv::imshow("Fatigue Detection", frame);

        if (cv::waitKey(1) == 27) break; // 按下 ESC 退出
    }

    // 清理资源
    camera->release();
    cm->stop();

    return 0;
}
