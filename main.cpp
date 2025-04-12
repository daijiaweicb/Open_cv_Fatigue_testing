#include <iostream>
#include <cstdio>
#include <vector>
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
    // 降低分辨率以提升性能
    const int width = 640;
    const int height = 480;
    const int frame_size = width * height * 3 / 2;  // YUV420
    std::vector<uchar> buffer(frame_size);
    cv::Mat yuvImg(height + height / 2, width, CV_8UC1);
    cv::Mat bgrImg, gray;
    
    // 启动 libcamera-vid，降低分辨率和帧率
    FILE* pipe = popen("libcamera-vid --width 640 --height 480 --framerate 10 "
                       "--lens-position 0.2 --codec yuv420 --nopreview --timeout 0 -o -", "r");
    if (!pipe) {
        std::cerr << "无法启动 libcamera-vid" << std::endl;
        return -1;
    }
    
    // 加载关键点预测模型
    dlib::shape_predictor predictor;
    try {
        dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> predictor;
    } catch (std::exception& e) {
        std::cerr << "无法加载关键点模型：" << e.what() << std::endl;
        return -1;
    }
    
    // 加载 Haar 人脸检测器
    cv::CascadeClassifier face_cascade;
    if (!face_cascade.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")) {
        std::cerr << "无法加载 Haar 模型" << std::endl;
        return -1;
    }
    
    const float EAR_THRESHOLD = 0.25f;
    const int EYES_CLOSED_FRAMES = 15;
    int counter = 0;
    
    // 用于近距离检测的优化参数
    float scaleFactor = 1.05;     // 更小的缩放因子，提高检测精度
    int minNeighbors = 2;         // 降低阈值以更容易检测到人脸
    cv::Size minFaceSize(40, 40); // 更小的最小检测尺寸
    
    int frame_count = 0;
    
    while (true) {
        size_t read_bytes = fread(buffer.data(), 1, frame_size, pipe);
        if (read_bytes != frame_size) {
            std::cerr << "读取失败或结束" << std::endl;
            break;
        }
        
        memcpy(yuvImg.data, buffer.data(), frame_size);
        cv::cvtColor(yuvImg, bgrImg, cv::COLOR_YUV2BGR_I420);
        cv::cvtColor(bgrImg, gray, cv::COLOR_BGR2GRAY);
        
        // 每隔几帧进行一次完整的人脸检测，减少计算负担
        frame_count++;
        std::vector<cv::Rect> faces;
        
        if (frame_count % 3 == 0) { // 每3帧检测一次人脸
            face_cascade.detectMultiScale(gray, faces, scaleFactor, minNeighbors, 0, minFaceSize);
        }
        
        for (const auto& face : faces) {
            cv::rectangle(bgrImg, face, cv::Scalar(255, 0, 0), 2);
            
            dlib::cv_image<dlib::bgr_pixel> cimg(bgrImg);
            dlib::rectangle dlib_rect(face.x, face.y, face.x + face.width, face.y + face.height);
            dlib::full_object_detection shape = predictor(cimg, dlib_rect);
            
            // 只在每3帧计算一次EAR，减少CPU负载
            if (frame_count % 3 == 0) {
                auto left_eye = extract_eye(shape, true);
                auto right_eye = extract_eye(shape, false);
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
                
                // 简化眼睛关键点绘制，只绘制眼睛轮廓的几个关键点
                for (int i = 0; i < 6; i += 2) {
                    cv::circle(bgrImg, left_eye[i], 2, cv::Scalar(0, 255, 0), -1);
                    cv::circle(bgrImg, right_eye[i], 2, cv::Scalar(0, 255, 0), -1);
                }
            }
        }
        
        cv::imshow("Fatigue Detection", bgrImg);
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;
    }
    
    pclose(pipe);
    return 0;
}