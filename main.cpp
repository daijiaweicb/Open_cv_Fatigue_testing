#include <iostream>
#include <cstdio>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>

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
    // 可以调整分辨率以获得更好的近距离效果
    const int width = 1280;
    const int height = 720;
    const int frame_size = width * height * 3 / 2;  // YUV420
    std::vector<uchar> buffer(frame_size);
    cv::Mat yuvImg(height + height / 2, width, CV_8UC1);
    cv::Mat bgrImg, gray;
    
    // 启动 libcamera-vid，添加适合近距离拍摄的参数
    // --lens-position 调整焦距，数值越大越适合近距离
    FILE* pipe = popen("libcamera-vid --width 1280 --height 720 --framerate 15 "
                        "--lens-position 0.2 --codec yuv420 --nopreview --timeout 0 -o -", "r");
    if (!pipe) {
        std::cerr << "无法启动 libcamera-vid" << std::endl;
        return -1;
    }
    
    // 加载 dlib 的人脸检测器，比 Haar 更适合近距离和不同角度的人脸
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    
    // 加载关键点预测模型
    dlib::shape_predictor predictor;
    try {
        dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> predictor;
    } catch (std::exception& e) {
        std::cerr << "无法加载关键点模型：" << e.what() << std::endl;
        return -1;
    }
    
    // 保留 Haar 检测器作为备选
    cv::CascadeClassifier face_cascade;
    if (!face_cascade.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")) {
        std::cerr << "无法加载 Haar 模型" << std::endl;
        return -1;
    }
    
    const float EAR_THRESHOLD = 0.25f;
    const int EYES_CLOSED_FRAMES = 15;
    int counter = 0;
    
    // 用于跟踪上一帧检测到的人脸位置
    cv::Rect lastFaceRect;
    bool faceTracked = false;
    
    // 用于自适应调整检测参数
    float scaleFactor = 1.1;
    int minNeighbors = 3;
    cv::Size minFaceSize(60, 60);  // 降低最小人脸尺寸以便更好地检测近距离人脸
    
    while (true) {
        size_t read_bytes = fread(buffer.data(), 1, frame_size, pipe);
        if (read_bytes != frame_size) {
            std::cerr << "读取失败或结束" << std::endl;
            break;
        }
        
        memcpy(yuvImg.data, buffer.data(), frame_size);
        cv::cvtColor(yuvImg, bgrImg, cv::COLOR_YUV2BGR_I420);
        cv::cvtColor(bgrImg, gray, cv::COLOR_BGR2GRAY);
        
        // 增强对比度以提高检测率
        cv::equalizeHist(gray, gray);
        
        std::vector<cv::Rect> faces;
        std::vector<dlib::rectangle> dlib_faces;
        
        // 使用 dlib 的人脸检测器
        dlib::cv_image<unsigned char> dlib_gray(gray);
        dlib_faces = detector(dlib_gray);
        
        // 如果 dlib 没检测到人脸，尝试使用 Haar 级联分类器
        if (dlib_faces.empty()) {
            // 如果有之前的人脸位置，先在附近区域搜索
            if (faceTracked) {
                // 扩大搜索区域，确保能捕捉到移动的人脸
                cv::Rect searchROI = lastFaceRect;
                searchROI.x = std::max(0, searchROI.x - searchROI.width/4);
                searchROI.y = std::max(0, searchROI.y - searchROI.height/4);
                searchROI.width = std::min(gray.cols - searchROI.x, searchROI.width*3/2);
                searchROI.height = std::min(gray.rows - searchROI.y, searchROI.height*3/2);
                
                cv::Mat roi = gray(searchROI);
                std::vector<cv::Rect> localFaces;
                face_cascade.detectMultiScale(roi, localFaces, scaleFactor, minNeighbors, 0, minFaceSize);
                
                // 调整检测到的人脸坐标到原图坐标系
                for (auto& face : localFaces) {
                    face.x += searchROI.x;
                    face.y += searchROI.y;
                    faces.push_back(face);
                }
            }
            
            // 如果ROI中没有检测到，尝试在整个图像中检测
            if (faces.empty()) {
                face_cascade.detectMultiScale(gray, faces, scaleFactor, minNeighbors, 0, minFaceSize);
            }
            
            // 转换OpenCV人脸矩形到dlib格式
            for (const auto& face : faces) {
                dlib_faces.push_back(dlib::rectangle(face.x, face.y, face.x + face.width, face.y + face.height));
            }
        } else {
            // 将dlib检测结果转换为OpenCV格式用于显示
            for (const auto& face : dlib_faces) {
                faces.push_back(cv::Rect(face.left(), face.top(), face.width(), face.height()));
            }
        }
        
        // 如果检测到人脸，更新跟踪状态
        if (!faces.empty()) {
            lastFaceRect = faces[0]; // 假设第一个是主要人脸
            faceTracked = true;
            
            // 自适应调整最小人脸尺寸 - 如果人脸很大，说明很近，可以提高最小尺寸要求
            if (lastFaceRect.width > 0.4 * width) {
                minFaceSize = cv::Size(100, 100);
                scaleFactor = 1.05; // 更精细的缩放步长
            } else {
                minFaceSize = cv::Size(60, 60);
                scaleFactor = 1.1;
            }
        } else {
            // 如果连续多帧没检测到，重置跟踪状态
            static int noFaceCounter = 0;
            if (++noFaceCounter > 10) {
                faceTracked = false;
                noFaceCounter = 0;
            }
        }
        
        for (size_t i = 0; i < dlib_faces.size(); i++) {
            const auto& face_rect = faces[i];
            const auto& dlib_rect = dlib_faces[i];
            
            // 显示人脸框
            cv::rectangle(bgrImg, face_rect, cv::Scalar(255, 0, 0), 2);
            
            // 使用dlib提取关键点
            dlib::cv_image<dlib::bgr_pixel> cimg(bgrImg);
            dlib::full_object_detection shape = predictor(cimg, dlib_rect);
            
            // 检查是否成功检测到所有关键点
            if (shape.num_parts() == 68) {
                auto left_eye = extract_eye(shape, true);
                auto right_eye = extract_eye(shape, false);
                
                // 绘制眼睛关键点
                for (const auto& pt : left_eye) cv::circle(bgrImg, pt, 2, cv::Scalar(0, 255, 0), -1);
                for (const auto& pt : right_eye) cv::circle(bgrImg, pt, 2, cv::Scalar(0, 255, 0), -1);
                
                // 计算EAR
                float left_ear = eye_aspect_ratio(left_eye);
                float right_ear = eye_aspect_ratio(right_eye);
                float ear = (left_ear + right_ear) / 2.0f;
                
                // 在图像上显示EAR值
                cv::putText(bgrImg, "EAR: " + std::to_string(ear), cv::Point(10, 30),
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
                
                // 检测疲劳
                if (ear < EAR_THRESHOLD) {
                    counter++;
                    cv::putText(bgrImg, "Eyes Closed!", cv::Point(10, 70),
                                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
                    
                    if (counter >= EYES_CLOSED_FRAMES) {
                        cv::putText(bgrImg, "DROWSINESS ALERT!", cv::Point(10, 110),
                                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
                    }
                } else {
                    counter = 0;
                }
            }
        }
        
        // 显示当前状态
        std::string status = faceTracked ? "Face Tracked" : "Searching Face";
        cv::putText(bgrImg, status, cv::Point(width - 200, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
        
        cv::imshow("Fatigue Detection (NO VCam)", bgrImg);
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;
    }
    
    pclose(pipe);
    return 0;
}