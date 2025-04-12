#include <iostream>
#include <cstdio>
#include <vector>
#include <opencv2/opencv.hpp>

// 计算眼睛宽高比的简化版本（基于眼睛轮廓矩形）
float eye_aspect_ratio(const cv::Rect& eye) {
    // 使用眼睛检测器得到的矩形，计算高宽比
    float height = static_cast<float>(eye.height);
    float width = static_cast<float>(eye.width);
    
    // 返回高宽比的倒数，值越小表示眼睛越闭合
    return height / width;
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
    
    // 加载眼睛检测器
    cv::CascadeClassifier eye_cascade;
    if (!eye_cascade.load("/usr/share/opencv4/haarcascades/haarcascade_eye.xml")) {
        std::cerr << "无法加载眼睛检测模型" << std::endl;
        return -1;
    }
    
    // 备选：闭合眼睛检测器（可能在某些OpenCV版本中有）
    cv::CascadeClassifier closed_eye_cascade;
    bool has_closed_eye_detector = closed_eye_cascade.load("/usr/share/opencv4/haarcascades/haarcascade_eye_tree_eyeglasses.xml");
    
    const float EAR_THRESHOLD = 0.35f;  // 眼睛高宽比阈值，需要根据实际测试调整
    const int EYES_CLOSED_FRAMES = 15;  // 连续闭眼帧数阈值
    int closed_counter = 0;
    
    // 检测参数
    float scaleFactor = 1.05;     // 较小的缩放因子提高检测精度
    int minNeighbors = 3;         // 眼睛检测的邻居数
    cv::Size minEyeSize(20, 20);  // 最小眼睛尺寸
    cv::Size maxEyeSize(80, 80);  // 最大眼睛尺寸，限制近距离时过大的检测框
    
    // 用于记录上一帧检测到的眼睛位置
    std::vector<cv::Rect> last_eyes;
    bool eye_tracked = false;
    
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
        
        // 增强对比度，有助于检测眼睛
        cv::equalizeHist(gray, gray);
        
        frame_count++;
        
        // 定义ROI（兴趣区域）- 默认为整个图像
        cv::Rect roi(0, 0, gray.cols, gray.rows);
        
        // 如果之前检测到眼睛，重用上一帧眼睛位置作为ROI优化
        if (eye_tracked && frame_count % 2 != 0) {
            if (last_eyes.size() >= 2) {
                // 找到包含两只眼睛的最小矩形
                int min_x = width, min_y = height, max_x = 0, max_y = 0;
                for (const auto& eye : last_eyes) {
                    min_x = std::min(min_x, eye.x);
                    min_y = std::min(min_y, eye.y);
                    max_x = std::max(max_x, eye.x + eye.width);
                    max_y = std::max(max_y, eye.y + eye.height);
                }
                
                // 扩大搜索区域
                int padding_x = (max_x - min_x) / 2;
                int padding_y = (max_y - min_y) / 2;
                
                roi.x = std::max(0, min_x - padding_x);
                roi.y = std::max(0, min_y - padding_y);
                roi.width = std::min(width - roi.x, max_x - min_x + 2 * padding_x);
                roi.height = std::min(height - roi.y, max_y - min_y + 2 * padding_y);
            }
        }
        
        // 获取ROI区域的图像
        cv::Mat roi_img = gray(roi);
        
        // 每隔几帧进行一次完整的眼睛检测，减少计算负担
        std::vector<cv::Rect> eyes;
        if (frame_count % 3 == 0) {
            eye_cascade.detectMultiScale(roi_img, eyes, scaleFactor, minNeighbors, 0, minEyeSize, maxEyeSize);
            
            // 调整眼睛坐标到原始图像坐标
            for (auto& eye : eyes) {
                eye.x += roi.x;
                eye.y += roi.y;
            }
            
            // 如果检测到眼睛，更新跟踪状态
            if (eyes.size() >= 1) {
                last_eyes = eyes;
                eye_tracked = true;
            } else {
                // 连续多帧没检测到眼睛，尝试在整个图像中重新检测
                static int no_eye_counter = 0;
                if (++no_eye_counter > 5) {
                    eye_tracked = false;
                    no_eye_counter = 0;
                    // 在整个图像中重新检测
                    eye_cascade.detectMultiScale(gray, eyes, scaleFactor, minNeighbors, 0, minEyeSize, maxEyeSize);
                    if (eyes.size() >= 1) {
                        last_eyes = eyes;
                        eye_tracked = true;
                    }
                }
            }
        } else {
            // 非检测帧使用上一帧的结果
            eyes = last_eyes;
        }
        
        // 分析眼睛状态
        int open_eyes = 0;
        int closed_eyes = 0;
        
        for (const auto& eye : eyes) {
            // 绘制眼睛检测框
            cv::rectangle(bgrImg, eye, cv::Scalar(0, 255, 0), 2);
            
            // 计算眼睛宽高比
            float ear = eye_aspect_ratio(eye);
            
            // 在眼睛附近显示高宽比
            cv::putText(bgrImg, std::to_string(ear).substr(0, 4), 
                       cv::Point(eye.x, eye.y - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);
            
            // 判断眼睛是否闭合
            if (ear < EAR_THRESHOLD) {
                closed_eyes++;
                cv::putText(bgrImg, "Closed", 
                           cv::Point(eye.x, eye.y - 20),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
            } else {
                open_eyes++;
            }
        }
        
        // 判断疲劳状态
        // 如果检测到的眼睛数量大于等于1，并且闭眼数量占多数
        if (eyes.size() >= 1 && closed_eyes > open_eyes) {
            closed_counter++;
            cv::putText(bgrImg, "Eyes Closing: " + std::to_string(closed_counter), 
                       cv::Point(10, 30),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
            
            if (closed_counter >= EYES_CLOSED_FRAMES) {
                cv::putText(bgrImg, "DROWSINESS ALERT!", cv::Point(10, 70),
                           cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
            }
        } else {
            closed_counter = std::max(0, closed_counter - 1); // 缓慢减少计数器
        }
        
        // 显示检测到的眼睛数量
        cv::putText(bgrImg, "Eyes: " + std::to_string(eyes.size()), 
                   cv::Point(width - 150, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
        
        cv::imshow("Eye Detection for Fatigue Monitoring", bgrImg);
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;
    }
    
    pclose(pipe);
    return 0;
}