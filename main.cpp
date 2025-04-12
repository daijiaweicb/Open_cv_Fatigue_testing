#include <iostream>
#include <cstdio>
#include <vector>
#include <opencv2/opencv.hpp>

// 实际计算眼睛开合状态的函数
bool is_eye_closed(const cv::Mat& eye_region, float& ratio) {
    // 转换为灰度图（如果不是）
    cv::Mat gray_eye;
    if (eye_region.channels() > 1)
        cv::cvtColor(eye_region, gray_eye, cv::COLOR_BGR2GRAY);
    else
        gray_eye = eye_region.clone();
    
    // 二值化以区分眼白和眼球/睫毛
    cv::Mat binary_eye;
    cv::threshold(gray_eye, binary_eye, 70, 255, cv::THRESH_BINARY_INV);
    
    // 计算白色像素（眼睛闭合区域）占比
    int total_pixels = binary_eye.rows * binary_eye.cols;
    int white_pixels = cv::countNonZero(binary_eye);
    ratio = (float)white_pixels / total_pixels;
    
    // 比例超过阈值认为是闭眼
    return (ratio > 0.30f); // 需要根据实际情况调整阈值
}

// 计算矩形的纵横比
float aspect_ratio(const cv::Rect& r) {
    return (float)r.height / r.width;
}

// 过滤假眼睛
std::vector<cv::Rect> filter_eyes(const std::vector<cv::Rect>& detected_eyes, const cv::Mat& frame) {
    std::vector<cv::Rect> valid_eyes;
    
    // 如果检测到的眼睛太多（可能有假阳性），只保留最可能的两个
    if (detected_eyes.size() > 2) {
        // 对检测到的眼睛副本进行排序
        std::vector<cv::Rect> sorted_eyes = detected_eyes;
        
        // 按面积从大到小排序（通常真实眼睛区域会较大）
        std::sort(sorted_eyes.begin(), sorted_eyes.end(), 
                 [](const cv::Rect& a, const cv::Rect& b) {
                     return a.area() > b.area();
                 });
        
        // 优先选择合理纵横比的眼睛
        std::vector<cv::Rect> ratio_filtered;
        for (const auto& eye : sorted_eyes) {
            float ratio = aspect_ratio(eye);
            // 眼睛通常宽度大于高度，比例在0.3到0.7之间较合理
            if (ratio > 0.3f && ratio < 0.7f) {
                ratio_filtered.push_back(eye);
            }
        }
        
        // 如果有纵横比合适的，选择它们，否则回退到按大小排序的结果
        std::vector<cv::Rect>& candidate_eyes = ratio_filtered.empty() ? sorted_eyes : ratio_filtered;
        
        // 选择最可能的两个眼睛
        for (size_t i = 0; i < std::min(size_t(2), candidate_eyes.size()); ++i) {
            valid_eyes.push_back(candidate_eyes[i]);
        }
    } else {
        // 如果检测到的眼睛数量合理，过滤不合理的纵横比
        for (const auto& eye : detected_eyes) {
            float ratio = aspect_ratio(eye);
            if (ratio > 0.3f && ratio < 0.7f) {
                valid_eyes.push_back(eye);
            }
        }
    }
    
    return valid_eyes;
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
    
    // 可选：也加载人脸检测器来限制眼睛检测区域
    cv::CascadeClassifier face_cascade;
    bool has_face_detector = face_cascade.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml");
    
    const int EYES_CLOSED_FRAMES = 15;  // 连续闭眼帧数阈值
    int closed_counter = 0;
    
    // 检测参数
    float scaleFactor = 1.1;     // 缩放因子
    int minNeighbors = 4;        // 增加邻居数减少假阳性
    cv::Size minEyeSize(25, 15); // 最小眼睛尺寸 (宽>高)
    cv::Size maxEyeSize(80, 45); // 最大眼睛尺寸
    
    // 用于记录上一帧检测到的有效眼睛
    std::vector<cv::Rect> last_valid_eyes;
    bool eyes_tracked = false;
    
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
        
        // 增强对比度
        cv::equalizeHist(gray, gray);
        
        frame_count++;
        
        // 定义眼睛搜索区域
        cv::Rect search_area(0, 0, gray.cols, gray.rows);
        
        // 如果有人脸检测器并且是每5帧的关键帧，尝试检测人脸
        std::vector<cv::Rect> faces;
        if (has_face_detector && frame_count % 5 == 0) {
            face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(80, 80));
            
            // 如果检测到人脸，缩小搜索区域到上半部分
            if (!faces.empty()) {
                // 使用最大的人脸
                auto largest_face = *std::max_element(faces.begin(), faces.end(),
                    [](const cv::Rect& a, const cv::Rect& b) {
                        return a.area() < b.area();
                    });
                
                // 只关注人脸上半部分
                search_area = cv::Rect(
                    largest_face.x,
                    largest_face.y,
                    largest_face.width,
                    largest_face.height / 2
                );
                
                // 在图像上标记人脸区域
                cv::rectangle(bgrImg, largest_face, cv::Scalar(255, 105, 65), 2);
            }
        }
        
        // 使用上一帧跟踪信息优化搜索区域
        if (eyes_tracked && frame_count % 3 != 0 && !last_valid_eyes.empty()) {
            // 计算包含所有眼睛的区域
            int min_x = width, min_y = height, max_x = 0, max_y = 0;
            for (const auto& eye : last_valid_eyes) {
                min_x = std::min(min_x, eye.x);
                min_y = std::min(min_y, eye.y);
                max_x = std::max(max_x, eye.x + eye.width);
                max_y = std::max(max_y, eye.y + eye.height);
            }
            
            // 扩大搜索区域
            int padding_x = (max_x - min_x);
            int padding_y = (max_y - min_y);
            
            cv::Rect eye_area(
                std::max(0, min_x - padding_x),
                std::max(0, min_y - padding_y),
                std::min(width - (min_x - padding_x), max_x - min_x + 2 * padding_x),
                std::min(height - (min_y - padding_y), max_y - min_y + 2 * padding_y)
            );
            
            // 如果已经有人脸区域，取交集
            if (search_area.width < gray.cols || search_area.height < gray.rows) {
                // 计算交集
                int x1 = std::max(search_area.x, eye_area.x);
                int y1 = std::max(search_area.y, eye_area.y);
                int x2 = std::min(search_area.x + search_area.width, eye_area.x + eye_area.width);
                int y2 = std::min(search_area.y + search_area.height, eye_area.y + eye_area.height);
                
                if (x2 > x1 && y2 > y1) {
                    search_area = cv::Rect(x1, y1, x2 - x1, y2 - y1);
                }
            } else {
                search_area = eye_area;
            }
        }
        
        // 确保搜索区域在图像内
        search_area &= cv::Rect(0, 0, gray.cols, gray.rows);
        
        // 获取搜索区域的图像
        cv::Mat roi_img = gray(search_area);
        
        // 每帧或每隔几帧检测眼睛
        std::vector<cv::Rect> detected_eyes;
        if (frame_count % 3 == 0 || !eyes_tracked) {
            eye_cascade.detectMultiScale(roi_img, detected_eyes, scaleFactor, minNeighbors, 0, minEyeSize, maxEyeSize);
            
            // 调整眼睛坐标到原始图像坐标
            for (auto& eye : detected_eyes) {
                eye.x += search_area.x;
                eye.y += search_area.y;
            }
            
            // 过滤假眼睛
            std::vector<cv::Rect> valid_eyes = filter_eyes(detected_eyes, bgrImg);
            
            // 如果检测到有效眼睛，更新跟踪状态
            if (!valid_eyes.empty()) {
                last_valid_eyes = valid_eyes;
                eyes_tracked = true;
            } else if (frame_count % 10 == 0) {
                // 定期在整个图像中重新检测
                std::vector<cv::Rect> full_detected_eyes;
                eye_cascade.detectMultiScale(gray, full_detected_eyes, scaleFactor, minNeighbors, 0, minEyeSize, maxEyeSize);
                std::vector<cv::Rect> full_valid_eyes = filter_eyes(full_detected_eyes, bgrImg);
                
                if (!full_valid_eyes.empty()) {
                    last_valid_eyes = full_valid_eyes;
                    eyes_tracked = true;
                } else {
                    eyes_tracked = false;
                }
            }
        }
        
        // 分析眼睛状态
        int closed_eyes_count = 0;
        
        // 绘制搜索区域
        cv::rectangle(bgrImg, search_area, cv::Scalar(50, 50, 255), 1);
        
        for (const auto& eye : last_valid_eyes) {
            // 确保眼睛区域在图像内
            cv::Rect safe_eye = eye & cv::Rect(0, 0, bgrImg.cols, bgrImg.rows);
            
            if (safe_eye.width > 0 && safe_eye.height > 0) {
                // 获取眼睛区域
                cv::Mat eye_roi = bgrImg(safe_eye);
                
                // 计算眼睛开合状态
                float closure_ratio;
                bool is_closed = is_eye_closed(eye_roi, closure_ratio);
                
                // 绘制眼睛检测框
                cv::Scalar color = is_closed ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
                cv::rectangle(bgrImg, safe_eye, color, 2);
                
                // 显示闭合比例
                cv::putText(bgrImg, 
                           "Ratio: " + std::to_string(closure_ratio).substr(0, 5),
                           cv::Point(safe_eye.x, safe_eye.y - 5),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
                
                // 显示状态
                std::string status = is_closed ? "Closed" : "Open";
                cv::putText(bgrImg, status,
                           cv::Point(safe_eye.x, safe_eye.y - 20),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
                
                if (is_closed) {
                    closed_eyes_count++;
                }
            }
        }
        
        // 疲劳判断：比例超过一半的眼睛是闭合的
        if (!last_valid_eyes.empty() && 
            closed_eyes_count >= last_valid_eyes.size() / 2) {
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
        
        // 显示有效眼睛数量
        cv::putText(bgrImg, "Valid Eyes: " + std::to_string(last_valid_eyes.size()), 
                   cv::Point(width - 170, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
        
        cv::imshow("Eye Fatigue Detection", bgrImg);
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;
    }
    
    pclose(pipe);
    return 0;
}