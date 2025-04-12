#include <iostream>
#include <cstdio>
#include <vector>
#include <opencv2/opencv.hpp>

// 计算眼睛开合状态
bool is_eye_closed(const cv::Mat& eye_region, float& ratio) {
    // 转换为灰度图（如果不是）
    cv::Mat gray_eye;
    if (eye_region.channels() > 1)
        cv::cvtColor(eye_region, gray_eye, cv::COLOR_BGR2GRAY);
    else
        gray_eye = eye_region.clone();
    
    // 应用高斯模糊减少噪声
    cv::GaussianBlur(gray_eye, gray_eye, cv::Size(5, 5), 0);
    
    // 二值化以区分眼白和眼球/睫毛
    cv::Mat binary_eye;
    cv::threshold(gray_eye, binary_eye, 70, 255, cv::THRESH_BINARY_INV);
    
    // 计算白色像素（眼睛闭合区域）占比
    int total_pixels = binary_eye.rows * binary_eye.cols;
    int white_pixels = cv::countNonZero(binary_eye);
    ratio = (float)white_pixels / total_pixels;
    
    // 比例超过阈值认为是闭眼
    return (ratio > 0.25f); // 需要根据实际情况调整阈值
}

// 检查是否是真实的眼睛而不是鼻孔
bool is_real_eye(const cv::Mat& roi, const cv::Rect& eye_rect) {
    // 转换为灰度图
    cv::Mat gray;
    if (roi.channels() > 1)
        cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    else
        gray = roi.clone();
    
    // 1. 检查宽高比 - 眼睛通常比鼻孔更宽
    float aspect = (float)eye_rect.width / eye_rect.height;
    if (aspect < 1.2f) // 眼睛宽高比通常大于1.2
        return false;
    
    // 2. 分析强度分布 - 眼睛通常有明显的对比度
    cv::Scalar mean, stddev;
    cv::meanStdDev(gray, mean, stddev);
    if (stddev[0] < 35.0) // 对比度过低，可能是鼻孔
        return false;
    
    // 3. 检查水平梯度 - 眼睛有明显的水平边缘
    cv::Mat sobelX;
    cv::Sobel(gray, sobelX, CV_16S, 1, 0);
    cv::convertScaleAbs(sobelX, sobelX);
    cv::Scalar meanGradient = cv::mean(sobelX);
    if (meanGradient[0] < 20.0) // 水平梯度值过低
        return false;
    
    // 4. 检查形状特征 - 分析上下区域的明暗差异
    int h = gray.rows;
    int w = gray.cols;
    
    // 从上到下分三部分检查亮度
    cv::Mat top = gray(cv::Rect(0, 0, w, h/3));
    cv::Mat middle = gray(cv::Rect(0, h/3, w, h/3));
    cv::Mat bottom = gray(cv::Rect(0, 2*h/3, w, h/3));
    
    double topMean = cv::mean(top)[0];
    double middleMean = cv::mean(middle)[0];
    double bottomMean = cv::mean(bottom)[0];
    
    // 眼睛中间部分通常比上下部分暗
    if (!(middleMean < topMean && middleMean < bottomMean))
        return false;
    
    return true;
}

// 优化的眼睛过滤函数
std::vector<cv::Rect> filter_eyes(const std::vector<cv::Rect>& detected_eyes, const cv::Mat& frame) {
    std::vector<cv::Rect> valid_eyes;
    std::vector<float> confidences;
    
    // 第一步: 基于基本几何特征进行过滤
    for (const auto& eye : detected_eyes) {
        float ratio = (float)eye.height / eye.width;
        
        // 眼睛高宽比应在合理范围内
        if (ratio > 0.4f && ratio < 0.7f) {
            // 眼睛大小应适中（太小可能是噪声，太大可能是误检）
            int area = eye.width * eye.height;
            if (area > 400 && area < 4000) {
                // 检查是否是真实的眼睛
                cv::Rect safe_eye = eye & cv::Rect(0, 0, frame.cols, frame.rows);
                if (safe_eye.width > 0 && safe_eye.height > 0) {
                    cv::Mat eye_roi = frame(safe_eye);
                    if (is_real_eye(eye_roi, safe_eye)) {
                        valid_eyes.push_back(eye);
                        
                        // 计算置信度 (基于大小和形状的加权分数)
                        float size_score = std::min(1.0f, area / 2000.0f);
                        float shape_score = 1.0f - std::abs(0.55f - ratio) / 0.55f;
                        confidences.push_back(0.6f * size_score + 0.4f * shape_score);
                    }
                }
            }
        }
    }
    
    // 如果没有满足条件的眼睛，返回空结果
    if (valid_eyes.empty())
        return valid_eyes;
    
    // 第二步: 如果检测到超过2个有效眼睛，按置信度选择最佳两个
    if (valid_eyes.size() > 2) {
        // 创建索引数组并按置信度排序
        std::vector<size_t> indices(valid_eyes.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), 
                 [&confidences](size_t a, size_t b) {
                     return confidences[a] > confidences[b];
                 });
        
        // 提取前两个最佳眼睛
        std::vector<cv::Rect> best_eyes;
        best_eyes.push_back(valid_eyes[indices[0]]);
        best_eyes.push_back(valid_eyes[indices[1]]);
        
        // 第三步: 检查两眼的水平位置关系
        if (best_eyes[0].x > best_eyes[1].x)
            std::swap(best_eyes[0], best_eyes[1]);
        
        // 检查眼睛是否在合理的水平位置
        int dx = best_eyes[1].x - (best_eyes[0].x + best_eyes[0].width);
        if (dx < 0 || dx > best_eyes[0].width * 2) {
            // 如果水平距离不合理，可能是检测到了错误的"眼睛"对
            // 尝试找到下一个最佳候选，如果有的话
            if (indices.size() > 2) {
                for (size_t i = 2; i < indices.size(); ++i) {
                    cv::Rect candidate = valid_eyes[indices[i]];
                    
                    // 检查与第一个眼睛的水平距离是否合理
                    int dx1 = std::abs(candidate.x - best_eyes[0].x);
                    int dy1 = std::abs(candidate.y - best_eyes[0].y);
                    
                    // 检查与第二个眼睛的水平距离是否合理
                    int dx2 = std::abs(candidate.x - best_eyes[1].x);
                    int dy2 = std::abs(candidate.y - best_eyes[1].y);
                    
                    // 如果与其中一个眼睛形成更好的水平对齐，替换另一个
                    if (dy1 < best_eyes[0].height && dx1 > best_eyes[0].width * 0.5 && 
                        dx1 < best_eyes[0].width * 3) {
                        best_eyes[1] = candidate;
                        break;
                    } else if (dy2 < best_eyes[1].height && dx2 > best_eyes[1].width * 0.5 && 
                               dx2 < best_eyes[1].width * 3) {
                        best_eyes[0] = candidate;
                        break;
                    }
                }
            }
        }
        
        // 重新确保左右顺序
        if (best_eyes[0].x > best_eyes[1].x)
            std::swap(best_eyes[0], best_eyes[1]);
        
        return best_eyes;
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
    
    // 加载人脸检测器
    cv::CascadeClassifier face_cascade;
    bool has_face_detector = face_cascade.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml");
    
    const int EYES_CLOSED_FRAMES = 15;  // 连续闭眼帧数阈值
    int closed_counter = 0;
    
    // 检测参数
    float scaleFactor = 1.1;     // 缩放因子
    int minNeighbors = 5;        // 进一步增加邻居数减少假阳性
    cv::Size minEyeSize(30, 15); // 增大最小眼睛尺寸避开鼻孔
    cv::Size maxEyeSize(90, 45); // 最大眼睛尺寸
    
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
        
        // 对比度增强
        cv::equalizeHist(gray, gray);
        
        frame_count++;
        
        // 默认搜索整个图像
        cv::Rect search_area(0, 0, gray.cols, gray.rows);
        
        // 人脸检测来限制眼睛搜索区域
        std::vector<cv::Rect> faces;
        if (has_face_detector && frame_count % 5 == 0) {
            face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(80, 80));
            
            if (!faces.empty()) {
                // 使用最大的人脸
                auto largest_face = *std::max_element(faces.begin(), faces.end(),
                    [](const cv::Rect& a, const cv::Rect& b) {
                        return a.area() < b.area();
                    });
                
                // 眼睛通常在人脸上部，设置搜索区域为人脸上半部
                search_area = cv::Rect(
                    largest_face.x,
                    largest_face.y,
                    largest_face.width,
                    largest_face.height / 2
                );
                
                // 排除人脸下部以避开鼻孔区域
                cv::rectangle(bgrImg, largest_face, cv::Scalar(255, 105, 65), 2);
                cv::rectangle(bgrImg, search_area, cv::Scalar(0, 165, 255), 1);
            }
        }
        
        // 确保搜索区域在图像内
        search_area &= cv::Rect(0, 0, gray.cols, gray.rows);
        
        // 获取搜索区域的图像
        cv::Mat roi_img = gray(search_area);
        
        // 眼睛检测
        std::vector<cv::Rect> detected_eyes;
        if (frame_count % 3 == 0 || !eyes_tracked) {
            eye_cascade.detectMultiScale(roi_img, detected_eyes, scaleFactor, minNeighbors, 0, minEyeSize, maxEyeSize);
            
            // 调整眼睛坐标到原始图像坐标
            for (auto& eye : detected_eyes) {
                eye.x += search_area.x;
                eye.y += search_area.y;
            }
            
            // 过滤假眼睛和鼻孔
            std::vector<cv::Rect> valid_eyes = filter_eyes(detected_eyes, bgrImg);
            
            // 如果检测到有效眼睛，更新跟踪状态
            if (!valid_eyes.empty()) {
                last_valid_eyes = valid_eyes;
                eyes_tracked = true;
            } else if (frame_count % 10 == 0) {
                // 定期重新检测
                eyes_tracked = false;
            }
        }
        
        // 分析眼睛状态
        int closed_eyes_count = 0;
        
        // 绘制检测到的眼睛和状态
        for (const auto& eye : last_valid_eyes) {
            cv::Rect safe_eye = eye & cv::Rect(0, 0, bgrImg.cols, bgrImg.rows);
            
            if (safe_eye.width > 0 && safe_eye.height > 0) {
                // 获取眼睛区域
                cv::Mat eye_roi = bgrImg(safe_eye);
                
                // 计算眼睛开合状态
                float closure_ratio;
                bool is_closed = is_eye_closed(eye_roi, closure_ratio);
                
                // 绘制眼睛检测框和状态
                cv::Scalar color = is_closed ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
                cv::rectangle(bgrImg, safe_eye, color, 2);
                
                // 显示闭合比例
                cv::putText(bgrImg, 
                           std::to_string(closure_ratio).substr(0, 5),
                           cv::Point(safe_eye.x, safe_eye.y - 5),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
                
                std::string status = is_closed ? "Closed" : "Open";
                cv::putText(bgrImg, status,
                           cv::Point(safe_eye.x, safe_eye.y - 20),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
                
                if (is_closed) {
                    closed_eyes_count++;
                }
            }
        }
        
        // 显示所有初步检测到的眼睛候选区域
        if (frame_count % 3 == 0) {
            for (const auto& eye : detected_eyes) {
                // 排除那些已经被确认为真实眼睛的区域
                bool is_valid = false;
                for (const auto& valid_eye : last_valid_eyes) {
                    if (std::abs(eye.x - valid_eye.x) < 5 && std::abs(eye.y - valid_eye.y) < 5) {
                        is_valid = true;
                        break;
                    }
                }
                
                // 用黄色虚线标记被过滤掉的候选区域
                if (!is_valid) {
                    cv::Rect safe_eye = eye & cv::Rect(0, 0, bgrImg.cols, bgrImg.rows);
                    cv::rectangle(bgrImg, safe_eye, cv::Scalar(0, 215, 255), 1);
                    
                    // 标记为过滤掉的区域
                    cv::putText(bgrImg, "Filtered",
                               cv::Point(safe_eye.x, safe_eye.y - 5),
                               cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 215, 255), 1);
                }
            }
        }
        
        // 疲劳判断
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
            closed_counter = std::max(0, closed_counter - 1);
        }
        
        // 显示信息
        cv::putText(bgrImg, "Valid Eyes: " + std::to_string(last_valid_eyes.size()), 
                   cv::Point(width - 170, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
        
        cv::imshow("Enhanced Eye Detection", bgrImg);
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;
    }
    
    pclose(pipe);
    return 0;
}