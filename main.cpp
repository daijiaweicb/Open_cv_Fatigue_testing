#include <iostream>
#include <vector>
#include <cstdio>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>

// 全局变量
cv::Mat shared_frame;
std::mutex mtx;
std::condition_variable cv_frame_ready;
std::atomic<bool> running(true);

void detection_thread_func(
    dlib::frontal_face_detector detector,
    dlib::shape_predictor predictor
) {
    int frame_counter = 0;
    const int detect_interval = 20;
    const float EAR_THRESHOLD = 0.25f;
    const int EYES_CLOSED_FRAMES = 15;
    int counter = 0;

    std::vector<dlib::rectangle> last_faces;
    dlib::full_object_detection last_shape;

    while (running) {
        std::unique_lock<std::mutex> lock(mtx);
        cv_frame_ready.wait(lock, [] { return !shared_frame.empty() || !running; });

        if (!running) break;

        cv::Mat frame_copy = shared_frame.clone();
        lock.unlock();

        frame_counter++;

        if (frame_counter % detect_interval == 0) {
            dlib::cv_image<dlib::bgr_pixel> cimg(frame_copy);
            last_faces = detector(cimg);
            if (!last_faces.empty()) {
                last_shape = predictor(cimg, last_faces[0]);
            }
        }

        if (!last_faces.empty()) {
            const dlib::rectangle& face = last_faces[0];
            cv::Rect face_rect(cv::Point(face.left(), face.top()),
                               cv::Point(face.right(), face.bottom()));
            cv::rectangle(frame_copy, face_rect, cv::Scalar(255, 0, 0), 2);

            if (last_shape.num_parts() == 68) {
                auto get_eye = [](const dlib::full_object_detection& s, bool left) {
                    std::vector<cv::Point2f> eye;
                    int start = left ? 36 : 42;
                    for (int i = 0; i < 6; ++i)
                        eye.emplace_back(s.part(start + i).x(), s.part(start + i).y());
                    return eye;
                };

                auto eye_aspect_ratio = [](const std::vector<cv::Point2f>& eye) {
                    float A = cv::norm(eye[1] - eye[5]);
                    float B = cv::norm(eye[2] - eye[4]);
                    float C = cv::norm(eye[0] - eye[3]);
                    return (A + B) / (2.0f * C);
                };

                auto left_eye = get_eye(last_shape, true);
                auto right_eye = get_eye(last_shape, false);
                float ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0f;

                if (ear < EAR_THRESHOLD) {
                    counter++;
                    if (counter >= EYES_CLOSED_FRAMES) {
                        cv::putText(frame_copy, "DROWSINESS ALERT! " + std::to_string(counter),
                                    cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX,
                                    1.0, cv::Scalar(0, 0, 255), 2);
                    }
                } else {
                    counter = 0;
                }

                for (size_t i = 0; i < 6; ++i) {
                    cv::circle(frame_copy, left_eye[i], 2, cv::Scalar(0, 255, 0), -1);
                    cv::circle(frame_copy, right_eye[i], 2, cv::Scalar(0, 255, 0), -1);
                    cv::line(frame_copy, left_eye[i], left_eye[(i + 1) % 6], cv::Scalar(255, 255, 0), 1);
                    cv::line(frame_copy, right_eye[i], right_eye[(i + 1) % 6], cv::Scalar(255, 255, 0), 1);
                }
            }
        }

        cv::imshow("Fatigue Detection", frame_copy);
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) {
            running = false;
        }
    }
}

int main() {
    // 加载 dlib 模型
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor predictor;
    try {
        dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> predictor;
    } catch (std::exception& e) {
        std::cerr << "无法加载模型：" << e.what() << std::endl;
        return -1;
    }

    // 启动 libcamera-vid
    FILE* pipe = popen("libcamera-vid -t 0 --codec yuv420 --width 640 --height 480 -n -o -", "r");
    if (!pipe) {
        std::cerr << "无法启动 libcamera-vid，请检查摄像头连接状态" << std::endl;
        return -1;
    }

    // 视频帧缓冲
    int width = 640, height = 480;
    int frame_size = width * height * 3 / 2;
    std::vector<unsigned char> buffer(frame_size);
    cv::Mat yuvImg(height + height / 2, width, CV_8UC1);
    cv::Mat bgrImg;

    // 启动检测线程
    std::thread detection_thread(detection_thread_func, detector, predictor);

    while (running) {
        size_t read_bytes = fread(buffer.data(), 1, frame_size, pipe);
        if (read_bytes != static_cast<size_t>(frame_size)) {
            std::cerr << "读取失败" << std::endl;
            running = false;
            break;
        }

        memcpy(yuvImg.data, buffer.data(), frame_size);
        cv::cvtColor(yuvImg, bgrImg, cv::COLOR_YUV2BGR_I420);

        {
            std::lock_guard<std::mutex> lock(mtx);
            shared_frame = bgrImg.clone();
        }
        cv_frame_ready.notify_one();
    }

    pclose(pipe);
    cv_frame_ready.notify_all();
    if (detection_thread.joinable()) detection_thread.join();

    return 0;
}
