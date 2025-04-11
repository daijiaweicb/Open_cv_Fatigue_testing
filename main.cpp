#include <iostream>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;
using namespace dlib;

// 计算EAR
float eye_aspect_ratio(const vector<Point2f>& eye) {
    float A = norm(eye[1] - eye[5]);
    float B = norm(eye[2] - eye[4]);
    float C = norm(eye[0] - eye[3]);
    return (A + B) / (2.0 * C);
}

vector<Point2f> extract_eye(const full_object_detection& shape, bool left) {
    vector<Point2f> eye;
    int start = left ? 36 : 42;
    for (int i = 0; i < 6; ++i)
        eye.emplace_back(shape.part(start + i).x(), shape.part(start + i).y());
    return eye;
}

int main() {
    // 启动 libcamera-vid
    FILE* pipe = popen("libcamera-vid -t 0 --codec yuv420 --width 640 --height 480 -n -o -", "r");
    if (!pipe) {
        cerr << "无法启动 libcamera-vid，请检查摄像头连接状态" << endl;
        return -1;
    }

    // 初始化 dlib
    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor predictor;
    deserialize("shape_predictor_68_face_landmarks.dat") >> predictor;

    int width = 640, height = 480;
    int frame_size = width * height * 3 / 2;
    vector<uchar> buffer(frame_size);
    Mat yuvImg(height + height / 2, width, CV_8UC1);
    Mat bgrImg;

    const float EAR_THRESHOLD = 0.25;
    const int EYES_CLOSED_FRAMES = 15;
    int counter = 0;

    while (true) {
        size_t read_bytes = fread(buffer.data(), 1, frame_size, pipe);
        if (read_bytes != frame_size) {
            cerr << "读取失败，可能摄像头断开或输出结束" << endl;
            break;
        }

        memcpy(yuvImg.data, buffer.data(), frame_size);
        cvtColor(yuvImg, bgrImg, COLOR_YUV2BGR_I420);

        // dlib 图像转换
        cv_image<bgr_pixel> cimg(bgrImg);
        std::vector<rectangle> faces = detector(cimg);

        for (auto& face : faces) {
            full_object_detection shape = predictor(cimg, face);
            auto left_eye = extract_eye(shape, true);
            auto right_eye = extract_eye(shape, false);

            float ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0;

            if (ear < EAR_THRESHOLD) {
                counter++;
                if (counter >= EYES_CLOSED_FRAMES) {
                    putText(bgrImg, "DROWSINESS ALERT!", Point(50, 50),
                            FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2);
                }
            } else {
                counter = 0;
            }

            for (const auto& pt : left_eye)
                circle(bgrImg, pt, 2, Scalar(0, 255, 0), -1);
            for (const auto& pt : right_eye)
                circle(bgrImg, pt, 2, Scalar(0, 255, 0), -1);
        }

        imshow("Fatigue Detection", bgrImg);
        if (waitKey(1) == 'q') break;
    }

    pclose(pipe);
    return 0;
}
