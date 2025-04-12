#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <iostream>

using namespace std;
using namespace dlib;

double compute_ear(const std::vector<point>& eye) {
    double A = length(eye[1] - eye[5]);
    double B = length(eye[2] - eye[4]);
    double C = length(eye[0] - eye[3]);
    return (A + B) / (2.0 * C);
}

int main() {
    cv::VideoCapture cap("/dev/video0", cv::CAP_V4L2); // 从树莓派摄像头读取
    if (!cap.isOpened()) {
        cerr << "摄像头打开失败！" << endl;
        return -1;
    }

    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor pose_model;
    deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

    const double EAR_THRESHOLD = 0.25;
    const int EAR_CONSEC_FRAMES = 20;

    int frame_counter = 0;
    bool alarm_on = false;

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        cv_image<bgr_pixel> cimg(frame);
        std::vector<rectangle> faces = detector(cimg);

        for (auto face : faces) {
            full_object_detection shape = pose_model(cimg, face);

            std::vector<point> left_eye{
                shape.part(36), shape.part(37), shape.part(38),
                shape.part(39), shape.part(40), shape.part(41)
            };
            std::vector<point> right_eye{
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
                        cout << "⚠ 疲劳检测：眨眼时间过长！" << endl;
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

    return 0;
}
