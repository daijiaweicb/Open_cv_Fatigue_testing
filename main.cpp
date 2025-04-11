#include <iostream>
#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include "libcam2opencv.h"

float eye_aspect_ratio(const std::vector<cv::Point2f> &eye)
{
    float A = cv::norm(eye[1] - eye[5]);
    float B = cv::norm(eye[2] - eye[4]);
    float C = cv::norm(eye[0] - eye[3]);
    return (A + B) / (2.0f * C);
}

std::vector<cv::Point2f> extract_eye(const dlib::full_object_detection &shape, bool left)
{
    std::vector<cv::Point2f> eye;
    int start = left ? 36 : 42;
    for (int i = 0; i < 6; ++i)
        eye.emplace_back(shape.part(start + i).x(), shape.part(start + i).y());
    return eye;
}

Libcam2OpenCV cam;

class FatigueDetector : public Libcam2OpenCV::Callback
{
public:
    FatigueDetector()
    {
        dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> predictor;
        face_cascade.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml");
        cv::namedWindow("Fatigue Detection", cv::WINDOW_NORMAL);
        cv::resizeWindow("Fatigue Detection", 1280, 960); // 强制放大
    }

    void hasFrame(const cv::Mat &frame, const libcamera::ControlList &) override
    {
        if (frame.empty())
        {
            std::cerr << "[ERROR] Empty frame!" << std::endl;
            return;
        }

        if (cv::getWindowProperty("Fatigue Detection", cv::WND_PROP_VISIBLE) < 1)
        {
            cam.stop();
            return;
        }

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(100, 100));

        for (const auto &face : faces)
        {
            cv::rectangle(frame, face, cv::Scalar(255, 0, 0), 2);

            dlib::cv_image<dlib::bgr_pixel> cimg(frame);
            dlib::rectangle dlib_rect(face.x, face.y, face.x + face.width, face.y + face.height);
            dlib::full_object_detection shape = predictor(cimg, dlib_rect);

            auto left_eye = extract_eye(shape, true);
            auto right_eye = extract_eye(shape, false);
            float ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0f;

            if (ear < EAR_THRESHOLD)
            {
                counter++;
                if (counter >= EYES_CLOSED_FRAMES)
                {
                    cv::putText(frame, "DROWSINESS ALERT!", cv::Point(50, 50),
                                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
                }
            }
            else
            {
                counter = 0;
            }

            for (const auto &pt : left_eye)
                cv::circle(frame, pt, 2, cv::Scalar(0, 255, 0), -1);
            for (const auto &pt : right_eye)
                cv::circle(frame, pt, 2, cv::Scalar(0, 255, 0), -1);
        }

        try
        {
            cv::imshow("Fatigue Detection", frame);
        }
        catch (const cv::Exception &e)
        {
            std::cerr << "[OpenCV Exception] " << e.what() << std::endl;
        }

        if (cv::waitKey(1) == 'q')
        {
            cam.stop();
        }
    }

private:
    dlib::shape_predictor predictor;
    cv::CascadeClassifier face_cascade;
    const float EAR_THRESHOLD = 0.25f;
    const int EYES_CLOSED_FRAMES = 15;
    int counter = 0;
};

bool cam_running()
{
    cv::waitKey(1);
    return cv::getWindowProperty("Fatigue Detection", cv::WND_PROP_VISIBLE) >= 1;
}

int main()
{
    FatigueDetector detector;
    cam.registerCallback(&detector);

    Libcam2OpenCVSettings settings;
    settings.width = 1280;
    settings.height = 960;
    settings.framerate = 15;
    settings.brightness = 0.0;
    settings.contrast = 1.0;

    cam.start();

    while (cam_running())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return 0;
}