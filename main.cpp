#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <libcamera/libcamera.h>
#include <iostream>
#include <vector>
#include <chrono>

double computeEAR(const std::vector<cv::Point>& eye) {
    double A = cv::norm(eye[1] - eye[5]);
    double B = cv::norm(eye[2] - eye[4]);
    double C = cv::norm(eye[0] - eye[3]);
    return (A + B) / (2.0 * C);
}

std::vector<cv::Point> getEyePoints(const dlib::full_object_detection& shape, bool left) {
    std::vector<cv::Point> points;
    int start = left ? 36 : 42;
    for (int i = 0; i < 6; ++i) {
        auto pt = shape.part(start + i);
        points.emplace_back(cv::Point(pt.x(), pt.y()));
    }
    return points;
}

int main() {
    // Initialize Dlib detector and model
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor predictor;
    dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> predictor;

    // Initialize libcamera
    libcamera::CameraManager cameraManager;
    cameraManager.start();
    auto cameras = cameraManager.cameras();

    if (cameras.empty()) {
        std::cerr << "No available cameras!" << std::endl;
        return -1;
    }

    auto camera = cameras.front();  // Use the first available camera

    // Set up camera configuration
    auto config = camera->generateConfiguration({libcamera::StreamRole::VideoRecording});
    if (!config) {
        std::cerr << "Failed to configure camera!" << std::endl;
        return -1;
    }
    config->at(0).size = libcamera::Size(640, 480);  // Set the resolution

    if (camera->configure(config.get()) != 0) {
        std::cerr << "Failed to configure the camera stream!" << std::endl;
        return -1;
    }

    // Start the camera
    if (camera->start() != 0) {
        std::cerr << "Failed to start the camera!" << std::endl;
        return -1;
    }

    // Create a request to capture frames
    libcamera::Request *request = camera->createRequest();
    if (!request) {
        std::cerr << "Failed to create capture request!" << std::endl;
        return -1;
    }

    const double EAR_THRESHOLD = 0.21;
    const int EAR_CONSEC_FRAMES = 15;
    int frame_counter = 0;
    bool fatigued = false;

    while (true) {
        // Capture a frame from the camera
        libcamera::FrameBuffer *frameBuffer = nullptr;
        camera->queueRequest(request);
        if (frameBuffer == nullptr) {
            std::cerr << "Frame capture failed!" << std::endl;
            break;
        }

        // Retrieve frame data and convert to OpenCV Mat
        auto plane = frameBuffer->planes()[0];  // Get the first plane
        cv::Mat frame(plane.height, plane.width, CV_8UC3, plane.data);

        dlib::cv_image<dlib::bgr_pixel> dlib_img(frame);
        std::vector<dlib::rectangle> faces = detector(dlib_img);

        for (auto face : faces) {
            auto shape = predictor(dlib_img, face);

            auto leftEye = getEyePoints(shape, true);
            auto rightEye = getEyePoints(shape, false);

            double leftEAR = computeEAR(leftEye);
            double rightEAR = computeEAR(rightEye);
            double ear = (leftEAR + rightEAR) / 2.0;

            // Visualize eye contours
            for (const auto& pt : leftEye)
                cv::circle(frame, pt, 2, cv::Scalar(0, 255, 0), -1);
            for (const auto& pt : rightEye)
                cv::circle(frame, pt, 2, cv::Scalar(0, 255, 0), -1);

            if (ear < EAR_THRESHOLD) {
                frame_counter++;
                if (frame_counter >= EAR_CONSEC_FRAMES) {
                    fatigued = true;
                    cv::putText(frame, "FATIGUE DETECTED!", cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                                cv::Scalar(0, 0, 255), 2);
                }
            } else {
                frame_counter = 0;
                fatigued = false;
            }

            // Display EAR value
            char text[50];
            sprintf(text, "EAR: %.2f", ear);
            cv::putText(frame, text, cv::Point(10, frame.rows - 10), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        fatigued ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0), 2);
        }

        // Show the processed frame
        cv::imshow("Fatigue Detection", frame);
        if (cv::waitKey(1) == 'q') break;
    }

    return 0;
}
