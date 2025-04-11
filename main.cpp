#include <iostream>
#include <thread>
#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <libcamera/libcamera.h>
#include <libcamera/camera_manager.h>
#include <sys/mman.h>

using namespace libcamera;

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
    const int width = 1640;
    const int height = 1232;

    // dlib & OpenCV 加载
    dlib::shape_predictor predictor;
    dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> predictor;
    cv::CascadeClassifier face_cascade;
    face_cascade.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml");

    // libcamera 初始化
    auto cm = std::make_unique<CameraManager>();
    cm->start();
    if (cm->cameras().empty()) {
        std::cerr << "无摄像头可用" << std::endl;
        return -1;
    }

    auto camera = cm->get(cm->cameras()[0]->id());
    camera->acquire();
    auto config = camera->generateConfiguration({ StreamRole::Viewfinder });
    auto &streamConfig = config->at(0);
    streamConfig.size.width = width;
    streamConfig.size.height = height;
    streamConfig.pixelFormat = formats::XRGB8888;
    config->validate();
    camera->configure(config.get());

    Stream *stream = streamConfig.stream();
    FrameBufferAllocator allocator(camera);
    allocator.allocate(stream);

    std::map<FrameBuffer*, cv::Mat> bufferMats;
    for (auto &buffer : allocator.buffers(stream)) {
        size_t length = buffer->planes()[0].length;
        void *memory = mmap(nullptr, length, PROT_READ | PROT_WRITE, MAP_SHARED, buffer->planes()[0].fd.get(), 0);
        if (memory == MAP_FAILED) {
            perror("mmap failed");
            return -1;
        }
        bufferMats[buffer.get()] = cv::Mat(height, width, CV_8UC4, memory);
    }

    std::vector<std::unique_ptr<Request>> requests;
    for (auto &buffer : allocator.buffers(stream)) {
        auto request = camera->createRequest();
        request->addBuffer(stream, buffer.get());
        requests.push_back(std::move(request));
    }

    const float EAR_THRESHOLD = 0.25f;
    const int EYES_CLOSED_FRAMES = 15;
    int counter = 0;
    bool exit_requested = false;

    camera->requestCompleted.connect([&](Request *request) {
        auto buffers = request->buffers();
        for (auto &[_, buffer] : buffers) {
            auto &rgba = bufferMats[buffer];
            cv::Mat bgr;
            cv::cvtColor(rgba, bgr, cv::COLOR_BGRA2BGR);

            cv::Mat gray;
            cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
            std::vector<cv::Rect> faces;
            face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(80, 80));

            for (const auto& face : faces) {
                dlib::cv_image<dlib::bgr_pixel> cimg(bgr);
                dlib::rectangle dlib_rect(face.x, face.y, face.x + face.width, face.y + face.height);
                dlib::full_object_detection shape = predictor(cimg, dlib_rect);

                auto left_eye = extract_eye(shape, true);
                auto right_eye = extract_eye(shape, false);
                float ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0f;

                if (ear < EAR_THRESHOLD) {
                    if (++counter >= EYES_CLOSED_FRAMES)
                        cv::putText(bgr, "DROWSINESS ALERT!", {50, 50}, cv::FONT_HERSHEY_SIMPLEX, 1.0, {0, 0, 255}, 2);
                } else counter = 0;

                for (const auto& pt : left_eye)  cv::circle(bgr, pt, 2, {0, 255, 0}, -1);
                for (const auto& pt : right_eye) cv::circle(bgr, pt, 2, {0, 255, 0}, -1);
            }

            cv::imshow("Fatigue Detection", bgr);
            if (cv::waitKey(1) == 'q') exit_requested = true;
        }

        request->reuse(Request::ReuseBuffers);
        camera->queueRequest(request);
    });

    camera->start();
    for (auto &req : requests)
        camera->queueRequest(req.get());

    std::cout << "按 q 退出..." << std::endl;
    while (!exit_requested)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

    camera->stop();
    camera->release();
    cm->stop();
    return 0;
}
