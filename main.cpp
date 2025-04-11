#include <iostream>
#include <cstdio>
#include <map>
#include <vector>
#include <thread>
#include <chrono>
#include <memory>
#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <libcamera/libcamera.h>
#include <sys/mman.h>

// --------- libcam2opencv core (minimal inline) ------------
class Libcam2OpenCV {
public:
    struct Callback {
        virtual void hasFrame(const cv::Mat &frame, const libcamera::ControlList &metadata) = 0;
        virtual ~Callback() {}
    };

    void registerCallback(Callback* cb) { callback = cb; }

    void start(unsigned int width = 640, unsigned int height = 480, unsigned int framerate = 15) {
        cm = std::make_unique<libcamera::CameraManager>();
        cm->start();

        if (cm->cameras().empty()) {
            std::cerr << "No camera found." << std::endl;
            return;
        }

        camera = cm->cameras()[0];
        camera->acquire();
        config = camera->generateConfiguration({ libcamera::StreamRole::Viewfinder });
        libcamera::StreamConfiguration &streamConfig = config->at(0);
        streamConfig.size.width = width;
        streamConfig.size.height = height;
        streamConfig.pixelFormat = libcamera::formats::BGR888;
        config->validate();
        camera->configure(config.get());

        allocator = new libcamera::FrameBufferAllocator(camera);
        stream = streamConfig.stream();
        allocator->allocate(stream);

        for (auto &buffer : allocator->buffers(stream)) {
            size_t buffer_size = 0;
            for (unsigned i = 0; i < buffer->planes().size(); ++i) {
                const auto &plane = buffer->planes()[i];
                buffer_size += plane.length;
                if (i == buffer->planes().size() - 1 || plane.fd.get() != buffer->planes()[i + 1].fd.get()) {
                    void *memory = mmap(nullptr, buffer_size, PROT_READ | PROT_WRITE, MAP_SHARED, plane.fd.get(), 0);
                    mapped_buffers[buffer.get()].emplace_back(static_cast<uint8_t *>(memory), buffer_size);
                    buffer_size = 0;
                }
            }
        }

        for (auto &buffer : allocator->buffers(stream)) {
            auto request = camera->createRequest();
            request->addBuffer(stream, buffer.get());
            requests.push_back(std::move(request));
        }

        camera->requestCompleted.connect(this, &Libcam2OpenCV::requestComplete);

        if (framerate > 0) {
            int64_t frame_time = 1000000 / framerate;
            controls.set(libcamera::controls::FrameDurationLimits, {frame_time, frame_time});
        }

        camera->start(&controls);
        for (auto &req : requests)
            camera->queueRequest(req.get());
    }

    void stop() {
        camera->stop();
        allocator->free(stream);
        camera->release();
        camera.reset();
        cm->stop();
        delete allocator;
    }

private:
    std::shared_ptr<libcamera::Camera> camera;
    std::unique_ptr<libcamera::CameraManager> cm;
    std::unique_ptr<libcamera::CameraConfiguration> config;
    libcamera::Stream *stream = nullptr;
    libcamera::FrameBufferAllocator *allocator = nullptr;
    std::map<libcamera::FrameBuffer *, std::vector<libcamera::Span<uint8_t>>> mapped_buffers;
    std::vector<std::unique_ptr<libcamera::Request>> requests;
    libcamera::ControlList controls;
    Callback *callback = nullptr;
    cv::Mat frame;

    std::vector<libcamera::Span<uint8_t>> Mmap(libcamera::FrameBuffer *buffer) const {
        auto it = mapped_buffers.find(buffer);
        if (it != mapped_buffers.end()) return it->second;
        return {};
    }

    void requestComplete(libcamera::Request *request) {
        if (!request || request->status() == libcamera::Request::RequestCancelled)
            return;

        const auto &meta = request->metadata();
        const auto &buffers = request->buffers();

        for (auto &[stream, buffer] : buffers) {
            auto mem = Mmap(buffer);
            auto &cfg = config->at(0);
            unsigned int w = cfg.size.width, h = cfg.size.height, stride = cfg.stride;
            frame.create(h, w, CV_8UC3);
            uint8_t *ptr = mem[0].data();
            for (unsigned int i = 0; i < h; ++i, ptr += stride)
                memcpy(frame.ptr(i), ptr, w * 3);
            if (callback) callback->hasFrame(frame, meta);
        }

        request->reuse(libcamera::Request::ReuseBuffers);
        camera->queueRequest(request);
    }
};

// ----------- drowsiness logic callback --------------
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

class FatigueCallback : public Libcam2OpenCV::Callback {
public:
    FatigueCallback() {
        dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> predictor;
        face_cascade.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml");
    }

    void hasFrame(const cv::Mat &frame, const libcamera::ControlList &) override {
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(80, 80));

        for (const auto &face : faces) {
            dlib::cv_image<dlib::bgr_pixel> cimg(frame);
            dlib::rectangle dlib_rect(face.x, face.y, face.x + face.width, face.y + face.height);
            dlib::full_object_detection shape = predictor(cimg, dlib_rect);

            auto left_eye = extract_eye(shape, true);
            auto right_eye = extract_eye(shape, false);
            float ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0f;

            if (ear < 0.25f) {
                if (++counter >= 15)
                    cv::putText(frame, "DROWSINESS ALERT!", {50, 50}, cv::FONT_HERSHEY_SIMPLEX, 1.0, {0, 0, 255}, 2);
            } else counter = 0;

            for (const auto &pt : left_eye) cv::circle(frame, pt, 2, {0, 255, 0}, -1);
            for (const auto &pt : right_eye) cv::circle(frame, pt, 2, {0, 255, 0}, -1);
        }

        cv::imshow("Fatigue Detection", frame);
        if (cv::waitKey(1) == 'q') exit_requested = true;
    }

    bool exit() const { return exit_requested; }

private:
    dlib::shape_predictor predictor;
    cv::CascadeClassifier face_cascade;
    int counter = 0;
    bool exit_requested = false;
};

// ----------- main ----------------------------------
int main() {
    Libcam2OpenCV cam;
    FatigueCallback cb;
    cam.registerCallback(&cb);
    cam.start(640, 480, 15);

    std::cout << "按下 q 键退出..." << std::endl;
    while (!cb.exit()) std::this_thread::sleep_for(std::chrono::milliseconds(100));

    cam.stop();
    return 0;
}
