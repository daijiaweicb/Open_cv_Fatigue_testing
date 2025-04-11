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
#include <libcamera/property_ids.h>
#include <libcamera/control_ids.h>
#include <sys/mman.h>
#include <functional>  // 添加这个

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
        streamConfig.pixelFormat = libcamera::formats::XRGB8888;
        config->validate();
        camera->configure(config.get());

        allocator = new libcamera::FrameBufferAllocator(camera);
        stream = streamConfig.stream();
        allocator->allocate(stream);

        std::cout << "[INFO] Buffers allocated: " << allocator->buffers(stream).size() << std::endl;

        for (auto &buffer : allocator->buffers(stream)) {
            size_t buffer_size = 0;
            for (unsigned i = 0; i < buffer->planes().size(); ++i) {
                const auto &plane = buffer->planes()[i];
                buffer_size += plane.length;
                if (i == buffer->planes().size() - 1 || plane.fd.get() != buffer->planes()[i + 1].fd.get()) {
                    void *memory = mmap(nullptr, buffer_size, PROT_READ | PROT_WRITE, MAP_SHARED, plane.fd.get(), 0);
                    if (memory == MAP_FAILED) {
                        perror("mmap failed");
                        exit(1);
                    }
                    mapped_buffers[buffer.get()].emplace_back(static_cast<uint8_t *>(memory), buffer_size);
                    buffer_size = 0;
                }
            }
        }

        auto &buffers = allocator->buffers(stream);
        size_t count = std::max(buffers.size(), size_t(4));  // 最少使用 4 个 request 循环

        for (size_t i = 0; i < count; ++i) {
            auto buffer = buffers[i % buffers.size()];
            auto request = camera->createRequest();
            request->addBuffer(stream, buffer.get());
            requests.push_back(std::move(request));
        }

        // ✅ 更稳定的信号连接方式（避免某些编译器错误）
        using namespace std::placeholders;
        camera->requestCompleted.connect(std::bind(&Libcam2OpenCV::requestComplete, this, _1));

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
        std::cout << "[DEBUG] Frame received" << std::endl;

        if (!request || request->status() == libcamera::Request::RequestCancelled)
            return;

        const auto &meta = request->metadata();
        const auto &buffers = request->buffers();

        for (auto &[stream, buffer] : buffers) {
            auto mem = Mmap(buffer);
            auto &cfg = config->at(0);
            unsigned int w = cfg.size.width, h = cfg.size.height, stride = cfg.stride;

            frame.create(h, w, CV_8UC4);
            uint8_t *ptr = mem[0].data();
            for (unsigned int i = 0; i < h; ++i, ptr += stride)
                memcpy(frame.ptr(i), ptr, w * 4);

            cv::Mat bgr;
            cv::cvtColor(frame, bgr, cv::COLOR_BGRA2BGR);

            if (callback) callback->hasFrame(bgr, meta);
        }

        request->reuse(libcamera::Request::ReuseBuffers);
        camera->queueRequest(request);
    }
};
