#include <libcamera/libcamera.h>
#include <libcamera/camera_manager.h>
#include <libcamera/camera.h>
#include <libcamera/request.h>
#include <libcamera/stream.h>
#include <libcamera/control_ids.h>
#include <libcamera/framebuffer_allocator.h>

#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>

#include <iostream>
#include <memory>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>

using namespace libcamera;
using namespace std;

Span<uint8_t> mmapBuffer(FrameBuffer *buffer) {
    const FrameBuffer::Plane &plane = buffer->planes()[0];
    void *memory = mmap(nullptr, plane.length, PROT_READ | PROT_WRITE, MAP_SHARED, plane.fd.get(), 0);
    return Span<uint8_t>(static_cast<uint8_t *>(memory), plane.length);
}

int main() {
    // 初始化 libcamera
    std::unique_ptr<CameraManager> cm = std::make_unique<CameraManager>();
    cm->start();

    if (cm->cameras().empty()) {
        cerr << "没有检测到摄像头\n";
        return -1;
    }

    shared_ptr<Camera> camera = cm->get(cm->cameras()[0]->id());
    camera->acquire();

    std::unique_ptr<CameraConfiguration> config = camera->generateConfiguration({StreamRole::Viewfinder});
    StreamConfiguration &streamConfig = config->at(0);
    streamConfig.pixelFormat = formats::BGR888;
    streamConfig.size.width = 640;
    streamConfig.size.height = 480;
    config->validate();
    camera->configure(config.get());

    Stream *stream = streamConfig.stream();
    FrameBufferAllocator allocator(camera);
    allocator.allocate(stream);

    // mmap 映射缓冲
    std::map<FrameBuffer *, Span<uint8_t>> mmapBuffers;
    const std::vector<std::unique_ptr<FrameBuffer>> &buffers = allocator.buffers(stream);
    for (const auto &buffer : buffers) {
        mmapBuffers[buffer.get()] = mmapBuffer(buffer.get());
    }

    // 准备 dlib 检测器
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor shape;
    dlib::deserialize("/path/to/shape_predictor_68_face_landmarks.dat") >> shape; // 你需要修改成正确的路径

    // 设置曝光、帧率等参数（可选）
    ControlList controls(camera->controls());
    controls.set(controls::Brightness, 0.5);
    controls.set(controls::Contrast, 1.0);
    camera->start(&controls);

    // 预填充 Request 队列
    std::vector<std::unique_ptr<Request>> requests;
    for (const auto &buffer : buffers) {
        std::unique_ptr<Request> request = camera->createRequest();
        request->addBuffer(stream, buffer.get());
        requests.push_back(std::move(request));
    }

    for (auto &request : requests)
        camera->queueRequest(request.get());

    // 主循环
    int frameCount = 0;
    while (true) {
        // 等待请求完成
        Request *completed = nullptr;
        while (!completed) {
            camera->requestCompleted.connect(
                [&](Request *req) {
                    completed = req;
                }
            );
            usleep(1000); // 等待 1ms
        }

        if (completed->status() == Request::RequestCancelled)
            continue;

        FrameBuffer *buffer = completed->buffers().begin()->second;
        Span<uint8_t> data = mmapBuffers[buffer];

        int width = streamConfig.size.width;
        int height = streamConfig.size.height;
        int stride = streamConfig.stride;

        // 复制内存为 OpenCV 图像
        cv::Mat frame(height, width, CV_8UC3);
        uint8_t *ptr = data.data();
        for (int i = 0; i < height; ++i, ptr += stride) {
            memcpy(frame.ptr(i), ptr, width * 3);
        }

        // dlib 图像转换
        dlib::cv_image<dlib::bgr_pixel> dlibImg(frame);
        std::vector<dlib::rectangle> faces = detector(dlibImg);

        for (auto &face : faces) {
            dlib::full_object_detection landmarks = shape(dlibImg, face);
            for (int i = 36; i <= 41; ++i) {
                cv::circle(frame, cv::Point(landmarks.part(i).x(), landmarks.part(i).y()), 2, cv::Scalar(0, 255, 0), -1);
            }
            for (int i = 42; i <= 47; ++i) {
                cv::circle(frame, cv::Point(landmarks.part(i).x(), landmarks.part(i).y()), 2, cv::Scalar(255, 0, 0), -1);
            }
        }

        cv::imshow("Fatigue Detection", frame);
        if (cv::waitKey(1) == 27) break; // 按 Esc 退出

        // 重新入队
        completed->reuse(Request::ReuseBuffers);
        camera->queueRequest(completed);

        frameCount++;
    }

    // 清理
    camera->stop();
    allocator.free(stream);
    camera->release();
    cm->stop();

    return 0;
}
