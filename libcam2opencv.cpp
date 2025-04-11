#include "libcam2opencv.h"

void Libcam2OpenCV::requestComplete(libcamera::Request *request) {
    if (nullptr == request) return;
    if (request->status() == libcamera::Request::RequestCancelled)
	return;

    /*
     * When a request has completed, it is populated with a metadata control
     * list that allows an application to determine various properties of
     * the completed request. This can include the timestamp of the Sensor
     * capture, or its gain and exposure values, or properties from the IPA
     * such as the state of the 3A algorithms.
     *
     * ControlValue types have a toString, so to examine each request, print
     * all the metadata for inspection. A custom application can parse each
     * of these items and process them according to its needs.
     */
    const libcamera::ControlList &requestMetadata = request->metadata();
    
    /*
     * Each buffer has its own FrameMetadata to describe its state, or the
     * usage of each buffer. While in our simple capture we only provide one
     * buffer per request, a request can have a buffer for each stream that
     * is established when configuring the camera.
     *
     * This allows a viewfinder and a still image to be processed at the
     * same time, or to allow obtaining the RAW capture buffer from the
     * sensor along with the image as processed by the ISP.
     */
    const libcamera::Request::BufferMap &buffers = request->buffers();
    for (auto bufferPair : buffers) {
	libcamera::FrameBuffer *buffer = bufferPair.second;
	libcamera::StreamConfiguration &streamConfig = config->at(0);
	unsigned int vw = streamConfig.size.width;
	unsigned int vh = streamConfig.size.height;
	unsigned int vstr = streamConfig.stride;
	auto mem = Mmap(buffer);
	frame.create(vh,vw,CV_8UC3);
	uint ls = vw*3;
	uint8_t *ptr = mem[0].data();
	for (unsigned int i = 0; i < vh; i++, ptr += vstr) {
	    memcpy(frame.ptr(i),ptr,ls);
	}
	if (nullptr != callback) {
	    callback->hasFrame(frame, requestMetadata);
	}
    }

    // in case the request has been cancelled in the meantime
    // this is a hack because libcamera should wait till a request has finisehd but doesn't
    if (nullptr == request) return;
    if (request->status() == libcamera::Request::RequestCancelled)
	return;
    /* Re-queue the Request to the camera. */
    request->reuse(libcamera::Request::ReuseBuffers);
    camera->queueRequest(request);
}

void Libcam2OpenCV::start(Libcam2OpenCVSettings settings) {
    cm = std::make_unique<libcamera::CameraManager>();
    cm->start();

    if (cm->cameras().empty()) {
        std::cerr << "❌ 未发现可用相机！" << std::endl;
        return;
    }

    camera = cm->cameras()[0];
    camera->acquire();

    // 申请 viewfinder 流配置
    config = camera->generateConfiguration({ libcamera::StreamRole::Viewfinder });
    libcamera::StreamConfiguration &streamConfig = config->at(0);

    // 设置分辨率
    if (settings.width > 0 && settings.height > 0) {
        streamConfig.size.width = settings.width;
        streamConfig.size.height = settings.height;
    }

    // ✅ 使用兼容的像素格式（IMX219不支持BGR888）
    streamConfig.pixelFormat = libcamera::formats::YUV420;  // 或 RGB888

    // ⚠️ validate 在 configure 前调用
    config->validate();

    // 应用配置
    if (camera->configure(config.get()) < 0) {
        std::cerr << "❌ 摄像头配置失败！" << std::endl;
        return;
    }

    // 分配缓冲区
    allocator = new libcamera::FrameBufferAllocator(camera);
    stream = streamConfig.stream();
    if (allocator->allocate(stream) < 0) {
        std::cerr << "❌ 缓冲区分配失败！" << std::endl;
        return;
    }

    for (const auto &buffer : allocator->buffers(stream)) {
        size_t buffer_size = 0;
        for (const auto &plane : buffer->planes()) {
            buffer_size += plane.length;
            void *memory = mmap(nullptr, plane.length, PROT_READ | PROT_WRITE, MAP_SHARED, plane.fd.get(), 0);
            mapped_buffers[buffer.get()].emplace_back(reinterpret_cast<uint8_t *>(memory), plane.length);
        }
    }

    // 创建请求
    for (const auto &buffer : allocator->buffers(stream)) {
        std::unique_ptr<libcamera::Request> request = camera->createRequest();
        if (!request) {
            std::cerr << "❌ 创建请求失败！" << std::endl;
            return;
        }

        if (request->addBuffer(stream, buffer.get()) < 0) {
            std::cerr << "❌ 向请求添加 buffer 失败！" << std::endl;
            return;
        }

        requests.push_back(std::move(request));
    }

    // 连接信号槽
    camera->requestCompleted.connect(this, &Libcam2OpenCV::requestComplete);

    // 设置控制参数
    if (settings.framerate > 0) {
        int64_t frame_time = 1000000 / settings.framerate;
        controls.set(libcamera::controls::FrameDurationLimits, libcamera::Span<const int64_t, 2>({frame_time, frame_time}));
    }
    controls.set(libcamera::controls::Brightness, settings.brightness);
    controls.set(libcamera::controls::Contrast, settings.contrast);

    // 启动摄像头
    if (camera->start(&controls) < 0) {
        std::cerr << "❌ 摄像头启动失败！" << std::endl;
        return;
    }

    // 初始请求入队
    for (auto &request : requests) {
        if (camera->queueRequest(request.get()) < 0) {
            std::cerr << "❌ 入队请求失败！" << std::endl;
        }
    }

    std::cout << "✅ 摄像头启动成功，正在捕获帧..." << std::endl;
}


void Libcam2OpenCV::stop() {
    /*
     * --------------------------------------------------------------------
     * Clean Up
     *
     * Stop the Camera, release resources and stop the CameraManager.
     * libcamera has now released all resources it owned.
     */
    camera->stop();
    allocator->free(stream);
    camera->release();
    camera.reset();
    cm->stop();
    delete allocator;
}
