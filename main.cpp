#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <libcamera/libcamera.h>

using namespace cv;
using namespace std;

// 眼睛纵横比计算
float eye_aspect_ratio(vector<Point2f>& eye) {
    float A = norm(eye[1] - eye[5]);
    float B = norm(eye[2] - eye[4]);
    float C = norm(eye[0] - eye[3]);
    return (A + B) / (2.0 * C);
}

int main() {
    // 初始化libcamera
    libcamera::CameraManager manager;
    manager.start();
    
    // 获取相机设备
    auto camera = manager.get("0");
    camera->acquire();
    
    // 配置相机参数
    libcamera::CameraConfiguration* config = camera->generateConfiguration();
    config->at(0).pixelFormat = libcamera::formats::BGR888;
    config->at(0).size = {640, 480};
    config->at(0).bufferCount = 4;
    config->validate();
    camera->configure(config);

    // 分配帧缓冲区
    libcamera::FrameBufferAllocator allocator(camera);
    allocator.allocate(config->at(0).stream());

    // 启动相机
    camera->start();
    
    // 加载人脸和眼睛检测模型
    CascadeClassifier face_cascade;
    CascadeClassifier eyes_cascade;
    if(!face_cascade.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml") ||
       !eyes_cascade.load("/usr/share/opencv4/haarcascades/haarcascade_eye_tree_eyeglasses.xml")) {
        cerr << "Error loading cascades!" << endl;
        return -1;
    }

    // 疲劳检测参数
    const float EAR_THRESHOLD = 0.25;
    const int EYES_CLOSED_FRAMES = 15;
    int counter = 0;
    bool alarm = false;

    while (true) {
        // 获取帧数据
        libcamera::FrameBuffer* buffer = allocator.buffers().front().get();
        camera->requestCompleted.connect([&](libcamera::Request* request) {
            buffer = request->buffers().begin()->second;
        });
        
        // 转换到OpenCV Mat
        Mat frame(config->at(0).size.height, config->at(0).size.width, CV_8UC3, buffer->planes()[0].data());

        // 转换为灰度图
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        equalizeHist(gray, gray);

        // 人脸检测
        vector<Rect> faces;
        face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, Size(100,100));

        for (auto& face : faces) {
            // 眼睛检测
            Mat faceROI = gray(face);
            vector<Rect> eyes;
            eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30,30));

            // 计算EAR
            if (eyes.size() >= 2) {
                vector<Point2f> eye_points;
                for (auto& eye : eyes) {
                    // 转换眼睛坐标到全局坐标系
                    Rect global_eye(eye.x + face.x, eye.y + face.y, eye.width, eye.height);
                    // 提取特征点（简化处理，实际应使用更精确的定位）
                    for (int i=0; i<6; ++i) {
                        eye_points.emplace_back(
                            global_eye.x + i*global_eye.width/5,
                            global_eye.y + global_eye.height/2
                        );
                    }
                }

                float ear = eye_aspect_ratio(eye_points);
                
                // 判断眼睛状态
                if (ear < EAR_THRESHOLD) {
                    counter++;
                    if (counter >= EYES_CLOSED_FRAMES) {
                        alarm = true;
                        putText(frame, "DROWSINESS ALERT!", Point(10,30),
                               FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,0,255), 2);
                    }
                } else {
                    counter = 0;
                    alarm = false;
                }
            }
        }

        // 显示结果
        imshow("Fatigue Detection", frame);
        if (waitKey(1) == 27) break;  // ESC退出
    }

    // 清理资源
    camera->stop();
    camera->release();
    manager.stop();
    return 0;
}