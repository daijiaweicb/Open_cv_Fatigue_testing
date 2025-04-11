#include "window.h"

Window::Window() {
    myCallback.window = this;
    camera.registerCallback(&myCallback);

    thermo = new QwtThermo;
    thermo->setScale(0, 255);
    thermo->setFillBrush(QBrush(Qt::red));
    thermo->show();

    image = new QLabel;
    label_fps = new QLabel("FPS: ");
    label_ear = new QLabel("EAR: ");
    label_alarm = new QLabel("状态: 正常");

    QVBoxLayout *vInfo = new QVBoxLayout;
    vInfo->addWidget(thermo);
    vInfo->addWidget(label_fps);
    vInfo->addWidget(label_ear);
    vInfo->addWidget(label_alarm);

    hLayout = new QHBoxLayout;
    hLayout->addLayout(vInfo);
    hLayout->addWidget(image);
    setLayout(hLayout);

    camera.start();
}

Window::~Window() {
    camera.stop();
}

void Window::updateImage(const cv::Mat &mat, float fps, float ear, bool alarm) {
    QImage qimg(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_BGR888);
    image->setPixmap(QPixmap::fromImage(qimg));

    QColor color = qimg.pixelColor(mat.cols / 2, mat.rows / 2);
    thermo->setValue(color.lightness());

    label_fps->setText(QString("FPS: %1").arg(fps, 0, 'f', 2));
    label_ear->setText(QString("EAR: %1").arg(ear, 0, 'f', 3));
    label_alarm->setText(alarm ? "状态: 疲劳警报!" : "状态: 正常");

    update();
}
