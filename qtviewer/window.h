#ifndef WINDOW_H
#define WINDOW_H

#include <qwt/qwt_thermo.h>

#include <QBoxLayout>
#include <QPushButton>
#include <QLabel>

#include "libcam2opencv.h"

#include "fatigue_callback.h"

class Window : public QWidget {
    Q_OBJECT
public:
    Window();
    ~Window();
    void updateImage(const cv::Mat &mat, float fps, float ear, bool alarm);

private:
    QwtThermo *thermo;
    QLabel *image;
    QLabel *label_fps;
    QLabel *label_ear;
    QLabel *label_alarm;
    QHBoxLayout *hLayout;

    Libcam2OpenCV camera;
    FatigueCallback myCallback;
};

#endif // WINDOW_H
