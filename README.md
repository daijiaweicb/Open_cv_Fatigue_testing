sudo modprobe v4l2loopback devices=1 video_nr=0 card_label="VirtualCam" exclusive_caps=1
libcamera-vid -t 0 --width 640 --height 480 --framerate 15 --codec yuv420 -o /dev/video0
