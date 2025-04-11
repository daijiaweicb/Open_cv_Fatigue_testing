/usr/include/dlib/statistics/lda.h: In function ‘std::pair<double, double> dlib::equal_error_rate(const std::vector<double, std::allocator<double> >&, const std::vector<double, std::allocator<double> >&)’:
/usr/include/dlib/statistics/lda.h:199:5: note: parameter passing for argument of type ‘std::pair<double, double>’ when C++17 is enabled changed to match C++14 in GCC 10.1
  199 |     )
      |     ^
/home/helmet/Desktop/Open_cv_Fatigue_testing/main.cpp: In member function ‘void Libcam2OpenCV::start(unsigned int, unsigned int, unsigned int)’:
/home/helmet/Desktop/Open_cv_Fatigue_testing/main.cpp:43:69: error: no matching function for call to ‘libcamera::FrameBufferAllocator::FrameBufferAllocator(std::__shared_ptr<libcamera::Camera, __gnu_cxx::_S_atomic>::element_type*)’
   43 |         allocator = new libcamera::FrameBufferAllocator(camera.get());
      |                                                                     ^
In file included from /usr/include/libcamera/libcamera/libcamera.h:19,
                 from /home/helmet/Desktop/Open_cv_Fatigue_testing/main.cpp:11:
/usr/include/libcamera/libcamera/framebuffer_allocator.h:25:9: note: candidate: ‘libcamera::FrameBufferAllocator::FrameBufferAllocator(std::shared_ptr<libcamera::Camera>)’
   25 |         FrameBufferAllocator(std::shared_ptr<Camera> camera);
      |         ^~~~~~~~~~~~~~~~~~~~
/usr/include/libcamera/libcamera/framebuffer_allocator.h:25:54: note:   no known conversion for argument 1 from ‘std::__shared_ptr<libcamera::Camera, __gnu_cxx::_S_atomic>::element_type*’ {aka ‘libcamera::Camera*’} to ‘std::shared_ptr<libcamera::Camera>’
   25 |         FrameBufferAllocator(std::shared_ptr<Camera> camera);
      |                              ~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~
make[2]: *** [CMakeFiles/fatigue_detection.dir/build.make:76: CMakeFiles/fatigue_detection.dir/main.cpp.o] Error 1
make[1]: *** [CMakeFiles/Makefile2:83: CMakeFiles/fatigue_detection.dir/all] Error 2


