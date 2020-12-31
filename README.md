# Visual_feedback_algorithms
This project includes calibration and stereo_match. Before you run the codes, you need to configure the environments.  
Instructions for environment configuration of [OpenCV and OpenCV_contri](https://blog.csdn.net/weijifen000/article/details/93377143).(OpenCV 4.5.0 is preferred.)  
Instructions for environment configuration of [PCL](https://zhuanlan.zhihu.com/p/142955614?utm_source=wechat_session).  
***
## Calibration
### calibration.cpp
You can use this cpp to calibrate single camera. It refers to [calibration.cpp](https://github.com/opencv/opencv/blob/master/samples/cpp/calibration.cpp).
### stereo_calib.cpp
You can use stereo_calib.cpp to calibrate stereo camera. It refers to [stereo_calib.cpp](https://github.com/opencv/opencv/blob/master/samples/cpp/stereo_calib.cpp).
### EIH_Calibration.cpp
You can use EIH_Calibration.cpp to calculate the transformation matrix of camera in hand. This cpp is only suitable for the method of eye-in-hand.
***
## Stereo_match
### surf_match.cpp
You can use surf_match.cpp to realize feature points extraction based on [SURF](https://docs.opencv.org/4.5.0/d5/df7/classcv_1_1xfeatures2d_1_1SURF.html) and pointcloud match based on [ICP](https://pcl.readthedocs.io/projects/tutorials/en/latest/interactive_icp.html). You need to input the extrinsics and intrinsics of stereo camera, which can be required by stereo_calib.cpp. In addition, two photos by stereo camera and ply-document are also required.
### sift_match.cpp
It is based on [SIFT](https://docs.opencv.org/4.5.0/d7/d60/classcv_1_1SIFT.html) and its other parts are same as surf_match.cpp.
