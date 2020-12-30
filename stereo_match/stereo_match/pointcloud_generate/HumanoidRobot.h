#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
using namespace std;
using namespace cv;

class Humanoid_Robot
{
public:
	Humanoid_Robot();
	~Humanoid_Robot();

	//// 根据左右相机的对应性信息，内外参数值，对畸变进行校正，然后计算三维点
	//// const vector<Point2f>& pnts2d1, <in> 左相机的对应点信息，要求x坐标，y坐标都有
	/// const vector<Point2f>& pnts2d2, <in> 右相机的对应点信息，要求x坐标，y坐标都有
	/// const Mat& cameraMatrix1,  <in>  左相机的内参数
	/// const Mat& R1, <in> 左相机相对世界坐标系的旋转矩阵，若世界坐标系建立在摄像机坐标1 上，其为单位矩阵
	/// const Mat& T1, <in> 左相机相对世界坐标系的平移向量，若是世界坐标系建立在摄像机坐标1上，其为0矩阵
	/// const vector<double>& distCoeffs1, <in>
	/// const Mat& cameraMatrix2, <in>  右相机的内参数
	/// const Mat& R2, <in> 右相机相对世界坐标系的旋转矩阵，若世界坐标系建立在摄像机坐标系1上，其为相对于摄像机1的旋转矩阵
	/// const Mat& T2, <in> 右相机相对世界坐标系的平移矩阵，若世界坐标系建立在摄像机坐标系1上，其为相对于摄像机1的平移矩阵
	/// const vector<double>& distCoeffs2, <in>
	///	vector<Point3f>& pnts3d  <out>
	static void triangulatePnts(const vector<Point2f>& pnts2d1, const vector<cv::Point2f>& pnts2d2, const Mat& cameraMatrix1, const Mat& R1, const Mat& T1, const vector<double>& distCoeffs1, const Mat& cameraMatrix2, const Mat& R2, const Mat& T2, const vector<double>& distCoeffs2, vector<Point3f>& pnts3d);
};

