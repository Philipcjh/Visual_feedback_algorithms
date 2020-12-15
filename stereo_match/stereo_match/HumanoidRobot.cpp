#include "HumanoidRobot.h"

Humanoid_Robot::Humanoid_Robot()
{
}

Humanoid_Robot::~Humanoid_Robot()
{
}

void Humanoid_Robot::triangulatePnts(const vector<Point2f>& pnts2d1, const vector<cv::Point2f>& pnts2d2, const Mat& cameraMatrix1, const Mat& R1, const Mat& T1, const vector<double>& distCoeffs1, const Mat& cameraMatrix2, const Mat& R2, const Mat& T2, const vector<double>& distCoeffs2, vector<Point3f>& pnts3d)
{
    Mat _R1, _R2;
    R1.copyTo(_R1);
    R2.copyTo(_R2);
    if (R1.rows == 1 || R1.cols == 1)
        Rodrigues(_R1, _R1);
    if (R2.rows == 1 || R2.cols == 1)
        Rodrigues(_R2, _R2);

    Mat RT1(3, 4, CV_64F), RT2(3, 4, CV_64F);
    _R1.colRange(0, 3).copyTo(RT1.colRange(0, 3));
    _R2.colRange(0, 3).copyTo(RT2.colRange(0, 3));
    T1.copyTo(RT1.col(3));
    T2.copyTo(RT2.col(3));

    Mat P1, P2;//3x4
    P1 = cameraMatrix1 * RT1;
    P2 = cameraMatrix2 * RT2;

    vector<Point2f> _pnts2d1, _pnts2d2;

    undistortPoints(pnts2d1, _pnts2d1, cameraMatrix1, distCoeffs1, cameraMatrix1);
    undistortPoints(pnts2d2, _pnts2d2, cameraMatrix2, distCoeffs2, cameraMatrix2);

    Mat pnts4d;
    triangulatePoints(P1, P2, _pnts2d1, _pnts2d2, pnts4d);
	
	for (int i = 0; i < pnts4d.size().width; i++)
	{
		Point3f point_tmp(pnts4d.at<float>(0, i) / pnts4d.at<float>(3, i), pnts4d.at<float>(1, i) / pnts4d.at<float>(3, i), pnts4d.at<float>(2, i) / pnts4d.at<float>(3, i));
		pnts3d.push_back(point_tmp);
	}
}
