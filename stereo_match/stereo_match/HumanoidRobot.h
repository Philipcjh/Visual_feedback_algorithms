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

	//// ������������Ķ�Ӧ����Ϣ���������ֵ���Ի������У����Ȼ�������ά��
	//// const vector<Point2f>& pnts2d1, <in> ������Ķ�Ӧ����Ϣ��Ҫ��x���꣬y���궼��
	/// const vector<Point2f>& pnts2d2, <in> ������Ķ�Ӧ����Ϣ��Ҫ��x���꣬y���궼��
	/// const Mat& cameraMatrix1,  <in>  ��������ڲ���
	/// const Mat& R1, <in> ����������������ϵ����ת��������������ϵ���������������1 �ϣ���Ϊ��λ����
	/// const Mat& T1, <in> ����������������ϵ��ƽ��������������������ϵ���������������1�ϣ���Ϊ0����
	/// const vector<double>& distCoeffs1, <in>
	/// const Mat& cameraMatrix2, <in>  ��������ڲ���
	/// const Mat& R2, <in> ����������������ϵ����ת��������������ϵ���������������ϵ1�ϣ���Ϊ����������1����ת����
	/// const Mat& T2, <in> ����������������ϵ��ƽ�ƾ�������������ϵ���������������ϵ1�ϣ���Ϊ����������1��ƽ�ƾ���
	/// const vector<double>& distCoeffs2, <in>
	///	vector<Point3f>& pnts3d  <out>
	static void triangulatePnts(const vector<Point2f>& pnts2d1, const vector<cv::Point2f>& pnts2d2, const Mat& cameraMatrix1, const Mat& R1, const Mat& T1, const vector<double>& distCoeffs1, const Mat& cameraMatrix2, const Mat& R2, const Mat& T2, const vector<double>& distCoeffs2, vector<Point3f>& pnts3d);
};

