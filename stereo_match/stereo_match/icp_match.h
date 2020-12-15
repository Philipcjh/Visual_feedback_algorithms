#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>


using namespace std;

class icp_match
{
public:
	icp_match();
	~icp_match();

	static void icp_estimation(const string& filename, int iterations,const vector<cv::Point3f>& pnts3d, cv::Mat& Hcm);
};