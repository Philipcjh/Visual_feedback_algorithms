#include <iostream>
#include<opencv2/opencv.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "pointcloud_generate/HumanoidRobot.h"
#include "icp/icp_match.h"

using namespace cv;
using namespace cv::xfeatures2d;


int64 work_begin = 0;
int64 work_end = 0;


//R��TתRT����
Mat R_T2RT(Mat& R, Mat& T)
{
    Mat RT;
    Mat_<double> R1 = (cv::Mat_<double>(4, 3) << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
        R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
        R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2),
        0.0, 0.0, 0.0);
    cv::Mat_<double> T1 = (cv::Mat_<double>(4, 1) << T.at<double>(0, 0), T.at<double>(1, 0), T.at<double>(2, 0), 1.0);

    cv::hconcat(R1, T1, RT);//C=A+B����ƴ��
    return RT;
}

//RTתR��T����
void RT2R_T(Mat& RT, Mat& R, Mat& T)
{
    cv::Rect R_rect(0, 0, 3, 3);
    cv::Rect T_rect(3, 0, 1, 3);
    R = RT(R_rect);
    T = RT(T_rect);
}

//�ж��Ƿ�Ϊ��ת����
bool isRotationMatrix(const cv::Mat& R)
{
    cv::Mat tmp33 = R({ 0,0,3,3 });
    cv::Mat shouldBeIdentity;

    shouldBeIdentity = tmp33.t() * tmp33;

    cv::Mat I = cv::Mat::eye(3, 3, shouldBeIdentity.type());

    return  cv::norm(I, shouldBeIdentity) < 1e-6;
}

/** @brief ŷ���� -> 3*3 ��R
*	@param 	eulerAngle		�Ƕ�ֵ
*	@param 	seq				ָ��ŷ����xyz������˳���磺"xyz" "zyx"
*/
cv::Mat eulerAngleToRotatedMatrix(const cv::Mat& eulerAngle, const std::string& seq)
{
    CV_Assert(eulerAngle.rows == 1 && eulerAngle.cols == 3);

    eulerAngle /= 180 / CV_PI;
    cv::Matx13d m(eulerAngle);
    auto rx = m(0, 0), ry = m(0, 1), rz = m(0, 2);
    auto xs = std::sin(rx), xc = std::cos(rx);
    auto ys = std::sin(ry), yc = std::cos(ry);
    auto zs = std::sin(rz), zc = std::cos(rz);

    cv::Mat rotX = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, xc, -xs, 0, xs, xc);
    cv::Mat rotY = (cv::Mat_<double>(3, 3) << yc, 0, ys, 0, 1, 0, -ys, 0, yc);
    cv::Mat rotZ = (cv::Mat_<double>(3, 3) << zc, -zs, 0, zs, zc, 0, 0, 0, 1);

    cv::Mat rotMat;

    if (seq == "zyx")		rotMat = rotX * rotY * rotZ;
    else if (seq == "yzx")	rotMat = rotX * rotZ * rotY;
    else if (seq == "zxy")	rotMat = rotY * rotX * rotZ;
    else if (seq == "xzy")	rotMat = rotY * rotZ * rotX;
    else if (seq == "yxz")	rotMat = rotZ * rotX * rotY;
    else if (seq == "xyz")	rotMat = rotZ * rotY * rotX;
    else {
        cv::error(cv::Error::StsAssert, "Euler angle sequence string is wrong.",
            __FUNCTION__, __FILE__, __LINE__);
    }

    if (!isRotationMatrix(rotMat)) {
        cv::error(cv::Error::StsAssert, "Euler angle can not convert to rotated matrix",
            __FUNCTION__, __FILE__, __LINE__);
    }

    return rotMat;
    //cout << isRotationMatrix(rotMat) << endl;
}

/** @brief ��Ԫ��ת��ת����
*	@note  ��������double�� ��Ԫ������ q = w + x*i + y*j + z*k
*	@param q ��Ԫ������{w,x,y,z}����
*	@return ������ת����3*3
*/
cv::Mat quaternionToRotatedMatrix(const cv::Vec4d& q)
{
    double w = q[0], x = q[1], y = q[2], z = q[3];

    double x2 = x * x, y2 = y * y, z2 = z * z;
    double xy = x * y, xz = x * z, yz = y * z;
    double wx = w * x, wy = w * y, wz = w * z;

    cv::Matx33d res{
        1 - 2 * (y2 + z2),	2 * (xy - wz),		2 * (xz + wy),
        2 * (xy + wz),		1 - 2 * (x2 + z2),	2 * (yz - wx),
        2 * (xz - wy),		2 * (yz + wx),		1 - 2 * (x2 + y2),
    };
    return cv::Mat(res);
}

/** @brief ((��Ԫ��||ŷ����||��ת����) && ת������) -> 4*4 ��Rt
*	@param 	m				1*6 || 1*10�ľ���  -> 6  {x,y,z, rx,ry,rz}   10 {x,y,z, qw,qx,qy,qz, rx,ry,rz}
*	@param 	useQuaternion	�����1*10�ľ����ж��Ƿ�ʹ����Ԫ��������ת����
*	@param 	seq				���ͨ��ŷ���Ǽ�����ת������Ҫָ��ŷ����xyz������˳���磺"xyz" "zyx" Ϊ�ձ�ʾ��ת����
*/
cv::Mat attitudeVectorToMatrix(cv::Mat& m, bool useQuaternion, const std::string& seq)
{
    CV_Assert(m.total() == 6 || m.total() == 10);
    if (m.cols == 1)
        m = m.t();
    cv::Mat tmp = cv::Mat::eye(4, 4, CV_64FC1);

    //���ʹ����Ԫ��ת������ת�������ȡm����ĵڵ��ĸ���Ա����4������
    if (useQuaternion)	// normalized vector, its norm should be 1.
    {
        cv::Vec4d quaternionVec = m({ 3, 0, 4, 1 });
        quaternionToRotatedMatrix(quaternionVec).copyTo(tmp({ 0, 0, 3, 3 }));
        // cout << norm(quaternionVec) << endl; 
    }
    else
    {
        cv::Mat rotVec;
        if (m.total() == 6)
            rotVec = m({ 3, 0, 3, 1 });		//6
        else
            rotVec = m({ 7, 0, 3, 1 });		//10

        //���seqΪ�ձ�ʾ���������ת����������"xyz"����ϱ�ʾŷ����
        if (0 == seq.compare(""))
            cv::Rodrigues(rotVec, tmp({ 0, 0, 3, 3 }));
        else
            eulerAngleToRotatedMatrix(rotVec, seq).copyTo(tmp({ 0, 0, 3, 3 }));
    }
    tmp({ 3, 0, 1, 3 }) = m({ 0, 0, 3, 1 }).t();
    //std::swap(m,tmp);
    return tmp;
}

static void workBegin()
{
    work_begin = getTickCount();
}

static void workEnd()
{
    work_end = getTickCount() - work_begin;
}

static double getTime()
{
    return work_end / ((double)getTickFrequency()) * 1000.;
}

struct SURFDetector
{
    Ptr<Feature2D> surf;
    SURFDetector(double hessian = 1000.0)//800.0
    {
        surf = SURF::create(hessian);
    }
    template<class T>
    void operator()(const T& in, const T& mask, std::vector<cv::KeyPoint>& pts, T& descriptors, bool useProvided = false)
    {
        surf->detectAndCompute(in, mask, pts, descriptors, useProvided);
    }
};

template<class KPMatcher>
struct SURFMatcher
{
    KPMatcher matcher;
    template<class T>
    void match(const T& in1, const T& in2, std::vector<cv::DMatch>& matches)
    {
        matcher.match(in1, in2, matches);
    }
};



////////////////////////////////////////////////////

int main() {
    //Create SIFT class pointer
    Ptr<Feature2D> f2d = SIFT::create();
    Mat img_1 = imread("17-1.bmp");
    Mat img_2 = imread("17-2.bmp");

    vector<KeyPoint> keypoints_1, keypoints_2;
    f2d->detect(img_1, keypoints_1);
    f2d->detect(img_2, keypoints_2);

    Mat descriptors_1, descriptors_2;
    f2d->compute(img_1, keypoints_1, descriptors_1);
    f2d->compute(img_2, keypoints_2, descriptors_2);

    BFMatcher matcher;
    vector<DMatch> matches;
    matcher.match(descriptors_1, descriptors_2, matches);

    std::nth_element(matches.begin(),     
        matches.begin() + 19,  
        matches.end());       

    matches.erase(matches.begin() + 20, matches.end());


    vector<Point2f> pnts2d1, pnts2d2;
    for (auto& i : matches) {
        pnts2d1.push_back(keypoints_1[i.queryIdx].pt);
        pnts2d2.push_back(keypoints_2[i.trainIdx].pt);
    }
    std::vector<Point3f> pnts3d;

    FileStorage fs("intrinsics.yml", FileStorage::READ);
    Mat M1, D1, M2, D2;
    fs["M1"] >> M1;
    fs["D1"] >> D1;
    fs["M2"] >> M2;
    fs["D2"] >> D2;

    fs.open("extrinsics.yml", FileStorage::READ);
    Mat R2, T2;
    Mat_<double> R1 = (cv::Mat_<double>(3, 3) <<
        1, 0, 0,
        0, 1, 0,
        0, 0, 1);
    Mat_<double> T1 = (cv::Mat_<double>(3, 1) <<
        0,
        0,
        0);

    fs["R"] >> R2;
    fs["T"] >> T2;

    //transform
    Humanoid_Robot::triangulatePnts(pnts2d1, pnts2d2, M1, R1, T1, D1, M2, R2, T2, D2, pnts3d);

    int iterations = 10;//��������
    std::string plyname = "�ʺ��ǵ���.ply";
    cv::Mat_<double> Hmc(4, 4);
    icp_match::icp_estimation(plyname, iterations, pnts3d, Hmc);

    Mat Hbg, Hbm;
    Mat_<double> InitialPose = (cv::Mat_<double>(1, 6) << 488.9, -132.94, 131.8, 2.95, 1.09, -0.018);
    Hbg = attitudeVectorToMatrix(InitialPose, false, "");

    cv::Mat Hgc = (cv::Mat_<double>(4, 4) <<
        -0.6729891476240062, -0.701462731035616, 0.2345967692624271, 24.51141485578401,
        0.6636722323716063, -0.7127005341235328, -0.2271499870986561, 177.9937467913584,
        0.3265344930619152, 0.002825885363233083, 0.9451810616028615, 197.2348458029795,
        0, 0, 0, 1);

    Hbm = Hbg * Hgc * Hmc.inv();
    //cout << Hbm << endl;
    Mat Rbm, Tbm;
    RT2R_T(Hbm, Rbm, Tbm);
    Mat_<double> T_com = (cv::Mat_<double>(3, 1) << 0, 0, 320);
    Tbm += T_com;
    cout << "Translation Vector" << endl;
    cout << Tbm << endl;
    waitKey(0);
    return EXIT_SUCCESS;
}
