#include <iostream>
#include<opencv2/opencv.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "pointcloud_generate/HumanoidRobot.h"
#include "icp/icp_match.h"

using namespace cv;
using namespace cv::xfeatures2d;

const int LOOP_NUM = 1;
const int GOOD_PTS_MAX = 20;
const float GOOD_PORTION = 0.5f;

int64 work_begin = 0;
int64 work_end = 0;


//R和T转RT矩阵
Mat R_T2RT(Mat& R, Mat& T)
{
    Mat RT;
    Mat_<double> R1 = (cv::Mat_<double>(4, 3) << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
        R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
        R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2),
        0.0, 0.0, 0.0);
    cv::Mat_<double> T1 = (cv::Mat_<double>(4, 1) << T.at<double>(0, 0), T.at<double>(1, 0), T.at<double>(2, 0), 1.0);

    cv::hconcat(R1, T1, RT);//C=A+B左右拼接
    return RT;
}

//RT转R和T矩阵
void RT2R_T(Mat& RT, Mat& R, Mat& T)
{
    cv::Rect R_rect(0, 0, 3, 3);
    cv::Rect T_rect(3, 0, 1, 3);
    R = RT(R_rect);
    T = RT(T_rect);
}

//判断是否为旋转矩阵
bool isRotationMatrix(const cv::Mat& R)
{
    cv::Mat tmp33 = R({ 0,0,3,3 });
    cv::Mat shouldBeIdentity;

    shouldBeIdentity = tmp33.t() * tmp33;

    cv::Mat I = cv::Mat::eye(3, 3, shouldBeIdentity.type());

    return  cv::norm(I, shouldBeIdentity) < 1e-6;
}

/** @brief 欧拉角 -> 3*3 的R
*	@param 	eulerAngle		角度值
*	@param 	seq				指定欧拉角xyz的排列顺序如："xyz" "zyx"
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

/** @brief 四元数转旋转矩阵
*	@note  数据类型double； 四元数定义 q = w + x*i + y*j + z*k
*	@param q 四元数输入{w,x,y,z}向量
*	@return 返回旋转矩阵3*3
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

/** @brief ((四元数||欧拉角||旋转向量) && 转移向量) -> 4*4 的Rt
*	@param 	m				1*6 || 1*10的矩阵  -> 6  {x,y,z, rx,ry,rz}   10 {x,y,z, qw,qx,qy,qz, rx,ry,rz}
*	@param 	useQuaternion	如果是1*10的矩阵，判断是否使用四元数计算旋转矩阵
*	@param 	seq				如果通过欧拉角计算旋转矩阵，需要指定欧拉角xyz的排列顺序如："xyz" "zyx" 为空表示旋转向量
*/
cv::Mat attitudeVectorToMatrix(cv::Mat& m, bool useQuaternion, const std::string& seq)
{
    CV_Assert(m.total() == 6 || m.total() == 10);
    if (m.cols == 1)
        m = m.t();
    cv::Mat tmp = cv::Mat::eye(4, 4, CV_64FC1);

    //如果使用四元数转换成旋转矩阵则读取m矩阵的第第四个成员，读4个数据
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

        //如果seq为空表示传入的是旋转向量，否则"xyz"的组合表示欧拉角
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

static Mat drawGoodMatches(
    const Mat& img1,
    const Mat& img2,
    const std::vector<KeyPoint>& keypoints1,
    const std::vector<KeyPoint>& keypoints2,
    std::vector<DMatch>& matches,
    std::vector<Point2f>& scene_corners_
)
{
    //-- Sort matches and preserve top 10% matches
    std::sort(matches.begin(), matches.end());
    std::vector< DMatch > good_matches;
    double minDist = matches.front().distance;
    double maxDist = matches.back().distance;

    const int ptsPairs = std::min(GOOD_PTS_MAX, (int)(matches.size() * GOOD_PORTION));
    for (int i = 0; i < ptsPairs; i++)
    {
        good_matches.push_back(matches[i]);
    }
    std::cout << "\nMax distance: " << maxDist << std::endl;
    std::cout << "Min distance: " << minDist << std::endl;

    std::cout << "Calculating homography using " << ptsPairs << " point pairs." << std::endl;

    // drawing the results
    Mat img_matches;

    drawMatches(img1, keypoints1, img2, keypoints2,
        good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
        std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    //-- Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;

    for (size_t i = 0; i < good_matches.size(); i++)
    {
        //-- Get the keypoints from the good matches
        obj.push_back(keypoints1[good_matches[i].queryIdx].pt);
        scene.push_back(keypoints2[good_matches[i].trainIdx].pt);
    }
    //-- Get the corners from the image_1 ( the object to be "detected" )
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = Point(0, 0);
    obj_corners[1] = Point(img1.cols, 0);
    obj_corners[2] = Point(img1.cols, img1.rows);
    obj_corners[3] = Point(0, img1.rows);
    std::vector<Point2f> scene_corners(4);

    Mat H = findHomography(obj, scene, RANSAC);
    perspectiveTransform(obj_corners, scene_corners, H);

    scene_corners_ = scene_corners;

    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
    line(img_matches,
        scene_corners[0] + Point2f((float)img1.cols, 0), scene_corners[1] + Point2f((float)img1.cols, 0),
        Scalar(0, 255, 0), 2, LINE_AA);
    line(img_matches,
        scene_corners[1] + Point2f((float)img1.cols, 0), scene_corners[2] + Point2f((float)img1.cols, 0),
        Scalar(0, 255, 0), 2, LINE_AA);
    line(img_matches,
        scene_corners[2] + Point2f((float)img1.cols, 0), scene_corners[3] + Point2f((float)img1.cols, 0),
        Scalar(0, 255, 0), 2, LINE_AA);
    line(img_matches,
        scene_corners[3] + Point2f((float)img1.cols, 0), scene_corners[0] + Point2f((float)img1.cols, 0),
        Scalar(0, 255, 0), 2, LINE_AA);
    return img_matches;
}

////////////////////////////////////////////////////
// This program demonstrates the usage of SURF_OCL.
// use cpu findHomography interface to calculate the transformation matrix
int main(int argc, char* argv[])
{
    const char* keys =
        "{ h help     |                  | print help message  }"
        "{ l left     | 3-1.bmp          | specify left image  }"
        "{ r right    | 3-2.bmp | specify right image }"
        "{ o output   | SURF_output.jpg  | specify output save path }"
        "{ m cpu_mode |                  | run without OpenCL }";

    //std::string plyname = "彩虹糖点云.ply";
    //std::string plyname = "杯子.ply";

    CommandLineParser cmd(argc, argv, keys);

    UMat img1, img2;

    std::string outpath = cmd.get<std::string>("o");

    std::string leftName = cmd.get<std::string>("l");
    imread(leftName, IMREAD_GRAYSCALE).copyTo(img1);
    if (img1.empty())
    {
        std::cout << "Couldn't load " << leftName << std::endl;
        cmd.printMessage();
        return EXIT_FAILURE;
    }

    std::string rightName = cmd.get<std::string>("r");
    imread(rightName, IMREAD_GRAYSCALE).copyTo(img2);
    if (img2.empty())
    {
        std::cout << "Couldn't load " << rightName << std::endl;
        cmd.printMessage();
        return EXIT_FAILURE;
    }

    double surf_time = 0.;

    //declare input/output
    std::vector<KeyPoint> keypoints1, keypoints2;
    std::vector<DMatch> matches;

    UMat _descriptors1, _descriptors2;
    Mat descriptors1 = _descriptors1.getMat(ACCESS_RW),
        descriptors2 = _descriptors2.getMat(ACCESS_RW);

    //instantiate detectors/matchers
    SURFDetector surf;

    SURFMatcher<BFMatcher> matcher;

    //-- start of timing section

    for (int i = 0; i <= LOOP_NUM; i++)
    {
        if (i == 1) workBegin();
        surf(img1.getMat(ACCESS_READ), Mat(), keypoints1, descriptors1);
        surf(img2.getMat(ACCESS_READ), Mat(), keypoints2, descriptors2);
        matcher.match(descriptors1, descriptors2, matches);
    }

    workEnd();
    std::cout << "FOUND " << keypoints1.size() << " keypoints on first image" << std::endl;
    std::cout << "FOUND " << keypoints2.size() << " keypoints on second image" << std::endl;

    surf_time = getTime();
    std::cout << "SURF run time: " << surf_time / LOOP_NUM << " ms" << std::endl << "\n";


    std::vector<Point2f> corner;
    Mat img_matches = drawGoodMatches(img1.getMat(ACCESS_READ), img2.getMat(ACCESS_READ), keypoints1, keypoints2, matches, corner);


    int match_ele = 20;
    std::nth_element(matches.begin(),
        matches.begin() + match_ele - 1,
        matches.end());

    matches.erase(matches.begin() + match_ele, matches.end());

    std::vector<Point2f> pnts2d1, pnts2d2;
    std::vector<Point3f> pnts3d;

    for (auto& i : matches) {
        pnts2d1.push_back(keypoints1[i.queryIdx].pt);
        pnts2d2.push_back(keypoints2[i.trainIdx].pt);
    }


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
    //-- Show detected matches
    //cout << pnts3d << endl;
    namedWindow("surf matches", 0);
    resizeWindow("surf matches", Size(612 * 2, 512));
    imshow("surf matches", img_matches);
    imwrite(outpath, img_matches);

    float px=0, py=0, pz=0;
    for (int i = 0; i < pnts3d.size()-1; i++)
    {
        px += pnts3d[i].x;
        py += pnts3d[i].y;
        pz += pnts3d[i].z;
    }

    px = px / pnts3d.size();
    py = py / pnts3d.size();
    pz = pz / pnts3d.size()-36;

    Mat_<double> PT1 = (cv::Mat_<double>(4, 1) <<
        px,
        py,
        pz,
        1);
    //int iterations = 10;//迭代次数

    //cv::Mat_<double> Hmc(4, 4);
    //icp_match::icp_estimation(plyname, iterations, pnts3d, Hmc);

    Mat Hbg, Hbm;
    Mat_<double> InitialPose = (cv::Mat_<double>(1, 6) << 534.89, -73.82, 463.45, 2.9302, 1.2036, -0.1236);
    Hbg = attitudeVectorToMatrix(InitialPose, false, "");

    //cv::Mat Hgc = (cv::Mat_<double>(4, 4) <<
    //    -0.6729891476240062, -0.701462731035616, 0.2345967692624271, 24.51141485578401,
    //    0.6636722323716063, -0.7127005341235328, -0.2271499870986561, 177.9937467913584,
    //    0.3265344930619152, 0.002825885363233083, 0.9451810616028615, 197.2348458029795,
    //    0, 0, 0, 1);

    cv::Mat Hgc = (cv::Mat_<double>(4, 4) <<
        -0.6777047146178363, -0.692194773990834, 0.2481586481356945, -2.247135056563309,
        0.6496175744645655, -0.7217103228125286, -0.2390213732966373, 151.3791275849701,
        0.3445480035227774, -0.0007776924932453855, 0.9387683784953849, 124.2921276699383,
        0, 0, 0, 1);

    Hbm = Hbg * Hgc * PT1;
    Mat Tbm = Hbm;
    //cout << Hbm << endl;
    //Mat Rbm, Tbm;
    //RT2R_T(Hbm, Rbm, Tbm);
    //Mat_<double> T_com = (cv::Mat_<double>(3, 1) <<-45, 0, 320);//固定补偿（软体手长度）
    Mat_<double> T_com = (cv::Mat_<double>(4, 1) << 0, 0, 245,0);//固定补偿（软体手长度）
    Tbm = Tbm+T_com;
    cout << "Translation Vector" << endl;
    cout << Tbm << endl;
    waitKey(0);
    return EXIT_SUCCESS;
}
