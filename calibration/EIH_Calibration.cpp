#define _CRT_SECURE_NO_DEPRECATE
#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/opencv.hpp>

#include <cctype>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <iostream>

using namespace cv;
using namespace std;

const char* usage =
" \nexample command line for calibration from a live feed.\n"
"   calibration  -w=4 -h=5 -s=0.025 -o=camera.yml -op -oe\n"
" \n"
" example command line for calibration from a list of stored images:\n"
"   imagelist_creator image_list.xml *.png\n"
"   calibration -w=4 -h=5 -s=0.025 -o=camera.yml -op -oe image_list.xml\n"
" where image_list.xml is the standard OpenCV XML/YAML\n"
" use imagelist_creator to create the xml or yaml list\n"
" file consisting of the list of strings, e.g.:\n"
" \n"
"<?xml version=\"1.0\"?>\n"
"<opencv_storage>\n"
"<images>\n"
"view000.png\n"
"view001.png\n"
"<!-- view002.png -->\n"
"view003.png\n"
"view010.png\n"
"one_extra_view.jpg\n"
"</images>\n"
"</opencv_storage>\n";




const char* liveCaptureHelp =
"When the live video from camera is used as input, the following hot-keys may be used:\n"
"  <ESC>, 'q' - quit the program\n"
"  'g' - start capturing images\n"
"  'u' - switch undistortion on/off\n";

static void help(char** argv)
{
    printf("This is a camera calibration sample.\n"
        "Usage: %s\n"
        "     -w=<board_width>         # the number of inner corners per one of board dimension\n"
        "     -h=<board_height>        # the number of inner corners per another board dimension\n"
        "     [-pt=<pattern>]          # the type of pattern: chessboard or circles' grid\n"
        "     [-n=<number_of_frames>]  # the number of frames to use for calibration\n"
        "                              # (if not specified, it will be set to the number\n"
        "                              #  of board views actually available)\n"
        "     [-d=<delay>]             # a minimum delay in ms between subsequent attempts to capture a next view\n"
        "                              # (used only for video capturing)\n"
        "     [-s=<squareSize>]       # square size in some user-defined units (1 by default)\n"
        "     [-o=<out_camera_params>] # the output filename for intrinsic [and extrinsic] parameters\n"
        "     [-op]                    # write detected feature points\n"
        "     [-oe]                    # write extrinsic parameters\n"
        "     [-oo]                    # write refined 3D object points\n"
        "     [-zt]                    # assume zero tangential distortion\n"
        "     [-a=<aspectRatio>]      # fix aspect ratio (fx/fy)\n"
        "     [-p]                     # fix the principal point at the center\n"
        "     [-v]                     # flip the captured images around the horizontal axis\n"
        "     [-V]                     # use a video file, and not an image list, uses\n"
        "                              # [input_data] string for the video file name\n"
        "     [-su]                    # show undistorted images after calibration\n"
        "     [-ws=<number_of_pixel>]  # Half of search window for cornerSubPix (11 by default)\n"
        "     [-dt=<distance>]         # actual distance between top-left and top-right corners of\n"
        "                              # the calibration grid. If this parameter is specified, a more\n"
        "                              # accurate calibration method will be used which may be better\n"
        "                              # with inaccurate, roughly planar target.\n"
        "     [input_data]             # input data, one of the following:\n"
        "                              #  - text file with a list of the images of the board\n"
        "                              #    the text file can be generated with imagelist_creator\n"
        "                              #  - name of video file with a video of the board\n"
        "                              # if input_data not specified, a live view from the camera is used\n"
        "\n", argv[0]);
    printf("\n%s", usage);
    printf("\n%s", liveCaptureHelp);
}

enum { DETECTION = 0, CAPTURING = 1, CALIBRATED = 2 };
enum Pattern { CHESSBOARD, CIRCLES_GRID, ASYMMETRIC_CIRCLES_GRID };

static double computeReprojectionErrors(
    const vector<vector<Point3f> >& objectPoints,
    const vector<vector<Point2f> >& imagePoints,
    const vector<Mat>& rvecs, const vector<Mat>& tvecs,
    const Mat& cameraMatrix, const Mat& distCoeffs,
    vector<float>& perViewErrors)
{
    vector<Point2f> imagePoints2;
    int i, totalPoints = 0;
    double totalErr = 0, err;
    perViewErrors.resize(objectPoints.size());

    for (i = 0; i < (int)objectPoints.size(); i++)
    {
        projectPoints(Mat(objectPoints[i]), rvecs[i], tvecs[i],
            cameraMatrix, distCoeffs, imagePoints2);
        err = norm(Mat(imagePoints[i]), Mat(imagePoints2), NORM_L2);
        int n = (int)objectPoints[i].size();
        perViewErrors[i] = (float)std::sqrt(err * err / n);
        totalErr += err * err;
        totalPoints += n;
    }

    return std::sqrt(totalErr / totalPoints);
}

static void calcChessboardCorners(Size boardSize, float squareSize, vector<Point3f>& corners, Pattern patternType = CHESSBOARD)
{
    corners.resize(0);

    switch (patternType)
    {
    case CHESSBOARD:
    case CIRCLES_GRID:
        for (int i = 0; i < boardSize.height; i++)
            for (int j = 0; j < boardSize.width; j++)
                corners.push_back(Point3f(float(j * squareSize),
                    float(i * squareSize), 0));
        break;

    case ASYMMETRIC_CIRCLES_GRID:
        for (int i = 0; i < boardSize.height; i++)
            for (int j = 0; j < boardSize.width; j++)
                corners.push_back(Point3f(float((2 * j + i % 2) * squareSize),
                    float(i * squareSize), 0));
        break;

    default:
        CV_Error(Error::StsBadArg, "Unknown pattern type\n");
    }
}

static bool runCalibration(vector<vector<Point2f> > imagePoints,
    Size imageSize, Size boardSize, Pattern patternType,
    float squareSize, float aspectRatio,
    float grid_width, bool release_object,
    int flags, Mat& cameraMatrix, Mat& distCoeffs,
    vector<Mat>& rvecs, vector<Mat>& tvecs,
    vector<float>& reprojErrs,
    vector<Point3f>& newObjPoints,
    double& totalAvgErr)
{
    cameraMatrix = Mat::eye(3, 3, CV_64F);
    if (flags & CALIB_FIX_ASPECT_RATIO)
        cameraMatrix.at<double>(0, 0) = aspectRatio;

    distCoeffs = Mat::zeros(8, 1, CV_64F);

    vector<vector<Point3f> > objectPoints(1);
    calcChessboardCorners(boardSize, squareSize, objectPoints[0], patternType);
    objectPoints[0][boardSize.width - 1].x = objectPoints[0][0].x + grid_width;
    newObjPoints = objectPoints[0];

    objectPoints.resize(imagePoints.size(), objectPoints[0]);

    double rms;
    int iFixedPoint = -1;
    if (release_object)
        iFixedPoint = boardSize.width - 1;
    rms = calibrateCameraRO(objectPoints, imagePoints, imageSize, iFixedPoint,
        cameraMatrix, distCoeffs, rvecs, tvecs, newObjPoints,
        flags | CALIB_FIX_K3 | CALIB_USE_LU);
    printf("RMS error reported by calibrateCamera: %g\n", rms);

    bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

    if (release_object) {
        cout << "New board corners: " << endl;
        cout << newObjPoints[0] << endl;
        cout << newObjPoints[boardSize.width - 1] << endl;
        cout << newObjPoints[boardSize.width * (boardSize.height - 1)] << endl;
        cout << newObjPoints.back() << endl;
    }

    objectPoints.clear();
    objectPoints.resize(imagePoints.size(), newObjPoints);
    totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints,
        rvecs, tvecs, cameraMatrix, distCoeffs, reprojErrs);

    return ok;
}


static void saveCameraParams(const string& filename,
    Size imageSize, Size boardSize,
    float squareSize, float aspectRatio, int flags,
    const Mat& cameraMatrix, const Mat& distCoeffs,
    const vector<Mat>& rvecs, const vector<Mat>& tvecs,
    const vector<float>& reprojErrs,
    const vector<vector<Point2f> >& imagePoints,
    const vector<Point3f>& newObjPoints,
    double totalAvgErr,
     Mat& bigmat1)
{
    FileStorage fs(filename, FileStorage::WRITE);

    time_t tt;
    time(&tt);
    struct tm* t2 = localtime(&tt);
    char buf[1024];
    strftime(buf, sizeof(buf) - 1, "%c", t2);

    fs << "calibration_time" << buf;

    if (!rvecs.empty() || !reprojErrs.empty())
        fs << "nframes" << (int)std::max(rvecs.size(), reprojErrs.size());
    fs << "image_width" << imageSize.width;
    fs << "image_height" << imageSize.height;
    fs << "board_width" << boardSize.width;
    fs << "board_height" << boardSize.height;
    fs << "square_size" << squareSize;

    if (flags & CALIB_FIX_ASPECT_RATIO)
        fs << "aspectRatio" << aspectRatio;

    if (flags != 0)
    {
        sprintf(buf, "flags: %s%s%s%s",
            flags & CALIB_USE_INTRINSIC_GUESS ? "+use_intrinsic_guess" : "",
            flags & CALIB_FIX_ASPECT_RATIO ? "+fix_aspectRatio" : "",
            flags & CALIB_FIX_PRINCIPAL_POINT ? "+fix_principal_point" : "",
            flags & CALIB_ZERO_TANGENT_DIST ? "+zero_tangent_dist" : "");
        //cvWriteComment( *fs, buf, 0 );
    }

    fs << "flags" << flags;

    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;

    fs << "avg_reprojection_error" << totalAvgErr;
    if (!reprojErrs.empty())
        fs << "per_view_reprojection_errors" << Mat(reprojErrs);

    if (!rvecs.empty() && !tvecs.empty())
    {
        CV_Assert(rvecs[0].type() == tvecs[0].type());
        Mat bigmat((int)rvecs.size(), 6, rvecs[0].type());
        for (int i = 0; i < (int)rvecs.size(); i++)
        {
            //Mat r = bigmat(Range(i, i + 1), Range(0, 3));
            //Mat t = bigmat(Range(i, i + 1), Range(3, 6));
            Mat t = bigmat(Range(i, i + 1), Range(0, 3));
            Mat r = bigmat(Range(i, i + 1), Range(3, 6));

            CV_Assert(rvecs[i].rows == 3 && rvecs[i].cols == 1);
            CV_Assert(tvecs[i].rows == 3 && tvecs[i].cols == 1);
            //*.t() is MatExpr (not Mat) so we can use assignment operator
            r = rvecs[i].t();
            t = tvecs[i].t();

        }
        //cvWriteComment( *fs, "a set of 6-tuples (rotation vector + translation vector) for each view", 0 );
        fs << "extrinsic_parameters" << bigmat;
        bigmat1 = bigmat;
    }

    if (!imagePoints.empty())
    {
        Mat imagePtMat((int)imagePoints.size(), (int)imagePoints[0].size(), CV_32FC2);
        for (int i = 0; i < (int)imagePoints.size(); i++)
        {
            Mat r = imagePtMat.row(i).reshape(2, imagePtMat.cols);
            Mat imgpti(imagePoints[i]);
            imgpti.copyTo(r);
        }
        fs << "image_points" << imagePtMat;
    }

    if (!newObjPoints.empty())
    {
        fs << "grid_points" << newObjPoints;
    }
}

static bool readStringList(const string& filename, vector<string>& l)
{
    l.resize(0);
    FileStorage fs(filename, FileStorage::READ);
    if (!fs.isOpened())
        return false;
    size_t dir_pos = filename.rfind('/');
    if (dir_pos == string::npos)
        dir_pos = filename.rfind('\\');
    FileNode n = fs.getFirstTopLevelNode();
    if (n.type() != FileNode::SEQ)
        return false;
    FileNodeIterator it = n.begin(), it_end = n.end();
    for (; it != it_end; ++it)
    {
        string fname = (string)*it;
        if (dir_pos != string::npos)
        {
            string fpath = samples::findFile(filename.substr(0, dir_pos + 1) + fname, false);
            if (fpath.empty())
            {
                fpath = samples::findFile(fname);
            }
            fname = fpath;
        }
        else
        {
            fname = samples::findFile(fname);
        }
        l.push_back(fname);
    }
    return true;
}


static bool runAndSave(const string& outputFilename,
    const vector<vector<Point2f> >& imagePoints,
    Size imageSize, Size boardSize, Pattern patternType, float squareSize,
    float grid_width, bool release_object,
    float aspectRatio, int flags, Mat& cameraMatrix,
    Mat& distCoeffs, bool writeExtrinsics, bool writePoints, bool writeGrid, Mat& bigmat)
{
    vector<Mat> rvecs, tvecs;
    vector<float> reprojErrs;
    double totalAvgErr = 0;
    vector<Point3f> newObjPoints;
    bool ok = runCalibration(imagePoints, imageSize, boardSize, patternType, squareSize,
        aspectRatio, grid_width, release_object, flags, cameraMatrix, distCoeffs,
        rvecs, tvecs, reprojErrs, newObjPoints, totalAvgErr);
    printf("%s. avg reprojection error = %.7f\n",
        ok ? "Calibration succeeded" : "Calibration failed",
        totalAvgErr);

    if (ok)
        saveCameraParams(outputFilename, imageSize,
            boardSize, squareSize, aspectRatio,
            flags, cameraMatrix, distCoeffs,
            writeExtrinsics ? rvecs : vector<Mat>(),
            writeExtrinsics ? tvecs : vector<Mat>(),
            writeExtrinsics ? reprojErrs : vector<float>(),
            writePoints ? imagePoints : vector<vector<Point2f> >(),
            writeGrid ? newObjPoints : vector<Point3f>(),
            totalAvgErr,
            bigmat);
    return ok;
}

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


//机械臂末端位姿,x,y,z,rx,ry,rz
Mat_<double> ToolPose = (cv::Mat_<double>(14, 6) <<
    447, -115.57, 428, 3.02, 1.29, -0.24,
    459.7, 23.49, 424.62, 2.6, 1.51, -0.431,
    593.1, 8.67, 398.24, 2.824, 1.245, -0.782,
    597.94, -131.82, 373.1, 3.2255, 0.8633, -0.6992,
    445.81, -294.27, 384.55, 3.0189, -2.0398, -0.5623,
    316.52, -214.95, 457.98, 2.8558, -2.3499, 0.0086,
    -170.14, -147.92, 409.21, 2.9162, 1.15, 0.5607,
    170.33, -58.04, 455.66, 2.8791, 0.9914, 0.3427,
    198.17, -15.27, 410.8, 2.9496, 0.6984, 0.4194,
    197.55, -30.28, 431.95, 3.1987, -0.4058, 0.6042,
    190.75, -36.49, 462.62, 3.1394, -0.8955, 0.6264,
    314.1, -83.88, 463.21, 3.1618, 0.4633, 0.218,
    315.26, -62.68, 484.87, 3.2642, -0.2393, 0.2022,
    311.43, -85.4, 480.2, 2.0044, 2.1832, -0.0565);




int main(int argc, char** argv)
{
    

    Size boardSize, imageSize;
    float squareSize, aspectRatio = 1;
    Mat cameraMatrix, distCoeffs;
    Mat bigmat;
    string outputFilename = "out_camera_data.yml";
    string inputFilename = "";

    int i, nframes;
    bool writeExtrinsics, writePoints;
    bool undistortImage = false;
    int flags = 0;
    VideoCapture capture;
    bool flipVertical;
    bool showUndistorted;
    bool videofile;
    int delay;
    clock_t prevTimestamp = 0;
    int mode = DETECTION;
    int cameraId = 0;
    vector<vector<Point2f> > imagePoints;
    vector<string> imageList;
    Pattern pattern = CHESSBOARD;

    cv::CommandLineParser parser(argc, argv,
        "{help ||}{w|11|}{h|8|}{pt|chessboard|}{n|10|}{d|1000|}{s|10|}{o|out_camera_data.yml|}"
        "{op||}{oe|1|}{zt||}{a||}{p||}{v||}{V||}{su||}"
        "{oo||}{ws|11|}{dt||}"
        "{@input_data|imagelist2.yaml|}");
    if (parser.has("help"))
    {
        help(argv);
        return 0;
    }
    boardSize.width = parser.get<int>("w");
    boardSize.height = parser.get<int>("h");

    if (parser.has("pt"))
    {
        string val = parser.get<string>("pt");
        if (val == "circles")
            pattern = CIRCLES_GRID;
        else if (val == "acircles")
            pattern = ASYMMETRIC_CIRCLES_GRID;
        else if (val == "chessboard")
            pattern = CHESSBOARD;
        else
            return fprintf(stderr, "Invalid pattern type: must be chessboard or circles\n"), -1;
    }
    squareSize = parser.get<float>("s");
    nframes = parser.get<int>("n");
    delay = parser.get<int>("d");
    writePoints = parser.has("op");
    writeExtrinsics = parser.has("oe");
    bool writeGrid = parser.has("oo");
    

    if (parser.has("a")) {
        flags |= CALIB_FIX_ASPECT_RATIO;
        aspectRatio = parser.get<float>("a");
    }
    if (parser.has("zt"))
        flags |= CALIB_ZERO_TANGENT_DIST;
    if (parser.has("p"))
        flags |= CALIB_FIX_PRINCIPAL_POINT;
    flipVertical = parser.has("v");
    videofile = parser.has("V");
    if (parser.has("o"))
        outputFilename = parser.get<string>("o");
    showUndistorted = parser.has("su");
    if (isdigit(parser.get<string>("@input_data")[0]))
        cameraId = parser.get<int>("@input_data");
    else
        inputFilename = parser.get<string>("@input_data");
    int winSize = parser.get<int>("ws");
    float grid_width = squareSize * (boardSize.width - 1);
    bool release_object = false;
    if (parser.has("dt")) {
        grid_width = parser.get<float>("dt");
        release_object = true;
    }
    if (!parser.check())
    {
        help(argv);
        parser.printErrors();
        return -1;
    }
    if (squareSize <= 0)
        return fprintf(stderr, "Invalid board square width\n"), -1;
    if (nframes <= 3)
        return printf("Invalid number of images\n"), -1;
    if (aspectRatio <= 0)
        return printf("Invalid aspect ratio\n"), -1;
    if (delay <= 0)
        return printf("Invalid delay\n"), -1;
    if (boardSize.width <= 0)
        return fprintf(stderr, "Invalid board width\n"), -1;
    if (boardSize.height <= 0)
        return fprintf(stderr, "Invalid board height\n"), -1;

    if (!inputFilename.empty())
    {
        if (!videofile && readStringList(samples::findFile(inputFilename), imageList))
            mode = CAPTURING;
        else
            capture.open(samples::findFileOrKeep(inputFilename));
    }
    else
        capture.open(cameraId);

    if (!capture.isOpened() && imageList.empty())
        return fprintf(stderr, "Could not initialize video (%d) capture\n", cameraId), -2;

    if (!imageList.empty())
        nframes = (int)imageList.size();

    if (capture.isOpened())
        printf("%s", liveCaptureHelp);

    namedWindow("Image View", 0);
    resizeWindow("Image View", 612, 512);

    for (i = 0;; i++)
    {
        Mat view, viewGray;
        bool blink = false;

        if (capture.isOpened())
        {
            Mat view0;
            capture >> view0;
            view0.copyTo(view);
        }
        else if (i < (int)imageList.size())
            view = imread(imageList[i], 1);
        //resize(view, view, Size(612, 512));

        if (view.empty())
        {
            if (imagePoints.size() > 0)
                runAndSave(outputFilename, imagePoints, imageSize,
                    boardSize, pattern, squareSize, grid_width, release_object, aspectRatio,
                    flags, cameraMatrix, distCoeffs,
                    writeExtrinsics, writePoints, writeGrid, bigmat);
            break;
        }

        imageSize = view.size();

        if (flipVertical)
            flip(view, view, 0);

        vector<Point2f> pointbuf;
        cvtColor(view, viewGray, COLOR_BGR2GRAY);

        bool found;
        switch (pattern)
        {
        case CHESSBOARD:
            found = findChessboardCorners(view, boardSize, pointbuf,
                CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
            break;
        case CIRCLES_GRID:
            found = findCirclesGrid(view, boardSize, pointbuf);
            break;
        case ASYMMETRIC_CIRCLES_GRID:
            found = findCirclesGrid(view, boardSize, pointbuf, CALIB_CB_ASYMMETRIC_GRID);
            break;
        default:
            return fprintf(stderr, "Unknown pattern type\n"), -1;
        }

        // improve the found corners' coordinate accuracy
        if (pattern == CHESSBOARD && found) cornerSubPix(viewGray, pointbuf, Size(winSize, winSize),
            Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.0001));

        if (mode == CAPTURING && found &&
            (!capture.isOpened() || clock() - prevTimestamp > delay * 1e-3 * CLOCKS_PER_SEC))
        {
            imagePoints.push_back(pointbuf);
            prevTimestamp = clock();
            blink = capture.isOpened();
        }

        if (found)
            drawChessboardCorners(view, boardSize, Mat(pointbuf), found);

        string msg = mode == CAPTURING ? "100/100" :
            mode == CALIBRATED ? "Calibrated" : "Press 'g' to start";
        int baseLine = 0;
        Size textSize = getTextSize(msg, 1, 1, 1, &baseLine);
        /*Point textOrigin(view.cols - 2 * textSize.width - 10, view.rows - 2 * baseLine - 10);*/
        Point textOrigin(view.cols - 8 * textSize.width - 120, view.rows - 8 * baseLine - 100);

        if (mode == CAPTURING)
        {
            if (undistortImage)
                msg = format("%d/%d Undist", (int)imagePoints.size(), nframes);
            else
                msg = format("%d/%d", (int)imagePoints.size(), nframes);
        }

        /*putText(view, msg, textOrigin, 1, 1,
            mode != CALIBRATED ? Scalar(0, 0, 255) : Scalar(0, 255, 0));*/
        putText(view, msg, textOrigin, 4, 4,
            mode != CALIBRATED ? Scalar(0, 0, 255) : Scalar(0, 255, 0), 4);

        if (blink)
            bitwise_not(view, view);

        if (mode == CALIBRATED && undistortImage)
        {
            Mat temp = view.clone();
            undistort(temp, view, cameraMatrix, distCoeffs);
        }

        imshow("Image View", view);
        char key = (char)waitKey(capture.isOpened() ? 50 : 500);

        if (key == 27)
            break;

        if (key == 'u' && mode == CALIBRATED)
            undistortImage = !undistortImage;

        if (capture.isOpened() && key == 'g')
        {
            mode = CAPTURING;
            imagePoints.clear();
        }

        if (mode == CAPTURING && imagePoints.size() >= (unsigned)nframes)
        {
            if (runAndSave(outputFilename, imagePoints, imageSize,
                boardSize, pattern, squareSize, grid_width, release_object, aspectRatio,
                flags, cameraMatrix, distCoeffs,
                writeExtrinsics, writePoints, writeGrid, bigmat))
                mode = CALIBRATED;
            else
                mode = DETECTION;
            if (!capture.isOpened())
                break;
        }
    }

    if (!capture.isOpened() && showUndistorted)
    {
        Mat view, rview, map1, map2;
        initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
            getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
            imageSize, CV_16SC2, map1, map2);

        for (i = 0; i < (int)imageList.size(); i++)
        {
            view = imread(imageList[i], 1);
            if (view.empty())
                continue;
            //undistort( view, rview, cameraMatrix, distCoeffs, cameraMatrix );
            remap(view, rview, map1, map2, INTER_LINEAR);
            imshow("Image View", rview);
            char c = (char)waitKey();
            if (c == 27 || c == 'q' || c == 'Q')
                break;
        }
    }

    //相机中30组标定板的位姿，x,y,z，rx,ry,rz,
    Mat_<double> CalPose = bigmat;

    //定义手眼标定矩阵
    std::vector<Mat> R_gripper2base;
    std::vector<Mat> t_gripper2base;
    std::vector<Mat> R_target2cam;
    std::vector<Mat> t_target2cam;
    Mat R_cam2gripper = (Mat_<double>(3, 3));
    Mat t_cam2gripper = (Mat_<double>(3, 1));

    vector<Mat> images;
    size_t num_images = 14;

    // 读取末端，标定板的姿态矩阵 4*4
    std::vector<cv::Mat> vecHg, vecHc;
    cv::Mat Hcg;//定义相机camera到末端grab的位姿矩阵
    Mat tempR, tempT;

    for (size_t i = 0; i < num_images; i++)//计算标定板位姿
    {
        cv::Mat tmp = attitudeVectorToMatrix(CalPose.row(i), false, ""); 
        vecHc.push_back(tmp);
        RT2R_T(tmp, tempR, tempT);
        R_target2cam.push_back(tempR);
        t_target2cam.push_back(tempT);
    }

    for (size_t i = 0; i < num_images; i++)//计算机械臂位姿
    {
        cv::Mat tmp = attitudeVectorToMatrix(ToolPose.row(i), false, ""); 
        vecHg.push_back(tmp);
        RT2R_T(tmp, tempR, tempT);
        R_gripper2base.push_back(tempR);
        t_gripper2base.push_back(tempT);
    }

    calibrateHandEye(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, R_cam2gripper, t_cam2gripper, CALIB_HAND_EYE_TSAI);

    Hcg = R_T2RT(R_cam2gripper, t_cam2gripper);//矩阵合并

    std::cout << "Hcg 矩阵为： " << std::endl;
    std::cout << Hcg << std::endl;
    cout << "是否为旋转矩阵：" << isRotationMatrix(Hcg) << std::endl << std::endl;//判断是否为旋转矩阵

    //Tool_In_Base*Hcg*Cal_In_Cam，用第一组和第二组进行对比验证
    cout << "第一组和第二组手眼数据验证：" << endl;
    cout << vecHg[0] * Hcg * vecHc[0] << endl << vecHg[1] * Hcg * vecHc[1] << endl << endl;//.inv()

    cout << "第三组和第四组手眼数据验证：" << endl;
    cout << vecHg[2] * Hcg * vecHc[2] << endl << vecHg[3] * Hcg * vecHc[3] << endl << endl;//.inv()


    cout << "标定板在相机中的位姿：" << endl;
    cout << vecHc[1] << endl;
    cout << "手眼系统反演的位姿为：" << endl;
    //用手眼系统预测第一组数据中标定板相对相机的位姿，是否与vecHc[1]相同
    cout << Hcg.inv() * vecHg[1].inv() * vecHg[0] * Hcg * vecHc[0] << endl << endl;

    cout << "----手眼系统测试----" << endl;
    cout << "机械臂下标定板XYZ为：" << endl;
    //FileStorage fs("CIH.yaml", FileStorage::WRITE);
    for (int i = 0; i < vecHc.size(); ++i)
    {
        cv::Mat cheesePos{ 0.0,0.0,0.0,1.0 };//4*1矩阵，单独求机械臂下，标定板的xyz
        cv::Mat worldPos = vecHg[i] * Hcg * vecHc[i] * cheesePos;
        cout << i+1 << ": " << worldPos.t() << endl;
        //fs << "No" << i;
        //fs << "Point" << worldPos.t();
    }
    getchar();
    
    return 0;
}