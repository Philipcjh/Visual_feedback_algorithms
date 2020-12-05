/*this creates a yaml or xml list of files from the command line args
 */

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <string>
#include <iostream>

using std::string;
using std::cout;
using std::endl;

using namespace cv;

static void help(char** av)
{
    cout << "\nThis creates a yaml or xml list of files from the command line args\n"
        "usage:\n./" << av[0] << " imagelist.yaml *.png\n"
        << "Try using different extensions.(e.g. yaml yml xml xml.gz etc...)\n"
        << "This will serialize this list of images or whatever with opencv's FileStorage framework" << endl;
}

int main(int ac, char** av)
{   
    ac = 42;
    av[0] = "Calibration.exe";
    av[1] = "imagelist.yaml";
    av[2] = "1-1.bmp";
    av[3] = "1-2.bmp";
    av[4] = "2-1.bmp";
    av[5] = "2-2.bmp";
    av[6] = "3-1.bmp";
    av[7] = "3-2.bmp";
    av[8] = "4-1.bmp";
    av[9] = "4-2.bmp";
    av[10] = "5-1.bmp";
    av[11] = "5-2.bmp";
    av[12] = "6-1.bmp";
    av[13] = "6-2.bmp";
    av[14] = "7-1.bmp"; 
    av[15] = "7-2.bmp"; 
    av[16] = "8-1.bmpp"; 
    av[17] = "8-2.bmp"; 
    av[18] = "9-1.bmp"; 
    av[19] = "9-2.bmp"; 
    av[20] = "10-1.bmp"; 
    av[21] = "10-2.bmp";
    av[22] = "11-1.bmp";
    av[23] = "11-2.bmp";
    av[24] = "12-1.bmp";
    av[25] = "12-2.bmp";
    av[26] = "13-1.bmp";
    av[27] = "13-2.bmp";
    av[28] = "14-1.bmp";
    av[29] = "14-2.bmp";
    av[30] = "15-1.bmp";
    av[31] = "15-2.bmp";
    av[32] = "16-1.bmp";
    av[33] = "16-2.bmp";
    av[34] = "17-1.bmp";
    av[35] = "17-2.bmp";
    av[36] = "18-1.bmp";
    av[37] = "18-2.bmp";
    av[38] = "19-1.bmp";
    av[39] = "19-2.bmp";
    av[40] = "20-1.bmp";
    av[41] = "20-2.bmp";
    //av[42] = "21-1.bmp";
    //av[43] = "21-2.bmp";
    //av[44] = "22-1.bmp";
    //av[45] = "22-2.bmp";
    //av[46] = "23-1.bmpp";
    //av[47] = "23-2.bmp";
    //av[48] = "24-1.bmp";
    //av[49] = "24-2.bmp";
    //av[50] = "25-1.bmp";
    //av[51] = "25-2.bmp";
    //av[52] = "26-1.bmp";
    //av[53] = "26-2.bmp";
    //av[54] = "27-1.bmp";
    //av[55] = "27-2.bmp";
    //av[56] = "28-1.bmp";
    //av[57] = "28-2.bmp";
    //av[58] = "29-1.bmp";
    //av[59] = "29-2.bmp";
    //av[60] = "30-1.bmp";
    //av[61] = "30-2.bmp";


    cv::CommandLineParser parser(ac, av, "{help h||}{@output||}");
    if (parser.has("help"))
    {
        help(av);
        return 0;
    }
    string outputname = parser.get<string>("@output");

    if (outputname.empty())
    {
        help(av);
        return 1;
    }

    Mat m = imread(outputname); //check if the output is an image - prevent overwrites!
    if (!m.empty()) {
        std::cerr << "fail! Please specify an output file, don't want to overwrite you images!" << endl;
        help(av);
        return 1;
    }

    FileStorage fs(outputname, FileStorage::WRITE);
    fs << "images" << "[";
    for (int i = 2; i < ac; i++) {
        fs << string(av[i]);
    }
    fs << "]";
    return 0;
}
