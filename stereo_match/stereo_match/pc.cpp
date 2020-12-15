// pcl_test.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
#include<pcl/visualization/cloud_viewer.h>
#include<iostream>//标准C++库中的输入输出类相关头文件。
#include<pcl/io/io.h>
#include<pcl/io/pcd_io.h>//pcd 读写类相关的头文件。
#include<pcl/io/ply_io.h>
#include<pcl/point_types.h> //PCL中支持的点类型头文件。

int main()
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new 	pcl::PointCloud<pcl::PointXYZ>);
	char strfilepath[256] = "彩虹糖点云.pcd";
	if (-1 == pcl::io::loadPCDFile(strfilepath, *cloud)) //打开点云文件
	{
		std::cout << "error input!" << std::endl;
		return -1;
	}
	std::cout << cloud->points.size() << std::endl;
	pcl::visualization::CloudViewer viewer("Cloud Viewer");     //创建viewer对象
	viewer.showCloud(cloud);
	system("pause");
	return 0;
}