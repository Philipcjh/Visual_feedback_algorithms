// pcl_test.cpp : ���ļ����� "main" ����������ִ�н��ڴ˴���ʼ��������
#include<pcl/visualization/cloud_viewer.h>
#include<iostream>//��׼C++���е�������������ͷ�ļ���
#include<pcl/io/io.h>
#include<pcl/io/pcd_io.h>//pcd ��д����ص�ͷ�ļ���
#include<pcl/io/ply_io.h>
#include<pcl/point_types.h> //PCL��֧�ֵĵ�����ͷ�ļ���

int main()
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new 	pcl::PointCloud<pcl::PointXYZ>);
	char strfilepath[256] = "�ʺ��ǵ���.pcd";
	if (-1 == pcl::io::loadPCDFile(strfilepath, *cloud)) //�򿪵����ļ�
	{
		std::cout << "error input!" << std::endl;
		return -1;
	}
	std::cout << cloud->points.size() << std::endl;
	pcl::visualization::CloudViewer viewer("Cloud Viewer");     //����viewer����
	viewer.showCloud(cloud);
	system("pause");
	return 0;
}