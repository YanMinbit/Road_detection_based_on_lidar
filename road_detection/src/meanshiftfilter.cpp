#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include<time.h>

using namespace cv;
using namespace std;
//颜色通道分离
static void MergeSeg(Mat& img
	, const Scalar& colorDiff = Scalar::all(1))
{
	CV_Assert(!img.empty());
	RNG rng = theRNG();
	// 定义掩码图像
	Mat mask(img.rows + 2, img.cols + 2, 
		CV_8UC1, Scalar::all(0));
	for (int y = 0; y < img.rows; y++)
	{
		for (int x = 0; x < img.cols; x++)
		{
			if (mask.at<uchar>(y + 1, x + 1) == 0)
			{
				// 颜色定义
				Scalar newVal(rng(256), rng(256), rng(256));
				// 泛洪合并
				floodFill(img, mask, Point(x, y)
					, newVal, 0, colorDiff, colorDiff);
			}
		}
	}
}
int main(int argc, char** argv)
{
	cv::Mat srcImg = imread("/home/minima/dev/catkin_ws/src/kitti_player-public/dataset/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000002.png");
	if (srcImg.empty())
		return -1;
	// 参数设定
	int spatialRad = 20;
	int colorRad = 20;
	int maxPyrLevel = 0;//这个值越小运行时间越长 一般都需要几秒
	cv::Mat resImg;
	// 均值漂移分割
	clock_t time_stt = clock();
	pyrMeanShiftFiltering(srcImg, resImg, 
		spatialRad, colorRad, maxPyrLevel);
    cv::imshow("filterImg", resImg);
    std::cout<<"image_process time used is "<<1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC<<"ms"<<std::endl;
	// 颜色通道分离合并
	MergeSeg(resImg, Scalar::all(2));
    std::cout<<"2 image_process time used is "<<1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC<<"ms"<<std::endl;
	cv::imshow("src", srcImg);
	cv::imshow("resImg", resImg);
	cv::waitKey();
	return 0;
}
