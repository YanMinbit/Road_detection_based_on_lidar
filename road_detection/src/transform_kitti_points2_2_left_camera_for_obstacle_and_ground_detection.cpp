/*
 * transform_kitti_points2_2_left_camera_for_obstacle_and_ground_detection.cpp
 *
 *  Created on: Apr 9, 2018
 *      Author: minima
 */



#include "ros/ros.h"
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_listener.h>
#include <tf/tf.h>
#include <pcl-1.7/pcl/filters/voxel_grid.h>
#include <pcl-1.7/pcl/point_cloud.h>
#include <pcl-1.7/pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <opencv2/core/core.hpp>
#include <Eigen/Core>//注意需要先安装这个库
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <pcl-1.7/pcl/filters/radius_outlier_removal.h>
#include <math.h>


#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

using namespace cv;
using namespace std;

class TransformPointsFrame
{
public:

	    ros::NodeHandle nh_; //定义ROS句柄
	    image_transport::ImageTransport it_; //定义一个image_transport实例
	    image_transport::Subscriber image_sub_; //订阅图像

	    image_transport::Publisher image_pub_; //发布图像检测最终的图像
	    Mat subscribed_image;
	    //Mat subscribed_image_h;//订阅到的图片的H通道
	    bool subscribed_image_new = false;
	    bool subscribed_points_new = false;
	    pcl::PointCloud<pcl::PointXYZI>::Ptr received_points = NULL;//成员变量定义不能直接用new初始化
	    double timepoints = 0;//记录点云的时间戳
	    double timeimg = 0;//记录图像的时间戳
	    std_msgs::Header points_header;


	std::string target_frame_link;//发布的点云信息的frameID
	std::string source_frame_link;//接收的点云信息的frameID
	double p_points_downsample_size_;

	ros::Publisher new_Points_publish;
	ros::Publisher delaunay_ground_Points_publish;
	ros::Publisher delaunay_obstacle_Points_publish;
	ros::Subscriber points_sub;
	tf::TransformListener tf_;
	tf::StampedTransform transform;

	Eigen::Matrix<double, 3,3> matrix_33_left_K;
	Mat left_K,left_D;//摄像头的内参
	Mat P2_velodyne_2_left_camera_2_image_plane;//将转到矫正摄像头参考系的点通过这个投影矩阵投到图像平面
	Eigen::Matrix<double, 3,3> matrix_33_P2;
	Eigen::Vector3d vector_T_for_P2;

	std::string sensor_param_filename;

	double thresh_theta;
	double cos_thresh_theta;
    Size imageSize;

    double obstacle_thresh_z;



	static void drawSubdiv(Mat& img, Subdiv2D& subdiv, Scalar delaunay_color)
	{//直接得到剖分三角形的顶点
	    vector<Vec6f> triangleList;
	    subdiv.getTriangleList(triangleList);
	    vector<Point> pt(3);

	    for (size_t i = 0; i < triangleList.size(); i++)//这个顺序好像没什么规律
	    {
	        Vec6f t = triangleList[i];
	        pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
	        pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
	        pt[2] = Point(cvRound(t[4]), cvRound(t[5]));
	        line(img, pt[0], pt[1], delaunay_color, 1, CV_AA, 0);
	        line(img, pt[1], pt[2], delaunay_color, 1, CV_AA, 0);
	        line(img, pt[2], pt[0], delaunay_color, 1, CV_AA, 0);
	    }
	}


	//画出连接同一个顶点所有边，这些边以给定边的起点为出发点
	static void draw_from_same_Org_point(Mat& img, int edge, Subdiv2D& subdiv, Scalar delaunay_color)
	{
		Point2f outer_vtx_from_edge;
		subdiv.edgeOrg(edge, &outer_vtx_from_edge);
		circle(img, outer_vtx_from_edge, 3, Scalar(), CV_FILLED, CV_AA, 0);//画出线的起始点
		int temp = edge;
		vector<Point2f> pt(2);
		do
		{
			subdiv.edgeOrg(temp, &pt[0]);//画给定线
			subdiv.edgeDst(temp, &pt[1]);
			line(img, pt[0], pt[1], delaunay_color, 1, CV_AA, 0);
			temp = subdiv.nextEdge(temp);//下一条同起点的线
			//cout<<temp<<endl;

		}while(edge != temp);



	}


	static void draw_from_same_Org_point_givenpoint(Mat& img, int vertex, Subdiv2D& subdiv, Scalar delaunay_color)
	{
		Point2f outer_vtx_from_edge;
		int edge_num;
		outer_vtx_from_edge =  subdiv.getVertex(vertex,&edge_num);//得到一条以给定点为起点的线

		circle(img, outer_vtx_from_edge, 3, Scalar(), CV_FILLED, CV_AA, 0);//画出线的起始点
		int temp = edge_num;
		vector<Point2f> pt(2);
		do
		{
			subdiv.edgeOrg(temp, &pt[0]);
			subdiv.edgeDst(temp, &pt[1]);
			line(img, pt[0], pt[1], delaunay_color, 1, CV_AA, 0);
			temp = subdiv.nextEdge(temp);
			//cout<<temp<<endl;

		}while(edge_num != temp);



	}

	//画出Voronoi图
	static void paintVoronoi(Mat& img, Subdiv2D& subdiv)
	{//直接得到顶点和中心点（其实就是给定用于进行三角剖分的点）
	    vector<vector<Point2f> > facets;
	    vector<Point2f> centers;
	    subdiv.getVoronoiFacetList(vector<int>(), facets, centers);//Voronoi图是当你调用calcVoronoi()一次性构建的。可选的是，你可以调用前面提到的getVoronoiFaceList()（它内部调用了calcVoronoi()）来随之更新。

	    vector<Point> ifacet;
	    vector<vector<Point> > ifacets(1);

	    for (size_t i = 0; i < facets.size(); i++)//这么多个多边形
	    {
	        ifacet.resize(facets[i].size());
	        for (size_t j = 0; j < facets[i].size(); j++)//这个多边形里面这么多个顶点
	            ifacet[j] = facets[i][j];

	        Scalar color;
	        color[0] = rand() & 255;
	        color[1] = rand() & 255;
	        color[2] = rand() & 255;
	        fillConvexPoly(img, ifacet, color, 8, 0);

	        ifacets[0] = ifacet;
	        //polylines(img, ifacets, true, Scalar(), 1, CV_AA, 0);//画边界线和点
	        //circle(img, centers[i], 3, Scalar(), CV_FILLED, CV_AA, 0);
	    }
	}


	TransformPointsFrame()
	 :it_(nh_)
	{
		ros::NodeHandle nh;
		ros::NodeHandle nh_priv("~");

		image_pub_ = it_.advertise("/gmm_em_road_last_result", 1);
		image_sub_ = it_.subscribe("/kitti_player_100/color/left/image_rect", 2, &TransformPointsFrame::imageCb, this);

		nh_priv.param("target_frame_link", target_frame_link, std::string("left_camera"));
		nh_priv.param("source_frame_link", source_frame_link, std::string("head_ground"));
		nh_priv.param("thresh_theta", thresh_theta, 77.0);//论文中用的角度大小
		nh_priv.param("obstacle_thresh_z", obstacle_thresh_z, -1.30);//激光雷达的高度：1.73m
		nh_priv.param("sensor_param_filename", sensor_param_filename, std::string("/home/bit/dev/catkin_ws1/src/my_serial/launch/param/velodyne_left_right_camera_parameters.xml"));
		nh.param("p_points_downsample_size_", p_points_downsample_size_, 0.1);
		points_sub = nh.subscribe("input_points", 1, &TransformPointsFrame::PointsCallback, this);
		new_Points_publish = nh.advertise<sensor_msgs::PointCloud2>("transformed_frame_points_left_camera", 10);
		delaunay_ground_Points_publish = nh.advertise<sensor_msgs::PointCloud2>("delaunay_ground_Points_head_ground", 10);
		delaunay_obstacle_Points_publish = nh.advertise<sensor_msgs::PointCloud2>("delaunay_obstacle_Points_head_ground", 10);
	    imageSize.width = 1242;
	    imageSize.height = 375;
		cos_thresh_theta = cos(thresh_theta * 3.1415926 / 180);
		received_points.reset(new pcl::PointCloud<pcl::PointXYZI>());//通过这种方式初始化


		const char* camera_parameters_file = sensor_param_filename.c_str();

		       FileStorage fs(camera_parameters_file, FileStorage::READ);
		       printf("read camera params!\n");
		       //注意实际传感器采用的是摄像头的内参，而kitti的数据使用的是投影矩阵
		       fs["left_K"]>>left_K;
		       fs["left_D"]>>left_D;
		       fs["P2_velodyne_2_left_camera_2_image_plane"]>>P2_velodyne_2_left_camera_2_image_plane;//这里复制粘贴忘记改到对应的量了，真是坑，以后细心一点


		       fs.release();

		       matrix_33_left_K << left_K.at<double>(0, 0),left_K.at<double>(0, 1),left_K.at<double>(0, 2),left_K.at<double>(1, 0),left_K.at<double>(1, 1),left_K.at<double>(1, 2),left_K.at<double>(2, 0),left_K.at<double>(2, 1),left_K.at<double>(2, 2);
		       matrix_33_P2<< P2_velodyne_2_left_camera_2_image_plane.at<double>(0, 0),P2_velodyne_2_left_camera_2_image_plane.at<double>(0, 1),P2_velodyne_2_left_camera_2_image_plane.at<double>(0, 2),P2_velodyne_2_left_camera_2_image_plane.at<double>(1, 0),P2_velodyne_2_left_camera_2_image_plane.at<double>(1, 1),P2_velodyne_2_left_camera_2_image_plane.at<double>(1, 2),P2_velodyne_2_left_camera_2_image_plane.at<double>(2, 0),P2_velodyne_2_left_camera_2_image_plane.at<double>(2, 1),P2_velodyne_2_left_camera_2_image_plane.at<double>(2, 2);
		       vector_T_for_P2 <<P2_velodyne_2_left_camera_2_image_plane.at<double>(0, 3),P2_velodyne_2_left_camera_2_image_plane.at<double>(1, 3),P2_velodyne_2_left_camera_2_image_plane.at<double>(2, 3);
		       //cout<<"left_K:"<<left_K<<endl;
		       //cout<<"left_D:"<<left_D<<endl;
		       //cout<<"matrix_33_left_K:"<< matrix_33_left_K  ;
		      // cout<<"matrix_33_P2:"<< matrix_33_P2<<endl  ;
		      // cout<<"vector_T_for_P2:"<< vector_T_for_P2<<endl  ;
		waitForTf();
		namedWindow("obstale_mask_3C",WINDOW_NORMAL);
		namedWindow("delaunay_draw",WINDOW_NORMAL);
		namedWindow("paintVoronoi",WINDOW_NORMAL);


		run();


	}
	~TransformPointsFrame()
		{
		   destroyAllWindows();
		}

	void run()
	{
		 ros::Rate rate(100);
		 bool status = ros::ok();
		 while (status) {
		     ros::spinOnce();

		     if (!(subscribed_points_new && subscribed_image_new  &&
		             fabs(timepoints - timeimg) < 0.05 ))
		     {
		    	 //cout<<"image and points are not sync or msg not received yet!"<<endl;//提示时间戳不同步
		    	 continue;
		     }
		     //收到统一时间戳的数据才执行下面


		      subscribed_points_new = false;
		      subscribed_image_new = false;
			  //操作之前做一下噪点滤除
			  pcl::RadiusOutlierRemoval<pcl::PointXYZI> outrem;// 创建滤波器
			  outrem.setInputCloud(received_points);              //设置输入点云
			  outrem.setRadiusSearch(0.3);              //设置在0.005半径的范围内找邻近点，按30万个点每秒，10Hz，一圈30000点，那么30米的半径，相邻两点之间的距离是2*3.14159*30/30000=0.0063，也就是至少能保证30米范围内正常扫描的点，能滤掉里面噪声点
			  outrem.setMinNeighborsInRadius(2);       //设置查询点的邻近点集数小于2的删除
			  outrem.filter (*received_points);//执行条件滤波，存储结果到cloud_filtered




			  std::vector<int> indices;
			  pcl::removeNaNFromPointCloud(*received_points, *received_points, indices);

			/*  pcl::VoxelGrid<pcl::PointXYZI> downSizeFilter;
			  downSizeFilter.setInputCloud(laserCloudIn);
			  downSizeFilter.setLeafSize(p_points_downsample_size_, p_points_downsample_size_, p_points_downsample_size_);
			  downSizeFilter.filter(*laserCloudIn);*/


			  int cloudSize = received_points->points.size();
			  //cout<<"test !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" <<endl;
			  pcl::PointXYZI point;
			  pcl::PointCloud<pcl::PointXYZI> result_points;
			  pcl::PointCloud<pcl::PointXYZI> result_delaunay_ground_points;
			  pcl::PointCloud<pcl::PointXYZI> result_delaunay_obstacle_points;
			  bool transform_successful = false;
			  transform_successful = tf_.canTransform(target_frame_link, source_frame_link, ros::Time());
			  if (transform_successful)
				{

					try{
							tf_.lookupTransform(target_frame_link, source_frame_link,
							ros::Time(0), transform);
						}
					catch(tf::TransformException &ex) {
							ROS_ERROR("%s",ex.what());
							ros::Duration(1.0).sleep();

						}

				}
			 //	cv::Mat R = (cv::Mat_<double>(3,3)<<0.999481849846212,-0.003980102561436,0.031940422846226,0.004092368279281,0.999985674064003,-0.003450244720207,-0.031926232941924,0.003579168948658,0.999483819378671);//stereoParams.RotationOfCamera2，应该是摄像头2相对摄像头1的旋转矩阵，opencv这个旋转矩阵应该是动到静的转化吧？这里两者应该是等价的吧，父参考系是1
			 // cv::Mat R = (cv::Mat_<double>(3,3)<<transform.getBasis()[0][0],transform.getBasis()[0][1],transform.getBasis()[0][2],transform.getBasis()[1][0],transform.getBasis()[1][1],transform.getBasis()[1][2],transform.getBasis()[2][0],transform.getBasis()[2][1],transform.getBasis()[2][2]);
			  //cv::Mat T = (cv::Mat_<double>(3,1)<<transform.getOrigin()[0],transform.getOrigin()[1],transform.getOrigin()[2]);//这个应该是只能进行初始化，但是在这里编译通不过，以为是只能常数初始化，也编译通不过
			Eigen::Matrix<double, 3,3> matrix_33_R;
			matrix_33_R << transform.getBasis()[0][0],transform.getBasis()[0][1],transform.getBasis()[0][2],transform.getBasis()[1][0],transform.getBasis()[1][1],transform.getBasis()[1][2],transform.getBasis()[2][0],transform.getBasis()[2][1],transform.getBasis()[2][2];
			Eigen::Vector3d  vector_3d_T,vector_point,vector_point_temp;//实质是3*1的矩阵
			vector_3d_T << transform.getOrigin()[0],transform.getOrigin()[1],transform.getOrigin()[2];



			Mat project_left_img = Mat::zeros(imageSize, CV_32FC4);//用于存放三维坐标和强度信息
			Mat mask = Mat::zeros(imageSize, CV_8UC1);//投影掩码
			Mat obstale_mask_3C = Mat::zeros(imageSize, CV_8UC3);//障碍掩码，未知区域是0，障碍区域是（255，0，0）地面区域（0，255，0），只是用于可视化
			Mat obstale_mask_1C = Mat::zeros(imageSize, CV_8UC1);//未知是0，障碍是255，地面是100

			Rect rect(0,0,imageSize.width,imageSize.height);

			Mat img(rect.size(), CV_8UC3);
			img = Scalar::all(0);
			Scalar  delaunay_color(255,255,255);

			clock_t time_stt = clock();
			//创建Delaunay剖分
			Subdiv2D subdiv(rect);
			Point2f point_to_inset;
			int inset_subdiv_point_num=0;


			  for (int i = 0; i < cloudSize; i++) {
				 // cv::Mat matpoint = (cv::Mat_<double>(3,1)<<laserCloudIn.points[i].x,laserCloudIn.points[i].y,laserCloudIn.points[i].z);

			  vector_point<<received_points->points[i].x,received_points->points[i].y,received_points->points[i].z;//head_ground坐标系下
			  vector_point = matrix_33_R * vector_point + vector_3d_T;


				  point.x = vector_point(0,0);//在摄像头坐标系下
				  point.y = vector_point(1,0);
				  point.z = vector_point(2,0);

				  point.intensity = received_points->points[i].intensity;//注意

				  result_points.push_back(point);

	//上面将点云从head_ground转到左摄像头，下面投到成像平面
				  if(point.z<=0)//注意z小于0时也能得到对应的图像坐标，事实上又是不可能看见的，在左摄像头参考系
					 {
						continue;

					 }
				  if ((received_points->points[i].x*received_points->points[i].x+received_points->points[i].y*received_points->points[i].y+received_points->points[i].z*received_points->points[i].z)>10000)
					 {//距离太远的点也不考虑，这里默认是100米，在head_ground参考系
						continue;

					 }

							vector_point_temp<<point.x,point.y,point.z;
							//vector_point_temp = matrix_33_left_K * vector_point_temp;
							vector_point_temp = matrix_33_P2 * vector_point_temp+vector_T_for_P2;
							int u = vector_point_temp(0,0) / vector_point_temp(2,0);
							int v = vector_point_temp(1,0) / vector_point_temp(2,0);
							//cout<<"u: "<<u<<"v: "<<v<<endl;
							if(u>0&&v>0&&u<(imageSize.width-1)&&v<(imageSize.height-1))
							{

								if (mask.at<uchar>(v, u)!=255)
								{
									project_left_img.at<Vec4f>(v, u)[0] = received_points->points[i].x;//在head_ground坐标系下
									project_left_img.at<Vec4f>(v, u)[1] = received_points->points[i].y;
									project_left_img.at<Vec4f>(v, u)[2] = received_points->points[i].z;
									project_left_img.at<Vec4f>(v, u)[3] = received_points->points[i].intensity;
									mask.at<uchar>(v, u) = 255;
									point_to_inset.x = u;//注意顺序
									point_to_inset.y = v;
									subdiv.insert(point_to_inset);
									inset_subdiv_point_num++;
								}
								else//考虑到有很多点投到了同样的图像坐标
								{
									for(int k =1;k<(imageSize.height-1);k++)
									{
										if (u>0&&v+k>0&&u<(imageSize.width-1)&&v+k<(imageSize.height-1)&&mask.at<uchar>(v+k, u)!=255)
										{
											int v_k = v+k;
											project_left_img.at<Vec4f>(v_k, u)[0] = received_points->points[i].x;//在head_ground坐标系下
											project_left_img.at<Vec4f>(v_k, u)[1] = received_points->points[i].y;
											project_left_img.at<Vec4f>(v_k, u)[2] = received_points->points[i].z;
											project_left_img.at<Vec4f>(v_k, u)[3] = received_points->points[i].intensity;
											mask.at<uchar>(v_k, u) = 255;
											point_to_inset.x = u;//注意顺序
											point_to_inset.y = v_k;
											subdiv.insert(point_to_inset);
											inset_subdiv_point_num++;
											break;
										}

									}
								}
							}

			  }

			  /*
			  drawSubdiv( img, subdiv, delaunay_color );
			  cv::imshow( "delaunay_draw", img );

			  img = Scalar::all(0);
			  paintVoronoi( img, subdiv );
			  cv::imshow( "paintVoronoi", img );
			   */

	//这里三角剖分检测障碍物,上面插入的时候已经剖分完了
	//对mask有投影的点进行遍历判断是否是障碍点，这个不好操作，可以从三角剖分的索引为4的点开始遍历
				  inset_subdiv_point_num = inset_subdiv_point_num+4;//根据这个好像不对，遍历边的最后会出现很大的边如：edge_num1040695090，或者出现负的边 edge_num-1181446616
				  cout<<"inset_subdiv_point_num: "<<inset_subdiv_point_num<<endl;	  	  	  	  	  	  	  	  	  	  	  	  	  //猜测可能原因是因为遮挡有些点投到了同一个位置，这个得处理一下！！！！可以考虑移一个像素点(现在是这么处理的)，或者取高度高的点
				  int edge_num;
				  //std::vector<Vec4f> edge_list;
				  //subdiv.getEdgeList(edge_list);//存的是起始点和终点的坐标
				  //cout<<"edge_list.size()"<<edge_list.size()<<endl;//实际的边数应该是这个的四倍吧,差不多是这样，但是不是正好，规律没找到，还是确定加入的点数靠谱
				  //int edge_size = edge_list.size()*4;








			////遍历判断是不是障碍物

				  bool search_state_flag=true;
				  for(int i=4;i<inset_subdiv_point_num ;i++)//遍历判断是不是障碍物
				  {
						Point2f outer_vtx_from_edge;
						int edge_num;
						outer_vtx_from_edge =  subdiv.getVertex(i,&edge_num);//得到一条以给定点为起点的线

						if (edge_num<16||!search_state_flag)//前16条边是虚边||edge_num>=edge_size+16
							{
								cout<<"edge search error!"<<" edge_num:"<<edge_num<<" search_state_flag:"<<search_state_flag<<endl;
								break;
							}

						//circle(img, outer_vtx_from_edge, 3, Scalar(), CV_FILLED, CV_AA, 0);//画出线的起始点
						int temp = edge_num;
						vector<Point2f> pt(2);
						 pcl::PointXYZ org_point_3d,dst_point_3d;
						 //cout<<"inset_subdiv_point_num:"<<inset_subdiv_point_num<<"   i org_point_3d.x:"<<i<<"  edge_num"<<edge_num<<endl;
						bool obstacle_flag = false;
						do
						{
							//int index_o = subdiv.edgeOrg(temp, &pt[0]);//这个点就是前面给的
							//if (index_o<4||index_o>=inset_subdiv_point_num)//前4个点是虚点
							//	{
							//	    search_state_flag = false;
							//		break;
							//	}
							int index_d = subdiv.edgeDst(temp, &pt[1]);
							if (index_d<0||index_d>=inset_subdiv_point_num)
								{
									search_state_flag = false;//不正常
									cout<<"vertex search error!"<<"index_d: "<<index_d<<endl;
									break;
								}
							if (index_d<4)//这条边可能连到虚点
							{
								temp = subdiv.nextEdge(temp);
								continue;
							}
							//下面判断距离，置位掩码
							org_point_3d.x = project_left_img.at<Vec4f>(outer_vtx_from_edge.y, outer_vtx_from_edge.x)[0];
							org_point_3d.y = project_left_img.at<Vec4f>(outer_vtx_from_edge.y, outer_vtx_from_edge.x)[1];
							org_point_3d.z = project_left_img.at<Vec4f>(outer_vtx_from_edge.y, outer_vtx_from_edge.x)[2];
							dst_point_3d.x = project_left_img.at<Vec4f>(pt[1].y, pt[1].x)[0];
							dst_point_3d.y = project_left_img.at<Vec4f>(pt[1].y, pt[1].x)[1];
							dst_point_3d.z = project_left_img.at<Vec4f>(pt[1].y, pt[1].x)[2];//

							double x_m = org_point_3d.x-dst_point_3d.x;
							double y_m = org_point_3d.y-dst_point_3d.y;
							double z_m = org_point_3d.z-dst_point_3d.z;

							double distance = sqrt(x_m*x_m+ y_m*y_m+z_m*z_m);
							double cos_m = fabs(z_m) / distance;
							if (cos_m>cos_thresh_theta||org_point_3d.z>obstacle_thresh_z)//这里加上其他特征进行判断，如Z(激光雷达的高度：1.73m,注意这里距离不一样，高度允许的值应该有差别，考虑横滚和俯仰角！！！！)，崎岖度评估数据！！！！
							{
								obstacle_flag = true;
								break;
							}



							//cout<<"org_point_3d.x: "<<org_point_3d.x<<endl;//经观察输出x都是正，应该在head_ground坐标系下
							//cout<<"org_point_3d.y: "<<org_point_3d.y<<endl;
							//cout<<"org_point_3d.z: "<<org_point_3d.z<<endl;
							//cout<<"dst_point_3d.x"<<dst_point_3d.x<<endl;
							//cout<<"dst_point_3d.y"<<dst_point_3d.y<<endl;
							//cout<<"dst_point_3d.z"<<dst_point_3d.z<<endl;



							//line(img, pt[0], pt[1], delaunay_color, 1, CV_AA, 0);
							temp = subdiv.nextEdge(temp);
							//cout<<"!!!!!!!!!!!!!!!!!!!!!!!temp"<<temp<<endl;

						}while(edge_num != temp);//查找所有和给定点相连的边，判断障碍物属性
						//cout<<"test7"<<endl;

						 if (obstacle_flag)
						 {
							 //置位障碍物掩码／／obstale_mask_3C
							obstale_mask_3C.at<Vec3b>(outer_vtx_from_edge.y, outer_vtx_from_edge.x)[0] = 255;
							obstale_mask_1C.at<uchar>(outer_vtx_from_edge.y, outer_vtx_from_edge.x) = 255;
							//这个地方一开始是Vec3b，所以出现了
							//  free(): invalid next size (normal): 0x00000000025cf120 ***

							// corrupted size vs. prev_size: 0x0000000002456160 ***

							// corrupted size vs. prev_size: 0x0000000001a62a20 ***

							//这样的问题  序列中有的还是能执行！
							/////////////////////////////////////////////////////////////
							 pcl::PointXYZI point_temp;
							point_temp.x = project_left_img.at<Vec4f>(outer_vtx_from_edge.y, outer_vtx_from_edge.x)[0];
							point_temp.y = project_left_img.at<Vec4f>(outer_vtx_from_edge.y, outer_vtx_from_edge.x)[1];
							point_temp.z = project_left_img.at<Vec4f>(outer_vtx_from_edge.y, outer_vtx_from_edge.x)[2];
							point_temp.intensity = project_left_img.at<Vec4f>(outer_vtx_from_edge.y, outer_vtx_from_edge.x)[3];
							result_delaunay_obstacle_points.push_back(point_temp);//准备发布的点云

						 }
						 else
						 {
							 //没有障碍物掩码
							obstale_mask_3C.at<Vec3b>(outer_vtx_from_edge.y, outer_vtx_from_edge.x)[1] = 255;
							obstale_mask_1C.at<uchar>(outer_vtx_from_edge.y, outer_vtx_from_edge.x) = 100;
							pcl::PointXYZI point_temp;
							point_temp.x = project_left_img.at<Vec4f>(outer_vtx_from_edge.y, outer_vtx_from_edge.x)[0];
							point_temp.y = project_left_img.at<Vec4f>(outer_vtx_from_edge.y, outer_vtx_from_edge.x)[1];
							point_temp.z = project_left_img.at<Vec4f>(outer_vtx_from_edge.y, outer_vtx_from_edge.x)[2];
							point_temp.intensity = project_left_img.at<Vec4f>(outer_vtx_from_edge.y, outer_vtx_from_edge.x)[3];
							result_delaunay_ground_points.push_back(point_temp);//准备发布的点云

						 }



				  }


				  ////根据voronoi图遍历，以前面得到的结果分区图像

				    vector<vector<Point2f> > facets;
				    vector<Point2f> centers;
				    subdiv.getVoronoiFacetList(vector<int>(), facets, centers);

				    vector<Point> ifacet;
				    vector<vector<Point> > ifacets(1);

				    Scalar ground_color(0, 255, 0), obstacle_color(255,0,0);

				    for( size_t i = 0; i < facets.size(); i++ )
				    {
				        ifacet.resize(facets[i].size());
				        for( size_t j = 0; j < facets[i].size(); j++ )
				            ifacet[j] = facets[i][j];

				       // Scalar color;
				       // color[0] = rand() & 255;
				       // color[1] = rand() & 255;
				       // color[2] = rand() & 255;


				        if(obstale_mask_1C.at<uchar>(centers[i].y, centers[i].x) == 255)//255表示是障碍点，100表示是地面点
				        	fillConvexPoly(img, ifacet, obstacle_color, 8, 0);//填充多边形
				        else
				        	fillConvexPoly(img, ifacet, ground_color, 8, 0);//填充多边形

				       // ifacets[0] = ifacet;
				       // polylines(img, ifacets, true, Scalar(), 1, CV_AA, 0);//画多边形
				       // circle(img, centers[i], 3, Scalar(), CV_FILLED, CV_AA, 0);//画中间点和边界线 注意这里的点和多边形是成对的

				    }


















				  cout<<"creating Delaunay time used is "<<1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC<<"ms"<<endl;//这个函数总共需要这么多时间

				/*

				//画出Delaunay剖分三角形
				Mat img_delaunay = img.clone();
				drawSubdiv(img_delaunay, subdiv, Scalar(255,255,255));
				cout<<"draw Delaunay time used is "<<1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC<<"ms"<<endl;
				imshow("Delaunay", img_delaunay);
				//imwrite("delaunay.jpg", img_delaunay);


				//画出Voronoi图
				Mat img_voronoi = img.clone();
				paintVoronoi(img_voronoi, subdiv);
				cout<<"draw Voronoi time used is "<<1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC<<"ms"<<endl;
				imshow("Voronoi", img_voronoi);
				//imwrite("voronoi.jpg", img_voronoi);

				Mat img_from_same_point = img.clone();
				draw_from_same_Org_point_givenpoint(img_from_same_point, 30, subdiv, Scalar(255,255,255));
				imshow("Img_from_same_point", img_from_same_point);
				//imwrite("img_from_same_point.jpg", img_from_same_point);

	*/

			  sensor_msgs::PointCloud2 laserCloudOutMsg;
			  pcl::toROSMsg(result_points, laserCloudOutMsg);
			  laserCloudOutMsg.header.stamp = points_header.stamp;
			  laserCloudOutMsg.header.frame_id = target_frame_link;

			  sensor_msgs::PointCloud2 laserCloudOutGroundMsg;
			  pcl::toROSMsg(result_delaunay_ground_points, laserCloudOutGroundMsg);
			  laserCloudOutGroundMsg.header.stamp = points_header.stamp;
			  laserCloudOutGroundMsg.header.frame_id = source_frame_link;

			  sensor_msgs::PointCloud2 laserCloudOutObstacleMsg;
			  pcl::toROSMsg(result_delaunay_obstacle_points, laserCloudOutObstacleMsg);
			  laserCloudOutObstacleMsg.header.stamp = points_header.stamp;
			  laserCloudOutObstacleMsg.header.frame_id = source_frame_link;

			  new_Points_publish.publish(laserCloudOutMsg);
			  delaunay_ground_Points_publish.publish(laserCloudOutGroundMsg);
			  delaunay_obstacle_Points_publish.publish(laserCloudOutObstacleMsg);

			  //namedWindow("project_to_left_image",WINDOW_NORMAL);
			  //namedWindow("project_to_left_image_mask",WINDOW_NORMAL);

			  //imshow("project_to_left_image", project_left_img);
			  //imshow("project_to_left_image_mask", mask);
			  imshow("obstale_mask_3C", obstale_mask_3C);
			  cv::imshow("received_img",subscribed_image);
			  cv::imshow("paintVoronoi",img);
			  cv::waitKey(3);


		      status = ros::ok();
		      rate.sleep();
		   }

		   return ;


	}


    void imageCb(const sensor_msgs::ImageConstPtr& msg)
    {
    	timeimg = msg->header.stamp.toSec();
    	//cout<<"image time stamp:"<<timeimg<<endl;
        cv_bridge::CvImagePtr cv_ptr; // 声明一个CvImage指针的实例
        try
        {   cv_ptr =  cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8); }
        catch(cv_bridge::Exception& e)  //异常处理
        {  ROS_ERROR("cv_bridge exception: %s", e.what());
            return;  }
        //cv::waitKey(3);
        GMM_image_process(cv_ptr->image);//实际没有在这个函数里面进行处理，在run主流程里面执行

    }

    void GMM_image_process(cv::Mat img)
    {
    	 subscribed_image = img;
    	// cv::imshow("received_img",img);
    	 //Mat hsv_;
    	 //cvtColor(subscribed_image, hsv_, COLOR_BGR2HSV);//注意传入图像是BGR8
    	 //vector< Mat >hsv(hsv_.channels());

    	      //下面测试HSV 和BGR的单通道效果都不行，可能结合多个通道和多个颜色空间效果会好些

    	      //原本是想用色调通道
    	      //split(hsv_, hsv);
    	      //subscribed_image_h = hsv[0];//用h通道试

    	      //试试BGR里面的三个通道试试
    	      //split(subscribed_image, hsv);
    	      //subscribed_image_h = hsv[0];//用h通道试


    	      //cvtColor(subscribed_image, subscribed_image_h, COLOR_BGR2GRAY);//用灰度信息试
    	      //imshow("befor_point_call_image", subscribed_image_h);
    	 subscribed_image_new = true;
    }



	void PointsCallback(const sensor_msgs::PointCloud2 &msg)
	    {
		  timepoints = msg.header.stamp.toSec();
		 // cout<<"points time stamp:"<<timepoints<<endl;
		  points_header = msg.header;
		  sensor_msgs::PointCloud2 temp = msg;
		  //pcl::PointCloud<pcl::PointXYZI>::Ptr  laserCloudIn(new pcl::PointCloud<pcl::PointXYZI>());
		  received_points->clear();
		  pcl::fromROSMsg(temp, *received_points);
		  subscribed_points_new = true;

	    }

	void waitForTf()
		 {
		   ros::Time start = ros::Time::now();
		   ROS_INFO("Waiting for tf transform data between frames %s and %s to become available", target_frame_link.c_str(), source_frame_link.c_str() );

		   bool transform_successful = false;

		   while (!transform_successful){
			 transform_successful = tf_.canTransform(target_frame_link, source_frame_link, ros::Time());
			 if (transform_successful) break;

			 ros::Time now = ros::Time::now();

			 if ((now-start).toSec() > 20.0){
			   ROS_WARN_ONCE("No transform between frames %s and %s available after %f seconds of waiting. This warning only prints once.", target_frame_link.c_str(), source_frame_link.c_str(), (now-start).toSec());
			 }

			 if (!ros::ok()) return;
			 ros::WallDuration(1.0).sleep();
		   }

		   ros::Time end = ros::Time::now();
		   ROS_INFO("Finished waiting for tf, waited %f seconds", (end-start).toSec());
		 }




	};



    int main (int argc, char** argv){
        //初始化节点
        ros::init(argc, argv, "transform_kitti_points2_2_left_camera_for_obstacle_and_ground_detection");

        TransformPointsFrame trans;

        	 ros::spin();
        	 return(0);

    }






