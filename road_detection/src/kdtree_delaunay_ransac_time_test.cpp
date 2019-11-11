#include <iostream>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <ros/ros.h>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/locale.hpp>
#include <boost/program_options.hpp>
#include <boost/progress.hpp>
#include <boost/tokenizer.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <sensor_msgs/distortion_models.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/PointCloud2.h>
#include <stereo_msgs/DisparityImage.h>
#include <std_msgs/Bool.h>
#include <tf/LinearMath/Transform.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <time.h>
#include <pcl-1.7/pcl/filters/voxel_grid.h>
#include <pcl-1.7/pcl/point_cloud.h>
#include <pcl-1.7/pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl-1.7/pcl/console/parse.h>
#include <pcl-1.7/pcl/filters/extract_indices.h>
#include <pcl-1.7/pcl/sample_consensus/ransac.h>
#include <pcl-1.7/pcl/sample_consensus/sac_model_plane.h>
#include <pcl-1.7/pcl/sample_consensus/sac_model_sphere.h>
#include <pcl-1.7/pcl/ModelCoefficients.h>
#include <pcl-1.7/pcl/sample_consensus/method_types.h>
#include <pcl-1.7/pcl/sample_consensus/model_types.h>
#include <pcl-1.7/pcl/segmentation/sac_segmentation.h>
#include <pcl-1.7/pcl/filters/passthrough.h>
#include <pcl-1.7/pcl/filters/project_inliers.h>

#include <Eigen/Core>//注意需要先安装这个库
#include <opencv2/imgproc/imgproc.hpp>
#include <pcl-1.7/pcl/filters/radius_outlier_removal.h>
#include <pcl-1.7/pcl/filters/statistical_outlier_removal.h>
#include <math.h>
 #include <pcl-1.7/pcl/kdtree/kdtree_flann.h>

using namespace std;
using namespace pcl;
using namespace ros;
using namespace tf;
using namespace std;


namespace po = boost::program_options;
struct kitti_road_options
{
    string  path;
    bool    train_test;//使用训练数据时为true,test时为false
};

cv::Mat convertTo3Channels(const cv::Mat& binImg)
{
	cv::Mat three_channel = cv::Mat::zeros(binImg.rows, binImg.cols, CV_8UC3);
	cv::vector<cv::Mat> channels;
	for (int i = 0;i<3;i++)
	{
		channels.push_back(binImg);
	}
	merge(channels, three_channel);
	return three_channel;
}

static void drawSubdiv(cv::Mat& img, cv::Subdiv2D& subdiv, cv::Scalar delaunay_color)
{//直接得到剖分三角形的顶点
    cv::vector<cv::Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);
    cv::vector<cv::Point> pt(3);

    for (size_t i = 0; i < triangleList.size(); i++)//这个顺序好像没什么规律
    {
        cv::Vec6f t = triangleList[i];
        pt[0] = cv::Point(cvRound(t[0]), cvRound(t[1]));
        pt[1] = cv::Point(cvRound(t[2]), cvRound(t[3]));
        pt[2] = cv::Point(cvRound(t[4]), cvRound(t[5]));
        cv::line(img, pt[0], pt[1], delaunay_color, 1, CV_AA, 0);
        cv::line(img, pt[1], pt[2], delaunay_color, 1, CV_AA, 0);
        cv::line(img, pt[2], pt[0], delaunay_color, 1, CV_AA, 0);
    }
}


//画出连接同一个顶点所有边，这些边以给定边的起点为出发点
static void draw_from_same_Org_point(cv::Mat& img, int edge, cv::Subdiv2D& subdiv, cv::Scalar delaunay_color)
{
	cv::Point2f outer_vtx_from_edge;
	subdiv.edgeOrg(edge, &outer_vtx_from_edge);
	cv::circle(img, outer_vtx_from_edge, 3, cv::Scalar(), CV_FILLED, CV_AA, 0);//画出线的起始点
	int temp = edge;
	cv::vector<cv::Point2f> pt(2);
	do
	{
		subdiv.edgeOrg(temp, &pt[0]);//画给定线
		subdiv.edgeDst(temp, &pt[1]);
		cv::line(img, pt[0], pt[1], delaunay_color, 1, CV_AA, 0);
		temp = subdiv.nextEdge(temp);//下一条同起点的线
		//cout<<temp<<endl;

	}while(edge != temp);



}


static void draw_from_same_Org_point_givenpoint(cv::Mat& img, int vertex, cv::Subdiv2D& subdiv, cv::Scalar delaunay_color)
{
	cv::Point2f outer_vtx_from_edge;
	int edge_num;
	outer_vtx_from_edge =  subdiv.getVertex(vertex,&edge_num);//得到一条以给定点为起点的线

	cv::circle(img, outer_vtx_from_edge, 3, cv::Scalar(), CV_FILLED, CV_AA, 0);//画出线的起始点
	int temp = edge_num;
	cv::vector<cv::Point2f> pt(2);
	do
	{
		subdiv.edgeOrg(temp, &pt[0]);
		subdiv.edgeDst(temp, &pt[1]);
		cv::line(img, pt[0], pt[1], delaunay_color, 1, CV_AA, 0);
		temp = subdiv.nextEdge(temp);
		//cout<<temp<<endl;

	}while(edge_num != temp);



}

//画出Voronoi图
static void paintVoronoi(cv::Mat& img, cv::Subdiv2D& subdiv)
{//直接得到顶点和中心点（其实就是给定用于进行三角剖分的点）
    cv::vector<cv::vector<cv::Point2f> > facets;
    cv::vector<cv::Point2f> centers;
    subdiv.getVoronoiFacetList(cv::vector<int>(), facets, centers);//Voronoi图是当你调用calcVoronoi()一次性构建的。可选的是，你可以调用前面提到的getVoronoiFaceList()（它内部调用了calcVoronoi()）来随之更新。

    cv::vector<cv::Point> ifacet;
    cv::vector<cv::vector<cv::Point> > ifacets(1);

    for (size_t i = 0; i < facets.size(); i++)//这么多个多边形
    {
        ifacet.resize(facets[i].size());
        for (size_t j = 0; j < facets[i].size(); j++)//这个多边形里面这么多个顶点
            ifacet[j] = facets[i][j];

        cv::Scalar color;
        color[0] = rand() & 255;
        color[1] = rand() & 255;
        color[2] = rand() & 255;
        cv::fillConvexPoly(img, ifacet, color, 8, 0);

        ifacets[0] = ifacet;
        //polylines(img, ifacets, true, Scalar(), 1, CV_AA, 0);//画边界线和点
        //circle(img, centers[i], 3, Scalar(), CV_FILLED, CV_AA, 0);
    }
}

int getCalibration_kitti_road(string full_filename_calib, double* P2,  double *R0_rect, double* Tr_velo_to_cam , double* Tr_cam_to_road)
{

    string calib_cam_to_cam = full_filename_calib;
    ifstream file_c2c(calib_cam_to_cam.c_str());
    if (!file_c2c.is_open())
        return false;

    ROS_INFO_STREAM("Reading camera calibration from " << calib_cam_to_cam);

    typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
    boost::char_separator<char> sep {" "};

    string line = "";
    unsigned char index = 0;
    tokenizer::iterator token_iterator;

    while (getline(file_c2c, line))
    {
        // Parse string phase 1, tokenize it using Boost.
        tokenizer tok(line, sep);

        // Move the iterator at the beginning of the tokenize vector and check for K/D/R/P matrices.
        token_iterator = tok.begin();
        if (strcmp((*token_iterator).c_str(), (string("P2:") ).c_str()) == 0)
        {
            index = 0; //should be 9 at the end
            ROS_DEBUG_STREAM("P2" );
            for (token_iterator++; token_iterator != tok.end(); token_iterator++)
            {
                //std::cout << *token_iterator << '\n';
                P2[index++] = boost::lexical_cast<double>(*token_iterator);
            }
        }
        token_iterator = tok.begin();
        if (strcmp((*token_iterator).c_str(), (string("R0_rect:") ).c_str()) == 0)
        {
            index = 0; //should be 9 at the end
            ROS_DEBUG_STREAM("R0_rect" );
            for (token_iterator++; token_iterator != tok.end(); token_iterator++)
            {
                //std::cout << *token_iterator << '\n';
            	R0_rect[index++] = boost::lexical_cast<double>(*token_iterator);
            }
        }
        token_iterator = tok.begin();
        if (strcmp((*token_iterator).c_str(), (string("Tr_velo_to_cam:") ).c_str()) == 0)
        {
            index = 0; //should be 9 at the end
            ROS_DEBUG_STREAM("Tr_velo_to_cam" );
            for (token_iterator++; token_iterator != tok.end(); token_iterator++)
            {
                //std::cout << *token_iterator << '\n';
            	Tr_velo_to_cam[index++] = boost::lexical_cast<double>(*token_iterator);
            }
        }
        token_iterator = tok.begin();
        if (strcmp((*token_iterator).c_str(), (string("Tr_cam_to_road:") ).c_str()) == 0)
        {
            index = 0; //should be 9 at the end
            ROS_DEBUG_STREAM("Tr_cam_to_road" );
            for (token_iterator++; token_iterator != tok.end(); token_iterator++)
            {
                //std::cout << *token_iterator << '\n';
            	Tr_cam_to_road[index++] = boost::lexical_cast<double>(*token_iterator);
            }
        }

    }
    ROS_INFO_STREAM("... ok");
    return true;
}



int main(int argc, char **argv)
{
    kitti_road_options options;
    po::variables_map vm;

    po::options_description desc("Kitti_road_detection!", 200);
    desc.add_options()
    ("help,h"                                                                                                    ,  "help message")
    ("directory ,d",  po::value<string>       (&options.path)->required()                                        ,  "*required* - path to the kitti road Directory")
	("train_test       ,t",  po::value<bool>         (&options.train_test)         ->default_value(0) ->implicit_value(1)   ,  "run on train or test data")//默认是在test上执行
    ;

    try // parse options
    {
        po::parsed_options parsed = po::command_line_parser(argc - 2, argv).options(desc).allow_unregistered().run();
        po::store(parsed, vm);
        po::notify(vm);

        vector<string> to_pass_further = po::collect_unrecognized(parsed.options, po::include_positional);

        // Can't handle __ros (ROS parameters ... )
        //        if (to_pass_further.size()>0)
        //        {
        //            ROS_WARN_STREAM("Unknown Options Detected, shutting down node\n");
        //            cerr << desc << endl;
        //            return 1;
        //        }
    }
    catch (...)
    {
        cerr << desc << endl;

        cout << "kitti_road needs a directory tree like the following:" << endl;
        cout << "└── ROAD                                  " << endl;
        cout << "    ├── data_road                         " << endl;
        cout << "    │   └── data_road                     " << endl;
        cout << "    │       └── testing                   " << endl;
        cout << "    │       │    └── calib                " << endl;
        cout << "    │       │    └── image_2              " << endl;
        cout << "    │       │    └── output               " << endl;
        cout << "    │       └── training                  " << endl;
        cout << "    │            └── calib                " << endl;
        cout << "    │            └── image_2              " << endl;
        cout << "    │            └── gt_image_2           " << endl;
        cout << "    │            └── output               " << endl;
        cout << "    │── data_road_velodyne                " << endl;
        cout << "            └── testing                   " << endl;
        cout << "            │    └── velodyne             " << endl;
        cout << "            └── training                  " << endl;
        cout << "                 └── velodyne             " << endl<< endl;


        ROS_WARN_STREAM("Parse error, shutting down node\n");
        return -1;
    }

    ros::init(argc, argv, "kitti_road_detection");
    ros::NodeHandle nh_priv("~");
    double thresh_theta;
    double cos_thresh_theta;
    cv::Size imageSize;
    cv::Size imageSize_3;

    double obstacle_thresh_z;
    double roughness_distance_thresh;//判断三角剖分连接的点的距离在这个范围内就用来统计之间的高度差，来判断是不是类似草地
    double roughness_z_thresh;
    double ransac_DistanceThreshold;
    double ransac_DistanceThreshold_m;

    double box_x_min,box_x_max,box_y_min,box_y_max,box_z_min,box_z_max;
    double box_x_min1,box_x_max1,box_y_min1,box_y_max1,box_z_min1,box_z_max1;//用来确定用于开始扩展的区域
    double ransac_DistanceThreshold1;
    double near_distanse_thresh;
    int  terminate_expand_num;
    double min_edge_distance;

    double threshold_low,threshold_up;//绿色：35到77  青色：78到99 蓝色：100到124
    double threshold_sv;//43到 255
    int erode_size;


    nh_priv.param("thresh_theta", thresh_theta, 77.0);//论文中用的角度大小
    nh_priv.param("obstacle_thresh_z", obstacle_thresh_z, -1.30);//激光雷达的高度：1.73m
    nh_priv.param("roughness_distance_thresh", roughness_distance_thresh, 0.1);
    nh_priv.param("roughness_z_thresh", roughness_z_thresh, 0.04);
    nh_priv.param("ransac_DistanceThreshold", ransac_DistanceThreshold, 0.04);
    nh_priv.param("ransac_DistanceThreshold_m", ransac_DistanceThreshold_m, 5.0);
    nh_priv.param("box_x_min", box_x_min, 2.45);
    nh_priv.param("box_x_max", box_x_max, 46.0);
    nh_priv.param("box_y_min", box_y_min, -2.0);
    nh_priv.param("box_y_max", box_y_max, 2.0);
    nh_priv.param("box_z_min", box_z_min, -2.7);
    nh_priv.param("box_z_max", box_z_max, 0.4);
    nh_priv.param("ransac_DistanceThreshold1", ransac_DistanceThreshold1, 0.04);
    nh_priv.param("box_x_min1", box_x_min1, 2.45);
    nh_priv.param("box_x_max1", box_x_max1, 10.0);
    nh_priv.param("box_y_min1", box_y_min1, -1.5);
    nh_priv.param("box_y_max1", box_y_max1, 1.5);
    nh_priv.param("box_z_min1", box_z_min1, -2.7);
    nh_priv.param("box_z_max1", box_z_max1, 0.4);
    nh_priv.param("near_distanse_thresh", near_distanse_thresh, 2.0);
    nh_priv.param("terminate_expand_num", terminate_expand_num, 5);
    nh_priv.param("threshold_low", threshold_low, 35.0);//通过纯颜色肯定不行，道路中很多像素的值也在这个区间
    nh_priv.param("threshold_up", threshold_up, 77.0);
    nh_priv.param("threshold_sv", threshold_sv, 43.0);
    nh_priv.param("min_edge_distance", min_edge_distance, 0.015);
    nh_priv.param("erode_size", erode_size, 3);

    cos_thresh_theta = cos(thresh_theta * 3.1415926 / 180);

//P2
    Eigen::Matrix<double, 3,3> matrix_33_P2;
    Eigen::Vector3d vector_T_for_P2;
// R0_rect
    Eigen::Matrix<double, 3,3> matrix_33_R0_rect;
//Tr_velo_to_cam
    Eigen::Matrix<double, 3,3> matrix_33_Tr_velo_to_cam;
    Eigen::Vector3d vector_T_for_Tr_velo_to_cam;
// Tr_cam_to_road
    Eigen::Matrix<double, 3,3> matrix_33_Tr_cam_to_road;
    Eigen::Vector3d vector_T_for_Tr_cam_to_road;
    Eigen::Matrix<double, 3,3> matrix_33_Tr_road_to_cam;//上面的逆
    Eigen::Vector3d vector_T_for_Tr_road_to_cam;


    /// This sets the logger level; use this to disable all ROS prints
    if ( ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info) )
        ros::console::notifyLoggerLevelsChanged();
    else
        std::cout << "Error while setting the logger level!" << std::endl;

    DIR *dir;
    struct dirent *ent;
    unsigned int total_entries = 0;        //number of elements to be played
    unsigned int len = 0;                  //counting elements support variable
    string dir_root             ;

    string dir_image02          ;
    string full_filename_image02;
    string dir_calib            ;
    string full_filename_calib;
    string dir_output           ;
    string full_filename_output;
    string full_filename_output_3C_mask;;
    string dir_velodyne_points  ;
    string full_filename_velodyne;
    string str_support;
    cv::Mat cv_image02;//这里不加图像大小 //注意道路数据uu和umm\um图像的大小是不一样的，有的是1241＊367  有的是1242＊375  注意的大小是不确定的，访问的时候需要注意


    //临时变量
    ros::Publisher map_pub           = nh_priv.advertise<pcl::PointCloud<pcl::PointXYZ> >  ("hdl64e", 1, true);



    if (vm.count("help"))
    {
        cout << desc << endl;

        cout << "kitti_road needs a directory tree like the following:" << endl;
        cout << "└── ROAD                                  " << endl;
        cout << "    ├── data_road                         " << endl;
        cout << "    │   └── data_road                     " << endl;
        cout << "    │       └── testing                   " << endl;
        cout << "    │       │    └── calib                " << endl;
        cout << "    │       │    └── image_2              " << endl;
        cout << "    │       │    └── output               " << endl;
        cout << "    │       └── training                  " << endl;
        cout << "    │            └── calib                " << endl;
        cout << "    │            └── image_2              " << endl;
        cout << "    │            └── gt_image_2           " << endl;
        cout << "    │            └── output               " << endl;
        cout << "    │── data_road_velodyne                " << endl;
        cout << "            └── testing                   " << endl;
        cout << "            │    └── velodyne             " << endl;
        cout << "            └── training                  " << endl;
        cout << "                 └── velodyne             " << endl<< endl;

        return 1;
    }



    dir_root             = options.path;
    dir_image02          = options.path;
    dir_calib            = options.path;
    dir_velodyne_points  = options.path;
    dir_output           = options.path;


    if(options.train_test)
    {
		(*(options.path.end() - 1) != '/' ? dir_root            = options.path + "/"                      : dir_root            = options.path);
		(*(options.path.end() - 1) != '/' ? dir_image02         = options.path + "/data_road/data_road/training/image_2/"        : dir_image02         = options.path + "data_road/data_road/training/image_2/");
		(*(options.path.end() - 1) != '/' ? dir_calib           = options.path + "/data_road/data_road/training/calib/"        : dir_calib         = options.path + "data_road/data_road/training/calib/");
		(*(options.path.end() - 1) != '/' ? dir_velodyne_points = options.path + "/data_road_velodyne/training/velodyne/" : dir_velodyne_points = options.path + "data_road_velodyne/training/velodyne/");
		(*(options.path.end() - 1) != '/' ? dir_output          = options.path + "/data_road/data_road/training/output/"        : dir_output         = options.path + "data_road/data_road/training/output/");
    }
    else
    {
		(*(options.path.end() - 1) != '/' ? dir_root            = options.path + "/"                      : dir_root            = options.path);
		(*(options.path.end() - 1) != '/' ? dir_image02         = options.path + "/data_road/data_road/testing/image_2/"        : dir_image02         = options.path + "data_road/data_road/testing/image_2/");
		(*(options.path.end() - 1) != '/' ? dir_calib           = options.path + "/data_road/data_road/testing/calib/"        : dir_calib         = options.path + "data_road/data_road/testing/calib/");
		(*(options.path.end() - 1) != '/' ? dir_velodyne_points = options.path + "/data_road_velodyne/testing/velodyne/" : dir_velodyne_points = options.path + "data_road_velodyne/testing/velodyne/");
		(*(options.path.end() - 1) != '/' ? dir_output          = options.path + "/data_road/data_road/testing/output/"        : dir_output         = options.path + "data_road/data_road/testing/output/");
    }

    // Check all the directories
    if ((opendir(dir_image02.c_str()) == NULL) || (opendir(dir_velodyne_points.c_str()) == NULL) || (opendir(dir_calib.c_str()) == NULL)|| (opendir(dir_output.c_str()) == NULL))
    {
        ROS_ERROR("Incorrect tree directory , use --help for details");
        nh_priv.shutdown();
        return -1;
    }
    else
    {
        ROS_INFO_STREAM ("Checking directories...");
        ROS_INFO_STREAM (options.path << "\t[OK]");
    }

    //count elements in the folder

    //cv::namedWindow("obstale_mask_3C",cv::WINDOW_NORMAL);
    //cv::namedWindow("paintVoronoi",cv::WINDOW_NORMAL);
    cv::namedWindow("result_3",cv::WINDOW_NORMAL);
    cv::namedWindow("result_32",cv::WINDOW_NORMAL);
    cv::namedWindow("result_33",cv::WINDOW_NORMAL);



        dir = opendir(dir_image02.c_str());
        while ((ent = readdir(dir))&& ros::ok())
        {
            //skip . & ..
            len = strlen (ent->d_name);
            //cout<<"ent->d_name:"<<ent->d_name<<endl;
            /*
             * 输出是下面这效果
             *  ent->d_name:.
				ent->d_name:..
				ent->d_name:umm_000041.png
				ent->d_name:uu_000085.png
             *
             * */

            //skip . & ..
            if (len > 2)//满足这个条件才是图片文件的名称
            {
            	 total_entries++;
            	 //下面对文件名称进行处理 参考http://www.jb51.net/article/55954.htm
            	  vector<string> vStr;
            	  boost::split( vStr, ent->d_name, boost::is_any_of( "_." ), boost::token_compress_on );
            	  string tag;//um uu umm
            	  string num;//数字
            	  string end;//后缀
            	  int seg_num = 0;
            	  for( vector<string>::iterator it = vStr.begin(); it != vStr.end(); ++ it )
            	  {
            		  if(seg_num==0)
            		  {
            			  tag = *it;
            		  	  //cout<<"tag:" << tag << endl;
            		  }
            		  if(seg_num==1)
            		  {
            			  num = *it;
            		  	  //cout<<"num:" << num << endl;
            		  }
            		  if(seg_num==2)
            		  {
            			  end = *it;
            		  	  //cout<<"end:" << end << endl;
            		  }

            		  seg_num++;

            	  }

            	  full_filename_image02 = dir_image02 + tag + "_" + num + ".png";
            	  //cout<<"full_filename_image02:"<<full_filename_image02<<endl;
            	  full_filename_calib = dir_calib + tag + "_" + num + ".txt";
            	  //cout<<"full_filename_calib:"<<full_filename_calib<<endl;
            	  full_filename_output = dir_output +  tag + "_road_" +  num + ".png";
            	  //cout<<"full_filename_output:"<<full_filename_output<<endl;
            	  full_filename_output_3C_mask = dir_output + "3C/" + tag + "_road_3C_maske_" +  num + ".png";
            	  full_filename_velodyne = dir_velodyne_points + tag + "_" + num + ".bin";
            	  //cout<<"full_filename_velodyne:"<<full_filename_velodyne<<endl;
            	  cv_image02 = cv::imread(full_filename_image02, CV_LOAD_IMAGE_UNCHANGED);
            	  //cv::imshow("image02",cv_image02);

            	    imageSize.width = cv_image02.cols;//注意道路数据uu和umm\um图像的大小是不一样的，有的是1241＊367  有的是1242＊375
            	    imageSize.height = cv_image02.rows;
            	    imageSize_3.width = cv_image02.cols;
            	    imageSize_3.height = cv_image02.rows*3;
               	  cv::Mat hsv_;//这个不能声明大小，因为读取的图片大小不确定
               	  cv::cvtColor(cv_image02, hsv_, cv::COLOR_BGR2HSV);//注意传入图像是BGR8
               	  cv::vector< cv::Mat >hsv(hsv_.channels());
               	  split(hsv_, hsv);//访问h通道hsv[0]



            	  pcl::PointCloud<pcl::PointXYZI>::Ptr velodyne_points (new pcl::PointCloud<pcl::PointXYZI>);
            	  pcl::PointCloud<pcl::PointXYZI>::Ptr velodyne_points_remove (new pcl::PointCloud<pcl::PointXYZI>);
            	    fstream input(full_filename_velodyne.c_str(), ios::in | ios::binary);
            	    if (!input.good())
            	    {
            	        ROS_ERROR_STREAM ( "Could not read file: " << full_filename_velodyne );
            	        return 0;
            	    }
            	    else
            	    {
            	        ROS_DEBUG_STREAM ("reading " << full_filename_velodyne);
            	        input.seekg(0, ios::beg);



            	        int i;
            	        for (i = 0; input.good() && !input.eof(); i++)
            	        {
            	            pcl::PointXYZI point;
            	            input.read((char *) &point.x, 3 * sizeof(float));
            	            input.read((char *) &point.intensity, sizeof(float));
            	            velodyne_points->push_back(point);
            	        }
            	        input.close();

            	        //发布读取的点云数据
            	       /* sensor_msgs::PointCloud2 pc2;
            	        pc2.header.frame_id = "base_link"; //ros::this_node::getName();
            	        pc2.header.stamp = ros::Time::now();
            	        velodyne_points->header = pcl_conversions::toPCL(pc2.header);
            	        map_pub.publish(velodyne_points);*/

            	        double P2[12],R0_rect[9],Tr_velo_to_cam[12],Tr_cam_to_road[12];
            	    if(!getCalibration_kitti_road(full_filename_calib, P2,  R0_rect, Tr_velo_to_cam ,  Tr_cam_to_road))
            	        {
            	                    ROS_ERROR_STREAM("Error reading calibration");
            	                    nh_priv.shutdown();
            	                    return -1;
            	        }
            	    //cout<< "P2: "<<P2[0]<<" "<<P2[1]<<" "<<P2[2]<<" "<<P2[3]<<" "<<P2[4]<<" "<<P2[5]<<" "<<P2[6]<<" "<<P2[7]<<" "<<P2[8]<<" "<<P2[9]<<" "<<P2[10]<<" "<<P2[11]<<endl;
            	    //cout<< "R0_rect: "<<R0_rect[0]<<endl;
            	   // cout<< "Tr_velo_to_cam: "<<Tr_velo_to_cam[0]<<endl;
            	   // cout<< "Tr_cam_to_road: "<<Tr_cam_to_road[0]<<endl;

            	    matrix_33_P2<<P2[0],P2[1],P2[2],P2[4],P2[5],P2[6],P2[8],P2[9],P2[10];
            	    vector_T_for_P2 <<P2[3],P2[7],P2[11];

            	    matrix_33_R0_rect<<R0_rect[0],R0_rect[1],R0_rect[2],R0_rect[3],R0_rect[4],R0_rect[5],R0_rect[6],R0_rect[7],R0_rect[8];

            	    matrix_33_Tr_velo_to_cam<<Tr_velo_to_cam[0],Tr_velo_to_cam[1],Tr_velo_to_cam[2],Tr_velo_to_cam[4],Tr_velo_to_cam[5],Tr_velo_to_cam[6],Tr_velo_to_cam[8],Tr_velo_to_cam[9],Tr_velo_to_cam[10];
            	    vector_T_for_Tr_velo_to_cam <<Tr_velo_to_cam[3],Tr_velo_to_cam[7],Tr_velo_to_cam[11];

            	    matrix_33_Tr_cam_to_road<<Tr_cam_to_road[0],Tr_cam_to_road[1],Tr_cam_to_road[2],Tr_cam_to_road[4],Tr_cam_to_road[5],Tr_cam_to_road[6],Tr_cam_to_road[8],Tr_cam_to_road[9],Tr_cam_to_road[10];
            	    vector_T_for_Tr_cam_to_road <<Tr_cam_to_road[3],Tr_cam_to_road[7],Tr_cam_to_road[11];


            	    matrix_33_Tr_road_to_cam = matrix_33_Tr_cam_to_road.transpose();//这个road坐标系不明确
            	    vector_T_for_Tr_road_to_cam = - matrix_33_Tr_road_to_cam * vector_T_for_Tr_cam_to_road;

            	    //地平线计算
            	    Eigen::Vector3d horizontal_point;
            	    horizontal_point<<20000,0,-1.73;//把它当成是激光雷达坐标系下
            	    horizontal_point = matrix_33_Tr_velo_to_cam * horizontal_point + vector_T_for_Tr_velo_to_cam;
            	    horizontal_point = matrix_33_R0_rect * horizontal_point;
            	    horizontal_point = matrix_33_P2 * horizontal_point + vector_T_for_P2;
            	    int horizontal_point_u = horizontal_point(0,0) / horizontal_point(2,0);
            	    int horizontal_point_v = horizontal_point(1,0) / horizontal_point(2,0);





            	    //上面实现了数据的读取 下面进行算法的计算

      	/*		  pcl::RadiusOutlierRemoval<pcl::PointXYZI> outrem;// 创建滤波器
      			  outrem.setInputCloud(velodyne_points);              //设置输入点云
      			  outrem.setRadiusSearch(0.3);              //设置在0.005半径的范围内找邻近点，按30万个点每秒，10Hz，一圈30000点，那么30米的半径，相邻两点之间的距离是2*3.14159*30/30000=0.0063，也就是至少能保证30米范围内正常扫描的点，能滤掉里面噪声点
      			  outrem.setMinNeighborsInRadius(2);       //设置查询点的邻近点集数小于2的删除

      			  outrem.filter (*velodyne_points);//执行条件滤波，存储结果到cloud_filtered

*/



      			  std::vector<int> indices;
      			  pcl::removeNaNFromPointCloud(*velodyne_points, *velodyne_points, indices);

      			clock_t time_stt1 = clock();
          	    pcl::StatisticalOutlierRemoval<pcl::PointXYZI> sor;// 创建滤波器对象

          	    sor.setInputCloud(velodyne_points);                        //设置呆滤波的点云

          	    sor.setMeanK(10);                                //设置在进行统计时考虑查询点邻近点数

          	    sor.setStddevMulThresh(10.0);                    //设置判断是否为离群点的阈值
          	    sor.setNegative(false);
          	    sor.filter(*velodyne_points);

          	    sor.setNegative(true);
          	    sor.filter(*velodyne_points_remove);//这个操作50个点需要1秒多，10个点需要0.5秒   这个操作不实时，就不用了

          	  cout<<"StatisticalOutlierRemoval time used is "<<1000*(clock()-time_stt1)/(double)CLOCKS_PER_SEC<<"ms"<<endl;//这个函数总共需要这么多时间


    			  int cloudSize = velodyne_points->points.size();
    			  //cout<<"test !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" <<endl;
    			  pcl::PointXYZI point;
    			  pcl::PointCloud<pcl::PointXYZI> result_points;//转到摄像头坐标系的点云
    			  pcl::PointCloud<pcl::PointXYZHSV>::Ptr result_delaunay_ground_points (new pcl::PointCloud<pcl::PointXYZHSV>);//点云类型从原来的PointXYZI换成PointXYZHSV 为了将对应的图像坐标存起来
    			  pcl::PointCloud<pcl::PointXYZHSV>::Ptr result_delaunay_obstacle_points (new pcl::PointCloud<pcl::PointXYZHSV>);//存储顺序是三维坐标x,y,z,激光雷达强度信息，然后是对应的图像坐标信息x,y
    			  pcl::PointCloud<pcl::PointXYZHSV>::Ptr cloud_ransac_seg_ground_in_range(new pcl::PointCloud<pcl::PointXYZHSV>);//用来存储RANSAC拟合的地面点云
    			  pcl::PointCloud<pcl::PointXYZHSV>::Ptr cloud_ransac_seg_ground_out_range(new pcl::PointCloud<pcl::PointXYZHSV>);
    			  pcl::PointCloud<pcl::PointXYZHSV>::Ptr cloud_ransac_seg_not_ground_in_range(new pcl::PointCloud<pcl::PointXYZHSV>);//用来存储RANSAC拟合的非地面点云
    			  pcl::PointCloud<pcl::PointXYZHSV>::Ptr cloud_ransac_seg_not_ground_out_range(new pcl::PointCloud<pcl::PointXYZHSV>);//用来存储RANSAC拟合的非地面点云
    			  pcl::PointCloud<pcl::PointXYZHSV>::Ptr cloud_used_to_expand(new pcl::PointCloud<pcl::PointXYZHSV>);
    				cv::Mat project_left_img = cv::Mat::zeros(imageSize, CV_32FC4);//用于存放三维坐标和强度信息 由于这些图片采用采样的坐标关系，所以需要多的数据量再声明一个就行，不需要增加这里的通道数
    				cv::Mat mask = cv::Mat::zeros(imageSize, CV_8UC1);//投影掩码
    				cv::Mat result_3(imageSize_3, CV_8UC3);
    				cv::Mat result_32(imageSize_3, CV_8UC1);
    				cv::Mat result_33(imageSize_3, CV_8UC3);

    				cv::Mat obstale_mask_3C = cv::Mat::zeros(imageSize, CV_8UC3);//障碍掩码，未知区域是0，障碍区域是（255，0，0）地面区域（0，255，0），只是用于可视化
    				cv::Mat obstale_mask_1C = cv::Mat::zeros(imageSize, CV_8UC1);//未知是0，障碍是255，地面是100
    				cv::Mat obstale_mask_3C_inv(imageSize, CV_8UC3);
    				cv::bitwise_not(obstale_mask_3C, obstale_mask_3C_inv);
    				//cv::imshow("obstale_mask_3C_inv",obstale_mask_3C_inv);

    				cv::Rect rect(0,0,imageSize.width,imageSize.height);

    				cv::Mat img_3C(rect.size(), CV_8UC3);//画彩色分割结果，用于查看
    				cv::Mat img_1C(rect.size(), CV_8UC1);//画单通道分割结果，用于评估

    				img_3C = cv::Scalar::all(0);
    				img_1C = cv::Scalar::all(0);
    				cv::Scalar  delaunay_color(255,255,255);


    				clock_t time_stt = clock();
    				clock_t time_stt2 = clock();
    				//创建Delaunay剖分
    				cv::Subdiv2D subdiv(rect);
    				cv::Point2f point_to_inset;
    				int inset_subdiv_point_num=0;
    				Eigen::Vector3d  vector_point,vector_point_temp;//临时变量

    				  for (int i = 0; i < cloudSize; i++) {
    					 // cv::Mat matpoint = (cv::Mat_<double>(3,1)<<laserCloudIn.points[i].x,laserCloudIn.points[i].y,laserCloudIn.points[i].z);

    					  vector_point<<velodyne_points->points[i].x,velodyne_points->points[i].y,velodyne_points->points[i].z;//head_ground坐标系下
    					  vector_point = matrix_33_Tr_velo_to_cam * vector_point + vector_T_for_Tr_velo_to_cam;//从激光雷达转到(non-rectified) camera coordinates
    					  vector_point = matrix_33_R0_rect * vector_point;

    					  point.x = vector_point(0,0);//在摄像头坐标系下
    					  point.y = vector_point(1,0);
    					  point.z = vector_point(2,0);

    					  point.intensity = velodyne_points->points[i].intensity;//注意

    					 // result_points.push_back(point);

    		//上面将点云从head_ground转到左摄像头，下面投到成像平面
    					  if(point.z<=0)//注意z小于0时也能得到对应的图像坐标，事实上又是不可能看见的，在左摄像头参考系
    						 {
    							continue;

    						 }
    					  if ((velodyne_points->points[i].x*velodyne_points->points[i].x+velodyne_points->points[i].y*velodyne_points->points[i].y+velodyne_points->points[i].z*velodyne_points->points[i].z)>3600)
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


    										point_to_inset.x = u;//注意顺序
    										point_to_inset.y = v;
    										subdiv.insert(point_to_inset);
    										inset_subdiv_point_num++;

    								}

    				  }

    				  cout<<"creating Delaunay time used is "<<1000*(clock()-time_stt2)/(double)CLOCKS_PER_SEC<<"ms"<<endl;//这个函数总共需要这么多时间
    				  //这个需要50几毫秒

      				clock_t time_stt3 = clock();

      				pcl::KdTreeFLANN<pcl::PointUV>::Ptr kdtreeImagePoint(new pcl::KdTreeFLANN<pcl::PointUV>());
      				 pcl::PointCloud<pcl::PointUV>::Ptr ImagePoint(new pcl::PointCloud<pcl::PointUV>());
      				 int inset_kd_tree_point_num=0;
      				  for (int i = 0; i < cloudSize; i++) {
      					 // cv::Mat matpoint = (cv::Mat_<double>(3,1)<<laserCloudIn.points[i].x,laserCloudIn.points[i].y,laserCloudIn.points[i].z);

      					  vector_point<<velodyne_points->points[i].x,velodyne_points->points[i].y,velodyne_points->points[i].z;//head_ground坐标系下
      					  vector_point = matrix_33_Tr_velo_to_cam * vector_point + vector_T_for_Tr_velo_to_cam;//从激光雷达转到(non-rectified) camera coordinates
      					  vector_point = matrix_33_R0_rect * vector_point;

      					  point.x = vector_point(0,0);//在摄像头坐标系下
      					  point.y = vector_point(1,0);
      					  point.z = vector_point(2,0);

      					  point.intensity = velodyne_points->points[i].intensity;//注意

      					  //result_points.push_back(point);

      		//上面将点云从head_ground转到左摄像头，下面投到成像平面
      					  if(point.z<=0)//注意z小于0时也能得到对应的图像坐标，事实上又是不可能看见的，在左摄像头参考系
      						 {
      							continue;

      						 }
      					  if ((velodyne_points->points[i].x*velodyne_points->points[i].x+velodyne_points->points[i].y*velodyne_points->points[i].y+velodyne_points->points[i].z*velodyne_points->points[i].z)>3600)
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

      			           	            pcl::PointUV point;
      			           	            point.u = u;
      			           	       	    point.v = v;
      			           	            ImagePoint->push_back(point);


      			           	       	    inset_kd_tree_point_num++;

      								}

      				  }
      				  kdtreeImagePoint->setInputCloud(ImagePoint);
      				  cout<<"creating kd-tree time used is "<<1000*(clock()-time_stt3)/(double)CLOCKS_PER_SEC<<"ms"<<endl;//这个函数总共需要这么多时间
      				  //这个只需要6、7毫秒


					  inset_subdiv_point_num = inset_subdiv_point_num+4;//根据这个好像不对，遍历边的最后会出现很大的边如：edge_num1040695090，或者出现负的边 edge_num-1181446616
					  cout<<"inset_subdiv_point_num: "<<inset_subdiv_point_num<<endl;	  	  	  	  	  	  	  	  	  	  	  	  	  //猜测可能原因是因为遮挡有些点投到了同一个位置，这个得处理一下！！！！可以考虑移一个像素点(现在是这么处理的)，或者取高度高的点
					  int edge_num;


				////遍历判断是不是障碍物
					  clock_t time_stt4 = clock();
					  bool search_state_flag=true;
					  int expand_point_num = 0;
					  cout<<"inset_subdiv_point_num: "<<inset_subdiv_point_num<<endl;
					  for(int i=4;i<inset_subdiv_point_num ;i++)//遍历判断是不是障碍物   这样遍历很快，那先对点定位然后遍历呢？
					  {

							cv::Point2f outer_vtx_from_edge;
							int edge_num;
							outer_vtx_from_edge =  subdiv.getVertex(i,&edge_num);//得到一条以给定点(点给的是索引号)为起点的线

							/*if (edge_num<16||!search_state_flag)//前16条边是虚边||edge_num>=edge_size+16
								{
									cout<<"edge search error!"<<" edge_num:"<<edge_num<<" search_state_flag:"<<search_state_flag<<endl;
									break;
								}*/

							//circle(img, outer_vtx_from_edge, 3, Scalar(), CV_FILLED, CV_AA, 0);//画出线的起始点
							int temp = edge_num;
							cv::vector<cv::Point2f> pt(2);
							 pcl::PointXYZ org_point_3d,dst_point_3d;
							 //cout<<"inset_subdiv_point_num:"<<inset_subdiv_point_num<<"   i org_point_3d.x:"<<i<<"  edge_num"<<edge_num<<endl;
							bool obstacle_flag = false;


							do
							{

								int index_d = subdiv.edgeDst(temp, &pt[1]);
								/*if (index_d<0||index_d>=inset_subdiv_point_num)
									{
										search_state_flag = false;//不正常
										cout<<"vertex search error!"<<"index_d: "<<index_d<<endl;
										break;
									}*/
								if (index_d<4)//这条边可能连到虚点
								{
									temp = subdiv.nextEdge(temp);
									continue;
								}
								expand_point_num++;
								//这里加上下一层连接的判断  注意第二层可能回到本身节点

      			    		  cv::Point2f outer_vtx_from_edge1;
      			    		  int edge_num_temp1;
      			    		  outer_vtx_from_edge1 =  subdiv.getVertex(index_d,&edge_num_temp1);//得到一条以给定点为起点的线
      			    		  //cout<<"outer_vtx_from_edge: "<<outer_vtx_from_edge<<endl;
      			    		  //cout<<"edge_num_temp: "<<edge_num_temp<<endl;

      							int temp2 = edge_num_temp1;
      							cv::vector<cv::Point2f> pt2(2);
      							do
      							{
      								int index_d2 = subdiv.edgeDst(temp2, &pt2[1]);//连接边的另一个端点
      								//cout<<"pt[1]: "<<pt[1]<<endl;
      								if (index_d2<4)//这条边可能连到虚点
      								{
      									temp2 = subdiv.nextEdge(temp2);
      									//cout<<"temp: "<<temp<<endl;
      									continue;
      								}


    								//第三层连接：目前看加第三层效果还没有提升的情况

            			    		  cv::Point2f outer_vtx_from_edge2;
            			    		  int edge_num_temp2;
            			    		  outer_vtx_from_edge2 =  subdiv.getVertex(index_d2,&edge_num_temp2);//得到一条以给定点为起点的线
            			    		  //cout<<"outer_vtx_from_edge: "<<outer_vtx_from_edge<<endl;
            			    		  //cout<<"edge_num_temp: "<<edge_num_temp<<endl;

            							int temp3 = edge_num_temp2;
            							cv::vector<cv::Point2f> pt3(2);
            							do
            							{
            								int index_d3 = subdiv.edgeDst(temp3, &pt3[1]);//连接边的另一个端点
            								//cout<<"pt[1]: "<<pt[1]<<endl;
            								if (index_d3<4)//这条边可能连到虚点
            								{
            									temp3 = subdiv.nextEdge(temp3);
            									//cout<<"temp: "<<temp<<endl;
            									continue;
            								}


          								expand_point_num++;
          								temp3 = subdiv.nextEdge(temp3);
          								        								//cout<<"temp: "<<temp<<endl;

          							}while(edge_num_temp2 != temp3);




      								expand_point_num++;
      								temp2 = subdiv.nextEdge(temp2);
      								        								//cout<<"temp: "<<temp<<endl;

      							}while(edge_num_temp1 != temp2);



								//line(img, pt[0], pt[1], delaunay_color, 1, CV_AA, 0);
								temp = subdiv.nextEdge(temp);
								//cout<<"!!!!!!!!!!!!!!!!!!!!!!!temp"<<temp<<endl;

							}while(edge_num != temp);//查找所有和给定点相连的边，判断障碍物属性


					  }

						cout<<"expand_point_num/(inset_subdiv_point_num-4): "<<expand_point_num/(inset_subdiv_point_num-4)<<endl;
						cout<<"Delaunay expanded time used is "<<1000*(clock()-time_stt4)/(double)CLOCKS_PER_SEC<<"ms"<<endl;//这个函数总共需要这么多时间


						int kd_tree_expand_num = expand_point_num/(inset_subdiv_point_num-4);
						 clock_t time_stt5 = clock();
						 std::vector<int> pointSearchInd;//kdtree查找最近点时记录索引和距离的平方
						 std::vector<float> pointSearchSqDis;
						 pcl::PointUV pointSel;

						for(int i =0;i<ImagePoint->size() ;i++)
						{

							pointSel = ImagePoint->points[i];
							kdtreeImagePoint->nearestKSearch(pointSel, kd_tree_expand_num, pointSearchInd, pointSearchSqDis);

						}
						cout<<"ImagePoint->size(): "<<ImagePoint->size()<<endl;
						cout<<"kd-tree expanded time used is "<<1000*(clock()-time_stt5)/(double)CLOCKS_PER_SEC<<"ms"<<endl;//这个函数总共需要这么多时间


						clock_t time_stt6 = clock();
						int expand_point_num1=0;
  			    	  for(int i =0;i<ImagePoint->size();i++)
  			    	  {

  			    		  cv::Point2f fp(ImagePoint->points[i].u,ImagePoint->points[i].v);
  			    		  //cv::Point2f fp(198,642);//注意顺序是x,y
  			    		  //cout<<"fp: "<<fp<<endl;
  			    		  int e0=0, vertex=0;
  			    		  subdiv.locate(fp, e0, vertex); //找到这个点的顶点索引和一条边，下面找到和他连接的点 注意这条边可能不是以给定点为起点！！！！
  			    		  //cout<<"e0: "<<e0<<endl;//这个输出是0，//前16条边是虚边

  			    		  //验证上面是不是对的,证明vertex得到的就是给定点的索引
  			    		  cv::Point2f outer_vtx_from_edge;
  			    		  int edge_num_temp;
  			    		  outer_vtx_from_edge =  subdiv.getVertex(vertex,&edge_num_temp);//得到一条以给定点为起点的线
  			    		  //cout<<"outer_vtx_from_edge: "<<outer_vtx_from_edge<<endl;
  			    		  //cout<<"edge_num_temp: "<<edge_num_temp<<endl;

  							int temp = edge_num_temp;
  							cv::vector<cv::Point2f> pt(2);
							do
							{

								int index_d = subdiv.edgeDst(temp, &pt[1]);
								/*if (index_d<0||index_d>=inset_subdiv_point_num)
									{
										search_state_flag = false;//不正常
										cout<<"vertex search error!"<<"index_d: "<<index_d<<endl;
										break;
									}*/
								if (index_d<4)//这条边可能连到虚点
								{
									temp = subdiv.nextEdge(temp);
									continue;
								}
								expand_point_num1++;
								//这里加上下一层连接的判断  注意第二层可能回到本身节点

      		    		  cv::Point2f outer_vtx_from_edge1;
      			    		  int edge_num_temp1;
      			    		  outer_vtx_from_edge1 =  subdiv.getVertex(index_d,&edge_num_temp1);//得到一条以给定点为起点的线
      			    		  //cout<<"outer_vtx_from_edge: "<<outer_vtx_from_edge<<endl;
      			    		  //cout<<"edge_num_temp: "<<edge_num_temp<<endl;

      							int temp2 = edge_num_temp1;
      							cv::vector<cv::Point2f> pt2(2);
      							do
      							{
      								int index_d2 = subdiv.edgeDst(temp2, &pt2[1]);//连接边的另一个端点
      								//cout<<"pt[1]: "<<pt[1]<<endl;
      								if (index_d2<4)//这条边可能连到虚点
      								{
      									temp2 = subdiv.nextEdge(temp2);
      									//cout<<"temp: "<<temp<<endl;
      									continue;
      								}


    								//第三层连接：目前看加第三层效果还没有提升的情况

            			    		  cv::Point2f outer_vtx_from_edge2;
            			    		  int edge_num_temp2;
            			    		  outer_vtx_from_edge2 =  subdiv.getVertex(index_d2,&edge_num_temp2);//得到一条以给定点为起点的线
            			    		  //cout<<"outer_vtx_from_edge: "<<outer_vtx_from_edge<<endl;
            			    		  //cout<<"edge_num_temp: "<<edge_num_temp<<endl;

            							int temp3 = edge_num_temp2;
            							cv::vector<cv::Point2f> pt3(2);
            							do
            							{
            								int index_d3 = subdiv.edgeDst(temp3, &pt3[1]);//连接边的另一个端点
            								//cout<<"pt[1]: "<<pt[1]<<endl;
            								if (index_d3<4)//这条边可能连到虚点
            								{
            									temp3 = subdiv.nextEdge(temp3);
            									//cout<<"temp: "<<temp<<endl;
            									continue;
            								}


            								expand_point_num1++;
          								temp3 = subdiv.nextEdge(temp3);
          								        								//cout<<"temp: "<<temp<<endl;

          							}while(edge_num_temp2 != temp3);




            							expand_point_num1++;
      								temp2 = subdiv.nextEdge(temp2);
      								        								//cout<<"temp: "<<temp<<endl;

      							}while(edge_num_temp1 != temp2);



								//line(img, pt[0], pt[1], delaunay_color, 1, CV_AA, 0);
								temp = subdiv.nextEdge(temp);
								//cout<<"!!!!!!!!!!!!!!!!!!!!!!!temp"<<temp<<endl;

							}while(edge_num_temp != temp);//查找所有和给定点相连的边，判断障碍物属性


					  }

						cout<<"expand_point_num1/ImagePoint->size(): "<<expand_point_num1/ImagePoint->size()<<endl;
						cout<<"locate Delaunay expanded time used is "<<1000*(clock()-time_stt6)/(double)CLOCKS_PER_SEC<<"ms"<<endl;//这个函数总共需要这么多时间






















            	}


            cv::waitKey(10);//临时

       }

    }
        closedir (dir);
        cout<<"total_entries"<<total_entries<<endl;
        ROS_INFO_STREAM("Done!");
        cv::destroyAllWindows();
        nh_priv.shutdown();

        return 0;


}

