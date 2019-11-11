//之前的版本kitti_road_detection_by_lidar是通过先根据三角剖分分析障碍物属性，
//然后通过ransac平面拟合来判断点是不是属于地面，然后根据三角剖分连接属性扩展得到道路区域
//这里将算法细化  构建三角剖分（看kd-tree的时间消耗，然后看要不要用kd-tree来查找邻域点来判断障碍物属性，已经验证kd-tree（二维图像坐标）对于64线的激光雷达数据
//的构建时间大概是6到9毫秒 ，而对应的三角剖分需要40几到90几毫秒）
//，先小区域拟合平面，然后用来扩展
//查找速度分析（注意这里的三角剖分查找是根据索引查找的，这个应该比对于给定点需要先定位然后查找要快,实验证明定位速度很快，对所有投影点进行定位只需要3ms左右）(三角剖分对图像上投影点进行两层扩展（大概每个点搜索42个点）遍历需要大概10ms左右，而kd-tree搜索
//三角剖分两层连接的平均数，遍历所有投影点需要70ms左右)（三层连接，对应每个点需要查找大约270个点，三角为70ms左右，而kd-tree为700ms左右）
//（得出结论，三角剖分查找时间消耗比较少，时间和查找点数呈线性的关系，而kd-tree查找是比较耗时的，
//而且是非线性的,两层查找时是7倍，三层时是10倍时间，也就是查找的点数越多，kd_tree越没有优势，本身查找相对已经很慢了，所有采用三角剖分只是构建的时候需要一点时间，查找是很快速的）
//为了实时只能遍历两层连接
//一边扩展一边拟合小区域平面来判断点是不是在路面（如果有提升可以加上三角剖分的地面属性判断，阈值留大点去掉一部分显然不可能是道路的点）
//基于图像的腐蚀膨胀等后处理大概在10ms内


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
    double ransac_DistanceThreshold_local;
    double near_distanse_thresh;
    int  terminate_expand_num;
    double min_edge_distance;

    double threshold_low,threshold_up;//绿色：35到77  青色：78到99 蓝色：100到124
    double threshold_sv;//43到 255
    int erode_size;
    int local_ransac_min_num;
    int local_ransac_max_num;


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
    nh_priv.param("ransac_DistanceThreshold_local", ransac_DistanceThreshold_local, 0.01);
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
    nh_priv.param("local_ransac_min_num", local_ransac_min_num, 10);
    nh_priv.param("local_ransac_max_num", local_ransac_max_num, 50);

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
    cv::namedWindow("test1",cv::WINDOW_NORMAL);
    cv::namedWindow("test2",cv::WINDOW_NORMAL);
    cv::namedWindow("test3",cv::WINDOW_NORMAL);
    cv::namedWindow("test4",cv::WINDOW_NORMAL);
    cv::namedWindow("test5",cv::WINDOW_NORMAL);



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
            	  //pcl::PointCloud<pcl::PointXYZI>::Ptr velodyne_points_remove (new pcl::PointCloud<pcl::PointXYZI>);
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

            	    //数据读取完毕，算法操作

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




    			  pcl::PointCloud<pcl::PointXYZHSV>::Ptr project_to_image_points (new pcl::PointCloud<pcl::PointXYZHSV>);
    			  pcl::PointCloud<pcl::PointXYZHSV>::Ptr cloud_used_to_expand(new pcl::PointCloud<pcl::PointXYZHSV>);


    			    cv::Mat project_left_img = cv::Mat::zeros(imageSize, CV_32FC4);//用于存放三维坐标和强度信息 由于这些图片采用采样的坐标关系，所以需要多的数据量再声明一个就行，不需要增加这里的通道数



    			    cv::Mat mask = cv::Mat::zeros(imageSize, CV_8UC1);//投影掩码
    				cv::Mat result_3(imageSize_3, CV_8UC3);
    				cv::Mat result_32(imageSize_3, CV_8UC1);
    				cv::Mat result_33(imageSize_3, CV_8UC3);

    				cv::Mat obstale_mask_3C = cv::Mat::zeros(imageSize, CV_8UC3);//障碍掩码，未知区域是0，障碍区域是（255，0，0）地面区域（0，255，0），只是用于可视化



    				cv::Mat obstale_mask_1C = cv::Mat::zeros(imageSize, CV_8UC1);//用于扩展的时候状态确定  未知是0，障碍是255，地面是100
    				cv::Mat obstale_mask_3C_inv(imageSize, CV_8UC3);//也是为了可视化，白色背景，点画的时候加粗
    				cv::bitwise_not(obstale_mask_3C, obstale_mask_3C_inv);
    				//cv::imshow("obstale_mask_3C_inv",obstale_mask_3C_inv);

    				cv::Rect rect(0,0,imageSize.width,imageSize.height);

    				cv::Mat img_3C(rect.size(), CV_8UC3);//画彩色分割结果，用于查看
    				cv::Mat img_1C(rect.size(), CV_8UC1);//画单通道分割结果，用于评估

    				img_3C = cv::Scalar::all(0);
    				img_1C = cv::Scalar::all(0);
    				cv::Scalar  delaunay_color(255,255,255);


    				clock_t time_stt = clock();
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

    					  result_points.push_back(point);

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

    									if (mask.at<uchar>(v, u)!=255)
    									{

    										project_left_img.at<cv::Vec4f>(v, u)[0] = velodyne_points->points[i].x;//在head_ground坐标系下
    										project_left_img.at<cv::Vec4f>(v, u)[1] = velodyne_points->points[i].y;//有网页总结说，mat变量的访问时间在release的模式下差不多
    										project_left_img.at<cv::Vec4f>(v, u)[2] = velodyne_points->points[i].z;
    										project_left_img.at<cv::Vec4f>(v, u)[3] = velodyne_points->points[i].intensity;
    										mask.at<uchar>(v, u) = 255;
    										point_to_inset.x = u;//注意顺序
    										point_to_inset.y = v;
    										subdiv.insert(point_to_inset);
    										inset_subdiv_point_num++;

    	    								pcl::PointXYZHSV point_temp;
    	    								point_temp.x = velodyne_points->points[i].x;
    	    								point_temp.y = velodyne_points->points[i].y;
    	    								point_temp.z = velodyne_points->points[i].z;
    	    								point_temp.h = velodyne_points->points[i].intensity;
    	    								point_temp.s = u;
    	    								point_temp.v = v;
    	    								project_to_image_points->push_back(point_temp);



    									}
    									else//考虑到有很多点投到了同样的图像坐标
    									{
    										for(int k =1;k<(imageSize.height-1);k++)
    										{
    											if (u>0&&v+k>0&&u<(imageSize.width-1)&&v+k<(imageSize.height-1)&&mask.at<uchar>(v+k, u)!=255)
    											{
    												int v_k = v+k;
    												project_left_img.at<cv::Vec4f>(v_k, u)[0] = velodyne_points->points[i].x;//在head_ground坐标系下
    												project_left_img.at<cv::Vec4f>(v_k, u)[1] = velodyne_points->points[i].y;
    												project_left_img.at<cv::Vec4f>(v_k, u)[2] = velodyne_points->points[i].z;
    												project_left_img.at<cv::Vec4f>(v_k, u)[3] = velodyne_points->points[i].intensity;
    												mask.at<uchar>(v_k, u) = 255;
    												point_to_inset.x = u;//注意顺序
    												point_to_inset.y = v_k;
    												subdiv.insert(point_to_inset);
    												inset_subdiv_point_num++;
    	    	    								pcl::PointXYZHSV point_temp;
    	    	    								point_temp.x = velodyne_points->points[i].x;
    	    	    								point_temp.y = velodyne_points->points[i].y;
    	    	    								point_temp.z = velodyne_points->points[i].z;
    	    	    								point_temp.h = velodyne_points->points[i].intensity;
    	    	    								point_temp.s = u;
    	    	    								point_temp.v = v_k;
    	    	    								project_to_image_points->push_back(point_temp);
    												break;
    											}

    										}
    									}
    								}

    				  }




    		//上面构建三角剖分并得到联合向量，由图像坐标可以得到三维点云坐标，由点云坐标也能得到图像坐标，再结合原始图像能得到颜色信息

              	    //TODO 拟合小范围地面用来进行扩展


              	  pcl::PointCloud<pcl::PointXYZHSV>::Ptr little_range_for_expanding_points (new pcl::PointCloud<pcl::PointXYZHSV>);

  					  for (int i = 0; i < project_to_image_points->size(); i++) {
  					         double temp_x = project_to_image_points->points[i].x;
  					         double temp_y = project_to_image_points->points[i].y;
  					         double temp_z = project_to_image_points->points[i].z;

  					         if(temp_x>box_x_min1&&temp_x<box_x_max1&&temp_y>box_y_min1&&temp_y<box_y_max1&&temp_z>box_z_min1&&temp_z<box_z_max1)
  					          		{
  					        	 	 	 little_range_for_expanding_points->push_back(project_to_image_points->points[i]);
  					          		}

  					    }

					    pcl::ModelCoefficients::Ptr coefficients_to_expand (new pcl::ModelCoefficients);
					    //inliers表示误差能容忍的点 记录的是点云的序号
					    pcl::PointIndices::Ptr inliers_to_expand (new pcl::PointIndices);
					    // 创建一个分割器
					    pcl::SACSegmentation<pcl::PointXYZHSV> seg_to_expand;
					    // Optional
					    seg_to_expand.setOptimizeCoefficients (true);
					    // Mandatory-设置目标几何形状
					    seg_to_expand.setModelType (pcl::SACMODEL_PLANE);
					    //分割方法：随机采样法
					    seg_to_expand.setMethodType (pcl::SAC_RANSAC);
					    //设置误差容忍范围
					    seg_to_expand.setDistanceThreshold (ransac_DistanceThreshold1);
					    //输入点云
					    seg_to_expand.setInputCloud (little_range_for_expanding_points);
					    //分割点云
					    seg_to_expand.segment (*inliers_to_expand, *coefficients_to_expand);//a * x + b * y + c * z = d 注意这个参数的第四位是-d 花了好多时间才发现

					    if (inliers_to_expand->indices.size () == 0)//具体inliers怎么使用，参考：http://www.pclcn.org/study/shownews.php?lang=cn&id=72
					      {
					        PCL_ERROR ("Could not estimate a planar model for the given dataset.");
					        return (-1);
					      }
					      pcl::ExtractIndices<pcl::PointXYZHSV> extract_to_expand;
					      extract_to_expand.setInputCloud(little_range_for_expanding_points);      //设置输入点云
					      extract_to_expand.setIndices(inliers_to_expand);                 //设置分割后的内点为需要提取的点集
					      extract_to_expand.setNegative(false);                  //设置提取内点而非外点
					      extract_to_expand.filter(*cloud_used_to_expand);
       				      for (int i = 0; i < cloud_used_to_expand->size(); i++) {//这些认为是地面点

        					    	  obstale_mask_3C.at<cv::Vec3b>(cloud_used_to_expand->points[i].v, cloud_used_to_expand->points[i].s)[0] = 0;//这个用于彩色可视化
        					    	  obstale_mask_3C.at<cv::Vec3b>(cloud_used_to_expand->points[i].v, cloud_used_to_expand->points[i].s)[1] = 255;
        					    	  obstale_mask_3C.at<cv::Vec3b>(cloud_used_to_expand->points[i].v, cloud_used_to_expand->points[i].s)[2] = 0;
        					    	  obstale_mask_1C.at<uchar>(cloud_used_to_expand->points[i].v, cloud_used_to_expand->points[i].s) = 255;//状态确定  这个用于扩展的时候状态确定  由于初始化的时候是0，这里先用255表示是地面点
        					    	  cv::circle(obstale_mask_3C_inv, cv::Point2f(cloud_used_to_expand->points[i].s,cloud_used_to_expand->points[i].v), 1, cv::Scalar(0,255,0), CV_FILLED, CV_AA, 0);


        					      }

       				   cout<<"before expand time used is "<<1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC<<"ms"<<endl;//这个函数总共需要这么多时间
       				clock_t time_stt2 = clock();
              	    //TODO 根据三角剖分连接情况通过两层连接地面拟合进行扩展
              	    //如果有提升可以加上三角剖分的地面属性判断，阈值留大点去掉一部分显然不可能是道路的点

    					  pcl::PointCloud<pcl::PointXYZHSV>::Ptr cloud_ransac_seg_ground_in_range_to_expanded_new(new pcl::PointCloud<pcl::PointXYZHSV>);
    					  pcl::PointCloud<pcl::PointXYZHSV>::Ptr cloud_ransac_seg_ground_in_range_expanded_ground(new pcl::PointCloud<pcl::PointXYZHSV>);//记录扩展得到的点
    					  pcl::PointCloud<pcl::PointXYZHSV>::Ptr cloud_ransac_seg_ground_in_range_expanded_ground_new(new pcl::PointCloud<pcl::PointXYZHSV>);
    			    	  pcl::PointCloud<pcl::PointXYZHSV>::Ptr cloud_ransac_seg_ground_in_range_expanded_not_ground(new pcl::PointCloud<pcl::PointXYZHSV>);
    			    	  cloud_ransac_seg_ground_in_range_to_expanded_new = cloud_used_to_expand;

    			    	  //cout<<"cloud_ransac_seg_ground_in_range_to_expanded_new->size(): "<<cloud_ransac_seg_ground_in_range_to_expanded_new->size()<<endl;
    			    	  //cout<<"cloud_ransac_seg_ground_in_range->size(): "<<cloud_ransac_seg_ground_in_range->size()<<endl;

    			    	  do
    			    	  {
        			    	  for(int i =0;i<cloud_ransac_seg_ground_in_range_to_expanded_new->size();i++)
        			    	  {

        			    		  cv::Point2f fp(cloud_ransac_seg_ground_in_range_to_expanded_new->points[i].s,cloud_ransac_seg_ground_in_range_to_expanded_new->points[i].v);
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


        							do//从给定点开始索引
        							{
        								int index_d = subdiv.edgeDst(temp, &pt[1]);//连接边的另一个端点  如果想一层一层的扩展，用一个向量存下这个索引就行了
        								//cout<<"pt[1]: "<<pt[1]<<endl;
        								if (index_d<4)//这条边可能连到虚点
        								{
        									temp = subdiv.nextEdge(temp);
        									//cout<<"temp: "<<temp<<endl;
        									continue;
        								}

        								//一层一层来，只要出现待处理点一次性解决  注意存在非地面点时加点扩展约束 注意第二层第三层扩展有重复的情况

        								//找到标记是0的点，然后查找邻域是255的点平面拟合看这个点是不是地面点，是就用来进一步扩展，不是就对应为障碍物进行标记
        								if(obstale_mask_1C.at<uchar>(pt[1].y, pt[1].x)==0)//是待处理的点
        								{//下面是对待处理点进行处理  有些刚开始的内部点没有待处理的点，但是没关系，给的范围每个点遍历后会有待处理点的
        									//整一个点云存储连接标记为255的点用来进行平面拟合

        	        						  vector<int> first_lay_index;//存扩展的边另一个端点的索引
        	        						  vector<int> second_lay_index;
        	        						  vector<int> third_lay_index;
        	        						  cv::vector<cv::Point2f> first_pt;//存扩展边的另一个顶点的坐标
        	        						  cv::vector<cv::Point2f> second_pt;
        	        						  cv::vector<cv::Point2f> third_pt;
        	        						  cv::Mat local_expand_mask_1C = cv::Mat::zeros(imageSize, CV_8UC1);//每扩展一个点新建一个

        									   pcl::PointCloud<pcl::PointXYZHSV>::Ptr local_for_ransac (new pcl::PointCloud<pcl::PointXYZHSV>);
        									   pcl::PointCloud<pcl::PointXYZHSV>::Ptr local_to_process (new pcl::PointCloud<pcl::PointXYZHSV>);
        									   bool obstacle_flag1 = false;

        									  	  //待处理的点坐标
      						    	  	       double temp_x = project_left_img.at<cv::Vec4f>(pt[1].y, pt[1].x)[0];//最后这个点需要处理一下，延伸的点也需要处理一下
      						    	  	       double temp_y = project_left_img.at<cv::Vec4f>(pt[1].y, pt[1].x)[1];
      						    	  	       double temp_z = project_left_img.at<cv::Vec4f>(pt[1].y, pt[1].x)[2];


        									//实验证明加延伸判断效果会变差，一层就是，两层就更加，因为平面假设已经很强
        									    cv::Point2f outer_vtx_from_edge1;
        										int edge_num_temp1;
        										outer_vtx_from_edge1 =  subdiv.getVertex(index_d,&edge_num_temp1);

        										int temp1 = edge_num_temp1;
        	        							cv::vector<cv::Point2f> pt1(2);

        	        							do//对待处理点进行延伸处理  处理第一层连接
        	        							{
        	        								int index_d1 = subdiv.edgeDst(temp1, &pt1[1]);//连接边的另一个端点
        	        								if (index_d1<4)//这条边可能连到虚点
        	        								{
        	        									temp1 = subdiv.nextEdge(temp1);
        	        									continue;
        	        								}
        	        								first_lay_index.push_back(index_d1);
        	        								first_pt.push_back(pt1[1]);

        	        								if(obstale_mask_1C.at<uchar>(pt1[1].y, pt1[1].x)==255)//待处理的点连接的已经认定是地面的点
        	        								{
        	        						    	  	 /* double temp_x1 = project_left_img.at<cv::Vec4f>(pt1[1].y, pt1[1].x)[0];
        	        						    	  	  double temp_y1 = project_left_img.at<cv::Vec4f>(pt1[1].y, pt1[1].x)[1];
        	        						    	  	  double temp_z1 = project_left_img.at<cv::Vec4f>(pt1[1].y, pt1[1].x)[2];
        	        	    							  double x_m = temp_x1-temp_x;
        	        	    							  double y_m = temp_y1-temp_y;
        	        	    							  double z_m = temp_z1-temp_z;

        	        	    							  double distance = sqrt(x_m*x_m+ y_m*y_m+z_m*z_m);
        	        	    							//  if(distance<near_distanse_thresh)//距离太远没有参考价值
        	        	    							  {
        	        	    								 // cout<<"distance1: "<<distance<<endl;
        	        	    								  obstacle_flag1  = true;
        	        	    								  break;
        	        	    							  }*/
        	    	    								pcl::PointXYZHSV point_temp;
        	    	    								point_temp.x = project_left_img.at<cv::Vec4f>(pt1[1].y, pt1[1].x)[0];
        	    	    								point_temp.y = project_left_img.at<cv::Vec4f>(pt1[1].y, pt1[1].x)[1];
        	    	    								point_temp.z = project_left_img.at<cv::Vec4f>(pt1[1].y, pt1[1].x)[2];
        	    	    								point_temp.h = project_left_img.at<cv::Vec4f>(pt1[1].y, pt1[1].x)[3];
        	    	    								point_temp.s = pt1[1].x;
        	    	    								point_temp.v = pt1[1].y;
        	    	    								local_for_ransac->push_back(point_temp);
        	    	    								local_expand_mask_1C.at<uchar>(pt1[1].y, pt1[1].x)=1;//被遍历过了
        	        								}
        	        								else
        	        								{
        	    	    								pcl::PointXYZHSV point_temp;
        	    	    								point_temp.x = project_left_img.at<cv::Vec4f>(pt1[1].y, pt1[1].x)[0];
        	    	    								point_temp.y = project_left_img.at<cv::Vec4f>(pt1[1].y, pt1[1].x)[1];
        	    	    								point_temp.z = project_left_img.at<cv::Vec4f>(pt1[1].y, pt1[1].x)[2];
        	    	    								point_temp.h = project_left_img.at<cv::Vec4f>(pt1[1].y, pt1[1].x)[3];
        	    	    								point_temp.s = pt1[1].x;
        	    	    								point_temp.v = pt1[1].y;
        	    	    								local_to_process->push_back(point_temp);  //这个变量后面也没有用到
        	    	    								local_expand_mask_1C.at<uchar>(pt1[1].y, pt1[1].x)=1;//被遍历过了
        	        								}



        	        								temp1 = subdiv.nextEdge(temp1);
        	        								//cout<<"!!!!!!!!!!!!!!!!!!!!!!!temp"<<temp<<endl;

        	        							}while(edge_num_temp1 != temp1);//&&local_for_ransac->size()<local_ransac_max_num  第一层全用


        	        							//处理第二层连接

    	        								//延伸判断增加下一层连接   注意这里有可能一个点被多次加入
        	        						for(int j=0;j<first_lay_index.size();j++)
        	        						{

    	                			    		  cv::Point2f outer_vtx_from_edge2;
    	                			    		  int edge_num_temp2;
    	                			    		  outer_vtx_from_edge2 =  subdiv.getVertex(first_lay_index[j],&edge_num_temp2);//得到一条以给定点为起点的线
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
    	                								if(local_expand_mask_1C.at<uchar>(pt3[1].y, pt3[1].x)!=1)
    	                								{
    	        	        								second_lay_index.push_back(index_d3);
    	        	        								second_pt.push_back(pt3[1]);
    	                								}

            	        								if(obstale_mask_1C.at<uchar>(pt3[1].y, pt3[1].x)==255&&local_expand_mask_1C.at<uchar>(pt3[1].y, pt3[1].x)!=1)
            	        								{

          	        						    	  	 /* double temp_x1 = project_left_img.at<cv::Vec4f>(pt3[1].y, pt3[1].x)[0];
          	        						    	  	  double temp_y1 = project_left_img.at<cv::Vec4f>(pt3[1].y, pt3[1].x)[1];
          	        						    	  	  double temp_z1 = project_left_img.at<cv::Vec4f>(pt3[1].y, pt3[1].x)[2];
          	        	    							  double x_m = temp_x1-temp_x;
          	        	    							  double y_m = temp_y1-temp_y;
          	        	    							  double z_m = temp_z1-temp_z;

          	        	    							  double distance = sqrt(x_m*x_m+ y_m*y_m+z_m*z_m);
          	        	    							  //if(distance<near_distanse_thresh)//距离太远没有参考价值   这个值是不是合适？
          	        	    							  {
          	        	    								 // cout<<"distance2: "<<distance<<endl;
          	        	    								  obstacle_flag1  = true;
          	        	    								  break;
          	        	    							  }*/

            	    	    								pcl::PointXYZHSV point_temp;
            	    	    								point_temp.x = project_left_img.at<cv::Vec4f>(pt3[1].y, pt3[1].x)[0];
            	    	    								point_temp.y = project_left_img.at<cv::Vec4f>(pt3[1].y, pt3[1].x)[1];
            	    	    								point_temp.z = project_left_img.at<cv::Vec4f>(pt3[1].y, pt3[1].x)[2];
            	    	    								point_temp.h = project_left_img.at<cv::Vec4f>(pt3[1].y, pt3[1].x)[3];
            	    	    								point_temp.s = pt3[1].x;
            	    	    								point_temp.v = pt3[1].y;
            	    	    								local_for_ransac->push_back(point_temp);
            	    	    								local_expand_mask_1C.at<uchar>(pt3[1].y, pt3[1].x)=1;



            	        								}
            	        								if(obstale_mask_1C.at<uchar>(pt3[1].y, pt3[1].x)==0&&local_expand_mask_1C.at<uchar>(pt3[1].y, pt3[1].x)!=1)
            	        								{
            	    	    								pcl::PointXYZHSV point_temp;
            	    	    								point_temp.x = project_left_img.at<cv::Vec4f>(pt3[1].y, pt3[1].x)[0];
            	    	    								point_temp.y = project_left_img.at<cv::Vec4f>(pt3[1].y, pt3[1].x)[1];
            	    	    								point_temp.z = project_left_img.at<cv::Vec4f>(pt3[1].y, pt3[1].x)[2];
            	    	    								point_temp.h = project_left_img.at<cv::Vec4f>(pt3[1].y, pt3[1].x)[3];
            	    	    								point_temp.s = pt3[1].x;
            	    	    								point_temp.v = pt3[1].y;
            	    	    								local_to_process->push_back(point_temp);
            	    	    								local_expand_mask_1C.at<uchar>(pt3[1].y, pt3[1].x)=1;
            	        								}
    	              								    temp3 = subdiv.nextEdge(temp3);
    	              								        								//cout<<"temp: "<<temp<<endl;

    	              							   }while(edge_num_temp2 != temp3);//&&local_for_ransac->size()<local_ransac_max_num 第二层全用 这样第一层就不需要用来进行扩展了


        	        						}
//一次处理多层虽然速度快了，但是效果不行了
        	        						//cout<<"1 local_for_ransac->size()"<<local_for_ransac->size()<<endl;
        	        						for(int j=0;j<second_lay_index.size();j++)//这里只是为了进一步找到第三层连接的已知地面点用来进行地面点拟合，达到我们设定的最小点数
        	        						{ //第三层进入更进一步的三层处理
        									    cv::Point2f outer_vtx_from_edge1;
        										int edge_num_temp1;
        										outer_vtx_from_edge1 =  subdiv.getVertex(second_lay_index[j],&edge_num_temp1);

        										int temp1 = edge_num_temp1;
        	        							cv::vector<cv::Point2f> pt1(2);

        	        							do
        	        							{
        	        								int index_d1 = subdiv.edgeDst(temp1, &pt1[1]);//连接边的另一个端点
        	        								if (index_d1<4)//这条边可能连到虚点
        	        								{
        	        									temp1 = subdiv.nextEdge(temp1);
        	        									continue;
        	        								}

        	        								if(obstale_mask_1C.at<uchar>(pt1[1].y, pt1[1].x)==255&&local_expand_mask_1C.at<uchar>(pt1[1].y, pt1[1].x)!=1)//待处理的点连接的已经认定是地面的点
        	        								{
        	        						    	  	 /* double temp_x1 = project_left_img.at<cv::Vec4f>(pt1[1].y, pt1[1].x)[0];
        	        						    	  	  double temp_y1 = project_left_img.at<cv::Vec4f>(pt1[1].y, pt1[1].x)[1];
        	        						    	  	  double temp_z1 = project_left_img.at<cv::Vec4f>(pt1[1].y, pt1[1].x)[2];
        	        	    							  double x_m = temp_x1-temp_x;
        	        	    							  double y_m = temp_y1-temp_y;
        	        	    							  double z_m = temp_z1-temp_z;

        	        	    							  double distance = sqrt(x_m*x_m+ y_m*y_m+z_m*z_m);
        	        	    							//  if(distance<near_distanse_thresh)//距离太远没有参考价值
        	        	    							  {
        	        	    								 // cout<<"distance1: "<<distance<<endl;
        	        	    								  obstacle_flag1  = true;
        	        	    								  break;
        	        	    							  }*/
        	    	    								pcl::PointXYZHSV point_temp;
        	    	    								point_temp.x = project_left_img.at<cv::Vec4f>(pt1[1].y, pt1[1].x)[0];
        	    	    								point_temp.y = project_left_img.at<cv::Vec4f>(pt1[1].y, pt1[1].x)[1];
        	    	    								point_temp.z = project_left_img.at<cv::Vec4f>(pt1[1].y, pt1[1].x)[2];
        	    	    								point_temp.h = project_left_img.at<cv::Vec4f>(pt1[1].y, pt1[1].x)[3];
        	    	    								point_temp.s = pt1[1].x;
        	    	    								point_temp.v = pt1[1].y;
        	    	    								local_for_ransac->push_back(point_temp);
        	    	    								local_expand_mask_1C.at<uchar>(pt1[1].y, pt1[1].x)=1;

            	        								//延伸判断增加下一层连接   注意这里有可能一个点被多次加入

            	                			    		  cv::Point2f outer_vtx_from_edge2;
            	                			    		  int edge_num_temp2;
            	                			    		  outer_vtx_from_edge2 =  subdiv.getVertex(index_d1,&edge_num_temp2);//得到一条以给定点为起点的线
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

                    	        								if(obstale_mask_1C.at<uchar>(pt3[1].y, pt3[1].x)==255&&local_expand_mask_1C.at<uchar>(pt3[1].y, pt3[1].x)!=1)
                    	        								{

                  	        						    	  	 /* double temp_x1 = project_left_img.at<cv::Vec4f>(pt3[1].y, pt3[1].x)[0];
                  	        						    	  	  double temp_y1 = project_left_img.at<cv::Vec4f>(pt3[1].y, pt3[1].x)[1];
                  	        						    	  	  double temp_z1 = project_left_img.at<cv::Vec4f>(pt3[1].y, pt3[1].x)[2];
                  	        	    							  double x_m = temp_x1-temp_x;
                  	        	    							  double y_m = temp_y1-temp_y;
                  	        	    							  double z_m = temp_z1-temp_z;

                  	        	    							  double distance = sqrt(x_m*x_m+ y_m*y_m+z_m*z_m);
                  	        	    							  //if(distance<near_distanse_thresh)//距离太远没有参考价值   这个值是不是合适？
                  	        	    							  {
                  	        	    								 // cout<<"distance2: "<<distance<<endl;
                  	        	    								  obstacle_flag1  = true;
                  	        	    								  break;
                  	        	    							  }*/

                    	    	    								pcl::PointXYZHSV point_temp;
                    	    	    								point_temp.x = project_left_img.at<cv::Vec4f>(pt3[1].y, pt3[1].x)[0];
                    	    	    								point_temp.y = project_left_img.at<cv::Vec4f>(pt3[1].y, pt3[1].x)[1];
                    	    	    								point_temp.z = project_left_img.at<cv::Vec4f>(pt3[1].y, pt3[1].x)[2];
                    	    	    								point_temp.h = project_left_img.at<cv::Vec4f>(pt3[1].y, pt3[1].x)[3];
                    	    	    								point_temp.s = pt3[1].x;
                    	    	    								point_temp.v = pt3[1].y;
                    	    	    								local_for_ransac->push_back(point_temp);//这个变量只是用来拟合，重复什么的也没事
                    	    	    								local_expand_mask_1C.at<uchar>(pt3[1].y, pt3[1].x)=1;

                        	        								//第三层延伸判断   注意这里有可能一个点被多次加入

                        	                			    		  cv::Point2f outer_vtx_from_edge3;
                        	                			    		  int edge_num_temp3;
                        	                			    		  outer_vtx_from_edge3 =  subdiv.getVertex(index_d3,&edge_num_temp3);//得到一条以给定点为起点的线
                        	                			    		  //cout<<"outer_vtx_from_edge: "<<outer_vtx_from_edge<<endl;
                        	                			    		  //cout<<"edge_num_temp: "<<edge_num_temp<<endl;

                        	                							int temp4 = edge_num_temp3;
                        	                							cv::vector<cv::Point2f> pt4(2);
                        	                							do
                        	                							{
                        	                								int index_d4 = subdiv.edgeDst(temp4, &pt4[1]);//连接边的另一个端点
                        	                								//cout<<"pt[1]: "<<pt[1]<<endl;
                        	                								if (index_d4<4)//这条边可能连到虚点
                        	                								{
                        	                									temp4 = subdiv.nextEdge(temp4);
                        	                									//cout<<"temp: "<<temp<<endl;
                        	                									continue;
                        	                								}

                                	        								if(obstale_mask_1C.at<uchar>(pt4[1].y, pt4[1].x)==255&&local_expand_mask_1C.at<uchar>(pt4[1].y, pt4[1].x)!=1)
                                	        								{

                              	        						    	  	 /* double temp_x1 = project_left_img.at<cv::Vec4f>(pt3[1].y, pt3[1].x)[0];
                              	        						    	  	  double temp_y1 = project_left_img.at<cv::Vec4f>(pt3[1].y, pt3[1].x)[1];
                              	        						    	  	  double temp_z1 = project_left_img.at<cv::Vec4f>(pt3[1].y, pt3[1].x)[2];
                              	        	    							  double x_m = temp_x1-temp_x;
                              	        	    							  double y_m = temp_y1-temp_y;
                              	        	    							  double z_m = temp_z1-temp_z;

                              	        	    							  double distance = sqrt(x_m*x_m+ y_m*y_m+z_m*z_m);
                              	        	    							  //if(distance<near_distanse_thresh)//距离太远没有参考价值   这个值是不是合适？
                              	        	    							  {
                              	        	    								 // cout<<"distance2: "<<distance<<endl;
                              	        	    								  obstacle_flag1  = true;
                              	        	    								  break;
                              	        	    							  }*/

                                	    	    								pcl::PointXYZHSV point_temp;
                                	    	    								point_temp.x = project_left_img.at<cv::Vec4f>(pt4[1].y, pt4[1].x)[0];
                                	    	    								point_temp.y = project_left_img.at<cv::Vec4f>(pt4[1].y, pt4[1].x)[1];
                                	    	    								point_temp.z = project_left_img.at<cv::Vec4f>(pt4[1].y, pt4[1].x)[2];
                                	    	    								point_temp.h = project_left_img.at<cv::Vec4f>(pt4[1].y, pt4[1].x)[3];
                                	    	    								point_temp.s = pt4[1].x;
                                	    	    								point_temp.v = pt4[1].y;
                                	    	    								local_for_ransac->push_back(point_temp);
                                	    	    								local_expand_mask_1C.at<uchar>(pt4[1].y, pt4[1].x)=1;

                                	        								}
                        	              								    temp4 = subdiv.nextEdge(temp4);
                        	              								        								//cout<<"temp: "<<temp<<endl;

                        	              							   }while(edge_num_temp3 != temp4&&local_for_ransac->size()<local_ransac_max_num);




                    	        								}
            	              								    temp3 = subdiv.nextEdge(temp3);
            	              								        								//cout<<"temp: "<<temp<<endl;

            	              							   }while(edge_num_temp2 != temp3&&local_for_ransac->size()<local_ransac_max_num);

        	        								}


        	        								temp1 = subdiv.nextEdge(temp1);
        	        								//cout<<"!!!!!!!!!!!!!!!!!!!!!!!temp"<<temp<<endl;

        	        							}while(edge_num_temp1 != temp1&&local_for_ransac->size()<local_ransac_max_num);


        	        							if(local_for_ransac->size()>=local_ransac_max_num)
        	        							{
        	        							    break;
        	        							}


        	        						}

        	        						//下面对第一个待处理的点和第一层第二层点进行地面点判断，然后第二层认为是地面的点进行扩展


        	        							//这里通过地面平面拟合判断点是否属于地面
        	        							//cout<<"2 local_for_ransac->size()"<<local_for_ransac->size()<<endl;
        	        							if(local_for_ransac->size()<local_ransac_min_num)
        	        							{
        	        								obstacle_flag1 = true;
        	        							}
        	        							else
        	        							{

														pcl::ModelCoefficients::Ptr coefficients_local (new pcl::ModelCoefficients);
														//inliers表示误差能容忍的点 记录的是点云的序号
														pcl::PointIndices::Ptr inliers_local (new pcl::PointIndices);
														// 创建一个分割器
														pcl::SACSegmentation<pcl::PointXYZHSV> seg_local;
														// Optional
														seg_local.setOptimizeCoefficients (true);
														// Mandatory-设置目标几何形状
														seg_local.setModelType (pcl::SACMODEL_PLANE);
														//分割方法：随机采样法
														seg_local.setMethodType (pcl::SAC_RANSAC);
														//设置误差容忍范围
														seg_local.setDistanceThreshold (ransac_DistanceThreshold_local);
														//输入点云
														seg_local.setInputCloud (local_for_ransac);
														//分割点云
														seg_local.segment (*inliers_local, *coefficients_local);//a * x + b * y + c * z = d 注意这个参数的第四位是-d 花了好多时间才发现

														if (inliers_local->indices.size () == 0)//具体inliers怎么使用，参考：http://www.pclcn.org/study/shownews.php?lang=cn&id=72
														  {
															PCL_ERROR ("Could not estimate a planar model for the given dataset.");
															return (-1);
														  }
													  double plane_a = coefficients_local->values[0];
													  double plane_b = coefficients_local->values[1];
													  double plane_c = coefficients_local->values[2];
													  double plane_d = -coefficients_local->values[3];

													  //处理扩展点一
													  double predict_plane = (plane_a*temp_x + plane_b * temp_y + plane_c*temp_z - plane_d);
													  bool expand_is_obstacle;
													 if(fabs(predict_plane )>ransac_DistanceThreshold_local )//如果只处理这个点需要300到500ms太慢了，但是效果好  可是这还有什么意义？？？
													 {
														   expand_is_obstacle = true;
		          	          					    	   obstale_mask_3C.at<cv::Vec3b>(pt[1].y, pt[1].x)[0] = 255;
		         	          					    	   obstale_mask_3C.at<cv::Vec3b>(pt[1].y, pt[1].x)[1] = 0;
		         	          					    	   obstale_mask_3C.at<cv::Vec3b>(pt[1].y, pt[1].x)[2] = 0;
		         	          					    	   cv::circle(obstale_mask_3C_inv, cv::Point2f(pt[1].x,pt[1].y), 1, cv::Scalar(255,0,0), CV_FILLED, CV_AA, 0);
		        	          					    	   obstale_mask_1C.at<uchar>(pt[1].y, pt[1].x) = 150;//有点愿意相信他是障碍点  最后把这些点补回来，这些点其实应该是地面点，但是为了延伸起点限制作用，就这样设置
		        	       								  /* pcl::PointXYZHSV point_temp;
		        	       								   point_temp.x = project_left_img.at<cv::Vec4f>(pt[1].y, pt[1].x)[0];
		        	       								   point_temp.y = project_left_img.at<cv::Vec4f>(pt[1].y, pt[1].x)[1];
		        	       								   point_temp.z = project_left_img.at<cv::Vec4f>(pt[1].y, pt[1].x)[2];
		        	       								   point_temp.h = project_left_img.at<cv::Vec4f>(pt[1].y, pt[1].x)[3];
		        	       								   point_temp.s = pt[1].x;
		        	       								   point_temp.v = pt[1].y;
		        	        							   cloud_ransac_seg_ground_in_range_expanded_not_ground->push_back(point_temp);//其实这个变量到后期没有起作用了
		        	        							   */
													 }
		        	        					     else
													 {
													//cout<<"!!!ground fabs(predict_plane ): "<<fabs(predict_plane )<<endl;
		        	        					    	   expand_is_obstacle = false;
		        	        					    	   obstale_mask_3C.at<cv::Vec3b>(pt[1].y, pt[1].x)[0] = 0;
		          	          					    	   obstale_mask_3C.at<cv::Vec3b>(pt[1].y, pt[1].x)[1] = 125;
		          	          					    	   obstale_mask_3C.at<cv::Vec3b>(pt[1].y, pt[1].x)[2] = 255;
		          	          					    	   cv::circle(obstale_mask_3C_inv, cv::Point2f(pt[1].x,pt[1].y), 1, cv::Scalar(0,125,255), CV_FILLED, CV_AA, 0);
		         	          					    	   obstale_mask_1C.at<uchar>(pt[1].y, pt[1].x) = 255;//状态确定
		         	       								   pcl::PointXYZHSV point_temp;
		         	       								   point_temp.x = project_left_img.at<cv::Vec4f>(pt[1].y, pt[1].x)[0];
		         	       								   point_temp.y = project_left_img.at<cv::Vec4f>(pt[1].y, pt[1].x)[1];
		         	       								   point_temp.z = project_left_img.at<cv::Vec4f>(pt[1].y, pt[1].x)[2];
		         	       								   point_temp.h = project_left_img.at<cv::Vec4f>(pt[1].y, pt[1].x)[3];
		         	       								   point_temp.s = pt[1].x;
		         	       								   point_temp.v = pt[1].y;
		         	       								   cloud_ransac_seg_ground_in_range_expanded_ground_new->push_back(point_temp);  //第二层才用来扩展
													 }

													// if(expand_is_obstacle == false)
													 {
														 bool expand_first_lay_has_obstacle;
												/*		 for(int k =0;k<first_pt.size();k++)

														 {
															 if(obstale_mask_1C.at<uchar>(first_pt[k].y, first_pt[k].x)==0)//第一层里面待处理的
															 {
																   double temp_x = project_left_img.at<cv::Vec4f>(first_pt[k].y, first_pt[k].x)[0];//最后这个点需要处理一下，延伸的点也需要处理一下
																   double temp_y = project_left_img.at<cv::Vec4f>(first_pt[k].y, first_pt[k].x)[1];
																   double temp_z = project_left_img.at<cv::Vec4f>(first_pt[k].y, first_pt[k].x)[2];
																   double predict_plane1 = (plane_a*temp_x + plane_b * temp_y + plane_c*temp_z - plane_d);
																	 if(fabs(predict_plane )>ransac_DistanceThreshold_local )
																	 {
																		   expand_first_lay_has_obstacle = true;
																		   obstale_mask_3C.at<cv::Vec3b>(first_pt[k].y, first_pt[k].x)[0] = 255;
																		   obstale_mask_3C.at<cv::Vec3b>(first_pt[k].y, first_pt[k].x)[1] = 0;
																		   obstale_mask_3C.at<cv::Vec3b>(first_pt[k].y, first_pt[k].x)[2] = 0;
																		   cv::circle(obstale_mask_3C_inv, cv::Point2f(first_pt[k].x, first_pt[k].y), 1, cv::Scalar(255,0,0), CV_FILLED, CV_AA, 0);
																		   obstale_mask_1C.at<uchar>(first_pt[k].y, first_pt[k].x) = 150;//有点愿意相信他是障碍点  最后把这些点补回来，这些点其实应该是地面点，但是为了延伸起点限制作用，就这样设置

																	 }
																	 else
																	 {//管到这一层需要200ms左右
																	//cout<<"!!!ground fabs(predict_plane ): "<<fabs(predict_plane )<<endl;
																		   obstale_mask_3C.at<cv::Vec3b>(first_pt[k].y, first_pt[k].x)[0] = 0;
																		   obstale_mask_3C.at<cv::Vec3b>(first_pt[k].y, first_pt[k].x)[1] = 125;
																		   obstale_mask_3C.at<cv::Vec3b>(first_pt[k].y, first_pt[k].x)[2] = 255;
																		   cv::circle(obstale_mask_3C_inv, cv::Point2f(first_pt[k].x, first_pt[k].y), 1, cv::Scalar(0,125,255), CV_FILLED, CV_AA, 0);
																		   obstale_mask_1C.at<uchar>(first_pt[k].y, first_pt[k].x) = 255;//状态确定
																		   pcl::PointXYZHSV point_temp;
																		   point_temp.x = project_left_img.at<cv::Vec4f>(first_pt[k].y, first_pt[k].x)[0];
																		   point_temp.y = project_left_img.at<cv::Vec4f>(first_pt[k].y, first_pt[k].x)[1];
																		   point_temp.z = project_left_img.at<cv::Vec4f>(first_pt[k].y, first_pt[k].x)[2];
																		   point_temp.h = project_left_img.at<cv::Vec4f>(first_pt[k].y, first_pt[k].x)[3];
																		   point_temp.s = first_pt[k].x;
																		   point_temp.v = first_pt[k].y;
																		   cloud_ransac_seg_ground_in_range_expanded_ground_new->push_back(point_temp);  //第二层才用来扩展
																	 }
															 }


														 }*/
														// if(expand_first_lay_has_obstacle==false)
														 {
														 //处理第二层  处理两层有点激进  原来只处理一点，现在等于同时处理三层了   管到这一层需要120到150ms左右
														/*	for(int k =0;k<second_pt.size();k++)
															{
																 if(obstale_mask_1C.at<uchar>(second_pt[k].y, second_pt[k].x)==0)//第一层里面待处理的
																 {
																	   double temp_x = project_left_img.at<cv::Vec4f>(second_pt[k].y, second_pt[k].x)[0];//最后这个点需要处理一下，延伸的点也需要处理一下
																	   double temp_y = project_left_img.at<cv::Vec4f>(second_pt[k].y, second_pt[k].x)[1];
																	   double temp_z = project_left_img.at<cv::Vec4f>(second_pt[k].y, second_pt[k].x)[2];
																	   double predict_plane1 = (plane_a*temp_x + plane_b * temp_y + plane_c*temp_z - plane_d);
																		 if(fabs(predict_plane )>ransac_DistanceThreshold_local )
																		 {
																			   obstale_mask_3C.at<cv::Vec3b>(second_pt[k].y, second_pt[k].x)[0] = 255;
																			   obstale_mask_3C.at<cv::Vec3b>(second_pt[k].y, second_pt[k].x)[1] = 0;
																			   obstale_mask_3C.at<cv::Vec3b>(second_pt[k].y, second_pt[k].x)[2] = 0;
																			   cv::circle(obstale_mask_3C_inv, cv::Point2f(second_pt[k].x, second_pt[k].y), 1, cv::Scalar(255,0,0), CV_FILLED, CV_AA, 0);
																			   obstale_mask_1C.at<uchar>(second_pt[k].y, second_pt[k].x) = 150;//有点愿意相信他是障碍点  最后把这些点补回来，这些点其实应该是地面点，但是为了延伸起点限制作用，就这样设置

																		 }
																		 else
																		 {
																		//cout<<"!!!ground fabs(predict_plane ): "<<fabs(predict_plane )<<endl;
																			   obstale_mask_3C.at<cv::Vec3b>(second_pt[k].y, second_pt[k].x)[0] = 0;
																			   obstale_mask_3C.at<cv::Vec3b>(second_pt[k].y, second_pt[k].x)[1] = 125;
																			   obstale_mask_3C.at<cv::Vec3b>(second_pt[k].y, second_pt[k].x)[2] = 255;
																			   cv::circle(obstale_mask_3C_inv, cv::Point2f(second_pt[k].x, second_pt[k].y), 1, cv::Scalar(0,125,255), CV_FILLED, CV_AA, 0);
																			   obstale_mask_1C.at<uchar>(second_pt[k].y, second_pt[k].x) = 255;//状态确定
																			   pcl::PointXYZHSV point_temp;
																			   point_temp.x = project_left_img.at<cv::Vec4f>(second_pt[k].y, second_pt[k].x)[0];
																			   point_temp.y = project_left_img.at<cv::Vec4f>(second_pt[k].y, second_pt[k].x)[1];
																			   point_temp.z = project_left_img.at<cv::Vec4f>(second_pt[k].y, second_pt[k].x)[2];
																			   point_temp.h = project_left_img.at<cv::Vec4f>(second_pt[k].y, second_pt[k].x)[3];
																			   point_temp.s = second_pt[k].x;
																			   point_temp.v = second_pt[k].y;
																			   cloud_ransac_seg_ground_in_range_expanded_ground_new->push_back(point_temp); //第二层才用来扩展
																		 }
																   }


															  }*/
														 }
													 }
        	        							}//if(local_for_ransac->size()<local_ransac_min_num)的else
        								}//if(obstale_mask_1C.at<uchar>(pt[1].y, pt[1].x)==0)//是待处理的点

        								temp = subdiv.nextEdge(temp);
        								//cout<<"temp: "<<temp<<endl;

        							}while(edge_num_temp != temp);


        			    	  }
        			    	  *cloud_ransac_seg_ground_in_range_expanded_ground += *cloud_ransac_seg_ground_in_range_expanded_ground_new;
        			    	  cloud_ransac_seg_ground_in_range_to_expanded_new->clear();
        			    	  *cloud_ransac_seg_ground_in_range_to_expanded_new += *cloud_ransac_seg_ground_in_range_expanded_ground_new;
        			    	  cloud_ransac_seg_ground_in_range_expanded_ground_new->clear();
        			    	 // cout<<"cloud_ransac_seg_ground_in_range_expanded_not_ground->size()"<<cloud_ransac_seg_ground_in_range_expanded_not_ground->size()<<endl;
        			    	 // cout<<"cloud_ransac_seg_ground_in_range_expanded_ground->size()"<<cloud_ransac_seg_ground_in_range_expanded_ground->size()<<endl;
        			    	  cout<<"cloud_ransac_seg_ground_in_range_to_expanded_new->size()"<<cloud_ransac_seg_ground_in_range_to_expanded_new->size()<<endl;
        			    	 // cout<<"cloud_ransac_seg_ground_out_range->size()"<<cloud_ransac_seg_ground_out_range->size()<<endl;

    			    	  }while(cloud_ransac_seg_ground_in_range_to_expanded_new->size()>=terminate_expand_num);//!=0


              	    //TODO 腐蚀膨胀后处理


    					  ////根据voronoi图遍历，以前面得到的障碍分类结果分区图像

    					    cv::vector<cv::vector<cv::Point2f> > facets;
    					    cv::vector<cv::Point2f> centers;
    					    subdiv.getVoronoiFacetList(cv::vector<int>(), facets, centers);

    					    cv:vector<cv::Point> ifacet;
    					    cv::vector<cv::vector<cv::Point> > ifacets(1);

    					    cv::Scalar ground_color(255, 0, 0), obstacle_color(0,50,0);

    					    for( size_t i = 0; i < facets.size(); i++ )
    					    {
    					        ifacet.resize(facets[i].size());
    					        for( size_t j = 0; j < facets[i].size(); j++ )
    					            ifacet[j] = facets[i][j];


    					        if(obstale_mask_1C.at<uchar>(centers[i].y, centers[i].x) == 255)
    					        {

    					        	fillConvexPoly(img_3C, ifacet, ground_color, 8, 0);//填充多边形
    					        	fillConvexPoly(img_1C, ifacet, 255, 8, 0);//之后将地平线以上的部分清0
    					        }
    					        else
    					        {
    					        	fillConvexPoly(img_3C, ifacet, obstacle_color, 8, 0);//填充多边形img_1C
    					        	//fillConvexPoly(img_1C, ifacet, 0, 8, 0);//本来就是0
    					        }


    					    }

    					  //处理掉地平线以上的部分
    					  cv::Rect rect_horizontal(0,0,imageSize.width,horizontal_point_v);

    					  img_1C(rect_horizontal) = cv::Scalar::all(0);
    					  img_3C(rect_horizontal) = cv::Scalar::all(0);

    					  cv::imshow("test1",img_1C);//地平线之后


    					  cout<<"expanded time used is "<<1000*(clock()-time_stt2)/(double)CLOCKS_PER_SEC<<"ms"<<endl;//这个函数总共需要这么多时间

    					  img_1C.copyTo(result_32(cv::Rect(0,0,cv_image02.cols,cv_image02.rows)));

    					  cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(erode_size, erode_size));




    					  cv::erode(img_1C,img_1C,element);//腐蚀白色

    					  cv::imshow("test2",img_1C);//腐蚀之后
    					  img_1C.copyTo(result_32(cv::Rect(0,cv_image02.rows,cv_image02.cols,cv_image02.rows)));

    					  //找最大的连通区域
    					  vector<vector<cv::Point>> contours ;
    					  cv::Rect rR;
    					  // 查找轮廓，对应连通域
    					  cv::findContours(img_1C,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
    					  // 寻找最大连通域
    					  double maxArea = 0;
    					  vector<cv::Point> maxContour;
    					  int maxContour_index;
    					  cv::Mat contours_mat=cv::Mat::zeros(imageSize, CV_8UC3);
    					  for(size_t i = 0; i < contours.size(); i++)
    					  {
    						  rR= boundingRect(cv::Mat(contours[i]));
    						  drawContours(contours_mat,contours,i,cv::Scalar(255,0,0),3);
    					      double area = cv::contourArea(contours[i]);
    					      if (area > maxArea&&((cv_image02.cols/2)>=rR.x)&&((cv_image02.cols/2)<=rR.x+rR.width))//加上这个条件是为了防止选择非车占区域，由于摄像头朝着车的正前方，所以图像中心应该在区域内
    					      {
    					          maxArea = area;
    					          maxContour = contours[i];
    					          maxContour_index = i;
    					          drawContours(contours_mat,contours,i,cv::Scalar(0,0,255),3);
    					      }
    					  }
    					  cv::imshow("test3",contours_mat);//画出轮廓
    					  cv::Mat img_1C_temp(rect.size(), CV_8UC1);
    					  cv::Mat img_3C_temp(cv_image02.size(), CV_8UC3);
    					  img_1C_temp = cv::Scalar::all(0);
    					  img_3C_temp = cv::Scalar::all(0);
    					  //cv::drawContours(img_1C_temp,contours,maxContour_index,cv::Scalar(255,0,0),1);
    					  cv::Point maxContour_points[1][maxContour.size()];
    					  for(int j=0;j<maxContour.size();j++)
    					  {
    						  maxContour_points[0][j] = maxContour[j];
    					  }


    					  const cv::Point* ppt[1] = { maxContour_points[0] };
    					  int npt[] = { maxContour.size() };
    					 // cv::polylines(img_1C_temp, ppt, npt, 1, 1, cv::Scalar(255,0,0), 3, 8, 0);
    					     		//    imshow("road_surface", out_img);
    					  cv::fillPoly(img_1C_temp, ppt, npt, 1, cv::Scalar(255, 0, 0));//填充
    					  cv::fillPoly(img_3C_temp, ppt, npt, 1, cv::Scalar(255, 0, 0));

    					  cv::imshow("test4",img_1C_temp);//填充选中轮廓
    					 // imshow("maxContour", img_1C_temp);



    					  cv::dilate(img_1C_temp,img_1C,element);
    					  cv::dilate(img_3C_temp,img_3C_temp,element);//这个函数能操作多通道数据吗？？可以
    					 // imshow("dilate_img_3C_temp", img_3C_temp);

    					  cv::imshow("test5",img_1C);//膨胀操作

    					  img_1C.copyTo(result_32(cv::Rect(0,cv_image02.rows*2,cv_image02.cols,cv_image02.rows)));



    					  cv::Mat show;
    					  addWeighted(cv_image02,1,img_3C_temp,0.5,0.,show);//这里图像大小可能不一样，已经改到一样了 有图像后处理的结果
    					  //imshow("result_show", show);



    					  cv::Mat show2;
    					  addWeighted(cv_image02,1,img_3C,0.5,0.,show2);//这里图像大小可能不一样，已经改到一样了 没有图像后处理的结果

    					  cv::imwrite(full_filename_output,img_1C);





    					  //cv::imwrite(full_filename_output_3C_mask,obstale_mask_3C);
    					  cv::circle(img_3C, cv::Point(horizontal_point_u,horizontal_point_v), 3, cv::Scalar(0,0,0), CV_FILLED, CV_AA, 0);
    					 // cout<<"horizontal_point_u,horizontal_point_v: "<<horizontal_point_u <<", "<<horizontal_point_v<<endl;

    					  cv::Rect temp_rect(0,0,cv_image02.cols,cv_image02.rows);//注意cv_image02的大小是不固定的
    					  //result_3(temp_rect)=cv_image02.clone();//没考过去，不知道为什么
    					  cv_image02.copyTo(result_3(temp_rect));//这样就考过来了 注意result_3的列数需要保证比cv_image02.cols大或者相等 这里叠加的是图像后处理过的结果
    					  obstale_mask_3C_inv.copyTo(result_3(cv::Rect(0,cv_image02.rows,cv_image02.cols,cv_image02.rows)));
    					  show.copyTo(result_3(cv::Rect(0,cv_image02.rows*2,cv_image02.cols,cv_image02.rows)));//注意这个是没有经过图像后处理的结果
    					  cv::imwrite(full_filename_output_3C_mask,result_3);


    					  show2.copyTo(result_33(temp_rect));
    					  show.copyTo(result_33(cv::Rect(0,cv_image02.rows,cv_image02.cols,cv_image02.rows)));
    					  obstale_mask_3C_inv.copyTo(result_33(cv::Rect(0,cv_image02.rows*2,cv_image02.cols,cv_image02.rows)));



    					  //imshow("obstale_mask_3C", obstale_mask_3C);
    					  //imshow("obstale_mask_3C_inv_", obstale_mask_3C_inv);
    					 // cv::imshow("cv_image02",cv_image02);
    					  //cv::imshow("paintVoronoi",img);
    					 // cv::imshow("paintVoronoi_1C",img_1C);
    					  cv::imshow("result_3",result_3);
    					  cv::imshow("result_32",result_32);
    					  cv::imshow("result_33",result_33);

    					  cv::waitKey(500000);



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

