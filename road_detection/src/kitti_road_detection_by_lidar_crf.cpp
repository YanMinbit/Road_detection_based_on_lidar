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
#include "densecrf.h"
#include "util_ppm.h"


using namespace std;
using namespace pcl;
using namespace ros;
using namespace tf;
using namespace std;


int nColors = 0;
int colors[255];//其实这个数组的大小有M 就行了   //给定的粗标签和最后优化的结果同一个标签的颜色还是取得一样的
unsigned int getColor( const unsigned char * c ){//用int型数据存一个像素的三位颜色信息，从字符转int
	return c[0] + 256*c[1] + 256*256*c[2];
}
// Certainty that the groundtruth is correct
const float GT_PROB = 0.5;

// Simple classifier that is 50% certain that the annotation is correct
//这个函数的作用是认定给定的标记图像是某个分类器的输出，然后分别赋予对应的能量数值用于CRF
//返回的这个变量是一个浮点数组，大小是W＊H＊M，也就是每个像素对应每个标签都会有一个能量值
//具体能量值是背景像素每个标签对应的位置赋予u_energy，然后有标记的像素对应标签位置赋予p_energy，非标签位置赋予n_energy

float * classify( const unsigned char * im, int W, int H, int M ){//传进来的M 是21，传进来的图片是粗标记数据
	const float u_energy = -log( 1.0f / M );//背景对应的能量   3.04   既然是背景，那么每个标签的概率是一样的 能量函数的定义是-log p.
	const float n_energy = -log( (1.0f - GT_PROB) / (M-1) );//非对应标签赋予能量   3.69  这个的解释是另外0.5的概率分摊到每个错误标签上的概率
	const float p_energy = -log( GT_PROB );////是这个标签，则对应赋予能量 这三个计算的是一些常量   0.69  其实这个的解释是表示0.5的概率确定这个给定的标记是正确的
	//使用方法：log（）是以e为底的对数函数，即数学中的ln（）； e = 2.718281828459
    //log（b）/log（a）是以a为底，b为指数的对数。
	float * res = new float[W*H*M];//这么大的数组
	for( int k=0; k<W*H; k++ ){
		// Map the color to a label
		int c = getColor( im + 3*k );//读三个字符存在int型变量里面
		int i;//这个变量记录的是当前这个像素对应的标签
		for( i=0;i<nColors && c!=colors[i]; i++ );//nColors初始化是0，所以第一次这里是不执行的，另外这个是一次只执行一幅图像，所以这个量全局使用只对应一幅图像的情况
		if (c && i==nColors){//第一次进来这里执行
			if (i<M)//一共最多有标签M个，否则多余的全部改成背景标记0
				colors[nColors++] = c;//这个colors数组记录给定标签图像里面给了的像素信息，每个类对应一种颜色，然后nColors记录给定的标签个数
			else
				c=0;
		}

		// Set the energy
		float * r = res + k*M;//每个位置每个标签对应都会有一个量
		if (c){//不是背景
			for( int j=0; j<M; j++ )
				r[j] = n_energy;//非对应标签赋予能量
			r[i] = p_energy;//是这个标签，则对应赋予能量
		}
		else{
			for( int j=0; j<M; j++ )
				r[j] = u_energy;//是背景的时候对应赋予的能量
		}
	}
	return res;//返回的这个变量是一个浮点数组，大小是W＊H＊M，也就是每个像素对应每个标签都会有一个能量值
}
void putColor( unsigned char * c, unsigned int cc ){//从 int转字符
	c[0] = cc&0xff; c[1] = (cc>>8)&0xff; c[2] = (cc>>16)&0xff;
}
// Produce a color image from a bunch of labels
unsigned char * colorize( const short * map, int W, int H ){//传进来的map应该记录的每个像素的标签
	unsigned char * r = new unsigned char[ W*H*3 ];
	for( int k=0; k<W*H; k++ ){
		int c = colors[ map[k] ];
		putColor( r+3*k, c );//将int转成字符存到连续的三个字符里面
	}
	return r;
}













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


    double ransac_DistanceThreshold;
    double box_x_min,box_x_max,box_y_min,box_y_max,box_z_min,box_z_max;
    double box_x_min1,box_x_max1,box_y_min1,box_y_max1,box_z_min1,box_z_max1;//用来确定用于开始扩展的区域
    double ransac_DistanceThreshold1;
    int  terminate_expand_num;
    double min_edge_distance;
    int erode_size;
    bool process_for_view;
    string total_time_save_filename;


    nh_priv.param("thresh_theta", thresh_theta, 77.0);//论文中用的角度大小
    nh_priv.param("ransac_DistanceThreshold", ransac_DistanceThreshold, 0.04);
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
    nh_priv.param("terminate_expand_num", terminate_expand_num, 5);
    nh_priv.param("min_edge_distance", min_edge_distance, 0.015);
    nh_priv.param("erode_size", erode_size, 3);
    nh_priv.param("process_for_view", process_for_view, true);

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
    string full_filename_output_3C_mask;
    string full_filename_output_crf;
    string full_filename_result_ppm;
    string full_filename_src_ppm;
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
            	  full_filename_output_crf = dir_output + "3C/" + tag + "_road_3C_crf_" +  num + ".png";
            	  full_filename_result_ppm = dir_output + "3C/" + tag + "_road_3C_result_" +  num + ".ppm";
            	  full_filename_src_ppm = dir_output + "3C/" + tag + "_road_3C_src_" +  num + ".ppm";
            	  total_time_save_filename = dir_output + "total_time.txt";
            	  full_filename_velodyne = dir_velodyne_points + tag + "_" + num + ".bin";
            	  //cout<<"full_filename_velodyne:"<<full_filename_velodyne<<endl;
            	  cv_image02 = cv::imread(full_filename_image02, CV_LOAD_IMAGE_UNCHANGED);
            	  //cv::imshow("image02",cv_image02);

            	    imageSize.width = cv_image02.cols;//注意道路数据uu和umm\um图像的大小是不一样的，有的是1241＊367  有的是1242＊375
            	    imageSize.height = cv_image02.rows;
            	    imageSize_3.width = cv_image02.cols;
            	    imageSize_3.height = cv_image02.rows*3;



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

      			  std::vector<int> indices;
      			  pcl::removeNaNFromPointCloud(*velodyne_points, *velodyne_points, indices);


          /*	    pcl::StatisticalOutlierRemoval<pcl::PointXYZI> sor;// 创建滤波器对象

          	    sor.setInputCloud(velodyne_points);                        //设置呆滤波的点云

          	    sor.setMeanK(50);                                //设置在进行统计时考虑查询点邻近点数

          	    sor.setStddevMulThresh(10.0);                    //设置判断是否为离群点的阈值
          	    sor.setNegative(false);
          	    sor.filter(*velodyne_points);

          	    sor.setNegative(true);
          	    sor.filter(*velodyne_points_remove);//这个操作50个点需要1秒多，10个点需要0.5秒   这个操作不实时，就不用了

*/


    			  int cloudSize = velodyne_points->points.size();
    			  pcl::PointXYZI point;
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
    										project_left_img.at<cv::Vec4f>(v, u)[1] = velodyne_points->points[i].y;
    										project_left_img.at<cv::Vec4f>(v, u)[2] = velodyne_points->points[i].z;
    										project_left_img.at<cv::Vec4f>(v, u)[3] = velodyne_points->points[i].intensity;
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
    												project_left_img.at<cv::Vec4f>(v_k, u)[0] = velodyne_points->points[i].x;//在head_ground坐标系下
    												project_left_img.at<cv::Vec4f>(v_k, u)[1] = velodyne_points->points[i].y;
    												project_left_img.at<cv::Vec4f>(v_k, u)[2] = velodyne_points->points[i].z;
    												project_left_img.at<cv::Vec4f>(v_k, u)[3] = velodyne_points->points[i].intensity;
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

    		//这里三角剖分检测障碍物,上面插入的时候已经剖分完了
    		//对mask有投影的点进行遍历判断是否是障碍点，这个不好操作，可以从三角剖分的索引为4的点开始遍历
    					  inset_subdiv_point_num = inset_subdiv_point_num+4;
    					  //cout<<"inset_subdiv_point_num: "<<inset_subdiv_point_num<<endl;
    					  int edge_num;


    				////遍历判断是不是障碍物
    					 // clock_t time_stt8 = clock();
    					  bool search_state_flag=true;
    					  for(int i=4;i<inset_subdiv_point_num ;i++)//遍历判断是不是障碍物
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
    								//下面判断距离，置位掩码
    								org_point_3d.x = project_left_img.at<cv::Vec4f>(outer_vtx_from_edge.y, outer_vtx_from_edge.x)[0];
    								org_point_3d.y = project_left_img.at<cv::Vec4f>(outer_vtx_from_edge.y, outer_vtx_from_edge.x)[1];
    								org_point_3d.z = project_left_img.at<cv::Vec4f>(outer_vtx_from_edge.y, outer_vtx_from_edge.x)[2];
    								dst_point_3d.x = project_left_img.at<cv::Vec4f>(pt[1].y, pt[1].x)[0];
    								dst_point_3d.y = project_left_img.at<cv::Vec4f>(pt[1].y, pt[1].x)[1];
    								dst_point_3d.z = project_left_img.at<cv::Vec4f>(pt[1].y, pt[1].x)[2];//

    								double x_m = org_point_3d.x-dst_point_3d.x;
    								double y_m = org_point_3d.y-dst_point_3d.y;
    								double z_m = org_point_3d.z-dst_point_3d.z;

    								double distance = sqrt(x_m*x_m+ y_m*y_m+z_m*z_m);
    								double cos_m = -z_m / distance;//fabs(z_m) / distance
    								/*if(distance<=min_edge_distance)
    								{
    									cout<<"distance： "<<distance<<endl;
    									cout<<"distance<=min_edge_distance"<<endl;
    								}*/
    								if (cos_m>cos_thresh_theta&&distance>min_edge_distance)//||org_point_3d.z>obstacle_thresh_z
    									//TODO 这里加上其他特征进行判断，如Z(激光雷达的高度：1.73m,注意这里距离不一样，高度允许的值应该有差别，考虑横滚和俯仰角！！！！)，崎岖度评估数据！！！！
    								{
    									obstacle_flag = true;
    									break;
    								}

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

        								dst_point_3d.x = project_left_img.at<cv::Vec4f>(pt2[1].y, pt2[1].x)[0];
        								dst_point_3d.y = project_left_img.at<cv::Vec4f>(pt2[1].y, pt2[1].x)[1];
        								dst_point_3d.z = project_left_img.at<cv::Vec4f>(pt2[1].y, pt2[1].x)[2];//

        								double x_m = org_point_3d.x-dst_point_3d.x;
        								double y_m = org_point_3d.y-dst_point_3d.y;
        								double z_m = org_point_3d.z-dst_point_3d.z;

        								double distance = sqrt(x_m*x_m+ y_m*y_m+z_m*z_m);
        								double cos_m = fabs(z_m) / distance;
        								/*if(distance>min_edge_distance)
        								{
        									cout<<"distance： "<<distance<<endl;
        									cout<<"distance>min_edge_distance"<<endl;
        								}*/
        								if (cos_m>cos_thresh_theta&&distance>min_edge_distance)
        									//TODO 这里加上其他特征进行判断，如Z(激光雷达的高度：1.73m,注意这里距离不一样，高度允许的值应该有差别，考虑横滚和俯仰角！！！！)，崎岖度评估数据！！！！
        								{
        									obstacle_flag = true;
        									break;
        								}

        								//第三层连接：目前看加第三层效果还没有提升的情况

                			    	/*	  cv::Point2f outer_vtx_from_edge2;
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

              								dst_point_3d.x = project_left_img.at<cv::Vec4f>(pt3[1].y, pt3[1].x)[0];
              								dst_point_3d.y = project_left_img.at<cv::Vec4f>(pt3[1].y, pt3[1].x)[1];
              								dst_point_3d.z = project_left_img.at<cv::Vec4f>(pt3[1].y, pt3[1].x)[2];//

              								double x_m = org_point_3d.x-dst_point_3d.x;
              								double y_m = org_point_3d.y-dst_point_3d.y;
              								double z_m = org_point_3d.z-dst_point_3d.z;

              								double distance = sqrt(x_m*x_m+ y_m*y_m+z_m*z_m);
              								double cos_m = fabs(z_m) / distance;
              								if (cos_m>cos_thresh_theta)
              									//TODO 这里加上其他特征进行判断，如Z(激光雷达的高度：1.73m,注意这里距离不一样，高度允许的值应该有差别，考虑横滚和俯仰角！！！！)，崎岖度评估数据！！！！
              								{
              									obstacle_flag = true;
              									break;
              								}
              								temp3 = subdiv.nextEdge(temp3);
              								        								//cout<<"temp: "<<temp<<endl;

              							}while(edge_num_temp2 != temp3);

                							if (obstacle_flag)
                							    break;*/



          								temp2 = subdiv.nextEdge(temp2);
          								        								//cout<<"temp: "<<temp<<endl;

          							}while(edge_num_temp1 != temp2);


          							if (obstacle_flag)
          								break;
    								//line(img, pt[0], pt[1], delaunay_color, 1, CV_AA, 0);
    								temp = subdiv.nextEdge(temp);
    								//cout<<"!!!!!!!!!!!!!!!!!!!!!!!temp"<<temp<<endl;

    							}while(edge_num != temp);//查找所有和给定点相连的边，判断障碍物属性
    							//cout<<"test7"<<endl;

    							 if (obstacle_flag)
    							 {
    								 //置位障碍物掩码／／obstale_mask_3C
    								if(process_for_view)
    								{
										obstale_mask_3C.at<cv::Vec3b>(outer_vtx_from_edge.y, outer_vtx_from_edge.x)[0] = 255;
										obstale_mask_3C.at<cv::Vec3b>(outer_vtx_from_edge.y, outer_vtx_from_edge.x)[1] = 0;
										obstale_mask_3C.at<cv::Vec3b>(outer_vtx_from_edge.y, outer_vtx_from_edge.x)[2] = 0;
										cv::circle(obstale_mask_3C_inv, outer_vtx_from_edge, 1, cv::Scalar(255,0,0), CV_FILLED, CV_AA, 0);//方便可视化加粗了点
    								}
    								obstale_mask_1C.at<uchar>(outer_vtx_from_edge.y, outer_vtx_from_edge.x) = 255;//确定是障碍点

    								pcl::PointXYZHSV point_temp;
    								point_temp.x = project_left_img.at<cv::Vec4f>(outer_vtx_from_edge.y, outer_vtx_from_edge.x)[0];
    								point_temp.y = project_left_img.at<cv::Vec4f>(outer_vtx_from_edge.y, outer_vtx_from_edge.x)[1];
    								point_temp.z = project_left_img.at<cv::Vec4f>(outer_vtx_from_edge.y, outer_vtx_from_edge.x)[2];
    								point_temp.h = project_left_img.at<cv::Vec4f>(outer_vtx_from_edge.y, outer_vtx_from_edge.x)[3];
    								point_temp.s = outer_vtx_from_edge.x;
    								point_temp.v = outer_vtx_from_edge.y;
    								result_delaunay_obstacle_points->push_back(point_temp);//确定是障碍点

    							 }
    							 else
    							 {
    								 //没有障碍物掩码
    								if(process_for_view)
    								{
										obstale_mask_3C.at<cv::Vec3b>(outer_vtx_from_edge.y, outer_vtx_from_edge.x)[2] = 255;
										cv::circle(obstale_mask_3C_inv, outer_vtx_from_edge, 1, cv::Scalar(0,0,255), CV_FILLED, CV_AA, 0);//方便可视化加粗了点
    								}
    								obstale_mask_1C.at<uchar>(outer_vtx_from_edge.y, outer_vtx_from_edge.x) = 100;
    								pcl::PointXYZHSV point_temp;
    								point_temp.x = project_left_img.at<cv::Vec4f>(outer_vtx_from_edge.y, outer_vtx_from_edge.x)[0];
    								point_temp.y = project_left_img.at<cv::Vec4f>(outer_vtx_from_edge.y, outer_vtx_from_edge.x)[1];
    								point_temp.z = project_left_img.at<cv::Vec4f>(outer_vtx_from_edge.y, outer_vtx_from_edge.x)[2];
    								point_temp.h = project_left_img.at<cv::Vec4f>(outer_vtx_from_edge.y, outer_vtx_from_edge.x)[3];
    								point_temp.s = outer_vtx_from_edge.x;
    								point_temp.v = outer_vtx_from_edge.y;
    								result_delaunay_ground_points->push_back(point_temp);//准备发布的点云

    							 }



    					  }

    					  pcl::PointCloud<pcl::PointXYZHSV>::Ptr filteredInputPclCloud_in_range(new pcl::PointCloud<pcl::PointXYZHSV>);
    					  pcl::PointCloud<pcl::PointXYZHSV>::Ptr filteredInputPclCloud_out_range(new pcl::PointCloud<pcl::PointXYZHSV>);
    					  pcl::PointCloud<pcl::PointXYZHSV>::Ptr filteredInputPclCloud_in_range1(new pcl::PointCloud<pcl::PointXYZHSV>);


    					  for (int i = 0; i < result_delaunay_ground_points->size(); i++) {
    					         double temp_x = result_delaunay_ground_points->points[i].x;
    					         double temp_y = result_delaunay_ground_points->points[i].y;
    					         double temp_z = result_delaunay_ground_points->points[i].z;


    					         if(temp_x>box_x_min&&temp_x<box_x_max&&temp_y>box_y_min&&temp_y<box_y_max&&temp_z>box_z_min&&temp_z<box_z_max)
    					         		{


    					         			filteredInputPclCloud_in_range->push_back(result_delaunay_ground_points->points[i]);
    					         		}
    					         	 else
    					         		{

    					         		    filteredInputPclCloud_out_range->push_back(result_delaunay_ground_points->points[i]);
    					         		}

    					         if(temp_x>box_x_min1&&temp_x<box_x_max1&&temp_y>box_y_min1&&temp_y<box_y_max1&&temp_z>box_z_min1&&temp_z<box_z_max1)
    					          		{


    					        	 	    filteredInputPclCloud_in_range1->push_back(result_delaunay_ground_points->points[i]);
    					          		}

    					    }
    					 // cout<<"filteredInputPclCloud_in_range1->size():"<<filteredInputPclCloud_in_range1->size()<<endl;

  					    pcl::ModelCoefficients::Ptr coefficients1 (new pcl::ModelCoefficients);
  					    //inliers表示误差能容忍的点 记录的是点云的序号
  					    pcl::PointIndices::Ptr inliers1 (new pcl::PointIndices);
  					    // 创建一个分割器
  					    pcl::SACSegmentation<pcl::PointXYZHSV> seg1;
  					    // Optional
  					    seg1.setOptimizeCoefficients (true);
  					    // Mandatory-设置目标几何形状
  					    seg1.setModelType (pcl::SACMODEL_PLANE);
  					    //分割方法：随机采样法
  					    seg1.setMethodType (pcl::SAC_RANSAC);
  					    //设置误差容忍范围
  					    seg1.setDistanceThreshold (ransac_DistanceThreshold1);
  					    //输入点云
  					    seg1.setInputCloud (filteredInputPclCloud_in_range1);
  					    //分割点云
  					    seg1.segment (*inliers1, *coefficients1);//a * x + b * y + c * z = d 注意这个参数的第四位是-d 花了好多时间才发现

  					    if (inliers1->indices.size () == 0)//具体inliers怎么使用，参考：http://www.pclcn.org/study/shownews.php?lang=cn&id=72
  					      {
  					        PCL_ERROR ("Could not estimate a planar model for the given dataset.");
  					        return (-1);
  					      }
  					      pcl::ExtractIndices<pcl::PointXYZHSV> extract1;
					      extract1.setInputCloud(filteredInputPclCloud_in_range1);      //设置输入点云
					      extract1.setIndices(inliers1);                 //设置分割后的内点为需要提取的点集
					      extract1.setNegative(false);                  //设置提取内点而非外点
					      extract1.filter(*cloud_used_to_expand);
					      //这个执行还是很快的，2千多个点，只需要1ms左右
					  	//cout<<"filteredInputPclCloud_in_range1->size(): "<<filteredInputPclCloud_in_range1->size()<<endl;
					  	//cout<<"SAC_RANSAC time used is "<<1000*(clock()-time_stt5)/(double)CLOCKS_PER_SEC<<"ms"<<endl;//这个函数总共需要这么多时间
    					  //std::cerr << "result_delaunay_ground_points has: "
    					   //   					             << result_delaunay_ground_points->points.size () << " data points." << std::endl;


    					  //TODO 可以考虑修改为前方一定范围点进行拟合，然后判断其他点是不是在平面上（或者采用高斯过程回归，因为很多时候不是一个平面能拟合的） 最后再想办法找边界
    					   // std::cerr << "PointCloud before seg has: "
    					    //         << filteredInputPclCloud_in_range->points.size () << " data points." << std::endl;
    					  //创建一个模型参数对象，用于记录结果
    					    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    					    //inliers表示误差能容忍的点 记录的是点云的序号
    					    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    					    // 创建一个分割器
    					    pcl::SACSegmentation<pcl::PointXYZHSV> seg;
    					    // Optional
    					    seg.setOptimizeCoefficients (true);
    					    // Mandatory-设置目标几何形状
    					    seg.setModelType (pcl::SACMODEL_PLANE);
    					    //分割方法：随机采样法
    					    seg.setMethodType (pcl::SAC_RANSAC);
    					    //设置误差容忍范围
    					    seg.setDistanceThreshold (ransac_DistanceThreshold);
    					    //输入点云
    					    seg.setInputCloud (filteredInputPclCloud_in_range);
    					    //分割点云
    					    seg.segment (*inliers, *coefficients);//a * x + b * y + c * z = d 注意这个参数的第四位是-d 花了好多时间才发现

    					    if (inliers->indices.size () == 0)//具体inliers怎么使用，参考：http://www.pclcn.org/study/shownews.php?lang=cn&id=72
    					      {
    					        PCL_ERROR ("Could not estimate a planar model for the given dataset.");
    					        return (-1);
    					      }

    					      //下面是提取前面拟合之外的点云
    					      pcl::ExtractIndices<pcl::PointXYZHSV> extract; //创建点云提取对象
    					      extract.setInputCloud(filteredInputPclCloud_in_range);      //设置输入点云
    					      extract.setIndices(inliers);                 //设置分割后的内点为需要提取的点集
    					      extract.setNegative(true);                  //设置提取内点而非外点(改成提取外面的点)
    					      extract.filter(*cloud_ransac_seg_not_ground_in_range);                    //提取输出存储到cloud_p，inpit的点云数量并不变化



    					      //pcl::ExtractIndices<pcl::PointXYZHSV> extract; //创建点云提取对象
    					      extract.setInputCloud(filteredInputPclCloud_in_range);      //设置输入点云
    					      extract.setIndices(inliers);                 //设置分割后的内点为需要提取的点集
    					      extract.setNegative(false);                  //设置提取内点而非外点
    					      extract.filter(*cloud_ransac_seg_ground_in_range);     //提取输出存储到cloud_p，input的点云数量并不变化


    					      double plane_a = coefficients->values[0];
    					      double plane_b = coefficients->values[1];
    					      double plane_c = coefficients->values[2];
    					      double plane_d = -coefficients->values[3];
    					     // double plane_squ = sqrt(plane_a*plane_a+plane_b*plane_b+plane_c*plane_c);//这个恒等于1
    					    //  cout<<"fabs(plane_squ ): "<<fabs(plane_squ )<<endl;

    					      for (int i = 0; i < filteredInputPclCloud_out_range->size(); i++) {//这些认为是障碍点，所以之前对应掩码图像的结果需要修改
        					    	  //a * x + b * y + c * z = d
        					    	  	  double temp_x = filteredInputPclCloud_out_range->points[i].x;
        					    	  	  double temp_y = filteredInputPclCloud_out_range->points[i].y;
        					    	  	  double temp_z = filteredInputPclCloud_out_range->points[i].z;
        					    	  	  double predict_plane = (plane_a*temp_x + plane_b * temp_y + plane_c*temp_z - plane_d);


    	    							  //double distance = sqrt(temp_x*temp_x+ temp_y*temp_y);
        					    	  	  //这个测试了上面拟合的地面点和非地面点，通过z阈值没办法分出地面点和非地面点，都有几厘米到0.5米的值，只是非地面点有1米多的值
        					    	  	  //通过predict_plane的阈值好像可以,但是为什么是3点多而不是0？？？
        					    	  	 // if(fabs(predict_plane )/distance>((ransac_DistanceThreshold+0.06)/2+0.001)/ransac_DistanceThreshold_m )
        					    	  	if(fabs(predict_plane )>((ransac_DistanceThreshold)/2)+0.061 )//if(fabs(predict_plane )/plane_squ>((ransac_DistanceThreshold)/2)+0.061 )
        					    	  	  {
        					    	  		  //cout<<"plane_a:"<<plane_a<<" ,plane_b: "<<plane_b<<" ,plane_c: "<<plane_c<<" ,plane_d: "<<plane_d<<endl;
        					    	  		  //cout<<"temp_x:"<<temp_x<<" ,temp_y: "<<temp_y<<" ,temp_z: "<<temp_z<<endl;
        					    	  		 //cout<<"fabs(predict_plane ): "<<fabs(predict_plane )<<endl;
        					    	  		cloud_ransac_seg_not_ground_out_range->push_back(filteredInputPclCloud_out_range->points[i]);
        					    	  	  }
        					    	  	  else
        					    	  	  {
        					    	  		//cout<<"!!!ground fabs(predict_plane ): "<<fabs(predict_plane )<<endl;
        					    	  		 cloud_ransac_seg_ground_out_range->push_back(filteredInputPclCloud_out_range->points[i]);
        					    	  	  }

        					        }



    					                 	  /*      sensor_msgs::PointCloud2 pc2;
    					                 	        pc2.header.frame_id = "base_link"; //ros::this_node::getName();
    					                 	        pc2.header.stamp = ros::Time::now();
    					                 	        velodyne_points_remove->header = pcl_conversions::toPCL(pc2.header);
    					                 	        map_pub.publish(velodyne_points_remove);*/


    				      for (int i = 0; i < cloud_ransac_seg_not_ground_in_range->size(); i++) {//这些认为是障碍点，所以之前对应掩码图像的结果需要修改
				    	  	  double temp_x = cloud_ransac_seg_not_ground_in_range->points[i].x;
				    	  	  double temp_y = cloud_ransac_seg_not_ground_in_range->points[i].y;
				    	  	  double temp_z = cloud_ransac_seg_not_ground_in_range->points[i].z;
				    	  	  double predict_plane = (plane_a*temp_x + plane_b * temp_y + plane_c*temp_z - plane_d);
				    	  		 if(fabs(predict_plane )>((ransac_DistanceThreshold)/2)+0.051)//这里只要不超过前面拟合的值，就没有可回收的点
    					    	 // obstale_mask_3C.at<cv::Vec3b>(cloud_ransac_seg_not_ground_in_range->points[i].v, cloud_ransac_seg_not_ground_in_range->points[i].s)[0] = 255;
    					    	//  obstale_mask_3C.at<cv::Vec3b>(cloud_ransac_seg_not_ground_in_range->points[i].v, cloud_ransac_seg_not_ground_in_range->points[i].s)[1] = 255;
    					    	  obstale_mask_1C.at<uchar>(cloud_ransac_seg_not_ground_in_range->points[i].v, cloud_ransac_seg_not_ground_in_range->points[i].s) = 200;//255  比较确定是障碍点
				    	  		 else
				    	  		  obstale_mask_1C.at<uchar>(cloud_ransac_seg_not_ground_in_range->points[i].v, cloud_ransac_seg_not_ground_in_range->points[i].s) = 100;//待处理

    					      }

    				      for (int i = 0; i < cloud_ransac_seg_not_ground_out_range->size(); i++) {//这些认为是障碍点，所以之前对应掩码图像的结果需要修改

    					    	//  obstale_mask_3C.at<cv::Vec3b>(cloud_ransac_seg_not_ground_out_range->points[i].v, cloud_ransac_seg_not_ground_out_range->points[i].s)[0] = 255;
    					    	//  obstale_mask_3C.at<cv::Vec3b>(cloud_ransac_seg_not_ground_out_range->points[i].v, cloud_ransac_seg_not_ground_out_range->points[i].s)[1] = 125;
    					    	  obstale_mask_1C.at<uchar>(cloud_ransac_seg_not_ground_out_range->points[i].v, cloud_ransac_seg_not_ground_out_range->points[i].s) = 200;//200 255都行暂时看这个值没有影响


    					      }

    				      for (int i = 0; i < cloud_ransac_seg_ground_in_range->size(); i++) {//这些认为是地面点

    					    	  //obstale_mask_3C.at<cv::Vec3b>(cloud_ransac_seg_ground_in_range->points[i].v, cloud_ransac_seg_ground_in_range->points[i].s)[0] = 0;
    					    	  //obstale_mask_3C.at<cv::Vec3b>(cloud_ransac_seg_ground_in_range->points[i].v, cloud_ransac_seg_ground_in_range->points[i].s)[1] = 255;
    					    	  //obstale_mask_3C.at<cv::Vec3b>(cloud_ransac_seg_ground_in_range->points[i].v, cloud_ransac_seg_ground_in_range->points[i].s)[2] = 0;
    					    	  obstale_mask_1C.at<uchar>(cloud_ransac_seg_ground_in_range->points[i].v, cloud_ransac_seg_ground_in_range->points[i].s) = 100;//待确认


    					      }

    				      for (int i = 0; i < cloud_ransac_seg_ground_out_range->size(); i++) {//这些认为是需要进一步处理的地面点

    					    	//  obstale_mask_3C.at<cv::Vec3b>(cloud_ransac_seg_ground_out_range->points[i].v, cloud_ransac_seg_ground_out_range->points[i].s)[0] = 0;
    					    	//  obstale_mask_3C.at<cv::Vec3b>(cloud_ransac_seg_ground_out_range->points[i].v, cloud_ransac_seg_ground_out_range->points[i].s)[1] = 0;
    					    	//  obstale_mask_3C.at<cv::Vec3b>(cloud_ransac_seg_ground_out_range->points[i].v, cloud_ransac_seg_ground_out_range->points[i].s)[2] = 255;
    					    	  obstale_mask_1C.at<uchar>(cloud_ransac_seg_ground_out_range->points[i].v, cloud_ransac_seg_ground_out_range->points[i].s) = 100;//待确认的点

    					      }

    				      //取一个小范围的点开始


       				      for (int i = 0; i < cloud_used_to_expand->size(); i++) {//这些认为是地面点
       				    	  	  	  if(process_for_view)
       				    	  	 	 {
										  obstale_mask_3C.at<cv::Vec3b>(cloud_used_to_expand->points[i].v, cloud_used_to_expand->points[i].s)[0] = 0;
										  obstale_mask_3C.at<cv::Vec3b>(cloud_used_to_expand->points[i].v, cloud_used_to_expand->points[i].s)[1] = 255;
										  obstale_mask_3C.at<cv::Vec3b>(cloud_used_to_expand->points[i].v, cloud_used_to_expand->points[i].s)[2] = 0;
       				    	  	 	 }
        					    	  obstale_mask_1C.at<uchar>(cloud_used_to_expand->points[i].v, cloud_used_to_expand->points[i].s) = 0;//状态确定
        					    	  cv::circle(obstale_mask_3C_inv, cv::Point2f(cloud_used_to_expand->points[i].s,cloud_used_to_expand->points[i].v), 1, cv::Scalar(0,255,0), CV_FILLED, CV_AA, 0);


        					      }


    				      //TODO 下面根据连接情况对cloud_ransac_seg_ground_out_range进行进一步处理 从ground_in_range出发，
    				      //找到标记是100待确认的点，看他的连接近邻有没有标记是255的点，没有就加入新ground 标记修正为0，其他加入新not_ground,
    				      //然后用同样的方法遍历新ground 直到找不到新的能加入的点为止

    					  pcl::PointCloud<pcl::PointXYZHSV>::Ptr cloud_ransac_seg_ground_in_range_to_expanded_new(new pcl::PointCloud<pcl::PointXYZHSV>);
    					  pcl::PointCloud<pcl::PointXYZHSV>::Ptr cloud_ransac_seg_ground_in_range_expanded_ground_new(new pcl::PointCloud<pcl::PointXYZHSV>);
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
        							do
        							{
        								int index_d = subdiv.edgeDst(temp, &pt[1]);//连接边的另一个端点
        								//cout<<"pt[1]: "<<pt[1]<<endl;
        								if (index_d<4)//这条边可能连到虚点
        								{
        									temp = subdiv.nextEdge(temp);
        									//cout<<"temp: "<<temp<<endl;
        									continue;
        								}

        								if(obstale_mask_1C.at<uchar>(pt[1].y, pt[1].x)==100)//是待处理的点
        								{
        									  bool obstacle_flag1 = false;

        									  	  //注释掉下面不进行延伸判断
      						    	  	   double temp_x = project_left_img.at<cv::Vec4f>(pt[1].y, pt[1].x)[0];
      						    	  	   double temp_y = project_left_img.at<cv::Vec4f>(pt[1].y, pt[1].x)[1];
      						    	  	   double temp_z = project_left_img.at<cv::Vec4f>(pt[1].y, pt[1].x)[2];

        									//实验证明加延伸判断效果会变差，一层就是，两层就更加，因为平面假设已经很强
        									cv::Point2f outer_vtx_from_edge1;
        										int edge_num_temp1;
        										outer_vtx_from_edge1 =  subdiv.getVertex(index_d,&edge_num_temp1);

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

        	        								if(obstale_mask_1C.at<uchar>(pt1[1].y, pt1[1].x)==255)//待处理的点有连接比较明确的障碍点，认为这个很可能是障碍点
        	        								{
        	        	    							obstacle_flag1  = true;
        	        	    							break;
        	        								}
        	        								//延伸判断增加下一层连接

        	                			    		/*  cv::Point2f outer_vtx_from_edge2;
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

                	        								if(obstale_mask_1C.at<uchar>(pt3[1].y, pt3[1].x)==255)//待处理的点有连接比较明确的障碍点，认为这个很可能是障碍点
                	        								{

              	        						    	  	  double temp_x1 = project_left_img.at<cv::Vec4f>(pt3[1].y, pt3[1].x)[0];
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
              	        	    							  }

                	        								}
        	              								    temp3 = subdiv.nextEdge(temp3);
        	              								        								//cout<<"temp: "<<temp<<endl;

        	              							   }while(edge_num_temp2 != temp3);

        	                							if (obstacle_flag1)
        	                							    break;
*/

        	        								temp1 = subdiv.nextEdge(temp1);
        	        								//cout<<"!!!!!!!!!!!!!!!!!!!!!!!temp"<<temp<<endl;

        	        							}while(edge_num_temp1 != temp1);
        	        							if(obstacle_flag1)
        	        							{
        	        								if(process_for_view)
        	        								{
													   obstale_mask_3C.at<cv::Vec3b>(pt[1].y, pt[1].x)[0] = 255;
													   obstale_mask_3C.at<cv::Vec3b>(pt[1].y, pt[1].x)[1] = 255;
													   obstale_mask_3C.at<cv::Vec3b>(pt[1].y, pt[1].x)[2] = 255;
													   cv::circle(obstale_mask_3C_inv, cv::Point2f(pt[1].x,pt[1].y), 1, cv::Scalar(0,0,255), CV_FILLED, CV_AA, 0);
        	        								}
         	          					    	   obstale_mask_1C.at<uchar>(pt[1].y, pt[1].x) = 150;//有点愿意相信他是障碍点  最后把这些点补回来，这些点其实应该是地面点，但是为了延伸起点限制作用，就这样设置

        	        							}
        	        							else
        	        							{
        	        								if(process_for_view)
        	        								{
													   obstale_mask_3C.at<cv::Vec3b>(pt[1].y, pt[1].x)[0] = 0;
													   obstale_mask_3C.at<cv::Vec3b>(pt[1].y, pt[1].x)[1] = 125;
													   obstale_mask_3C.at<cv::Vec3b>(pt[1].y, pt[1].x)[2] = 255;
													   cv::circle(obstale_mask_3C_inv, cv::Point2f(pt[1].x,pt[1].y), 1, cv::Scalar(0,125,255), CV_FILLED, CV_AA, 0);
        	        								}
          	          					    	   obstale_mask_1C.at<uchar>(pt[1].y, pt[1].x) = 0;//状态确定
         	       								   pcl::PointXYZHSV point_temp;
         	       								   point_temp.x = project_left_img.at<cv::Vec4f>(pt[1].y, pt[1].x)[0];
         	       								   point_temp.y = project_left_img.at<cv::Vec4f>(pt[1].y, pt[1].x)[1];
         	       								   point_temp.z = project_left_img.at<cv::Vec4f>(pt[1].y, pt[1].x)[2];
         	       								   point_temp.h = project_left_img.at<cv::Vec4f>(pt[1].y, pt[1].x)[3];
         	       								   point_temp.s = pt[1].x;
         	       								   point_temp.v = pt[1].y;
         	       								   cloud_ransac_seg_ground_in_range_expanded_ground_new->push_back(point_temp);
        	        							}


        								}

        								temp = subdiv.nextEdge(temp);
        								//cout<<"temp: "<<temp<<endl;

        							}while(edge_num_temp != temp);

        			    	  }

        			    	  cloud_ransac_seg_ground_in_range_to_expanded_new->clear();
        			    	  *cloud_ransac_seg_ground_in_range_to_expanded_new += *cloud_ransac_seg_ground_in_range_expanded_ground_new;
        			    	  cloud_ransac_seg_ground_in_range_expanded_ground_new->clear();

    			    	  }while(cloud_ransac_seg_ground_in_range_to_expanded_new->size()>=terminate_expand_num);//!=0


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


    					        if(obstale_mask_1C.at<uchar>(centers[i].y, centers[i].x) == 0||obstale_mask_1C.at<uchar>(centers[i].y, centers[i].x) == 150)//255表示是障碍点，0表示是地面点 ，100表示待处理
    					        {
    					        	if(process_for_view)
    					        	{
    					        		fillConvexPoly(img_3C, ifacet, ground_color, 8, 0);//填充多边形
    					        	}
    					        	fillConvexPoly(img_1C, ifacet, 255, 8, 0);//之后将地平线以上的部分清0
    					        }
    					        else
    					        {
    					        	if(process_for_view)
    					        	{
    					        		fillConvexPoly(img_3C, ifacet, obstacle_color, 8, 0);//填充多边形img_1C
    					        	}
    					        	//fillConvexPoly(img_1C, ifacet, 0, 8, 0);//本来就是0
    					        }


    					    }

    					  //处理掉地平线以上的部分
    					  cv::Rect rect_horizontal(0,0,imageSize.width,horizontal_point_v);

    					  img_1C(rect_horizontal) = cv::Scalar::all(0);

    					  if(process_for_view)
    					  img_3C(rect_horizontal) = cv::Scalar::all(0);




    					  img_1C.copyTo(result_32(cv::Rect(0,0,cv_image02.cols,cv_image02.rows)));


    					  cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(erode_size, erode_size));

    					  cv::erode(img_1C,img_1C,element);//腐蚀白色
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
    					  for(size_t i = 0; i < contours.size(); i++)
    					  {
    						  rR= boundingRect(cv::Mat(contours[i]));
    					      double area = cv::contourArea(contours[i]);
    					      if (area > maxArea&&((cv_image02.cols/2)>=rR.x)&&((cv_image02.cols/2)<=rR.x+rR.width))//加上这个条件是为了防止选择非车占区域，由于摄像头朝着车的正前方，所以图像中心应该在区域内
    					      {
    					          maxArea = area;
    					          maxContour = contours[i];
    					          maxContour_index = i;
    					      }
    					  }

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

    					  if(process_for_view)
    					  cv::fillPoly(img_3C_temp, ppt, npt, 1, cv::Scalar(255, 0, 0));

    					 // imshow("maxContour", img_1C_temp);



    					  cv::dilate(img_1C_temp,img_1C,element);
    					  //cout<<"total time used is "<<1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC<<"ms"<<endl;//这个函数总共需要这么多时间

   	                     std::ofstream fout(total_time_save_filename.c_str(), std::ios::out | std::ios::app);
   	                       if (!fout.is_open())
   	                       {
   	                           std::cerr << "Can't open " << total_time_save_filename.c_str()<< " file for output.\n";
   	                           exit(EXIT_FAILURE);
   	                       }

   	                       fout <<std::scientific<< 1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC<< "\n";
   	                       fout.close();




    					  if(process_for_view)
    					  cv::dilate(img_3C_temp,img_3C_temp,element);//这个函数能操作多通道数据吗？？可以
    					  //imshow("dilate_img_3C_temp", img_3C_temp);

    					  //检测结果 彩色表示 img_3C_temp， 原图 cv_image02

    					  //到这一步img_3C_temp这个变量只有两种像素值，一个是0，一个是蓝色的道路区域

    					  //类型转化,为了直接使用CRF代码

    					  //注意opencv对应的图像的像素信息顺序是bgr,而ppm的顺序是r,g,b


    					  int W = cv_image02.size().width;
    					  int H = cv_image02.size().height;
    					  unsigned char * src_image_p = new unsigned char[W*H*3];//_p表示是指针形式
    					  unsigned char * anno_image_p =  new unsigned char[W*H*3];

    					  for (int i = 0;i<H;i++)
    						  for(int j=0;j<W;j++)
    						  {
    							  src_image_p[i*W*3+j*3]=cv_image02.at<cv::Vec3b>(i, j)[2];//Vec3b误写成Vec3d了
    							  src_image_p[i*W*3+j*3+1]=cv_image02.at<cv::Vec3b>(i, j)[1];
    							  src_image_p[i*W*3+j*3+2]=cv_image02.at<cv::Vec3b>(i, j)[0];
    							  if(img_3C_temp.at<cv::Vec3b>(i, j)[0]!=0)//只要表示道路区域和非道路区域两种颜色，并且其中一种不是0就行。
    							  {
									  anno_image_p[i*W*3+j*3]=img_3C_temp.at<cv::Vec3b>(i, j)[2];
									  anno_image_p[i*W*3+j*3+1]=img_3C_temp.at<cv::Vec3b>(i, j)[1];
									  anno_image_p[i*W*3+j*3+2]=img_3C_temp.at<cv::Vec3b>(i, j)[0];
    							  }
    							  else
    							  {
									  anno_image_p[i*W*3+j*3]=0;
									  anno_image_p[i*W*3+j*3+1]=255;
									  anno_image_p[i*W*3+j*3+2]=0;
    							  }
    						  }
    					 // const char* temp_out_ppm = full_filename_result_ppm.c_str();
    					  //writePPM( temp_out_ppm, W, H, anno_image_p );
    					  writePPM( full_filename_result_ppm.c_str(), W, H, anno_image_p );
    					  writePPM( full_filename_src_ppm.c_str(), W, H, src_image_p );
    					  //试验结果，直接按下面的方法，颜色的作用太大，感觉得到的效果和混合高斯模型的效果差不多，总的来说就是利用的信息太弱了。可能需要采用别的
    					  //受阴影影响小的颜色空间并结合三维坐标信息进行CRF后处理，另外迭代次数，加权参数等也是需要调节或者学习的量
    					  //然后是基于颜色的分类器也是需要进一步研究的问题
    					  //另外是复现别的方法，效果真的有那么好？？？
    					    int M = 21;//只有两种类型，背景和道路

    					    nColors = 0;//一幅图片初始化一下
    						/////////// Put your own unary classifier here! ///////////
    						float * unary = classify( anno_image_p, W, H, M );//这里统计标签时是不考虑像素值为0的情况的。通过粗标记数据学习分类器  得到CRF 的unary部分
    						///////////////////////////////////////////////////////////



    						 clock_t time_stt_crf = clock();

    						// Setup the CRF model
    						DenseCRF2D crf(W, H, M);//采用的是2D 这个类
    						// Specify the unary potential as an array of size W*H*(#classes)
    						// packing order: x0y0l0 x0y0l1 x0y0l2 .. x1y0l0 x1y0l1 ...
    						crf.setUnaryEnergy( unary );
    						// add a color independent term (feature = pixel location 0..W-1, 0..H-1)
    						// x_stddev = 3
    						// y_stddev = 3
    						// weight = 3
    						crf.addPairwiseGaussian( 3, 3, 3 );//最难懂的就是这两个函数了，用了参考文献的方法   本来需要学习的参数（权值和标准差都是直接传进去的，也就是本文没有参数学习的部分，后一篇文章有参数学习代码也开源）
    						//这里这个权值的效果是大小变化影响还是挺大的，初步的感觉是这个权值越小越能容忍像素标记的离散程度，越大标记越趋向于成为团状


    						// add a color dependent term (feature = xyrgb)
    						// x_stddev = 60
    						// y_stddev = 60
    						// r_stddev = g_stddev = b_stddev = 20
    						// weight = 10
    						crf.addPairwiseBilateral( 60, 60, 20, 20, 20, src_image_p, 10 );//从代码实现上看，这个函数和上面的函数算法上没有看到区别啊，只是考虑的特征向量不同了，上面是纯位置，这个是位置和颜色
    						//运行发现代码运行时间和这里设置的迭代次数没有关系，1到十次的时间复杂度竟然是一样的，就差十来毫秒，当设置迭代次数为20次时反而运行时间变短了，而且迭代次数越多，时间反而在减少？？？？哦，原来我改的参数是权值
    						//从测试看权值设置为10的效果相对是最好的，这个权值越小离初始状态越近，成团状，当增加这个值时，接近10的时候是最接近椅子的状态的，再往后增加椅子的地方容易一步步被背景侵蚀，到100之后的效果几乎已经不再变化了

    						// Do map inference
    						short * map = new short[W*H];
    						crf.map(1, map);//map应该记录的每个像素的标签 10表示迭代次数
    						//迭代测试差不多40ms迭代一次，到10次的时候差不多已经收敛了，在执行几乎不再变化，150次的效果和10次差不多



    						std::cout<<"image_process time used is "<<1000*(clock()-time_stt_crf)/(double)CLOCKS_PER_SEC<<"ms"<<std::endl;//测试一张图片需要400多毫秒



    						// Store the result
    						unsigned char *res = colorize( map, W, H );
    						//writePPM( argv[3], W, H, res );

    						cv::Mat img_3C_temp_crf(cv_image02.size(), CV_8UC3);
      					  for (int i = 0;i<H;i++)
      						  for(int j=0;j<W;j++)
      						  {

      							img_3C_temp_crf.at<cv::Vec3b>(i, j)[0] = res[i*W*3+j*3+2];
      							img_3C_temp_crf.at<cv::Vec3b>(i, j)[1] = res[i*W*3+j*3+1];
      							img_3C_temp_crf.at<cv::Vec3b>(i, j)[2] = res[i*W*3+j*3+0];


      						  }
							  cv::imwrite(full_filename_output_crf,img_3C_temp_crf);





							  cout<<"nColors: "<<nColors<<endl;
    						delete[] src_image_p;//这些变量都是指针
    						delete[] anno_image_p;
    						delete[] res;
    						delete[] map;
    						delete[] unary;















    					  img_1C.copyTo(result_32(cv::Rect(0,cv_image02.rows*2,cv_image02.cols,cv_image02.rows)));



    					  cv::Mat show;
    					  addWeighted(cv_image02,1,img_3C_temp,0.5,0.,show);//这里图像大小可能不一样，已经改到一样了 有图像后处理的结果
    					  //imshow("result_show", show);



    					  cv::Mat show2;
    					  addWeighted(cv_image02,1,img_3C,0.5,0.,show2);//这里图像大小可能不一样，已经改到一样了 没有图像后处理的结果

    					  //后处理时间消耗10ms以内




    					  cv::imwrite(full_filename_output,img_1C);





    					  //cv::imwrite(full_filename_output_3C_mask,obstale_mask_3C);
    					  cv::circle(img_3C, cv::Point(horizontal_point_u,horizontal_point_v), 3, cv::Scalar(0,0,0), CV_FILLED, CV_AA, 0);
    					 // cout<<"horizontal_point_u,horizontal_point_v: "<<horizontal_point_u <<", "<<horizontal_point_v<<endl;

    					  cv::Rect temp_rect(0,0,cv_image02.cols,cv_image02.rows);//注意cv_image02的大小是不固定的
    					  //result_3(temp_rect)=cv_image02.clone();//没考过去，不知道为什么
    					  cv_image02.copyTo(result_3(temp_rect));//这样就考过来了 注意result_3的列数需要保证比cv_image02.cols大或者相等 这里叠加的是图像后处理过的结果
    					  obstale_mask_3C_inv.copyTo(result_3(cv::Rect(0,cv_image02.rows,cv_image02.cols,cv_image02.rows)));
    					  //img_3C.copyTo(result_3(cv::Rect(0,cv_image02.rows*2,cv_image02.cols,cv_image02.rows)));//注意这个是没有经过图像后处理的结果
    					  show.copyTo(result_3(cv::Rect(0,cv_image02.rows*2,cv_image02.cols,cv_image02.rows)));//注意这个是没有经过图像后处理的结果
    					  cv::imwrite(full_filename_output_3C_mask,result_3);


    					  show2.copyTo(result_33(temp_rect));
    					  show.copyTo(result_33(cv::Rect(0,cv_image02.rows,cv_image02.cols,cv_image02.rows)));
    					  obstale_mask_3C.copyTo(result_33(cv::Rect(0,cv_image02.rows*2,cv_image02.cols,cv_image02.rows)));



    					  //imshow("obstale_mask_3C", obstale_mask_3C);
    					  //imshow("obstale_mask_3C_inv_", obstale_mask_3C_inv);
    					 // cv::imshow("cv_image02",cv_image02);
    					  //cv::imshow("paintVoronoi",img);
    					 // cv::imshow("paintVoronoi_1C",img_1C);
    					  cv::imshow("result_3",result_3);
    					  cv::imshow("result_32",result_32);
    					  cv::imshow("result_33",result_33);

    					  cv::waitKey(10);



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

