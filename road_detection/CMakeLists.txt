cmake_minimum_required(VERSION 2.8.3)
project(road_detection)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
set(CMAKE_BUILD_TYPE Release)
set(BUILD_STATIC_LIBS ON)
set(BUILD_SHARED_LIBS OFF)
#set(OpenCV_DIR /usr/local/opencv320/share/OpenCV) #想使用2的版本这一句去掉及find_package(OpenCV 3.2 REQUIRED)改成find_package(OpenCV  REQUIRED) 即可
## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  pcl_conversions
  pcl_ros
  roscpp
  sensor_msgs
  serial
  std_msgs
  tf
  tf2
)

 find_package(PCL REQUIRED)
 #find_package(OpenCV 3.2 REQUIRED)
find_package(OpenCV  REQUIRED)
 include_directories(
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
   "/usr/include/eigen3"
	
)
link_directories(
  ${PCL_LIBRARY_DIRS}
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a run_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a run_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   sensor_msgs#   std_msgs
# )

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a run_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES road_detection
#  CATKIN_DEPENDS cv_bridge image_transport pcl_conversions pcl_ros roscpp sensor_msgs serial std_msgs tf tf2
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  #include
  ${catkin_INCLUDE_DIRS}
)

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/road_detection.cpp
# )

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
# add_executable(${PROJECT_NAME}_node src/road_detection_node.cpp)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
# target_link_libraries(${PROJECT_NAME}_node
#   ${catkin_LIBRARIES}
# )

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS ${PROJECT_NAME} ${PROJECT_NAME}_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_road_detection.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)

add_library( densecrf src/util.cpp src/bipartitedensecrf.cpp src/permutohedral.cpp src/densecrf.cpp src/filter.cpp )
add_executable( kitti_road_detection_by_lidar_crf  src/kitti_road_detection_by_lidar_crf.cpp src/util_ppm.cpp)
target_link_libraries( kitti_road_detection_by_lidar_crf densecrf ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} )


add_executable(transform_kitti_points2_2_left_camera_for_obstacle_and_ground_detection src/transform_kitti_points2_2_left_camera_for_obstacle_and_ground_detection.cpp)
target_link_libraries(transform_kitti_points2_2_left_camera_for_obstacle_and_ground_detection ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(kitti_road_detection_by_lidar src/kitti_road_detection_by_lidar.cpp)
target_link_libraries(kitti_road_detection_by_lidar ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})


add_executable(analyze_while_expand src/analyze_while_expand.cpp)
target_link_libraries(analyze_while_expand ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(kdtree_delaunay_ransac_time_test src/kdtree_delaunay_ransac_time_test.cpp)
target_link_libraries(kdtree_delaunay_ransac_time_test ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})


add_executable(analyze_while_expand_paper src/analyze_while_expand_paper.cpp)
target_link_libraries(analyze_while_expand_paper ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
add_executable(kitti_road_detection_by_lidar_paper src/kitti_road_detection_by_lidar_paper.cpp)
target_link_libraries(kitti_road_detection_by_lidar_paper ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})






#add_executable(opencv_GMM_EM_cluster src/opencv_GMM_EM_cluster.cpp)
#target_link_libraries(opencv_GMM_EM_cluster   ${OpenCV_LIBRARIES} )#${catkin_LIBRARIES} 有这个能编译，但是执行会有问题，去掉没问题，但是还是得用ROS 啊？？ 这是opencv3.0时候的注释

#add_executable(opencv_GMM_EM_picture_segment src/opencv_GMM_EM_picture_segment.cpp)
#target_link_libraries(opencv_GMM_EM_picture_segment ${OpenCV_LIBRARIES} )#注意$前得有空格

add_executable(opencv_Kmeans_cluster src/opencv_Kmeans_cluster.cpp)
target_link_libraries(opencv_Kmeans_cluster  ${OpenCV_LIBRARIES} )

add_executable(opencv_delaunay2 src/opencv_delaunay2.cpp)
target_link_libraries(opencv_delaunay2  ${OpenCV_LIBRARIES} )

add_executable(opencv_GMM_CvEM_cluster src/opencv_GMM_CvEM_cluster.cpp)
target_link_libraries(opencv_GMM_CvEM_cluster   ${OpenCV_LIBRARIES} )

add_executable(meanshiftfilter src/meanshiftfilter.cpp)
target_link_libraries(meanshiftfilter   ${OpenCV_LIBRARIES} )
