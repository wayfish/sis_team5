/*
#include "ros/ros.h"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>
#include <stdlib.h>
*/
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "pcl_ros/point_cloud.h"
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <cstring>
#include <iostream> 
#include <vector>
#include <pcl/io/ply_io.h>
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <tf/transform_listener.h>
#include <ros/ros.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <Eigen/Geometry>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Float32MultiArray.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/PoseStamped.h>
#include <iostream>
#include <tf/transform_broadcaster.h>
#include <pcl/PolygonMesh.h>
#include <tf/transform_listener.h>
#include "tf/transform_datatypes.h"
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/Pose.h"

using namespace pcl;

class PoseEstimator{
	private:
		//subscribers, subscribe for current point cloud and mask	
		ros::Subscriber scene_cloud_sub;		
		ros::Subscriber object_maskr_sub;
		ros::Subscriber object_maskg_sub;
		ros::Subscriber object_maskb_sub;
		
	
		//publishers
		ros::Publisher scene_publisher;
		ros::Publisher kinder_publisher;
		ros::Publisher kusan_publisher;
		ros::Publisher doublemint_publisher;
		ros::Publisher icp_result_publisher;
		ros::Publisher icp_result2_publisher;
		ros::Publisher icp_result3_publisher;
		ros::Publisher target_publisher;
		ros::Publisher target2_publisher;
		ros::Publisher target3_publisher;
		ros::Publisher test_publisher;
		ros::Publisher pose_publisher; 

		//pointers, points to object point cloud
		pcl::PointCloud<PointXYZRGB>::Ptr scene_cloud;
		pcl::PointCloud<PointXYZRGB>::Ptr target_cloud;
		pcl::PointCloud<PointXYZRGB>::Ptr target2_cloud;
		pcl::PointCloud<PointXYZRGB>::Ptr target3_cloud;
		pcl::PointCloud<PointXYZRGB>::Ptr model_cloud;

		pcl::PointCloud<PointXYZRGB>::Ptr kusan_model;
		pcl::PointCloud<PointXYZRGB>::Ptr doublemint_model;
		pcl::PointCloud<PointXYZRGB>::Ptr kinder_model;
		pcl::PointCloud<PointXYZRGB>::Ptr icp_result_cloud;
		pcl::PointCloud<PointXYZRGB>::Ptr icp_result2_cloud;
		pcl::PointCloud<PointXYZRGB>::Ptr icp_result3_cloud;
		pcl::PointCloud<PointXYZRGB>::Ptr test_cloud;

   		cv_bridge::CvImagePtr cv_ptr; //base_to_target
		tf::TransformBroadcaster br;    
		tf::Transform tf_scene;
		tf::Transform tf_base;
		tf::Transform tf_final;

		tf::TransformListener listener;
		Eigen::Matrix4f car_to_camera_transform;

		tf::Matrix3x3 arm_to_camera_rotation;

		double distance1, distance2, distance3;
		
		bool ifMove;
		
		void updateScenePoints(const sensor_msgs::PointCloud2::ConstPtr& cloud);
		void loadModelHelper(pcl::PointCloud<PointXYZRGB>::Ptr object, std::string object_name);
		void loadModels();
		void filterObjectPointCloud(PointCloud<PointXYZRGB>::Ptr cloud, cv_bridge::CvImagePtr mask, std::string object);
		std::vector<double> icpAlignByPointToPlaneMethod (PointCloud<PointXYZRGB>::Ptr sourceCloud, PointCloud<PointXYZRGB>::Ptr targetCloud, PointCloud<PointXYZRGB>::Ptr cloud_source_trans_normals );
		void addNormal(PointCloud<PointXYZRGB>::Ptr cloud, PointCloud<PointXYZRGBNormal>::Ptr cloud_with_normals);
		void getObjectsPointCloudWithMaskr(const sensor_msgs::Image::ConstPtr& mask);
		void getObjectsPointCloudWithMaskg(const sensor_msgs::Image::ConstPtr& mask);
		void getObjectsPointCloudWithMaskb(const sensor_msgs::Image::ConstPtr& mask);
		void pointCloudPreprocessing(PointCloud<PointXYZRGB>::Ptr object_cloud);
		void setTF();
		void publishFinalTf(std::vector<double> pose);

	public:	
		PoseEstimator();
};



