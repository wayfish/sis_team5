#include "pose_estimate.h"



int main(int argc, char** argv){
  ros::init(argc, argv, "demo");

  //tf::Matrix3x3 m;
  PoseEstimator pose_estimator;

/*
  pcl::PointCloud<PointXYZRGB>::Ptr test(new PointCloud<PointXYZRGB>);
  std::string model_folder_path("/home/evanyeh/Documents/NCTU/sis_lab/catkin_ws/src/project_test/models/");
  pcl::PLYReader Reader;

  //Load kinder model
  std::string kinder_path = model_folder_path + "kusan.ply";
  pcl::PolygonMesh mesh;
  pcl::io::loadPLYFile(kinder_path, mesh);
  pcl::fromPCLPointCloud2( mesh.cloud, *test );

//  pcl::io::loadPLYFile<pcl::PointXYZRGB>(kinder_path, *test);
*/


  ros::spin();

  return 0;
}
