#include "pose_estimate.h"
#include "std_msgs/String.h"
#include "grasp/data.h"
#include "std_msgs/Int64.h"

bool ob1_exi = true;
bool ob2_exi = true;
bool ob3_exi = true;

void PoseEstimator::publishFinalTf(std::vector<double> pose){
	tf_final.setRotation( tf::Quaternion(pose[0], pose[1], pose[2], pose[3]) );
	tf_final.setOrigin( tf::Vector3(pose[4], pose[5], pose[6]) );

  br.sendTransform(tf::StampedTransform(tf_final, ros::Time::now(), "/my_base_frame", "/final_result_frame"));


}

void PoseEstimator::setTF(){
	tf::Matrix3x3 rotation; 
	rotation.setEulerYPR(90*(3.14/180), 0, -142*(3.14/180));
  tf::Quaternion q;
  rotation.getRotation(q);

  //tf_scene.setRotation( tf::Quaternion(0, 0, 0, 1) );
	tf_scene.setOrigin( tf::Vector3(0, 0, 0.45) );
	tf_scene.setRotation(q);

	rotation.setEulerYPR(180*(3.14/180), 0, 0);
  rotation.getRotation(q);
 	tf_base.setOrigin( tf::Vector3(-0.12, 0.0, 0.0) );
  tf_base.setRotation(q);


}

PoseEstimator::PoseEstimator(){
	//initial point cloud
  	scene_cloud.reset(new PointCloud<PointXYZRGB>());
  	kusan_model.reset(new PointCloud<PointXYZRGB>());
  	doublemint_model.reset(new PointCloud<PointXYZRGB>());
  	kinder_model.reset(new PointCloud<PointXYZRGB>());
  	target1_cloud.reset(new PointCloud<PointXYZRGB>());
	target2_cloud.reset(new PointCloud<PointXYZRGB>());
	target3_cloud.reset(new PointCloud<PointXYZRGB>());
  	model_cloud.reset(new PointCloud<PointXYZRGB>());
	test_cloud.reset(new PointCloud<PointXYZRGB>());
	icp_result1_cloud.reset( new pcl::PointCloud<PointXYZRGB> );
	icp_result2_cloud.reset( new pcl::PointCloud<PointXYZRGB> );
	icp_result3_cloud.reset( new pcl::PointCloud<PointXYZRGB> );
	
	distance1 = 1000;
	distance2 = 1000;
	distance3 = 1000;
	
	//Memebers initial
	setTF();
  	loadModels();

	//initial subscribe and pulisher
  	ros::Time::init();
  	ros::NodeHandle nh;
  
  	scene_publisher = nh.advertise<sensor_msgs::PointCloud2> ("/camera/scene", 1);

	kinder_publisher = nh.advertise<sensor_msgs::PointCloud2> ("/model/kinder", 1);
  	kusan_publisher = nh.advertise<sensor_msgs::PointCloud2> ("/model/kusan", 1);
  	doublemint_publisher = nh.advertise<sensor_msgs::PointCloud2> ("/model/doublemint", 1);
  
  	icp_result1_publisher = nh.advertise<sensor_msgs::PointCloud2> ("/model/icp_result", 1);
	icp_result2_publisher = nh.advertise<sensor_msgs::PointCloud2> ("/model/icp_result", 1);
	icp_result3_publisher = nh.advertise<sensor_msgs::PointCloud2> ("/model/icp_result", 1);
	
	target1_publisher = nh.advertise<sensor_msgs::PointCloud2> ("/camera/target", 1);
	target2_publisher = nh.advertise<sensor_msgs::PointCloud2> ("/camera/target", 1);
	target3_publisher = nh.advertise<sensor_msgs::PointCloud2> ("/camera/target", 1);
	test_publisher = nh.advertise<sensor_msgs::PointCloud2> ("/model/test", 1);
	pose_publisher1 = nh.advertise<geometry_msgs::PoseStamped> ("kusan_position", 1);
	pose_publisher2 = nh.advertise<geometry_msgs::PoseStamped> ("kinder_position", 1);
	pose_publisher3 = nh.advertise<geometry_msgs::PoseStamped> ("double_mint_position", 1);
	
	scene_cloud_sub = nh.subscribe("/camera/depth_registered/points", 1, &PoseEstimator::updateScenePoints, this); //keyword: callback function in class 
	object_mask_sub1 = nh.subscribe("object_mask", 1, &PoseEstimator::getObjectsPointCloudWithMask1, this);
	object_mask_sub2 = nh.subscribe("object_mask", 1, &PoseEstimator::getObjectsPointCloudWithMask2, this);
	object_mask_sub3 = nh.subscribe("object_mask", 1, &PoseEstimator::getObjectsPointCloudWithMask3, this);
	
	get_id= nh.subscribe("object_id", 1,&PoseEstimator::getObjectsID, this);// 
	ros::ServiceClient client = nh.serviceClient<grasp::data>("grasp");
	
  	car_to_camera_transform = Eigen::Matrix4f::Identity();

	ifMove = false;

	std::cout << "initial finish" << std::endl;
}

void PoseEstimator::getObjectsID(const std_msgs::Int64::ConstPtr& msg_ID){
	static int ob_ID = msg_ID; 

	

}
//main callback
void PoseEstimator::getObjectsPointCloudWithMask1(const sensor_msgs::Image::ConstPtr& mask){
  int cloud_size_thres = 100; //if the number of the decteted object point cloud is smaller than the thres, threating them as nothing 
  cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(mask); 

  //Generate object point cloud
  pcl::copyPointCloud(*scene_cloud, *target_cloud);
  filterObjectPointCloud(target_cloud, cv_ptr, 1);
	//filter
  pcl::VoxelGrid<PointXYZRGB> sor;
  sor.setInputCloud (target_cloud);
  sor.setLeafSize (0.003f, 0.003f, 0.003f);
  sor.filter (*target_cloud);  
	
		if (target_cloud->points.size() > cloud_size_thres){
		  //std::vector<double> pose = icpAlignByPointToPlaneMethod( target_cloud, kusan_model, icp_result_cloud); 
		    			
			std::vector<double> pose = icpAlignByPointToPlaneMethod( kusan_model, target_cloud, icp_result_cloud);

			std::cout<<"pose:"<<std::endl;
			for(int i=0; i<pose.size(); i++){
			std::cout<<pose[i]<<" ";
			}
			std::cout<<std::endl;
			//distance1 = sqrt(pose[4]*pose[4] + pose[5]*pose[5] + pose[6]*pose[6]);
			if(ob1_exi){
			geometry_msgs::PoseStamped msg;	
			msg.pose.position.x = pose[4];
			msg.pose.position.y = pose[5];
			msg.pose.position.z = pose[6];
			pose_publisher1.publish(msg);
				
			if ( ob_ID==1){
				geometry_msgs::Pose msg_g;
				msg_g.position.x = pose[4];
				msg_g.position.y = pose[5];
				msg_g.position.z = pose[6];
				msg_g.orientation.x = 0;
				msg_g.orientation.y = 0;
				msg_g.orientation.z = 0;
				msg_g.orientation.w = 1;
				grasp::data srv;
				srv.request.pose = msg_g;
				srv.request.id = ob_ID;
				client.call(srv);}
			}
			publishFinalTf(pose);
		}
 
	target_cloud->header.frame_id = "/my_camera_frame";
	pcl_conversions::toPCL(ros::Time::now(), target_cloud->header.stamp);
	target_publisher.publish(*target_cloud);
	//printf("target cloud size: %ld\n", target_cloud->points.size());

	icp_result_cloud->header.frame_id = "/my_camera_frame";
	pcl_conversions::toPCL(ros::Time::now(), icp_result_cloud->header.stamp);
	icp_result_publisher.publish(*icp_result_cloud);
	
}

void PoseEstimator::getObjectsPointCloudWithMask2(const sensor_msgs::Image::ConstPtr& mask){
  int cloud_size_thres = 100; //if the number of the decteted object point cloud is smaller than the thres, threating them as nothing 
  cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(mask); 

  //Generate object point cloud
  pcl::copyPointCloud(*scene_cloud, *target_cloud);
  filterObjectPointCloud(target_cloud, cv_ptr, 2);
	//filter
  pcl::VoxelGrid<PointXYZRGB> sor;
  sor.setInputCloud (target_cloud);
  sor.setLeafSize (0.003f, 0.003f, 0.003f);
  sor.filter (*target_cloud);  
	
		if (target_cloud->points.size() > cloud_size_thres){
		  //std::vector<double> pose = icpAlignByPointToPlaneMethod( target_cloud, kusan_model, icp_result_cloud); 

			std::vector<double> pose = icpAlignByPointToPlaneMethod( kinder_model, target_cloud, icp_result_cloud);	

			std::cout<<"pose:"<<std::endl;
			for(int i=0; i<pose.size(); i++){
			std::cout<<pose[i]<<" ";
			}
			std::cout<<std::endl;
			//distance1 = sqrt(pose[4]*pose[4] + pose[5]*pose[5] + pose[6]*pose[6]);
			if(ob2_exi){
			geometry_msgs::PoseStamped msg;	
			msg.pose.position.x = pose[4];
			msg.pose.position.y = pose[5];
			msg.pose.position.z = pose[6];
			pose_publisher2.publish(msg);
				
			if ( ob_ID==2){
				geometry_msgs::Pose msg_g;
				msg_g.position.x = pose[4];
				msg_g.position.y = pose[5];
				msg_g.position.z = pose[6];
				msg_g.orientation.x = 0;
				msg_g.orientation.y = 0;
				msg_g.orientation.z = 0;
				msg_g.orientation.w = 1;
				grasp::data srv;
				srv.request.pose = msg_g;
				srv.request.id = ob_ID;
				client.call(srv);}
			}
			publishFinalTf(pose);
		}
 
	target_cloud->header.frame_id = "/my_camera_frame";
	pcl_conversions::toPCL(ros::Time::now(), target_cloud->header.stamp);
	target_publisher.publish(*target_cloud);
	//printf("target cloud size: %ld\n", target_cloud->points.size());

	icp_result_cloud->header.frame_id = "/my_camera_frame";
	pcl_conversions::toPCL(ros::Time::now(), icp_result_cloud->header.stamp);
	icp_result_publisher.publish(*icp_result_cloud);
	
}

void PoseEstimator::getObjectsPointCloudWithMask3(const sensor_msgs::Image::ConstPtr& mask){
  int cloud_size_thres = 100; //if the number of the decteted object point cloud is smaller than the thres, threating them as nothing 
  cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(mask); 

  //Generate object point cloud
  pcl::copyPointCloud(*scene_cloud, *target_cloud);
  filterObjectPointCloud(target_cloud, cv_ptr, 3);
	//filter
  pcl::VoxelGrid<PointXYZRGB> sor;
  sor.setInputCloud (target_cloud);
  sor.setLeafSize (0.003f, 0.003f, 0.003f);
  sor.filter (*target_cloud);  
	
		if (target_cloud->points.size() > cloud_size_thres){
		  //std::vector<double> pose = icpAlignByPointToPlaneMethod( target_cloud, kusan_model, icp_result_cloud); 

				
			std::vector<double> pose = icpAlignByPointToPlaneMethod( doublemint_model, target_cloud, icp_result_cloud);
				
			std::cout<<"pose:"<<std::endl;
			for(int i=0; i<pose.size(); i++){
			std::cout<<pose[i]<<" ";
			}
			std::cout<<std::endl;
			//distance1 = sqrt(pose[4]*pose[4] + pose[5]*pose[5] + pose[6]*pose[6]);
			if(ob3_exi){
			geometry_msgs::PoseStamped msg;	
			msg.pose.position.x = pose[4];
			msg.pose.position.y = pose[5];
			msg.pose.position.z = pose[6];
			pose_publisher3.publish(msg);
				
			if ( ob_ID==3){
				geometry_msgs::Pose msg_g;
				msg_g.position.x = pose[4];
				msg_g.position.y = pose[5];
				msg_g.position.z = pose[6];
				msg_g.orientation.x = 0;
				msg_g.orientation.y = 0;
				msg_g.orientation.z = 0;
				msg_g.orientation.w = 1;
				grasp::data srv;
				srv.request.pose = msg_g;
				srv.request.id = ob_ID;
				client.call(srv);}
			}
			publishFinalTf(pose);
		}
 
	target_cloud->header.frame_id = "/my_camera_frame";
	pcl_conversions::toPCL(ros::Time::now(), target_cloud->header.stamp);
	target_publisher.publish(*target_cloud);
	//printf("target cloud size: %ld\n", target_cloud->points.size());

	icp_result_cloud->header.frame_id = "/my_camera_frame";
	pcl_conversions::toPCL(ros::Time::now(), icp_result_cloud->header.stamp);
	icp_result_publisher.publish(*icp_result_cloud);
	
}

void PoseEstimator::updateScenePoints(const sensor_msgs::PointCloud2::ConstPtr& cloud){
  pcl::fromROSMsg (*cloud, *scene_cloud);
	pcl_conversions::toPCL(ros::Time::now(), scene_cloud->header.stamp);
	scene_cloud->header.frame_id = "/my_camera_frame";

	scene_publisher.publish(*scene_cloud);
	kinder_publisher.publish(*kinder_model);
  kusan_publisher.publish(*kusan_model);
  doublemint_publisher.publish(*doublemint_model);
  //doublemint

  br.sendTransform(tf::StampedTransform(tf_scene, ros::Time::now(), "/map", "/my_camera_frame"));
  br.sendTransform(tf::StampedTransform(tf_base, ros::Time::now(), "/map", "/my_base_frame"));

  //get tf		
 	try{
 			tf::StampedTransform transform;
 			listener.lookupTransform("/my_camera_frame", "/my_base_frame",  ros::Time(0), transform);
			//std::cout<< "transform: " << "x:" << transform.getOrigin().x() << " y:" << transform.getOrigin().y() << "z:" << transform.getOrigin().z() << std::endl;

			if(ifMove == false){
				tf::Matrix3x3 rotation = transform.getBasis();
				rotation = rotation.inverse();
				tf::Vector3 transisiton = transform.getOrigin();
				arm_to_camera_rotation = rotation;

				double yaw, pitch, roll;
				rotation.getEulerYPR(yaw, pitch, roll);
				Eigen::Affine3f tf_point = Eigen::Affine3f::Identity();
				tf_point.translation() << transisiton[0], transisiton[1], transisiton[2];
				tf_point.rotate (Eigen::AngleAxisf (roll - 75*(3.14/180), Eigen::Vector3f::UnitX())); 
				tf_point.rotate (Eigen::AngleAxisf (pitch , Eigen::Vector3f::UnitY())); 
				tf_point.rotate (Eigen::AngleAxisf (yaw, Eigen::Vector3f::UnitZ())); 
				pcl::transformPointCloud(*kusan_model, *kusan_model, tf_point);
				kusan_model->header.frame_id = "/my_camera_frame";
				ifMove = true;
			}
	
  }
  catch (tf::TransformException ex){
      ROS_ERROR("%s",ex.what());
      ros::Duration(1.0).sleep();
  }
}

void PoseEstimator::loadModelHelper(pcl::PointCloud<PointXYZRGB>::Ptr object, std::string object_name){
		std::string model_folder_path("/root/sis_mini_competition_2018/catkin_ws/src/pose_estimate_and_pick/models/");

    //Load kinder model
    std::string object_path = model_folder_path + object_name + ".ply";

  	pcl::PolygonMesh mesh;
  	pcl::io::loadPLYFile(object_path, mesh);
  	pcl::fromPCLPointCloud2( mesh.cloud, *object );

    object->header.frame_id = "/my_base_frame";
		pointCloudPreprocessing(object);

		//transform to orgin
		//Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
		Eigen::Affine3f transform_1 = Eigen::Affine3f::Identity();
		Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();
		Eigen::Affine3f transform_3 = Eigen::Affine3f::Identity();

		//move to origin
   	Eigen::Vector4f centroid_model;
    pcl::compute3DCentroid (*object, centroid_model);
		transform_1.translation() << -centroid_model[0], -centroid_model[1], -centroid_model[2];
		pcl::transformPointCloud(*object, *object, transform_1);

		std::cout << object_name;
    printf(" cloud size: %ld\n",object->points.size());
}

void PoseEstimator::loadModels(){
		loadModelHelper(kinder_model, "kinder");
	  loadModelHelper(kusan_model, "kusan");  
	  loadModelHelper(doublemint_model, "doublemint");
}

void PoseEstimator::filterObjectPointCloud(PointCloud<PointXYZRGB>::Ptr cloud, cv_bridge::CvImagePtr mask, int object_id){
	if(cloud->points.size() != (640*480)){
		printf("The size of point cloud is not eqaul to the size of the mask\n");
		return;
	}

	int count = 0;
	//int threshold = 20; 
	for (int row=0;row<480;row++){
		for(int column=0;column<640;column++){
				if(mask->image.at<uchar>(row,column) != object_id){
					cloud->points[640*row + column].x = std::numeric_limits<float>::quiet_NaN(); //output 1. ->for background (no color)
          			cloud->points[640*row + column].y = std::numeric_limits<float>::quiet_NaN(); //we may need assign points which don't have correct ID to 111  
                    cloud->points[640*row + column].z = std::numeric_limits<float>::quiet_NaN(); //or make the location of point cloud of backgroun (1,1,1)  
					count++;
				}
      }
	}

	if (object_id==1)
	{ 
		if(count >= (640*480-5) )
		{
			ob1_exi=false;	
		}
	} else if (object_id==2)
	{ 
		if(count >= (640*480-5) )
		{
			ob2_exi=false;	
		}
	} else if (object_id==3)
	{ 
		if(count >= (640*480-5) )
		{
			ob3_exi=false;	
		}
	}
	
	//std::cout<< "count: " << count << std::endl;
	//remove NAN
	std::vector<int> indices;
	pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);
}

void PoseEstimator::addNormal(PointCloud<PointXYZRGB>::Ptr cloud, PointCloud<PointXYZRGBNormal>::Ptr cloud_with_normals){
	/*
	std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);
	std::cout<< "(3)Filtered cloud points: " << cloud->points.size() << std::endl;
	std::cout<< "(3)NaN points: " << indices.size() << std::endl;
	*/

  pcl::PointCloud<pcl::Normal>::Ptr normals( new pcl::PointCloud<pcl::Normal> );

  pcl::search::KdTree<PointXYZRGB>::Ptr searchTree (new pcl::search::KdTree<pcl::PointXYZRGB>);
  searchTree->setInputCloud ( cloud );

  pcl::NormalEstimation<PointXYZRGB, Normal> normalEstimator;
  normalEstimator.setInputCloud ( cloud );
  normalEstimator.setSearchMethod ( searchTree );
  normalEstimator.setKSearch ( 15 );
  normalEstimator.compute ( *normals );
  pcl::concatenateFields( *cloud, *normals, *cloud_with_normals );
}

void PoseEstimator::pointCloudPreprocessing(PointCloud<PointXYZRGB>::Ptr object_cloud){
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*object_cloud, *object_cloud, indices);
  
  //Pointcloud downsampling
  pcl::VoxelGrid<PointXYZRGB> sor;
  sor.setInputCloud (object_cloud);
  sor.setLeafSize (0.005f, 0.005f, 0.005f);
  sor.filter (*object_cloud);  

  //Pointcloud Denoise
  pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor2;
  if(object_cloud->points.size() > 100){
    sor2.setInputCloud(object_cloud);
    sor2.setMeanK(50);
    sor2.setStddevMulThresh (0.55);
    sor2.filter(*object_cloud);
  }

  std::vector<int> indices2;
  pcl::removeNaNFromPointCloud(*object_cloud, *object_cloud, indices2);
}

std::vector<double> PoseEstimator::icpAlignByPointToPlaneMethod (PointCloud<PointXYZRGB>::Ptr cloud_source, //object
																											 PointCloud<PointXYZRGB>::Ptr cloud_target, //model
																											 PointCloud<PointXYZRGB>::Ptr final_cloud) 
{
  	pcl::PointCloud<PointXYZRGBNormal>::Ptr cloud_source_normals( new pcl::PointCloud<PointXYZRGBNormal> );
  	pcl::PointCloud<PointXYZRGBNormal>::Ptr cloud_target_normals( new pcl::PointCloud<PointXYZRGBNormal> );
  	pcl::PointCloud<PointXYZRGB>::Ptr translated_cloud_source(new pcl::PointCloud<PointXYZRGB>);
  	Eigen::Matrix4f transform_translation = Eigen::Matrix4f::Identity();

		//overlap by centroid
  	Eigen::Vector4f centroid_source;
		Eigen::Vector4f centroid_target;
    pcl::compute3DCentroid (*cloud_source, centroid_source);
    pcl::compute3DCentroid (*cloud_target, centroid_target);
    transform_translation(0,3) -= (centroid_source[0] - centroid_target[0]);
    transform_translation(1,3) -= (centroid_source[1] - centroid_target[1]);
    transform_translation(2,3) -= (centroid_source[2] - centroid_target[2]);
    pcl::transformPointCloud (*cloud_source, *translated_cloud_source, transform_translation);
		
		/*
		test_cloud = translated_cloud_source;
    test_cloud->header.frame_id = "/my_camera_frame";
		pcl_conversions::toPCL(ros::Time::now(),test_cloud->header.stamp);
		test_publisher.publish(*test_cloud);
		*/

		//PointCloud<PointXYZRGB>::Ptr final_cloud (new pcl::PointCloud<PointXYZRGB>);
		pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB>::Ptr icp ( new pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> () );
	  icp->setMaximumIterations ( 200 );
		icp->setInputSource ( cloud_source ); 
  	icp->setInputTarget ( cloud_target );
		icp->align ( *final_cloud ); 

	  if ( icp->hasConverged() ){
   		std::cout << "icp score: " << icp->getFitnessScore() << std::endl;
			//printf("cloud size: %ld\n", cloud_source_trans_normals->points.size());
   	}
   	else
   		std::cout << "Not converged." << std::endl;

    //Generate the transform matrix from model to object scene
    Eigen::Matrix4f inverse_transformation = icp->getFinalTransformation();
    Eigen::Matrix3f inverse_object_rotation_matrix;
    for(int row=0;row<3;row++){
      for(int col=0;col<3;col++)
        inverse_object_rotation_matrix(row,col) = inverse_transformation(row,col);
    }
    Eigen::Matrix3f object_rotation_matrix = inverse_object_rotation_matrix;
    Eigen::Matrix4f object_transform_matrix = Eigen::Matrix4f::Identity(); 
    for(int row=0;row<3;row++){
      object_transform_matrix(row,3) = inverse_transformation(row,3);
      for(int col=0;col<3;col++)
        object_transform_matrix(row,col) = object_rotation_matrix(row,col);
    }

    object_transform_matrix(0,3) += (centroid_source[0] - centroid_target[0]);
    object_transform_matrix(1,3) += (centroid_source[1] - centroid_target[1]);
    object_transform_matrix(2,3) += (centroid_source[2] - centroid_target[2]);

  	tf::Matrix3x3 tf3d;
  	tf3d.setValue((object_transform_matrix(0,0)), (object_transform_matrix(0,1)), (object_transform_matrix(0,2)), 
                  (object_transform_matrix(1,0)), (object_transform_matrix(1,1)), (object_transform_matrix(1,2)), 
                  (object_transform_matrix(2,0)), (object_transform_matrix(2,1)), (object_transform_matrix(2,2)));
  	tf::Quaternion tfqt;
  	tf3d.getRotation(tfqt);

    Eigen::Vector4f centroid_final;
		pcl::compute3DCentroid(*final_cloud, centroid_final);
		
		//cheating!!!!!
		double roll, yaw, pitch;
		arm_to_camera_rotation.getRPY(roll, pitch, yaw);
		//std::cout << "r:" << roll + 75*(3.14/180) << " " << 
 
    Eigen::Affine3f inv_rotation = Eigen::Affine3f::Identity();
		//inv_rotation.rotate (Eigen::AngleAxisf (-135*(3.14/180), Eigen::Vector3f::UnitX())); 	
		//inv_rotation.rotate (Eigen::AngleAxisf (-110*(3.14/180), Eigen::Vector3f::UnitZ())); 
		inv_rotation.rotate (Eigen::AngleAxisf (roll - 75*(3.14/180), Eigen::Vector3f::UnitX())); 	
		inv_rotation.rotate (Eigen::AngleAxisf (pitch, Eigen::Vector3f::UnitY())); 
		inv_rotation.rotate (Eigen::AngleAxisf (yaw, Eigen::Vector3f::UnitZ())); 
		Eigen::Vector4f u((centroid_source[0] - centroid_final[0]), (centroid_source[1] - centroid_final[1]), (centroid_source[2] - centroid_final[2]), 1);
		//std::cout << "u:" << u[0] << " " << u[1] << " " << u[2] << std::endl;

		Eigen::Matrix4f m4f_transform = inv_rotation.matrix();
		std::cout << m4f_transform << std::endl;
		u = m4f_transform*u;
		
		double roll2, yaw2, pitch2;
		tf3d.getRPY(roll2, pitch2, yaw2);
		if(yaw2 > 3.14/2) yaw2-=3.14;
		else if(yaw2 < -3.14/2) yaw2+=3.14;
    tf3d.setRPY(0, 0, yaw2);
		tf3d.getRotation(tfqt);

  	std::vector<double> rot_and_tra;
  	rot_and_tra.resize(7);
  	rot_and_tra[0]=tfqt[0];
  	rot_and_tra[1]=tfqt[1];
  	rot_and_tra[2]=tfqt[2];
  	rot_and_tra[3]=tfqt[3];
  	//rot_and_tra[4]=object_transform_matrix(0,3);
  	//rot_and_tra[5]=object_transform_matrix(1,3);
  	//rot_and_tra[6]=object_transform_matrix(2,3);
		//rot_and_tra[4] = -(centroid_source[0] - centroid_final[0]);
		//rot_and_tra[5] = -(centroid_source[1] - centroid_final[1]);
		//rot_and_tra[6] = -(centroid_source[2] - centroid_final[2]);
		rot_and_tra[4] = u[0];
		rot_and_tra[5] = u[1];
		rot_and_tra[6] = u[2];
	  
		return rot_and_tra;
}


























