cmake_minimum_required(VERSION 2.8.3)
project(grasp)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  pcl_ros
  pcl_conversions
  cv_bridge
  tf
)

catkin_package(
  INCLUDE_DIRS include
#  LIBRARIES project_test
  CATKIN_DEPENDS roscpp rospy std_msgs sensor_msgs pcl_ros
#  DEPENDS system_lib
)

add_service_files(
   FILES
   data.srv
 )
 
generate_messages(
  DEPENDENCIES
  std_msgs  # and/or other packages containing depended messages
)

add_library(pose_estimate_lib src/pose_estimate.cpp)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

add_executable(grasp src/pose_estimate.cpp)
add_dependencies( grasp ${PROJECT_NAME}_generate_messages_cpp )
target_link_libraries(grasp ${catkin_LIBRARIES})
