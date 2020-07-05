#!/usr/bin/env python

import rospy
#from pyrobot import Robot
import numpy as np
import time
import tf
from place.srv import data, dataResponse
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped

pose_pub = rospy.Publisher("ob_pose",PoseStamped,queue_size=1)

ob_pose = Pose()
ob_ps = PoseStamped()

ob_pose.position.x = 0.35
ob_pose.position.y = 0.0
ob_pose.position.z = 0.06 #0.14  
ob_pose.orientation.x = 0
ob_pose.orientation.y = 0.707
ob_pose.orientation.z = 0
ob_pose.orientation.w = 0.707

ob_ps.pose.position.x = 0.35
ob_ps.pose.position.y = 0.0
ob_ps.pose.position.z = 0.06 #0.14
ob_ps.pose.orientation.x = 0
ob_ps.pose.orientation.y = 0
ob_ps.pose.orientation.z = 0
ob_ps.pose.orientation.w = 1
ob_ps.header.frame_id = "map"

ob_id = 0

rospy.init_node('gp_node')


r = rospy.Rate(10)

counter = 0
while counter<300:
    pose_pub.publish(deck_ps)
    r.sleep()
    if counter%100 ==0:
    	#print("deck_pose(x)=",deck_pose.pose.position.x) 
        print(counter)
    counter+=1
    #r.sleep()

rospy.wait_for_service("grasp")
#robot = Robot('locobot')
place_OB = rospy.ServiceProxy("grasp", data)
place_OB(ob_pose,ob_id)

#print("place object")


#r = rospy.Rate(10)

#while not rospy.is_shutdown():




