#!/usr/bin/env python

import rospy
#from pyrobot import Robot
import numpy as np
import time
import tf
from place.srv import data, dataResponse
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped

pose_pub = rospy.Publisher("deck_pose",PoseStamped,queue_size=1)

deck_pose = Pose()
deck_ps = PoseStamped()

deck_pose.position.x = 0.35
deck_pose.position.y = 0.0 -0.02
deck_pose.position.z = 0.08  #0.14  
deck_pose.orientation.x = 0
deck_pose.orientation.y = 0.707
deck_pose.orientation.z = 0
deck_pose.orientation.w = 0.707

deck_ps.pose.position.x = 0.35
deck_ps.pose.position.y = 0.0
deck_ps.pose.position.z = 0.08  #0.14
deck_ps.pose.orientation.x = 0
deck_ps.pose.orientation.y = 0
deck_ps.pose.orientation.z = 0
deck_ps.pose.orientation.w = 1
deck_ps.header.frame_id = "map"

deck_id = 0

rospy.init_node('test_pose')


r = rospy.Rate(10)

counter = 0
while counter<1000:
    pose_pub.publish(deck_ps)
    r.sleep()
    if counter%60 ==0:
    	#print("deck_pose(x)=",deck_pose.pose.position.x) 
        print(counter)
    counter+=1
    #r.sleep()

rospy.wait_for_service("place")
#robot = Robot('locobot')
place_OB = rospy.ServiceProxy("place", data)
place_OB(deck_pose,deck_id)

#print("place object")


#r = rospy.Rate(10)

#while not rospy.is_shutdown():




