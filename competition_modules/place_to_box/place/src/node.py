#!/usr/bin/env python

import rospy
#from pyrobot import Robot
import numpy as np
import time
import tf
from place.srv import data, dataResponse
from geometry_msgs.msg import PoseStamped

pose_pub = rospy.Publisher("deck_pose",PoseStamped,queue_size=1)

deck_pose = PoseStamped()

deck_pose.pose.position.x = 0.36 - 0.03
deck_pose.pose.position.y = 0.0 - 0.02
deck_pose.pose.position.z = 0.052 - 0.14
deck_pose.pose.orientation.x = 0
deck_pose.pose.orientation.y = 0
deck_pose.pose.orientation.z = 0
deck_pose.pose.orientation.w = 1
deck_pose.header.frame_id = "deck"


rospy.init_node('test_pose')


r = rospy.Rate(10)

counter = 0
while counter<600:
    pose_pub.publish(deck_pose)
    r.sleep()
    if counter%60 ==0:
    	#print("deck_pose(x)=",deck_pose.pose.position.x) 
        print(counter)
    counter+=1
    #r.sleep()

rospy.wait_for_service("place")
#robot = Robot('locobot')
place_OB = rospy.ServiceProxy("place", data)
place_OB(deck_pose)

#print("place object")


#r = rospy.Rate(10)

#while not rospy.is_shutdown():




