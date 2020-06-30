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
deck_pose.pose.orientation.x
deck_pose.pose.orientation.y
deck_pose.pose.orientation.z
deck_pose.pose.orientation.w
deck_pose.id = 0

pose_pub.publish(deck_pose)


rospy.wait_for_service("grasp")
#robot = Robot('locobot')


place_OB = rospy.ServiceProxy("grasp", data)
place_OB(deck_pose)
#print("place object")

