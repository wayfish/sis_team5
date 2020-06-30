#!/usr/bin/env python

import math
import time
import numpy as np
import roslib
import rospy
import tf
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64
# from scipy.spatial.transform import Rotation as R
from camera_rotation.srv import find_plate
from tf.transformations import quaternion_matrix
from geometry_msgs.msg import Twist


class get_transform(object):
    def __init__(self):
        self.pan = 0
        self.state = 0
        self._trans = []
        self._rot = []
        self.trans = []
        self.rot = []
        self.pan_pub = rospy.Publisher("/teleop/pan", Float64, queue_size=1)
        self.cmd_vel = rospy.Publisher('/teleop/cmd_vel', Twist, queue_size=1) 
        rospy.Service("get_pose", find_plate, self.get_pose)      
    
    def get_pose(self, req):

        plate = req.plate
        
        while not rospy.is_shutdown():
            try:
                (_trans,_rot) = listener.lookupTransform('/camera_link', "/tag_" + str(plate + 4), rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    if self.state == 0:
                        self.pan -= 0.25
                        self.pan_pub.publish(self.pan)
                     
                    else:
                        self.pan += 0.25
                        self.pan_pub.publish(self.pan)
                      
                    if self.pan < -0.8:
                        self.state = 1 

                    if self.pan > 0.8:
                        self.state = 0

                    time.sleep(1)
                    continue

            (trans,rot) = listener.lookupTransform('/base_link', "/plate_" + str(plate), rospy.Time(0))

            pose = PoseStamped()
            pose.pose.position.x = trans[0]
            pose.pose.position.y = trans[1]
            pose.pose.position.z = trans[2]
            pose.pose.orientation.x = rot[0]
            pose.pose.orientation.y = rot[1]
            pose.pose.orientation.z = rot[2]
            pose.pose.orientation.w = rot[3]

## get yaw and spin
            x = rot[0]
            y = rot[1]
            z = rot[2]
            w = rot[3]
            sinr_cosp = 2 * (w * x + y * z)
            cosr_cosp = 1 - 2 * (x**2 + y**2)
            roll = np.arctan2(sinr_cosp, cosr_cosp)

            sinp = 2 * (w * y - z * x)
            pitch = np.where(np.abs(sinp) >= 1,
                            np.sign(sinp) * np.pi / 2,
                            np.arcsin(sinp))

            siny_cosp = 2 * (w * z + x * y)
            cosy_cosp = 1 - 2 * (y**2 + z**2)
            yaw = np.arctan2(siny_cosp, cosy_cosp)
            
            if self.state == 1:
                yaw = -yaw
            move_cmd = Twist()
            move_cmd.angular.z = yaw
            self.cmd_vel.publish(move_cmd)
## get yaw and spin

            self.pan = 0
            self.state = 0
            self._trans = []
            self._rot = []
            self.trans = []
            self.rot = []
            self.pan_pub.publish(self.pan)
            return ["successful", pose]


if __name__ == "__main__":
    rospy.init_node("get_transform")
    listener = tf.TransformListener()
    get_transform()
    rospy.spin()
