#!/usr/bin/env python

import rospy
import numpy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import ctypes
import struct
from geometry_msgs.msg import PoseStamped

dk_pose = np.zeros((4,3)) #xyz_4_point,lrup 

pose_pub = rospy.Publisher("dk_pose",PoseStamped,queue_size=5)

deck0 = PoseStamped()
deck1 = PoseStamped()
deck2 = PoseStamped()
deck3 = PoseStamped()
  
def pointcloud2_to_array(cloud_msg, squeeze=True):
    ''' Converts a rospy PointCloud2 message to a numpy recordarray 

    Reshapes the returned array to have shape (height, width), even if the height is 1.

    The reason for using np.fromstring rather than struct.unpack is speed... especially
    for large point clouds, this will be <much> faster.
    '''
    # construct a numpy record type equivalent to the point type of this cloud
    gen = pc2.read_points(PointCloud2, skip_nans=True)
    int_data = list(gen)
    bound_l = -0.10 
    bound_r = 0.10 
    bound_u = 0.40
    bound_d = 0.25
    vaild_l = 0.20
    vaild_r = -0.20   
    vaild_u = 0.05
    vaild_d = 0.6

    
    for x in int_data:
        test = x[3] 
        # cast float32 to int so that bitwise operations are possible
        s = struct.pack('>f' ,test)
        i = struct.unpack('>l',s)[0]
        # you can get back the float value by the inverse operations
        pack = ctypes.c_uint32(i).value
        r = (pack & 0x00FF0000)>> 16
        g = (pack & 0x0000FF00)>> 8
        b = (pack & 0x0000FF00)
    #x[0] ->x , x[1] -> y , x[2] -> z
        print(r,g,b)
        if ((r+g+b)>675) and (x[2]>0.05):
            if x[1]< bound_l:
                bound_l = x[1]
                dk_pose[0,:]= x[0:2]
            elif x[1]> bound_r:
                bound_r = x[1]
                dk_pose[1,:]= x[0:2]  
                
            if x[0]< bound_u:
                bound_u = x[0]
                dk_pose[2,:]= x[0:2]
            elif x[0]> bound_d:
                bound_d = x[0]
                dk_pose[3,:]= x[0:2] 
        else :
            if x[1]< vaild_l:
                vaild_l = x[1]
            elif x[1]> vaild_r:
                vaild_r = x[1]                
            if x[0]< vaild_u:
                vaild_u = x[0]
            elif x[0]> vaild_d:
                vaild_d = x[0]
                
        h_dem1 = np.sqrt(np.square(dk_pose[0,0]-dk_pose[2,0])+np.square(dk_pose[0,1]-dk_pose[2,1]))
        w_dem1 = np.sqrt(np.square(dk_pose[2,0]-dk_pose[1,0])+np.square(dk_pose[2,1]-dk_pose[1,1]))
        h_dem2 = np.sqrt(np.square(dk_pose[1,0]-dk_pose[3,0])+np.square(dk_pose[1,1]-dk_pose[3,1]))
        w_dem2 = np.sqrt(np.square(dk_pose[3,0]-dk_pose[0,0])+np.square(dk_pose[3,1]-dk_pose[0,1]))
        h_de = np.mean(h_dem1,h_dem2)
        w_de = np.mean(w_dem1,w_dem2)
        
        print("height",h_de)
        print("width",w_de)
        if np.min(h_dem,w_dem)<0.235: #need padding to long enough
            if vaild_u < bound_u: #grow up
                if h_dem<w_dem:
                    dk_pose[0,:] = dk_pose[2,:]+(0.24/h_dem)*(dk_pose[0,:]-dk_pose[2,:])
                    dk_pose[3,:] = dk_pose[1,:]+(0.24/h_dem)*(dk_pose[3,:]-dk_pose[1,:])
                else:
                    dk_pose[3,:] = dk_pose[0,:]+(0.24/w_dem)*(dk_pose[3,:]-dk_pose[0,:])
                    dk_pose[1,:] = dk_pose[2,:]+(0.24/w_dem)*(dk_pose[1,:]-dk_pose[2,:])                    
                    
            elif vaild_d > bound_d: #grow down
                if h_dem < w_dem:
                    dk_pose[1,:] = dk_pose[3,:]-(0.24/h_dem)*(dk_pose[3,:]-dk_pose[1,:])
                    dk_pose[2,:] = dk_pose[0,:]-(0.24/h_dem)*(dk_pose[0,:]-dk_pose[2,:])
                else:
                    dk_pose[2,:] = dk_pose[1,:]-(0.24/w_dem)*(dk_pose[1,:]-dk_pose[2,:])
                    dk_pose[0,:] = dk_pose[3,:]-(0.24/w_dem)*(dk_pose[3,:]-dk_pose[0,:])               
                        

    deck0.pose.position.x = dk_pose[0,0]
    deck0.pose.position.y = dk_pose[0,1]
    deck0.pose.position.z = dk_pose[0,2]
    deck0.pose.orientation.x = 0
    deck0.pose.orientation.y = 0
    deck0.pose.orientation.z = 0
    deck0.pose.orientation.w = 1
    deck0.header.frame_id = "map"
    

    deck1.pose.position.x = dk_pose[1,0]
    deck1.pose.position.y = dk_pose[1,1]
    deck1.pose.position.z = dk_pose[1,2]
    deck1.pose.orientation.x = 0
    deck1.pose.orientation.y = 0
    deck1.pose.orientation.z = 0
    deck1.pose.orientation.w = 1
    deck1.header.frame_id = "map"
    

    deck2.pose.position.x = dk_pose[2,0]
    deck2.pose.position.y = dk_pose[2,1]
    deck2.pose.position.z = dk_pose[2,2]
    deck2.pose.orientation.x = 0
    deck2.pose.orientation.y = 0
    deck2.pose.orientation.z = 0
    deck2.pose.orientation.w = 1
    deck2.header.frame_id = "map"

    
    deck3.pose.position.x = dk_pose[3,0]
    deck3.pose.position.y = dk_pose[3,1]
    deck3.pose.position.z = dk_pose[3,2]
    deck3.pose.orientation.x = 0
    deck3.pose.orientation.y = 0
    deck3.pose.orientation.z = 0
    deck3.pose.orientation.w = 1
    deck3.header.frame_id = "map"
    

rospy.init_node('get_PC')
rospy.Subscriber("/camera/depth_registered/points", PointCloud2, pointcloud2_to_array)
    # 
rospy.spin()

counter = 0
while counter<1000:
    pose_pub.publish(deck0)
    pose_pub.publish(deck1)
    pose_pub.publish(deck2)
    pose_pub.publish(deck3)
    r.sleep()
    if counter%100 ==0:
    	#print("deck_pose(x)=",deck_pose.pose.position.x) 
        print(counter)
    counter+=1


