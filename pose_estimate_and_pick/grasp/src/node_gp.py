#!/usr/bin/env python

import get_edge
import rospy
#from pyrobot import Robot
import numpy as np
import time
import tf
from place.srv import data, dataResponse
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped

import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torchvision.models.vgg import VGG
import pandas as pd
import scipy.misc
import random
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
                                                                                                                                                                                     
#import numpy as np
import os
#import time
#import rospy
#import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2


model_use  = "subt_model" # "subt_model"
n_class = 4


class FCN8s(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super(FCN8s, self).__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace = True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        
        # After the feature extraction layer of vgg, you can get the feature map. 
        # The size of the feature map after 5 max_pools are respectively
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)
        x2 = output['x2']  # size=(N, 128, x.H/4,  x.W/4)
        x1 = output['x1']  # size=(N, 64, x.H/2,  x.W/2)
        
#===========FCN16s model ==========================
#         score = self.relu(self.deconv1(x5))               # size=(N, 512, x.H/16, x.W/16)
#         score = self.bn1(score + x4)                      # element-wise add, size=(N, 512, x.H/16, x.W/16)
#         score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
#         score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
#         score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
#         score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
#         score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)
#===========FCN16s model ==========================
        
#===========Please design a FCN8s model ===========
        score = self.relu(self.deconv1(x5))               # size=(N, 512, x.H/16, x.W/16)
        score = self.bn1(score + x4)                      # element-wise add, size=(N, 512, x.H/16, x.W/16)
        
        score = self.relu(self.deconv2(score))            # size=(N, 256, x.H/8, x.W/8)
        score = self.bn2(score + x3)                      # element-wise add, size=(N, 256, x.H/8, x.W/8)
        
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)

        
#===========Please design a FCN8s model ===========


        return score  # size=(N, n_class, x.H/1, x.W/1)


# In[5]:


class VGGNet(VGG):
    def __init__(self, pretrained=False, model='vgg16', requires_grad=True, remove_fc=True, show_params=False):
        super(VGGNet, self).__init__(make_layers(cfg[model]))
        self.ranges = ranges[model]

        if pretrained:
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc:  # delete redundant fully-connected layer params, can save memory
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}

        # get the output of each maxpooling layer (5 maxpool in VGG net)
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):      
                x = self.features[layer](x)
            output["x%d"%(idx+1)] = x
        return output


# In[6]:


ranges = {
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

# cropped version from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)



# get data

    
# create dir for model
FullPath = os.getcwd()
model_dir = os.path.join(FullPath + "/models", model_use)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

use_gpu = torch.cuda.is_available()
num_gpu = list(range(torch.cuda.device_count()))

vgg_model = VGGNet(requires_grad=True, remove_fc=True)
fcn_model = FCN8s(pretrained_net=vgg_model, n_class=n_class)

if use_gpu:
    ts = time.time()
    vgg_model = vgg_model.cuda()
    fcn_model = fcn_model.cuda()
#     fcn_model = nn.DataParallel(fcn_model, device_ids=num_gpu)
    print("Finish cuda loading, time elapsed {}".format(time.time() - ts))
else:
#     nn.DataParallel(fcn_model)
    print("Use CPU to train.")



# means     = np.array([103.939, 116.779, 123.68]) / 255. # mean of three channels in the order of BGR
h, w      = 480, 640
val_h     = h
val_w     = w


def decode_segmap(image, nc=4): 
    label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=red, 2=green, 3=blue,
               (255, 0, 0), (0, 255, 0), (0, 0, 255)])
 
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
   
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
     
    rgb = np.stack([r, g, b], axis=2)
    return rgb

def prediction(model_name, inputs):
    
    # load pretrain models
              
    vgg_model = VGGNet(requires_grad=True, remove_fc=True).cuda()
    fcn_model = FCN8s(pretrained_net=vgg_model, n_class=n_class).cuda()   
    
    state_dict = torch.load(os.path.join(model_dir, model_name))
    check_point = state_dict['net']
    fcn_model.load_state_dict(check_point)

    
    output = fcn_model(inputs.cuda())
    output = output.data.cpu().numpy()

    N, _, h, w = output.shape
    pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis = 1).reshape(h, w).astype(np.uint8)
    kernel = np.ones((5,5),np.uint8)
    pred_open = cv2.morphologyEx(pred,cv2.MORPH_OPEN,kernel)
    
    #RGB_pred = decode_segmap(pred_open)

    return pred_open
    

def callback(data):
    # define picture to_down' coefficient of ratio
    scaling_factor = 0.5
    global count,bridge
    count = count + 1
    if count == 180:
        count = 0
        cv_img = bridge.imgmsg_to_cv2(data, "rgb8")/255.0
	cv_img = np.transpose(cv_img, (2,0,1))[None,:,:,:]
	cv_img = torch.from_numpy(cv_img).float()
	mask = prediction("best_model_dataset1.pkl", cv_img)
        cv2.imshow("frame" , mask)
        cv2.waitKey(1000)
    
    
    
    edge,label =  get_edge.get_range(np.array(mask))
    #edge_pla,label_p =  get_edge.get_range(pla_mask)
    edge_c = np.mean(edge[:,:,:],axis=0) #(2,3)

    
#----------------------------

    pose_pub1 = rospy.Publisher("red_pose",PoseStamped,queue_size=1)
    pose_pub2 = rospy.Publisher("green_pose",PoseStamped,queue_size=1)
    pose_pub3 = rospy.Publisher("blue_pose",PoseStamped,queue_size=1)

    green_pose = Pose()
    red_ps = PoseStamped()
    green_ps = PoseStamped()
    blue_ps = PoseStamped()

    green_pose.position.x = get_edge.l_get(edge_c[:,1])[0]
    green_pose.position.y = get_edge.l_get(edge_c[:,1])[1]
    green_pose.position.z = 0.045 #0.14  
    green_pose.orientation.x = 0
    green_pose.orientation.y = 0.707
    green_pose.orientation.z = 0
    green_pose.orientation.w = 0.707
    
    green_ps.pose.position.x = get_edge.l_get(edge_c[:,1])[0]+0.06
    green_ps.pose.position.y = get_edge.l_get(edge_c[:,1])[1]-0.02
    green_ps.pose.position.z = 0.11  #0.14
    green_ps.pose.orientation.x = 0
    green_ps.pose.orientation.y = 0
    green_ps.pose.orientation.z = 0
    green_ps.pose.orientation.w = 1
    green_ps.header.frame_id = "map"

    blue_ps.pose.position.x = get_edge.l_get(edge_c[:,2])[0]+0.06
    blue_ps.pose.position.y = get_edge.l_get(edge_c[:,2])[1]-0.02
    blue_ps.pose.position.z = 0.11  #0.14
    blue_ps.pose.orientation.x = 0
    blue_ps.pose.orientation.y = 0
    blue_ps.pose.orientation.z = 0
    blue_ps.pose.orientation.w = 1
    blue_ps.header.frame_id = "map"

    red_ps.pose.position.x = get_edge.l_get(edge_c[:,0])[0]+0.06
    red_ps.pose.position.y = get_edge.l_get(edge_c[:,0])[1]-0.02
    red_ps.pose.position.z = 0.11  #0.14
    red_ps.pose.orientation.x = 0
    red_ps.pose.orientation.y = 0
    red_ps.pose.orientation.z = 0
    red_ps.pose.orientation.w = 1
    red_ps.header.frame_id = "map"

    ob_id = 2

    #rospy.init_node('gp_node')


    r = rospy.Rate(10)

    counter = 0
    while counter<360:
        pose_pub1.publish(red_ps)
        pose_pub2.publish(green_ps)
        pose_pub3.publish(blue_ps)
        r.sleep()
        if counter%60 ==0:
    	#print("deck_pose(x)=",deck_pose.pose.position.x) 
            print(counter)
            counter+=1
    #r.sleep()

    rospy.wait_for_service("grasp")
#robot = Robot('locobot')
    grasp_OB = rospy.ServiceProxy("grasp", data)
    grasp_OB(green_pose,ob_id)
        
        
        
    else:
        pass
 
def displayWebcam():
    rospy.init_node('gp_node', anonymous=True)
 
    # make a video_object and init the video object
    global count,bridge
    count = 0
    bridge = CvBridge()
    rospy.Subscriber('/camera/color/image_rect_color', Image, callback)
    rospy.spin()
 
if __name__ == '__main__':
    displayWebcam()


