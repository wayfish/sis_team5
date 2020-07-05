#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
                                                                                                                                                                                    

import numpy as np
import time
import os
import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

model_use  = "subt_model" # "subt_model"
n_class = 6


# In[4]:
#publisher
rospy.init_node('object', anonymous=True)
mask_pub = rospy.Publisher('/object/mask', Image, queue_size=10)
ID_pub = rospy.Publisher('/object/ID', Int64, queue_size=10)

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


# In[7]:


# get data
FullPath = os.getcwd()
    
# create dir for model
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


# In[8]:


import csv
# means     = np.array([103.939, 116.779, 123.68]) / 255. # mean of three channels in the order of BGR
h, w      = 480, 640
val_h     = h
val_w     = w



def prediction(model_name, inputs):
    
    # load pretrain models
              
    vgg_model = VGGNet(requires_grad=True, remove_fc=True).cuda()
    fcn_model = FCN8s(pretrained_net=vgg_model, n_class=n_class).cuda()
#     fcn_model = nn.DataParallel(fcn_model)      
    
    state_dict = torch.load(os.path.join(model_dir, model_name))
    check_point = state_dict['net']
    fcn_model.load_state_dict(check_point)


    output = fcn_model(inputs.cuda())
    output = output.data.cpu().numpy()

    N, _, h, w = output.shape
    pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis = 1).reshape(h, w).astype(np.uint8)
    kernel = np.ones((5,5),np.uint8)
    pred_open = cv2.morphologyEx(pred,cv2.MORPH_OPEN,kernel)
 
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
	pred_open = prediction("best_model_dataset2_v3.pkl", cv_img)
        ID_list = np.unique(pred_open)
        if 1 in ID_list:
            ID = 1
        elif 2 in ID_list:
            ID = 2   
        elif 3 in ID_list:
            ID = 3
            
        mask_pub.publish(pred_open)
        ID_pub.publish(ID)    
        
    else:
        pass
 
def displayWebcam():
    rospy.init_node('camera_display', anonymous=True)
 
    # make a video_object and init the video object
    global count,bridge
    count = 0
    bridge = CvBridge()
    rospy.Subscriber('/camera/color/image_rect_color', Image, callback)
    rospy.spin()
 
if __name__ == '__main__':
    displayWebcam()

