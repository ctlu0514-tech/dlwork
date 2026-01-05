import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc1 = nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out

        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
        
    def forward(self, x):
        x_out = x * self.ca(x)
        x_out = x_out * self.sa(x_out)
        return x_out


class BasicBlock(nn.Module):
    def __init__(self, input_channel, out_channel=32):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels=input_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(out_channel)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.maxpool(x)

        return x


class AttBasicBlock(nn.Module):
    def __init__(self, input_channel, out_channel=32):
        super(AttBasicBlock, self).__init__()
        self.conv = BasicBlock(input_channel=input_channel, out_channel=out_channel)
        self.cbam = CBAM(out_channel, 2, 7)


    def forward(self, x):
        x = self.conv(x)
        x = self.cbam(x)
        return x


class CNN_3D(nn.Module):
    def __init__(self, input_size=[16, 256, 256], input_channels=1, n_classes=2):
        super(CNN_3D, self).__init__()
        self.conv1 = BasicBlock(input_channels, input_channels*2)
        self.conv2 = BasicBlock(input_channels*2, input_channels*4)
        self.conv3 = BasicBlock(input_channels*4, input_channels*8)
        
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten(start_dim=1)

        # self.fc1 = nn.Linear(input_channels * input_size[0] * input_size[1] * input_size[2] // 64, 128)
        self.fc1 = nn.Linear(65536, 128)
        self.fc2 = nn.Linear(128, 16)
        self.fc3 = nn.Linear(16, n_classes)


    def forward(self, x, feature=None, meta=None):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        
        # #输出最后一层的特征
        # feature = x.cpu().detach().numpy()
        feature_invol = x
        
        if feature is not None:
            x = torch.add(x, feature)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x, feature_invol 


class Att_CNN_3D(CNN_3D):
    def __init__(self, input_size=[16, 256, 256], input_channels=1, n_classes=2):
        super(CNN_3D, self).__init__()
        self.conv1 = AttBasicBlock(input_channels, input_channels*2)
        self.conv2 = AttBasicBlock(input_channels*2, input_channels*4)
        self.conv3 = AttBasicBlock(input_channels*4, input_channels*8)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten(start_dim=1)
        # self.fc1 = nn.Linear(input_channels*input_size[0]*input_size[1]*input_size[2]//64, 128) #[1,16,256,256]
        self.fc1 = nn.Linear(65536, 128) #[1,6,64,64]
        self.fc2 = nn.Linear(128, 16)
        self.fc3 = nn.Linear(16, n_classes)

        def forward(self, x, feature=None, meta=None):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.flatten(x)
        
            # #输出最后一层的特征
            # feature = x.cpu().detach().numpy()
            feature_invol = x
        
            if feature is not None:
                x = torch.add(x, feature)
            
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)
            return x, feature_invol


# ones = torch.rand(16, 1, 6, 64, 64)
# model = Att_CNN_3D()
# out, feature = model(ones) 
# print(out.size())
# print(feature.shape)