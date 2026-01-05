import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
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
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
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
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result

class MS_CAM(nn.Module):
    '''
    单特征进行通道注意力加权,作用类似SE模块
    '''

    def __init__(self, channels=64, r=4):
        super(MS_CAM, self).__init__()
        inter_channels = int(channels // r)

        # 局部注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels , inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels , inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei


class Channel_Att(nn.Module):
    def __init__(self, channels, t=16):
        super(Channel_Att, self).__init__()
        self.channels = channels
        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)
 
 
    def forward(self, x):
        residual = x
        x = self.bn2(x)
        # 式2的计算，即Mc的计算
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = torch.sigmoid(x) * residual #
        
        return x
 
 
class Att(nn.Module):
    def __init__(self, channels, out_channels=None, no_spatial=True):
        super(Att, self).__init__()
        self.Channel_Att = Channel_Att(channels)
  
    def forward(self, x):
        x_out1=self.Channel_Att(x)
        return x_out1

'''class AFF(nn.Module):
 

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo'''

class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel,attention = 'false'):
        super(BasicBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.ms = MS_CAM(channels=out_channel + in_channel)
        self.nam = Att(channels=out_channel + in_channel)
        self.att = attention

    def forward(self, x):
        x1 = self.conv(x)
        x1 = self.relu(x1)
        x1 = self.bn(x1)
        x1 = self.maxpool(x1)
        x2 = self.maxpool(x)
        out = torch.cat((x1, x2), dim=1)
        if(self.att == 'MAC'):
            out = self.ms(out)
        #if(self.att == 'nam'):
            #out = self.nam(out)
        return out


'''class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel,attention = 'false'):
        super(BasicBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channel + in_channel)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.ms = MS_CAM(out_channel)
        self.att = attention


    def forward(self, x):
        x1 = self.conv(x)
        x1 = self.relu(x1)
        x1 = self.bn(x1)
        x1 = self.maxpool(x1)
        x2 = self.maxpool(x)
        out = torch.cat((x1, x2), dim=1)
        if(self.att == 'MAC'):
            out = self.ms(out)
        return out'''


class BasicBlock3D(BasicBlock):
    def __init__(self, in_channel, out_channel):
        super(BasicBlock, self).__init__()

        self.conv = nn.Conv3d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(out_channel)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()


class ConvBlock(nn.Module):
    def __init__(self, input_channel, out_channel, kernel_size, attention = False):
        super(ConvBlock, self).__init__()
        self.layer1 = BasicBlock(input_channel + out_channel * 0, out_channel,attention = attention)
        self.layer2 = BasicBlock(input_channel + out_channel * 1, out_channel,attention = attention)
        self.layer3 = BasicBlock(input_channel + out_channel * 2, out_channel,attention = attention)
        self.layer4 = BasicBlock(input_channel + out_channel * 3, out_channel,attention = attention)

        self.conv = nn.Conv2d(in_channels=input_channel + out_channel * 4, out_channels=out_channel,
                               kernel_size=kernel_size, stride=kernel_size, padding=1, bias=False)
        
        self.cbam = CBAM(input_channel + out_channel * 4, 16, 7)

        self.bn = nn.BatchNorm2d(out_channel)
        self.flatten = nn.Flatten(start_dim=1)
        self.relu = nn.ReLU()

       


    def forward(self, x):
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)

        out = self.flatten(x)

        return x

class ConvBlock_outfeature(nn.Module):
    def __init__(self, input_channel, out_channel, kernel_size, attention = False):
        super(ConvBlock_outfeature, self).__init__()
        self.layer1 = BasicBlock(input_channel + out_channel * 0, out_channel,attention = attention)
        self.layer2 = BasicBlock(input_channel + out_channel * 1, out_channel,attention = attention)
        self.layer3 = BasicBlock(input_channel + out_channel * 2, out_channel,attention = attention)
        self.layer4 = BasicBlock(input_channel + out_channel * 3, out_channel,attention = attention)

        self.conv = nn.Conv2d(in_channels=input_channel + out_channel * 4, out_channels=out_channel,
                               kernel_size=kernel_size, stride=kernel_size, padding=1, bias=False)
        
        self.cbam = CBAM(input_channel + out_channel * 4, 16, 7)

        self.bn = nn.BatchNorm2d(out_channel)
        self.flatten = nn.Flatten(start_dim=1)
        self.relu = nn.ReLU()

       


    def forward(self, x):
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        xfeature = x
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)

        out = self.flatten(x)

        return x,xfeature

class ConvBlock3D(ConvBlock):
    def __init__(self, input_channel, out_channel, kernel_size, is_cbam = False):
        super(ConvBlock, self).__init__()
        self.layer1 = BasicBlock3D(input_channel + out_channel * 0, out_channel)
        self.layer2 = BasicBlock3D(input_channel + out_channel * 1, out_channel)
        self.layer3 = BasicBlock3D(input_channel + out_channel * 2, out_channel)
        # self.layer4 = BasicBlock3D(input_channel + out_channel * 3, out_channel)

        self.conv = nn.Conv3d(in_channels=input_channel + out_channel * 3, out_channels=out_channel,
                               kernel_size=kernel_size, stride=3, padding=1, bias=False)
        
        self.cbam = CBAM(input_channel + out_channel * 4, 16, 7)

        self.bn = nn.BatchNorm3d(out_channel)
        self.flatten = nn.Flatten(start_dim=1)
        self.relu = nn.ReLU()

        self.is_cbam = False

    def forward(self, x):
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        
        if self.is_cbam:
            x = self.cbam(x)
        
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)

        out = self.flatten(x)

        return x

class AttConvBlock(ConvBlock):
    def __init__(self, input_channel, out_channel, kernel_size):
        super(ConvBlock, self).__init__()
        self.layer1 = BasicBlock(input_channel + out_channel * 0, out_channel)
        self.layer2 = BasicBlock(input_channel + out_channel * 1, out_channel)
        self.layer3 = BasicBlock(input_channel + out_channel * 2, out_channel)
        self.layer4 = BasicBlock(input_channel + out_channel * 3, out_channel)
        
        self.cbam1 = CBAM(input_channel + out_channel * 1, 16, 7)
        self.cbam2 = CBAM(input_channel + out_channel * 2, 16, 7)
        self.cbam3 = CBAM(input_channel + out_channel * 3, 16, 7)
        self.cbam4 = CBAM(input_channel + out_channel * 4, 16, 7)
        
        self.conv = nn.Conv2d(in_channels=input_channel + out_channel * 4, out_channels=out_channel,
                               kernel_size=kernel_size, stride=kernel_size, padding=1, bias=False)
        
        self.bn = nn.BatchNorm2d(out_channel)
        self.flatten = nn.Flatten(start_dim=1)
        self.relu = nn.ReLU()


    def forward(self, x):
        
        x = self.layer1(x)
        x = self.cbam1(x)

        x = self.layer2(x)
        x = self.cbam2(x)

        x = self.layer3(x)
        x = self.cbam3(x)

        x = self.layer4(x)
        x = self.cbam4(x)
        
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)

        out = self.flatten(x)

        return x


class DCPCNN_single(nn.Module):
    def __init__(self, input_size=(3, 128, 128), out_channel=16, n_classes=2):
        super(DCPCNN_single, self).__init__()

        self.channel1 = ConvBlock(input_size[0], out_channel, input_size[1] // 16)

        self.fc1 = nn.Linear(out_channel, n_classes * 16)
        self.fc2 = nn.Linear(n_classes * 16, n_classes * 4)
        self.fc3 = nn.Linear(n_classes * 4, n_classes)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.4)
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x):
        
        x1 = self.channel1(x)
        x1 = self.flatten(x1)

        out = self.fc1(x1)
        out = self.relu(out)

        out = self.dropout(out)

        out = self.fc2(out)
        out = self.relu(out)

        out = self.fc3(out)
        out = self.softmax(out)

        return out


class DCPCNN(nn.Module):
    def __init__(self, input_size1=(3, 96, 96), input_size2=(3, 128, 128), out_channel=16, n_classes=2, is_cbam=False):
        super(DCPCNN, self).__init__()

        self.channel1 = ConvBlock(input_size1[0], out_channel, input_size1[1] // 16, is_cbam)
        self.channel2 = ConvBlock(input_size1[0], out_channel, input_size1[1] // 16, is_cbam)
        self.channel3 = ConvBlock(input_size2[0], out_channel, input_size2[1] // 16, is_cbam)
        self.channel4 = ConvBlock(input_size2[0], out_channel, input_size2[1] // 16, is_cbam)

        self.fc1 = nn.Linear(out_channel * 4, n_classes * 16)
        self.fc2 = nn.Linear(n_classes * 16, n_classes * 4)
        self.fc3 = nn.Linear(n_classes * 4, n_classes)

        self.cbam = CBAM(out_channel, 16, 7)
        self.flatten = nn.Flatten(start_dim=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.5)

        self.is_cbam = is_cbam

    def forward(self, x):

        x1 = self.channel1(x[:, 0, :, :96, :96])
        x2 = self.channel2(x[:, 1, :, :96, :96])
        x3 = self.channel3(x[:, 2])
        x4 = self.channel4(x[:, 3])

        if self.is_cbam:
            x1 = self.cbam(x1)
            x2 = self.cbam(x2)
            x3 = self.cbam(x3)
            x4 = self.cbam(x4)
        
        x1 = self.flatten(x1)
        x2 = self.flatten(x2)
        x3 = self.flatten(x3)
        x4 = self.flatten(x4)

        out = torch.cat((x1, x2, x3, x4), dim=1)
        #out = torch.cat((x1, x3, x4), dim=1)
        # out = torch.cat((out, c), dim=1)

        out = self.fc1(out)
        out = self.relu(out)
        # out = self.sigmoid(out)

        out = self.dropout(out)

        out = self.fc2(out)
        out = self.relu(out)
        # out = self.sigmoid(out)

        out = self.fc3(out)
        out = self.softmax(out)

        return out


class Att_DCPCNN(DCPCNN):
    def __init__(self, input_size1=(3, 96, 96), input_size2=(3, 128, 128), out_channel=16, n_classes=2):
        super(DCPCNN, self).__init__()

        self.channel1 = AttConvBlock(input_size1[0], out_channel, input_size1[1] // 16)
        self.channel2 = AttConvBlock(input_size1[0], out_channel, input_size1[1] // 16)
        self.channel3 = AttConvBlock(input_size2[0], out_channel, input_size2[1] // 16)
        self.channel4 = AttConvBlock(input_size2[0], out_channel, input_size2[1] // 16)

        self.fc1 = nn.Linear(out_channel * 4, n_classes * 16)
        self.fc2 = nn.Linear(n_classes * 16, n_classes * 4)
        self.fc3 = nn.Linear(n_classes * 4, n_classes)

        self.cbam = CBAM(out_channel, 16, 7)
        self.flatten = nn.Flatten(start_dim=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.4)

        self.is_cbam = False


class DCPCNN_3D(DCPCNN):
    def __init__(self, input_size=(8, 64, 64), out_channel=16, n_classes=2, attention='false'):
        super(DCPCNN, self).__init__()

        self.channel1 = ConvBlock(input_size[0], out_channel, input_size[1] // 16 ,attention = attention)
        self.channel2 = ConvBlock(input_size[0], out_channel, input_size[1] // 16,attention = attention)
        self.channel3 = ConvBlock(input_size[0], out_channel, input_size[1] // 16,attention = attention)
        self.channel4 = ConvBlock(input_size[0], out_channel, input_size[1] // 16,attention = attention)

        self.fc1 = nn.Linear(out_channel * 3, n_classes * 16)
        self.fc2 = nn.Linear(n_classes * 16, n_classes * 4)
        self.fc3 = nn.Linear(n_classes * 4, n_classes)

        self.cbam = CBAM(out_channel, 16, 7)
        self.flatten = nn.Flatten(start_dim=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.5)
        self.att = attention
        self.nam = Att(channels=out_channel)
       

    def forward(self, x):
        residual = x
        x1 = self.channel1(x[:, 0])
        x2 = self.channel2(x[:, 1])
        x3 = self.channel3(x[:, 2])
        x4 = self.channel4(x[:, 3])
        #out2 = torch.cat((x1_feature,x2_feature,x3_feature,x4_feature),dim=1)
        if(self.att == 'nam'):
            x1 = self.nam(x1)
            x2 = self.nam(x2)
            x3 = self.nam(x3)
            x4 = self.nam(x4)

        if self.att == 'cbam':
            x1 = self.cbam(x1)
            x2 = self.cbam(x2)
            x3 = self.cbam(x3)
            x4 = self.cbam(x4)
        
        x1 = self.flatten(x1)
        x2 = self.flatten(x2)
        x3 = self.flatten(x3)
        x4 = self.flatten(x4)
        out= torch.cat((x1, x3, x4), dim=1)
        #out1 = out.view(-1,self.fc1.in_features)
        #out1 = out
        #out = torch.cat((x1, x3, x4), dim=1)
        # out = torch.cat((out, c), dim=1)

        out = self.fc1(out)
        out = self.relu(out)
        # out = self.sigmoid(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.relu(out)
        # out = self.sigmoid(out)

        out = self.fc3(out)
        out = self.softmax(out)
       

        return out #return out,out1,out2


class DCPCNN_3D_single(DCPCNN_3D):
    def __init__(self, input_size=(8, 64, 64), out_channel=16, n_classes=2):
        super(DCPCNN_3D, self).__init__()

        self.channel1 = ConvBlock(input_size[0], out_channel, input_size[1] // 16)

        self.fc1 = nn.Linear(out_channel, n_classes * 16)
        self.fc2 = nn.Linear(n_classes * 16, n_classes * 4)
        self.fc3 = nn.Linear(n_classes * 4, n_classes)

        self.cbam = CBAM(out_channel, 16, 7)
        self.flatten = nn.Flatten(start_dim=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.5)

        self.is_cbam = False

    def forward(self, x):

        x1 = self.channel1(x[:, 0])

        if self.is_cbam:
            x1 = self.cbam(x1)
        
        out = self.flatten(x1)

        out = self.fc1(out)
        out = self.relu(out)
        # out = self.sigmoid(out)

        out = self.dropout(out)

        out = self.fc2(out)
        out = self.relu(out)
        # out = self.sigmoid(out)

        out = self.fc3(out)
        out = self.softmax(out)

        return out

# ones = torch.rand(8, 1, 8, 64, 64)
# model = DCPCNN_3D_single()
# out = model(ones)