import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_inplanes():
    return [64, 128, 256, 512]

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


class NAM(nn.Module):
    def __init__(self, channels=5, t=16):
        super(NAM, self).__init__()
        self.channels = channels
        self.bn2 = nn.BatchNorm3d(self.channels)

    def forward(self, x):
        residual = x
        x = self.bn2(x)
        weight_bn = self.bn2.weight.abs() / torch.sum(self.bn2.weight.abs())
        print(weight_bn)
        print(x)
        x = x.permute(0, 2, 3, 1, 4).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2, 4).contiguous()
        x = torch.sigmoid(x) * residual

        return x


class Att(nn.Module):
    def __init__(self, channels=3, out_channels=None, no_spatial=True):
        super(Att, self).__init__()
        self.Channel_Att = ChannelAttention(channels)

    def forward(self, x):
        x_out1 = self.Channel_Att(x)

        return x_out1


class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv3d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class addcbam_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.cbam = CBAM(planes, 16, 7)


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.cbam(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.cbam(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class addAFF_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.aff = AFF(channels=planes)


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.aff(out,residual)
        out += residual
        out = self.relu(out)

        return out


class addnam_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.nam = NAM(channels=planes)


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.nam(out)
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=4,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                #  groups=3,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=2,
                 attention='false'):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.dropout = nn.Dropout(p=0.65)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)
        self.softmax = nn.Softmax(dim=1)
        self.attention = attention
        self.cbam1 = CBAM(block_inplanes[0], 16 , 7)
        self.cbam2 = CBAM(block_inplanes[1], 16 , 7)
        self.cbam3 = CBAM(block_inplanes[2], 16 , 7)
        self.cbam4 = CBAM(block_inplanes[3], 16 , 7)
 
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)
        print(x.shape)
        x = self.layer1(x)
        print(x.shape)
        if(self.attention == 'cbam'):
            x = self.cbam1(x)
        x = self.layer2(x)
        if(self.attention == 'cbam'):
            x = self.cbam2(x)
        x = self.layer3(x)
        if(self.attention == 'cbam'):
            x = self.cbam3(x)
        x = self.layer4(x)
        if(self.attention == 'cbam'):
            x = self.cbam4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.softmax(x)

        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model

def resnet10(input_channels=1, num_classes=2, attention='flase'):
    if(attention == 'cbam'):
        return ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), n_input_channels=input_channels, n_classes=num_classes,attention='cbam')
    if(attention == 'nam'):
        return ResNet(addnam_BasicBlock, [1, 1, 1, 1], get_inplanes(), n_input_channels=input_channels, n_classes=num_classes)
    if(attention =='AFF'):
        return ResNet(addAFF_BasicBlock, [1, 1, 1, 1], get_inplanes(), n_input_channels=input_channels,n_classes=num_classes)
    if(attention == 'false'):
        return ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), n_input_channels=input_channels, n_classes=num_classes)

def resnet18(input_channels=1, num_classes=2 , attention='flase'):

    if(attention == 'cbam'):
        return ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), n_input_channels=input_channels, n_classes=num_classes,attention='cbam')
    if(attention == 'nam'):
        return ResNet(addnam_BasicBlock, [2, 2, 2, 2], get_inplanes(), n_input_channels=input_channels, n_classes=num_classes)
    if(attention =='AFF'):
        return ResNet(addAFF_BasicBlock, [2, 2, 2, 2], get_inplanes(), n_input_channels=input_channels,n_classes=num_classes)
    if(attention == 'false'):
        return ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), n_input_channels=input_channels, n_classes=num_classes)

def resnet34(input_channels=1, num_classes=2):
    return ResNet(BasicBlock, [3, 4, 6, 3],  get_inplanes(), n_input_channels=input_channels, n_classes=num_classes)

def resnet50(input_channels=1, num_classes=2, attention='flase'):
    if(attention == 'cbam'):
        return ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), n_input_channels=input_channels, n_classes=num_classes,attention='cbam')
    if(attention == 'nam'):
        return ResNet(addnam_BasicBlock, [3, 4, 6, 3], get_inplanes(), n_input_channels=input_channels, n_classes=num_classes)
    if(attention =='AFF'):
        return ResNet(addAFF_BasicBlock, [3, 4, 6, 3], get_inplanes(), n_input_channels=input_channels,n_classes=num_classes)
    if(attention == 'false'):
        return ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), n_input_channels=input_channels, n_classes=num_classes)
    
def resnet101(input_channels=1, num_classes=2, attention='flase'):
    if(attention == 'cbam'):
        return ResNet(BasicBlock, [3, 4, 23, 3], get_inplanes(), n_input_channels=input_channels, n_classes=num_classes,attention='cbam')
    if(attention == 'nam'):
        return ResNet(addnam_BasicBlock, [3, 4, 23, 3], get_inplanes(), n_input_channels=input_channels, n_classes=num_classes)
    if(attention =='AFF'):
        return ResNet(addAFF_BasicBlock, [3, 4, 23, 3], get_inplanes(), n_input_channels=input_channels,n_classes=num_classes)
    if(attention == 'false'):
        return ResNet(BasicBlock, [3, 4, 23, 3], get_inplanes(), n_input_channels=input_channels, n_classes=num_classes)