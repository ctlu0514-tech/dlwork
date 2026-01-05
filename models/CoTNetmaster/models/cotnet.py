import math
import numpy as np
import torch
from torch import nn as nn
from functools import partial
#from config import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
#from .helpers import build_model_with_cfg
from .layers import SelectiveKernelConv, ConvBnAct, create_attn
#from .registry import register_model
#from .resnet import ResNet
from .layers import Shiftlution
from ..cupy_layers.aggregation_zeropad import LocalConvolution
from .layers import create_act_layer
import torch.nn.functional as F
from torch import einsum
from .layers.activations import Swish
from .layers.tbconv import TBConv
from .layers.kerv2d import Kerv2d
from .layers import get_act_layer

'''
def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 2, 'input_size': (5, 8, 64, 64), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv1', 'classifier': 'fc',
        **kwargs
    }

default_cfgs = {
    'cot_basic': _cfg(
        url=''),
}
'''

def get_inplanes():
    return [64, 128, 256, 512]

def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)

class CotLayer(nn.Module):
    def __init__(self, dim, kernel_size):
        super(CotLayer, self).__init__()

        self.dim = dim
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
            nn.Conv3d(dim, dim, self.kernel_size, stride=1, padding=self.kernel_size//2, groups=4, bias=False),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True)
        )

        share_planes = 8
        factor = 2
        self.embed = nn.Sequential(
            nn.Conv3d(2*dim, dim//factor, 1, bias=False),
            nn.BatchNorm3d(dim//factor),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim//factor, pow(kernel_size, 2) * dim // share_planes, kernel_size=1),
            nn.GroupNorm(num_groups=dim // share_planes, num_channels=pow(kernel_size, 2) * dim // share_planes)
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm3d(dim)
        )

        self.local_conv = LocalConvolution(dim, dim, kernel_size=self.kernel_size, stride=1, padding=(self.kernel_size - 1) // 2, dilation=1)
        self.bn = nn.BatchNorm3d(dim)
        act = get_act_layer('swish')
        self.act = act(inplace=True)

        reduction_factor = 4
        self.radix = 2
        attn_chs = max(dim * self.radix // reduction_factor, 32)
        self.se = nn.Sequential(
            nn.Conv3d(dim, attn_chs, 1),
            nn.BatchNorm3d(attn_chs),
            nn.ReLU(inplace=True),
            nn.Conv3d(attn_chs, self.radix*dim, 1)
        )

    def forward(self, x):
        k = self.key_embed(x)
        qk = torch.cat([x, k], dim=1)
        b, c, qk_d, qk_hh, qk_ww = qk.size()

        w = self.embed(qk)
        w = w.view(b, 1, -1, self.kernel_size*self.kernel_size*self.kernel_size, qk_d, qk_hh, qk_ww)
        
        x = self.conv1x1(x)
        x = self.local_conv(x, w)  # Need to implement LocalConvolution3D
        x = self.bn(x)
        x = self.act(x)

        B, C, D, H, W = x.shape
        x = x.view(B, C, 1, D, H, W)
        k = k.view(B, C, 1, D, H, W)
        x = torch.cat([x, k], dim=2)

        x_gap = x.sum(dim=2)
        x_gap = x_gap.mean((2, 3, 4), keepdim=True)
        x_attn = self.se(x_gap)
        x_attn = x_attn.view(B, C, self.radix)
        x_attn = F.softmax(x_attn, dim=2)
        out = (x * x_attn.reshape((B, C, self.radix, 1, 1, 1))).sum(dim=2)
        
        return out.contiguous()

class CoXtLayer(nn.Module):
    def __init__(self, dim, kernel_size):
        super(CoXtLayer, self).__init__()

        self.dim = dim
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
            nn.Conv3d(dim, dim, self.kernel_size, stride=1, padding=self.kernel_size//2, groups=8, bias=False),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True)
        )

        self.dw_group = 2
        share_planes = 8
        factor = 2
        self.embed = nn.Sequential(
            nn.Conv3d(2*dim, dim//factor, 1, groups=self.dw_group, bias=False),
            nn.BatchNorm3d(dim//factor),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim//factor, pow(kernel_size, 2) * dim // share_planes, kernel_size=1, groups=self.dw_group),
            nn.GroupNorm(num_groups=dim // share_planes, num_channels=pow(kernel_size, 2) * dim // share_planes)
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=1, stride=1, padding=0, dilation=1, groups=self.dw_group, bias=False),
            nn.BatchNorm3d(dim)
        )

        self.local_conv = LocalConvolution(dim, dim, kernel_size=self.kernel_size, stride=1, padding=(self.kernel_size - 1) // 2, dilation=1)
        self.bn = nn.BatchNorm3d(dim)
        act = get_act_layer('swish')
        self.act = act(inplace=True)

        reduction_factor = 4
        self.radix = 2
        attn_chs = max(dim * self.radix // reduction_factor, 32)
        self.se = nn.Sequential(
            nn.Conv3d(dim, attn_chs, 1),
            nn.BatchNorm3d(attn_chs),
            nn.ReLU(inplace=True),
            nn.Conv3d(attn_chs, self.radix*dim, 1)
        )

    def forward(self, x):
        batch_size, channels, depth, height, width = x.size()
        k = self.key_embed(x)
        qk = torch.cat([x.unsqueeze(2), k.unsqueeze(2)], dim=2)
        qk = qk.view(batch_size, -1, depth, height, width)

        w = self.embed(qk)
        w = w.view(batch_size * 2, 1, -1, self.kernel_size*self.kernel_size, depth, height, width)
        
        x = self.conv1x1(x)
        x = x.view(batch_size * 2, -1, depth, height, width)
        x = self.local_conv(x, w)
        x = x.view(batch_size, -1, depth, height, width)
        x = self.bn(x)
        x = self.act(x)

        B, C, D, H, W = x.shape
        x = x.view(B, C, 1, D, H, W)
        k = k.view(B, C, 1, D, H, W)
        x = torch.cat([x, k], dim=2)

        x_gap = x.sum(dim=2)
        x_gap = x_gap.mean((2, 3, 4), keepdim=True)
        x_attn = self.se(x_gap)
        x_attn = x_attn.view(B, C, self.radix)
        x_attn = F.softmax(x_attn, dim=2)
        out = (x * x_attn.reshape((B, C, self.radix, 1, 1, 1))).sum(dim=2)
        
        return out.contiguous()


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
                 reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm3d,
                 attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(Bottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv3d(in_planes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)

        if stride > 1:
            self.avd = nn.AvgPool3d(3, 2, padding=1)
        else:
            self.avd = None
        
        self.conv2 = CotLayer(5, kernel_size=3) if cardinality == 1 else CoXtLayer(width, kernel_size=3)

        #self.conv2 = nn.Conv2d(
        #    first_planes, width, kernel_size=3, stride=1 if use_aa else stride,
        #    padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        #self.bn2 = norm_layer(width)
        #self.act2 = act_layer(inplace=True)
        #self.aa = aa_layer(channels=width, stride=stride) if use_aa else None

        self.conv3 = nn.Conv3d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        if self.avd is not None:
            x = self.avd(x)

        x = self.conv2(x)
        #x = self.bn2(x)
        #if self.drop_block is not None:
        #    x = self.drop_block(x)
        #x = self.act2(x)
        #if self.aa is not None:
        #    x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = self.act3(x)

        return x



class ResNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=1,
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
        
        #self.hs = hswish()
        
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # addAFF_BasicBlock  addnam_BasicBlock  addcbam_BasicBlock  addse_BasicBlock  block
        self.layer1 = self._make_layer(
                                            # addnam_BasicBlock, 
                                        block,
                                       block_inplanes[0], 
                                       layers[0],
                                       shortcut_type)
        
        self.layer2 = self._make_layer(
                                          block,
                                        # addnam_BasicBlock,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        
        self.layer3 = self._make_layer(
                                          block,
                                        #  addnam_BasicBlock,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        
        self.layer4 = self._make_layer(
                                        #  addnam_BasicBlock,
                                        block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)
        
        self.dropout = nn.Dropout(p=0.65)
        #self.dropout2 = nn.Dropout(p=0.25)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)
        #self.fc_256 = nn.Linear(256, n_classes)
        #self.fc_meta = nn.Linear(block_inplanes[3] * block.expansion+1, n_classes)
        #self.fc_pca_meta = nn.Linear(block_inplanes[3] * block.expansion + 4, n_classes)
        #self.fc_pca = nn.Linear(8, n_classes)
        self.softmax = nn.Softmax(dim=1)
        #self.sigmoid = nn.Sigmoid()
        #self.attention = attention
        # self.cbam1 = CBAM(block_inplanes[0], 16 , 7)
        # self.cbam2 = CBAM(block_inplanes[1], 16 , 7)
        # self.cbam3 = CBAM(block_inplanes[2], 16 , 7)
        # self.cbam4 = CBAM(block_inplanes[3], 16 , 7)
        # self.fc1 = nn.Linear(512, 2048)
        
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

    def forward(self, x, feature=None, meta=None):
        #print(x.shape)
        x = self.conv1(x)
        #print(x.shape)
        x = self.bn1(x)
        #x = self.hs(x)
        x = self.relu(x)

        if not self.no_max_pool:
            x = self.maxpool(x)
            
        #print(x.shape)
        x = self.layer1(x)
        #print(x.shape)
        x = self.layer2(x)
        #print(x.shape)
        x = self.layer3(x)
        #print(x.shape)
        x = self.layer4(x)
        #print(x.shape)
        
        x = self.avgpool(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.dropout(x)
        #print(x.shape)
        
        
        
        feature_invol = x
        
        if meta is not None:
            #print(meta)
            x = torch.cat((x, meta), dim=-1)
        
        if feature is not None and meta is not None:
            # Both feature and meta are present
            x = torch.add(x, feature)
            # pca = PCA(n_components=4)
            # pca_x = x.cpu().detach().numpy()
            # pca.fit(pca_x)
            # pca_x = pca.transform(pca_x)
            # x = torch.from_numpy(pca_x).to('cuda:0')
            # meta = self.meta_nn.forward(meta)
            # x = torch.cat((x, meta), dim=-1)
            # x = self.fc_pca_meta(x)
            # x = self.sigmoid(x)
            # x = self.dropout2(x)
            # x = F.relu(self.fc1(x))
            x = torch.cat((x, meta), dim=-1)
            # print(x.shape)
            # x = self.dropout2(x)
            # x = F.relu(self.fc2(x))

        if feature is not None:
            # Only feature is present
            x = self.res_vit(x)
            
            x = torch.cat((x, feature), dim=1)
            x = self.fc_256(x)
            
            # x = torch.add(x, feature)
            # x = self.fc3(x)

        x = self.fc(x)
        #print(x.shape)
        x = self.softmax(x)
        #print(x.shape)
        
        return x, feature_invol
        # return x

def cotnet50(input_channels=1, num_classes=2):
    return ResNet(Bottleneck, [3, 4, 6, 3],  get_inplanes(), n_input_channels=input_channels, n_classes=num_classes)
'''
def _create_cotnet(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(
        ResNet, variant, default_cfg=default_cfgs[variant], pretrained=pretrained, **kwargs)

@register_model
def cotnet50(pretrained=False, **kwargs):
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3],  **kwargs)
    return _create_cotnet('cot_basic', pretrained, **model_args)

@register_model
def cotnext50_2x48d(pretrained=False, **kwargs):
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], cardinality=2, base_width=48, **kwargs)
    return _create_cotnet('cot_basic', pretrained, **model_args)

@register_model
def cotnet101(pretrained=False, **kwargs):
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], **kwargs)
    return _create_cotnet('cot_basic', pretrained, **model_args)

@register_model
def cotnext101_2x48d(pretrained=False, **kwargs):
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=2, base_width=48, **kwargs)
    return _create_cotnet('cot_basic', pretrained, **model_args)
'''