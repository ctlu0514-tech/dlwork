import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

def get_inplanes():
    return [64, 128, 256, 512]

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out

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

class Bottleneck_MHSA(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = MHSA(n_dims=planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        #print(residual.shape)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        #print(out.shape)
        out = self.conv2(out)
        #print(out.shape)
        out = self.bn2(out)
        out = self.relu(out)
        #print(out.shape)
        out = self.conv3(out)
        out = self.bn3(out)
        #print(out.shape)

        if self.downsample is not None:
            residual = self.downsample(x)
            #print(residual.shape)
        #print(out.shape)

        out += residual
        out = self.relu(out)

        return out
    
    
class Bottleneck(nn.Module):
    expansion = 4

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
        #print(residual.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        #print(out.shape)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        #print(out.shape)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        #print(out.shape)
        out += residual
        out = self.relu(out)

        return out


#SE
class SEModule(nn.Module):

    def __init__(self, channels, reduction, kernel_size):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x
    
class addse_BasicBlock(nn.Module):
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
        self.se = SEModule(planes, 16, 7)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.se(out)
        out += residual
        out = self.relu(out)

        return out

class addse_Bottleneck(nn.Module):
    expansion = 4

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
        self.se = SEModule(planes * self.expansion, 16, 7)

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
        out = self.se(out)
        out += residual
        out = self.relu(out)

        return out

#CBAM
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

# NAM
class NAM(nn.Module):
    def __init__(self, channels=4, t=16):
        super(NAM, self).__init__()
        self.channels = channels
        self.bn2 = nn.BatchNorm3d(self.channels)

    def forward(self, x):
        residual = x
        x = self.bn2(x)
        weight_bn = self.bn2.weight.abs() / torch.sum(self.bn2.weight.abs())
        x = x.permute(0, 4, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 4, 2, 1, 3).contiguous()
        x = torch.sigmoid(x) * residual

        return x
    
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

# AFF
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

class addAFF_Bottleneck(nn.Module):
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
        self.aff = AFF(channels=planes)


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
        out = self.aff(out, residual)
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
        out = self.aff(out, residual)
        out += residual
        out = self.relu(out)

        return out



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
    
    

class MHSA(nn.Module):
    def __init__(self, n_dims, width=14, height=14, depth=14, heads=4):
        super(MHSA, self).__init__()
        self.heads = heads

        self.query = nn.Conv3d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv3d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv3d(n_dims, n_dims, kernel_size=1)

        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height, depth]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, width, 1, depth]), requires_grad=True)
        self.rel_d = nn.Parameter(torch.randn([1, heads, n_dims // heads, width, height, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height, depth = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)

        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)
        c1, c2, c3, c4 = content_content.size()
        
        content_position = (self.rel_h + self.rel_w + self.rel_d).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)
        content_position = torch.matmul(content_position, q)
        content_position = content_position if (content_content.shape == content_position.shape) else content_position[:, :, :c3, ]
        assert (content_content.shape == content_position.shape)
        
        #print(content_content.shape)
        #print(content_position.shape)
        
        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height, depth)

        return out

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
                                       stride=1)
        
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
        '''
        self.meta_nn = torch.nn.Sequential(
            torch.nn.Linear(1, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(inplace=True),

            # torch.nn.Linear(4*38, 512),
            # torch.nn.ReLU(inplace=True),
        )
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.25),
            torch.nn.Linear(512*2, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.25),
            torch.nn.Linear(512, 2)
        )
        
        self.classifier1 = torch.nn.Sequential(
            torch.nn.Dropout(p=0.25),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.25),
            torch.nn.Linear(512, 2)
        )
        
        self.classifier2 = torch.nn.Sequential(
            torch.nn.Linear(512 + 1, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.25),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.25),
            torch.nn.Linear(512, 2)
        )


        

        self.fc1 = nn.Linear(512, 256)
        # self.fc2 = nn.Linear(256+8*3, 128)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)
        
        self.res_vit = nn.Linear(512, 128)
        '''
        
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
        
        #return x, feature_invol
        return x


def resnet10(input_channels=1, num_classes=2):
    return ResNet(BasicBlock, [1, 1, 1, 1],  get_inplanes(), n_input_channels=input_channels, n_classes=num_classes)

def resnet18(input_channels=1, num_classes=2):
    return ResNet(BasicBlock, [2, 2, 2, 2],  get_inplanes(), n_input_channels=input_channels, n_classes=num_classes)

def resnet34(input_channels=1, num_classes=2):
    return ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), n_input_channels=input_channels, n_classes=num_classes)



def resnet50(input_channels=1, num_classes=2):
    return ResNet(Bottleneck, [3, 4, 6, 3],  get_inplanes(), n_input_channels=input_channels, n_classes=num_classes)

def resnet101(input_channels=1, num_classes=2):
    return ResNet(Bottleneck, [3, 4, 23, 3],  get_inplanes(), n_input_channels=input_channels, n_classes=num_classes)



# ones = torch.rand(16, 1, 8, 64 ,64)
# model = resnet34()
# out = model(ones)


