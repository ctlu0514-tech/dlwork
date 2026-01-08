import torch.nn as nn
import os
import torch
from models.LeNet3d import LeNet
from models.alexnet import AlexNet
from models.c3d import C3D
from models.CNN_3d import CNN_3D, Att_CNN_3D
from models.mobilenet import MobileNet
from models.shufflenet import ShuffleNet
from models.swin_transformer import SwinTransformer
from models.densenet import DenseNet
from models.mobilenet_V3 import MobileNetV3_Small,MobileNetV3_Large
from models.resnet_3d import resnet10, resnet18, resnet34, resnet50, resnet101
from models.resnext import resnext50,resnext101
from models.Resnext50 import Resnext50
from models.vit import VIT
from models.CoTNetmaster.models.cotnet import cotnet50
from models.vgg import VGG_Net
from models.densenet_fgpn import Densenet36_fgpn

def get_arch(model_name, attention='false', input_channels=5):
    
    if model_name == 'LeNet':
        model = LeNet(attention=attention)
    elif model_name == 'AlexNet':
        model = AlexNet()
    elif model_name == 'C3D':
        model = C3D()
    elif model_name == 'CNN_3D':
        model = CNN_3D()
    elif model_name == 'Att_CNN_3D':
        model = Att_CNN_3D()
    elif model_name == 'MobileNet':
        model = MobileNet()
    elif model_name == 'MobileNetV3_Small':
        model = MobileNetV3_Small()
    elif model_name == 'MobileNetV3_Large':
        model = MobileNetV3_Large()
    elif model_name == 'ShuffleNet':
        model = ShuffleNet(groups=3)
    elif model_name == 'DenseNet':
        model = DenseNet()
    elif model_name == 'resnet10':
        model = resnet10(input_channels=input_channels)
    elif model_name == 'resnet18':
        model = resnet18(input_channels=input_channels)
    elif model_name == 'resnet34':
        model = resnet34(input_channels=input_channels)
    elif model_name == 'resnet50':
        model = resnet50(input_channels=input_channels)
    elif model_name == 'resnet101':
        model = resnet101(input_channels=input_channels)
    elif model_name == 'resnext50':
        model = resnext50(input_channels=input_channels)
    elif model_name == 'resnext101':
        model = resnext101(input_channels=input_channels)
    elif model_name == 'Resnext50':
        model = Resnext50(2)
    elif model_name == 'cotnet50':
        model = cotnet50(input_channels=input_channels)
    elif model_name == 'vit':
        model = VIT(image_size=64, patch_size=8, num_classes=2, dim=128, depth=6, heads=8, mlp_dim=256, channels=input_channels)
    elif model_name == 'vgg':
        model = VGG_Net()
    elif model_name == 'SwinTransformer':
        model = SwinTransformer()
    elif model_name == 'Densenet36_fgpn':
        model = Densenet36_fgpn(in_channels=input_channels, num_classes=2)
    return model

