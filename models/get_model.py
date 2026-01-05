import torch.nn as nn
import os
import torch
from models.LeNet3d import LeNet
from models.alexnet import AlexNet
from models.c3d import C3D
from models.CNN_3d import CNN_3D, Att_CNN_3D
from models.mobilenet import MobileNet
from models.resnet_3d import resnet10, resnet18, resnet34, resnet50, resnet101
from models.vit import VIT

def get_arch(model_name, attention='false'):
    
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
    elif model_name == 'resnet10':
        model = resnet10()
    elif model_name == 'resnet18':
        model = resnet18()
    elif model_name == 'resnet34':
        model = resnet34()
    elif model_name == 'resnet50':
        model = resnet50()
    elif model_name == 'resnet101':
        model = resnet101()
    elif model_name == 'vit':
        model = VIT(image_size=64, patch_size=8, num_classes=2, dim=128, depth=6, heads=8, mlp_dim=256)
    return model

