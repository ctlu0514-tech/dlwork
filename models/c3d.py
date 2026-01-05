"""
This is the c3d implementation with batch norm.

References
----------
[1] Tran, Du, et al. "Learning spatiotemporal features with 3d convolutional networks." 
Proceedings of the IEEE international conference on computer vision. 2015.
"""

import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial


class C3D(nn.Module):
    def __init__(self,
                 sample_size = 256,
                 sample_duration = 16,
                 num_classes=2):

        super(C3D, self).__init__()
        self.group1 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2)))
        self.group2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2)))
        self.group3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2)))
        self.group4 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2)))
        self.group5 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1)))

        last_duration = int(math.floor(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        self.fc1 = nn.Sequential(
            nn.Linear(9216 , 4096),
            # nn.Linear((648 * last_duration * last_size * last_size) , 4096),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc3 = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.5))
        

        self.fc = nn.Sequential(
            nn.Linear(2048, num_classes))
        

        self.fc_meta = nn.Sequential(
            nn.Linear(9216+8 , 4096),
            # nn.Linear((648 * last_duration * last_size * last_size) , 4096),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x, feature=None, meta=None):
        out = self.group1(x)
        # print(out.shape)
        out = self.group2(out)
        # print(out.shape)
        out = self.group3(out)
        # print(out.shape)
        out = self.group4(out)
        # print(out.shape)
        out = self.group5(out)
        # print(out.shape)
        out = out.view(out.size(0), -1)
        
        
        out = self.fc1(out)
        
        out = self.fc2(out)

        out = self.fc3(out)
        feature_invol = out
        if feature is not None:
            
            out = torch.add(out, feature)
        
        out = self.fc(out)
        # elif feature is not None and meta is not None:

        #     out = torch.add(out, feature)
        #     out = torch.cat((out, meta), dim=-1)
        #     out = self.fc_meta(out)
        
        #     out = self.fc2(out)
        
        #     out = self.fc(out)

        # else:  
        #     out = self.fc1(out)
        
        #     out = self.fc2(out)
        
        #     out = self.fc(out)
        
        return out, feature_invol


def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()

    elif ft_portion == "last_layer":
        ft_module_names = []
        ft_module_names.append('fc')

        parameters = []
        for k, v in model.named_parameters():
            for ft_module in ft_module_names:
                if ft_module in k:
                    parameters.append({'params': v})
                    break
            else:
                parameters.append({'params': v, 'lr': 0.0})
        return parameters

    else:
        raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' expected")


# def get_model(**kwargs):
#     """
#     Returns the model.
#     """
#     model = C3D(**kwargs)
#     return model


# if __name__ == '__main__':
#     model = get_model(sample_size = 256, sample_duration = 16, num_classes=2)
#     model = model.cuda()

#     input_var = Variable(torch.randn(8, 1, 16, 256, 256))
#     output = model(input_var)
#     print(len(output))

# ones = torch.rand(16, 1, 8, 64, 64)
# model = C3D()
# out, feature = model(ones) 
# print(out.size())
# print(feature.shape)
# print(type(feature))

