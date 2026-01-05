import torch.nn as nn
import torch
import functools
import numpy as np
IMAGE_SIZE_FOR_NET = [64,64,10] 
# official pretrain weights
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=False):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(512*4*4, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)
        # N x 512*7*7
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_features(cfg: list):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg(model_name="vgg16", **kwargs):
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    cfg = cfgs[model_name]

    model = VGG(make_features(cfg), **kwargs)
    return model

'''class VGG_Net(nn.Module):
    def __init__(self, in_channels=3, n_targets=2):
        super(VGG_Net,self).__init__()
        self.conv11 = nn.Sequential(
            nn.Conv3d(in_channels,8,[3,3,1], padding = [1,1,0]),
            nn.ReLU(),
            nn.BatchNorm3d(8, affine=False),
        )
        
        self.conv12 = nn.Sequential(
            nn.Conv3d(8,8,[3,3,1], padding = [1,1,0]),
            nn.ReLU(),
            nn.BatchNorm3d(8, affine=False),            
        )
        self.conv13 = nn.Sequential(
            nn.Conv3d(8,8,[3,3,1], padding = [1,1,0]),
            nn.ReLU(),
            nn.BatchNorm3d(8, affine=False),
        )
        
        self.conv21 = nn.Sequential(           
            nn.MaxPool3d(2),            
            nn.Conv3d(8,64,[3,3,1], padding = [1,1,0]),
            nn.ReLU(),
            nn.BatchNorm3d(64, affine=False),
        )
        self.conv22 = nn.Sequential(          
            nn.Conv3d(64,64,[3,3,1], padding = [1,1,0]),
            nn.ReLU(),
            nn.BatchNorm3d(64, affine=False),
        )
        self.conv23 = nn.Sequential(            
            nn.Conv3d(64,64,[3,3,1], padding = [1,1,0]),
            nn.ReLU(),
            nn.BatchNorm3d(64, affine=False),
        )
        self.conv31 = nn.Sequential(                        
            nn.MaxPool3d(2),            
            nn.Conv3d(64,128,[3,3,1], padding = [1,1,0]),
            nn.ReLU(),
            nn.BatchNorm3d(128, affine=False),
        )
        self.conv32 = nn.Sequential(            
            nn.Conv3d(128,128,[3,3,1], padding = [1,1,0]),
            nn.ReLU(),
            nn.BatchNorm3d(128, affine=False),
        )
        self.conv33 = nn.Sequential(            
            nn.Conv3d(128,128,[3,3,1], padding = [1,1,0]), 
            nn.ReLU(),
            nn.BatchNorm3d(128, affine=False),
        )
            
        self.final_conv_depth = 128
        n_maxpools = 2
        size = [i//(2**n_maxpools) for i in IMAGE_SIZE_FOR_NET]
        print(size)
        self.flattened_size = int(functools.reduce(lambda x, y:x*y, size)) * self.final_conv_depth
        print(self.flattened_size)
        
        self.lin0 = nn.Sequential(
            nn.Linear(self.flattened_size,2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048, affine=False),
        )
        
        self.lin1 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024, affine=False),
        )
        self.lin2 = nn.Sequential(
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64, affine=False),
        )
        self.classifier_to_end = nn.Sequential(
            nn.Linear(64,n_targets),
        )
    def forward(self, x):
        activations = []
        x = self.conv11(x)
        activations.append(x.view(-1, np.prod(IMAGE_SIZE_FOR_NET) * 8))
        
        x = self.conv12(x)
        activations.append(x.view(-1, np.prod(IMAGE_SIZE_FOR_NET) * 8))
        
        x = self.conv13(x)
        activations.append(x.view(-1, np.prod(IMAGE_SIZE_FOR_NET) * 8))
        
        x = self.conv21(x)
        activations.append(x.view(-1, np.prod(IMAGE_SIZE_FOR_NET) * 8))
        
        x = self.conv22(x)
        activations.append(x.view(-1, np.prod(IMAGE_SIZE_FOR_NET) * 8))
        
        x = self.conv23(x)
        activations.append(x.view(-1, np.prod(IMAGE_SIZE_FOR_NET) * 8))
        
        x = self.conv31(x)
        activations.append(x.view(-1, np.prod(np.floor_divide(IMAGE_SIZE_FOR_NET,4))*128))
        
        x = self.conv32(x)
        activations.append(x.view(-1, np.prod(np.floor_divide(IMAGE_SIZE_FOR_NET,4))*128))
        
        x = self.conv33(x)
        activations.append(x.view(-1, np.prod(np.floor_divide(IMAGE_SIZE_FOR_NET,4))*128))
        
        x = x.view(-1, self.flattened_size)
        
        x = self.lin0(x)
        activations.append(x.view(-1, 2048))
        
        x = self.lin1(x)
        activations.append(x.view(-1, 1024))
        
        x = self.lin2(x)
        activations.append(x.view(-1, 64))
        
        return self.classifier_to_end(x)'''
    


class VGG_Net(nn.Module):
    def __init__(self, in_channels=5, n_targets=2):
        super(VGG_Net,self).__init__()
        self.conv11 = nn.Sequential(
            nn.Conv3d(in_channels,4,[3,3,1], padding = [1,1,0]),
            nn.ReLU(),
            nn.BatchNorm3d(4, affine=False),
        )
        
        self.conv12 = nn.Sequential(
            nn.Conv3d(4,4,[3,3,1], padding = [1,1,0]),
            nn.ReLU(),
            nn.BatchNorm3d(4, affine=False),            
        )
        self.conv13 = nn.Sequential(
            nn.Conv3d(4,4,[3,3,1], padding = [1,1,0]),
            nn.ReLU(),
            nn.BatchNorm3d(4, affine=False),
        )
        
        self.conv21 = nn.Sequential(           
            nn.MaxPool3d(2),           
            nn.Conv3d(4,8,[3,3,1], padding = [1,1,0]),
            nn.ReLU(),
            nn.BatchNorm3d(8, affine=False),
        )
        self.conv22 = nn.Sequential(          
            nn.Conv3d(8,8,[3,3,1], padding = [1,1,0]),
            nn.ReLU(),
            nn.BatchNorm3d(8, affine=False),
        )
        self.conv23 = nn.Sequential(            
            nn.Conv3d(8,8,[3,3,1], padding = [1,1,0]),
            nn.ReLU(),
            nn.BatchNorm3d(8, affine=False),
        )
        self.conv31 = nn.Sequential(                        
            nn.MaxPool3d(2),            
            nn.Conv3d(8,16,[3,3,1], padding = [1,1,0]),
            nn.ReLU(),
            nn.BatchNorm3d(16, affine=False),
        )
        self.conv32 = nn.Sequential(            
            nn.Conv3d(16,16,[3,3,1], padding = [1,1,0]),
            nn.ReLU(),
            nn.BatchNorm3d(16, affine=False),
        )
        self.conv33 = nn.Sequential(            
            nn.Conv3d(16,16,[3,3,1], padding = [1,1,0]), 
            nn.ReLU(),
            nn.BatchNorm3d(16, affine=False),
        )
        self.conv41 = nn.Sequential(    
            nn.MaxPool3d(2),                             
            nn.Conv3d(16,32,[3,3,1], padding = [1,1,0]),
            nn.ReLU(),
            nn.BatchNorm3d(32, affine=False),
        )
        self.conv42 = nn.Sequential(            
            nn.Conv3d(32,32,[3,3,1], padding = [1,1,0]),
            nn.ReLU(),
            nn.BatchNorm3d(32, affine=False),
        )
        self.conv43 = nn.Sequential(            
            nn.Conv3d(32,32,[3,3,1], padding = [1,1,0]), 
            nn.ReLU(),
            nn.BatchNorm3d(32, affine=False),
        )
        self.conv51 = nn.Sequential(                                  
            nn.Conv3d(32,64,[3,3,1], padding = [1,1,0]),
            nn.ReLU(),
            nn.BatchNorm3d(64, affine=False),
        )
        self.conv52 = nn.Sequential(            
            nn.Conv3d(64,64,[3,3,1], padding = [1,1,0]),
            nn.ReLU(),
            nn.BatchNorm3d(64, affine=False),
        )
        self.conv53 = nn.Sequential(            
            nn.Conv3d(64,64,[3,3,1], padding = [1,1,0]), 
            nn.ReLU(),
            nn.BatchNorm3d(64, affine=False),
        )
        self.conv61 = nn.Sequential(                                  
            nn.Conv3d(64,128,[3,3,1], padding = [1,1,0]),
            nn.ReLU(),
            nn.BatchNorm3d(128, affine=False),
        )
        self.conv62 = nn.Sequential(            
            nn.Conv3d(128,128,[3,3,1], padding = [1,1,0]),
            nn.ReLU(),
            nn.BatchNorm3d(128, affine=False),
        )
        self.conv63 = nn.Sequential(            
            nn.Conv3d(128,128,[3,3,1], padding = [1,1,0]), 
            nn.ReLU(),
            nn.BatchNorm3d(128, affine=False),
        )
        '''self.conv71 = nn.Sequential(                                
            nn.Conv3d(128,256,[3,3,1], padding = [1,1,0]),
            nn.ReLU(),
            nn.BatchNorm3d(256, affine=False),
        )
        self.conv72 = nn.Sequential(            
            nn.Conv3d(256,256,[3,3,1], padding = [1,1,0]),
            nn.ReLU(),
            nn.BatchNorm3d(256, affine=False),
        )
        self.conv73 = nn.Sequential(            
            nn.Conv3d(256,256,[3,3,1], padding = [1,1,0]), 
            nn.ReLU(),
            nn.BatchNorm3d(256, affine=False),
        )
        self.conv81 = nn.Sequential(     
            nn.MaxPool3d(2),                                  
            nn.Conv3d(256,512,[3,3,1], padding = [1,1,0]),
            nn.ReLU(),
            nn.BatchNorm3d(512, affine=False),
        )
        self.conv82 = nn.Sequential(            
            nn.Conv3d(512,512,[3,3,1], padding = [1,1,0]),
            nn.ReLU(),
            nn.BatchNorm3d(512, affine=False),
        )
        self.conv83 = nn.Sequential(            
            nn.Conv3d(512,512,[3,3,1], padding = [1,1,0]), 
            nn.ReLU(),
            nn.BatchNorm3d(512, affine=False),
        )'''
        self.final_conv_depth = 128
        n_maxpools = 3
        size = [i//(2**n_maxpools) for i in IMAGE_SIZE_FOR_NET]
        self.flattened_size = int(functools.reduce(lambda x, y:x*y, size)) * self.final_conv_depth

        
        self.lin0 = nn.Sequential(
            nn.Linear(self.flattened_size,2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048, affine=False),
        )
        
        self.lin1 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024, affine=False),
        )
        self.lin2 = nn.Sequential(
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64, affine=False),
        )
        self.classifier_to_end = nn.Sequential(
            nn.Linear(64,n_targets),
        )
    def forward(self, x):
        #activations = []
        x = self.conv11(x)
        #activations.append(x.view(-1, np.prod(IMAGE_SIZE_FOR_NET) * 8))
        
        x = self.conv12(x)
        #activations.append(x.view(-1, np.prod(IMAGE_SIZE_FOR_NET) * 8))
        
        x = self.conv13(x)
        #activations.append(x.view(-1, np.prod(IMAGE_SIZE_FOR_NET) * 8))
        
        x = self.conv21(x)
        #activations.append(x.view(-1, np.prod(IMAGE_SIZE_FOR_NET) * 8))
        
        x = self.conv22(x)
        #activations.append(x.view(-1, np.prod(IMAGE_SIZE_FOR_NET) * 8))
        
        x = self.conv23(x)
        #activations.append(x.view(-1, np.prod(IMAGE_SIZE_FOR_NET) * 8))
        
        x = self.conv31(x)
        #activations.append(x.view(-1, np.prod(np.floor_divide(IMAGE_SIZE_FOR_NET,4))*128))
        
        x = self.conv32(x)
        #activations.append(x.view(-1, np.prod(np.floor_divide(IMAGE_SIZE_FOR_NET,4))*128))
        
        x = self.conv33(x)
        #activations.append(x.view(-1, np.prod(np.floor_divide(IMAGE_SIZE_FOR_NET,4))*128))
        x = self.conv41(x)
        x = self.conv42(x)
        x = self.conv43(x)
        x = self.conv51(x)
        x = self.conv52(x)
        x = self.conv53(x)
        x = self.conv61(x)
        x = self.conv62(x)
        x = self.conv63(x)
        '''x = self.conv71(x)
        x = self.conv72(x)
        x = self.conv73(x)
        x = self.conv81(x)
        x = self.conv82(x)
        x = self.conv83(x)'''
        
        
        
        x = x.view(-1, self.flattened_size)
        
        x = self.lin0(x)
        #activations.append(x.view(-1, 2048))
        
        x = self.lin1(x)
        #activations.append(x.view(-1, 1024))
        
        x = self.lin2(x)
        #activations.append(x.view(-1, 64))
        
        return self.classifier_to_end(x)