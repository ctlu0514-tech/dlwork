import torch.nn as nn
import torch


class AlexNet(nn.Module):
    def __init__(self, num_classes=2, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 48, kernel_size=3, stride=4, padding=2),  
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(1, 0, 0)),
            nn.Conv3d(48, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(1, 0, 0)),
            nn.Conv3d(128, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(1, 0, 0)),
        )
        self.classifier = nn.Sequential(
            
            nn.Dropout(p=0.5),
            nn.Flatten(),
            nn.Linear(16384, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x, feature=None, meta=None):
        for module in self.features:
            x = module(x)
            # print(x.size())
        x = torch.flatten(x, start_dim=1)
        # print(x.size())
        feature = x.cpu().detach().numpy()
       

        for module in self.classifier:
                x = module(x)
                # print(x.size())
                #         
        return x, feature

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# ones = torch.rand(16, 1, 8, 64, 64)
# model = AlexNet()
# out, feature = model(ones) 
# print(type(out))
# print(out.size())