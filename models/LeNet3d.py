import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.decomposition import PCA
from models.resnet_3d import CBAM, SEModule, NAM, AFF
    


class LeNet(nn.Module):
    def __init__(self, mode="add", attention='false', planes=64):
        super(LeNet, self).__init__()
        assert mode in ["concat", "add"]
        self.mode = mode
        self.attention = attention
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(1, 0, 0))
        self.conv2 = nn.Conv3d(16, 64, kernel_size=3, stride=2, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(1, 0, 0))
        self.dropout = nn.Dropout(p=0.25)
        
        self.aff_residual = torch.nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64)
        )
        
        self.aff = AFF(channels=planes)
        self.nam = NAM(channels=planes)
        self.se = SEModule(planes, 16, 7)
        self.cbam = CBAM(planes, 16, 7)
        
        # self.fc1 = nn.Linear(2048, 120)
        # self.fc2 = nn.Linear(120, 512)
        # self.fc5 = nn.Linear(512, 84)
        # self.fc4 = nn.Linear(84,2)

        self.fc1 = nn.Linear(2048, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,2)
        


        self.fc_feature = nn.Linear(2048*2 if self.mode == "concat" else 2048, 120)
        self.fc1_meta = nn.Linear(2048+8*3, 120)
        self.fc1_pca = nn.Linear(8, 120)
        self.fc1_meta_pca = nn.Linear(8+8, 120)
        self.fc2_meta = nn.Linear(120+2, 84)



        self.meta_nn = torch.nn.Sequential(
            torch.nn.Linear(4, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 2048),
            torch.nn.ReLU(inplace=True),

            # torch.nn.Linear(4*38, 512),
            # torch.nn.ReLU(inplace=True),
        )

    def forward(self, x, feature=None, meta=None):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # if self.attention == 'AFF':
        #     residual = x
        #     x = self.aff_residual.forward(x)
        #     x = self.aff(x, residual)
        # elif self.attention == 'nam':
        #     x = self.nam(x)
        # elif self.attention == 'se':
        #     x = self.se(x)
        # elif self.attention == 'cbam':
        #     x = self.cbam(x)
            
        x = x.view(x.size(0), -1)
        # feature = x.cpu().detach().numpy()
        x = self.dropout(x)
        feature_invol = x
              
        if feature is not None:
            # x = torch.cat((x, feature), dim=-1)
            x = torch.add(x, feature)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        
        # elif feature is not None and meta is not None:
        #     # x = torch.add(x, feature)
        #     x = torch.cat((x, feature), dim=-1)
        #     # pca = PCA(n_components=8)
        #     # pca_x = x.cpu().detach().numpy()
        #     # pca.fit(pca_x)
        #     # pca_x = pca.transform(pca_x)
        #     # x = torch.from_numpy(pca_x).to('cuda:0')
        #     # x = torch.cat((x, meta), dim=-1)
        #     # x = self.fc1_meta_pca(x)
        #     x = F.relu(self.fc_feature(x))
        #     # x = self.fc1(x)
        #     # meta = self.meta_nn.forward(meta)
        #     x = torch.cat((x, meta), dim=-1)
        #     # x = torch.add(x, meta)
        #     x = F.relu(self.fc2_meta(x))
        #     # x = F.relu(self.fc2(x))
        #     x = self.fc3(x)
        # else:
        #     x = F.relu(self.fc1(x))
        #     x = F.relu(self.fc2(x))
        #     x = self.fc3(x)  

        # elif feature is not None and meta is not None:
        #     # x = torch.cat((x, feature), dim=-1)
        #     x = torch.add(x, feature)
        #     x = torch.cat((x, meta), dim=-1)
        #     x = F.relu(self.fc1_meta(x))
        #     x = F.relu(self.fc5(x))
        #     x = self.fc3(x)
        else:
            x = F.relu(self.fc1(x))
        
            x = F.relu(self.fc2(x))
        
            x = self.fc3(x)
        
        return x, feature_invol


# ones = torch.rand(16, 1, 8, 64, 64)
# # meta = torch.rand(16, 8)
# model = LeNet()
# out, feature = model(ones)
