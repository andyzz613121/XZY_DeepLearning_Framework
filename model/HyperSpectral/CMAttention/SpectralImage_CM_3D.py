import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
from model.HyperSpectral.Base_Network import HS_Base, HS_Base3D
from model.Self_Module.Attention import SPA_Att
class HS_SI_CM_3D(HS_Base3D):
    def __init__(self, input_channels, out_channels):
        print('Using HS_SI_CM_3D model')
        super(HS_SI_CM_3D,self).__init__(input_channels, out_channels)
        self.SI_conv_1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SI_conv_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.SI_conv_3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.GAP2D = torch.nn.AdaptiveAvgPool2d(1)

        # SA sideout mlps & compress conv空间
        self.SA_fc1 = self.get_mlp(3, [16, 256, out_channels])
        self.SA_fc2 = self.get_mlp(3, [32, 256, out_channels])
        self.SA_fc3 = self.get_mlp(3, [64, 256, out_channels])
        self.SA_CPR1 = nn.Sequential(
            nn.Conv2d(16*42, 16, kernel_size=1),
            nn.BatchNorm2d(16))
        self.SA_CPR2 = nn.Sequential(
            nn.Conv2d(32*36, 32, kernel_size=1),
            nn.BatchNorm2d(32))
        self.SA_CPR3 = nn.Sequential(
            nn.Conv2d(64*30, 64, kernel_size=1),
            nn.BatchNorm2d(64))

        # SP sideout mlps 光谱
        self.SP_fc1 = self.get_mlp(3, [16, 256, out_channels])
        self.SP_fc2 = self.get_mlp(3, [32, 256, out_channels])
        self.SP_fc3 = self.get_mlp(3, [64, 256, out_channels])

        # fuse side convs & mlps 融合
        self.Fuse_conv1 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1),
            nn.BatchNorm2d(16))
        self.Fuse_conv2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.BatchNorm2d(32))
        self.Fuse_conv3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64))
        self.Fuse_fc1 = self.get_mlp(3, [16, 256, out_channels])
        self.Fuse_fc2 = self.get_mlp(3, [32, 256, out_channels])
        self.Fuse_fc3 = self.get_mlp(3, [64, 256, out_channels])

        self.CM_epoch = torch.ones([out_channels, out_channels]).cuda()
        self.CM_iter = torch.zeros([out_channels, out_channels]).cuda()

        # CM mlps 融合
        # self.CM_fc1 = self.get_mlp(3, [out_channels*out_channels, 256, 32*2])
        self.CM_fc2 = self.get_mlp(3, [out_channels*out_channels, 64, 64*2])
        self.CM_fc3 = self.get_mlp(3, [out_channels*out_channels, 128, 128*2])

        # Fusion Paras
        self.fuse_para = nn.Parameter(torch.ones([9]))

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0)

    def get_mlp(self, layer_num, node_list, drop_rate=0.2):
        layers = []
        for layer in range(layer_num-1):
            layers.append(nn.Linear(node_list[layer], node_list[layer+1]))
            if layer+1 != (layer_num-1):  #Last layer
                layers.append(nn.Dropout(drop_rate))
                layers.append(nn.ReLU())
        mlp = nn.Sequential(*layers)
        for m in mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0)
        return mlp

    def sideout2d(self, feats, mlp):
        b, c, h, w = feats.size()
        return mlp(self.GAP2D(feats).view(b, -1))
    
    def CVT3d_2dCompress(self, feats, compress_conv):
        feats = torch.reshape(feats, (feats.shape[0], -1, feats.shape[3], feats.shape[4]))
        return compress_conv(feats)

    def sideout3d(self, feats, mlp, compress_conv):
        b, c, _, h, w = feats.size()
        return mlp(self.GAP2D(self.compress_conv(feats)).view(b, -1))

    def sideout3d_noGAP(self, feats, mlp):
        b, c, _, h, w = feats.size()
        return mlp(feats.view(b, -1))

    def forward(self, x):
        b, c, h, w = x.size()

        pth = int((h - 1)/2)
        ptw = int((w - 1)/2)
        pt = x[:,:,pth,ptw].view(b, 1, 1, -1)
        pt = F.interpolate(pt, size=(1, 121), mode='bilinear', align_corners=False)
        pt = pt.view(b, 1, 11, 11)

        x_3d = torch.unsqueeze(x, 1)
        SA1 = self.conv_1(x_3d)
        SP1 = self.SI_conv_1(pt)
        SA1_2D = self.CVT3d_2dCompress(SA1, self.SA_CPR1)
        SA_side1 = self.sideout2d(SA1_2D, self.SA_fc1)
        SP_side1 = self.sideout2d(SP1, self.SP_fc1)
        fuse_vec1 = torch.cat([SA1_2D, SP1], 1)
        Fuse_side1 = self.Fuse_fc1( self.GAP2D(self.Fuse_conv1(fuse_vec1)).view(b, -1) )

        SA2 = self.conv_2(SA1)
        SP2 = self.SI_conv_2(SP1)
        SA2_2D = self.CVT3d_2dCompress(SA2, self.SA_CPR2)
        SA_side2 = self.sideout2d(SA2_2D, self.SA_fc2)
        SP_side2 = self.sideout2d(SP2, self.SP_fc2)
        fuse_vec2 = torch.cat([SA2_2D, SP2], 1)
        # w2 = self.CM_fc2(self.CM_epoch.view(1, -1)).view(1, -1, 1, 1).repeat(b, 1, 1, 1)
        # fuse_vec2_w = fuse_vec2 * (1+self.sigmoid(w2))
        Fuse_side2 = self.Fuse_fc2( self.GAP2D(self.Fuse_conv2(fuse_vec2)).view(b, -1) )
        
        SA3 = self.conv_3(SA2)
        SP3 = self.SI_conv_3(SP2)
        SA3_2D = self.CVT3d_2dCompress(SA3, self.SA_CPR3)
        SA_side3 = self.sideout2d(SA3_2D, self.SA_fc3)
        SP_side3 = self.sideout2d(SP3, self.SP_fc3)
        fuse_vec3 = torch.cat([SA3_2D, SP3], 1)
        # w3 = self.CM_fc3(self.CM_epoch.view(1, -1)).view(1, -1, 1, 1).repeat(b, 1, 1, 1)
        # fuse_vec3_w = fuse_vec3 * (1+self.sigmoid(w3))
        Fuse_side3 = self.Fuse_fc3( self.GAP2D(self.Fuse_conv3(fuse_vec3)).view(b, -1) )

        Total_fuse = self.fuse_para[0]*Fuse_side3 + self.fuse_para[1]*Fuse_side2 + self.fuse_para[2]*Fuse_side1+\
                     self.fuse_para[3]*SP_side3 + self.fuse_para[4]*SP_side2 + self.fuse_para[5]*SP_side1+\
                     self.fuse_para[6]*SA_side3 + self.fuse_para[7]*SA_side2 + self.fuse_para[8]*SA_side1
        return [Total_fuse, Fuse_side3, Fuse_side2, Fuse_side1, SP_side3, SP_side2, SP_side1, SA_side3, SA_side2, SA_side1]
        # return [self.Fuse_fc3( self.GAP(self.Fuse_conv3(torch.cat([SA3, SP3], 1))).view(b, -1) )]

        # return [self.sideout3d_noGAP(SA3, self.SA_fc3)]
