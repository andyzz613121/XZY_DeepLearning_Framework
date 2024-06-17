import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.Self_Module.Attention_SRNet import AttachAttentionModule
from model.Self_Module.SRM_filter import spam11, minmax41

class SRNet(nn.Module):
    def __init__(self, in_channels=3):
        super(SRNet, self).__init__()
        print("Using SRNet in 'SRNet//SRNet_Base//'")
        # Layer 1
        self.layer1 = nn.Conv2d(in_channels=in_channels, out_channels=64,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.attention1 = AttachAttentionModule(64)

        # Layer 2
        self.layer2 = nn.Conv2d(in_channels=64, out_channels=16,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.attention2 = AttachAttentionModule(16)

        # Layer 3
        self.layer31 = nn.Conv2d(in_channels=16, out_channels=16,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn31 = nn.BatchNorm2d(16)
        self.layer32 = nn.Conv2d(in_channels=16, out_channels=16,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn32 = nn.BatchNorm2d(16)
        self.attention3 = AttachAttentionModule(16)

        # Layer 4
        self.layer41 = nn.Conv2d(in_channels=16, out_channels=16,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn41 = nn.BatchNorm2d(16)
        self.layer42 = nn.Conv2d(in_channels=16, out_channels=16,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn42 = nn.BatchNorm2d(16)
        self.attention4 = AttachAttentionModule(16)

        # Layer 5
        self.layer51 = nn.Conv2d(in_channels=16, out_channels=16,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn51 = nn.BatchNorm2d(16)
        self.layer52 = nn.Conv2d(in_channels=16, out_channels=16,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn52 = nn.BatchNorm2d(16)
        self.attention5 = AttachAttentionModule(16)
        
        # Layer 6
        self.layer61 = nn.Conv2d(in_channels=16, out_channels=16,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn61 = nn.BatchNorm2d(16)
        self.layer62 = nn.Conv2d(in_channels=16, out_channels=16,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn62 = nn.BatchNorm2d(16)
        self.attention6 = AttachAttentionModule(16)
        
        # Layer 7
        self.layer71 = nn.Conv2d(in_channels=16, out_channels=16,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn71 = nn.BatchNorm2d(16)
        self.layer72 = nn.Conv2d(in_channels=16, out_channels=16,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn72 = nn.BatchNorm2d(16)
        self.attention7 = AttachAttentionModule(16)
        
        # Layer 8
        self.layer81 = nn.Conv2d(in_channels=16, out_channels=16,
            kernel_size=1, stride=2, padding=0, bias=False)
        self.bn81 = nn.BatchNorm2d(16)
        self.layer82 = nn.Conv2d(in_channels=16, out_channels=16,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn82 = nn.BatchNorm2d(16)
        self.layer83 = nn.Conv2d(in_channels=16, out_channels=16,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn83 = nn.BatchNorm2d(16)
        self.pool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        
        # Layer 9
        self.layer91 = nn.Conv2d(in_channels=16, out_channels=64,
            kernel_size=1, stride=2, padding=0, bias=False)
        self.bn91 = nn.BatchNorm2d(64)
        self.layer92 = nn.Conv2d(in_channels=16, out_channels=64,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn92 = nn.BatchNorm2d(64)
        self.layer93 = nn.Conv2d(in_channels=64, out_channels=64,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn93 = nn.BatchNorm2d(64)
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        
        # Layer 10
        self.layer101 = nn.Conv2d(in_channels=64, out_channels=128,
            kernel_size=1, stride=2, padding=0, bias=False)
        self.bn101 = nn.BatchNorm2d(128)
        self.layer102 = nn.Conv2d(in_channels=64, out_channels=128,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn102 = nn.BatchNorm2d(128)
        self.layer103 = nn.Conv2d(in_channels=128, out_channels=128,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn103 = nn.BatchNorm2d(128)
        self.pool3 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        
        # Layer 11
        self.layer111 = nn.Conv2d(in_channels=128, out_channels=256,
            kernel_size=1, stride=2, padding=0, bias=False)
        self.bn111 = nn.BatchNorm2d(256)
        self.layer112 = nn.Conv2d(in_channels=128, out_channels=256,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn112 = nn.BatchNorm2d(256)
        self.layer113 = nn.Conv2d(in_channels=256, out_channels=256,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn113 = nn.BatchNorm2d(256)
        self.pool4 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        # Layer 12
        self.layer121 = nn.Conv2d(in_channels=256, out_channels=512,
            kernel_size=3, stride=2, padding=1, bias=False)
        self.bn121 = nn.BatchNorm2d(512)
        self.layer122 = nn.Conv2d(in_channels=512, out_channels=512,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn122 = nn.BatchNorm2d(512)
        
        self.deconv2 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.debn2   = nn.BatchNorm2d(512)
        #倒数2层的反卷积[1/16 --> 1/8],
        self.deconv1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.debn1   = nn.BatchNorm2d(256)
        #倒数3层的反卷积[1/8 --> 1/4],
        self.deconv0_1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.debn0_1   = nn.BatchNorm2d(128)
        #倒数4层的反卷积[1/4 --> 1/2],
        self.deconv0_2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.debn0_2   = nn.BatchNorm2d(64)
        #倒数5层的反卷积[1/2 --> 1/1],
        self.deconv0_3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.debn0_3   = nn.BatchNorm2d(32)

        self.classifier = nn.Conv2d(32, 2, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal(m.weight.data)
                if m.bias != None:
                    m.bias.data.fill_(0)    
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal(m.weight.data)
                if m.bias != None:
                    m.bias.data.fill_(0)

    def forward(self, inputs):
        # Layer 1
        conv = self.layer1(inputs)
        actv = F.relu(self.bn1(conv))
        actv = self.attention1(actv)
        
        # Layer 2
        conv = self.layer2(actv)
        actv = F.relu(self.bn2(conv))
        actv = self.attention2(actv)
        
        # Layer 3
        conv1 = self.layer31(actv)
        actv1 = F.relu(self.bn31(conv1))
        conv2 = self.layer32(actv1)
        bn = self.bn32(conv2)
        res = torch.add(actv, bn)
        res = self.attention3(res)
        
        # Layer 4
        conv1 = self.layer41(res)
        actv1 = F.relu(self.bn41(conv1))
        conv2 = self.layer42(actv1)
        bn = self.bn42(conv2)
        res = torch.add(res, bn)
        res = self.attention4(res)
        
        # Layer 5
        conv1 = self.layer51(res)
        actv1 = F.relu(self.bn51(conv1))
        conv2 = self.layer52(actv1)
        bn = self.bn52(conv2)
        res = torch.add(res, bn)
        res = self.attention5(res)
        
        # Layer 6
        conv1 = self.layer61(res)
        actv1 = F.relu(self.bn61(conv1))
        conv2 = self.layer62(actv1)
        bn = self.bn62(conv2)
        res = torch.add(res, bn)
        res = self.attention6(res)
        
        # Layer 7
        conv1 = self.layer71(res)
        actv1 = F.relu(self.bn71(conv1))
        conv2 = self.layer72(actv1)
        bn = self.bn72(conv2)
        res = torch.add(res, bn)
        res = self.attention7(res)
        
        # Layer 8
        convs = self.layer81(res)
        convs = self.bn81(convs)
        conv1 = self.layer82(res)
        actv1 = F.relu(self.bn82(conv1))
        conv2 = self.layer83(actv1)
        bn = self.bn83(conv2)
        pool = self.pool1(bn)
        res = torch.add(convs, pool)
        
        # Layer 9
        convs = self.layer91(res)
        convs = self.bn91(convs)
        conv1 = self.layer92(res)
        actv1 = F.relu(self.bn92(conv1))
        conv2 = self.layer93(actv1)
        bn = self.bn93(conv2)
        pool = self.pool2(bn)
        res = torch.add(convs, pool)
        
        # Layer 10
        convs = self.layer101(res)
        convs = self.bn101(convs)
        conv1 = self.layer102(res)
        actv1 = F.relu(self.bn102(conv1))
        conv2 = self.layer103(actv1)
        bn = self.bn103(conv2)
        pool = self.pool3(bn)
        res = torch.add(convs, pool)
        
        # Layer 11
        convs = self.layer111(res)
        convs = self.bn111(convs)
        conv1 = self.layer112(res)
        actv1 = F.relu(self.bn112(conv1))
        conv2 = self.layer113(actv1)
        bn = self.bn113(conv2)
        pool = self.pool4(bn)
        res = torch.add(convs, pool)
        
        # Layer 12
        conv1 = self.layer121(res)
        actv1 = F.relu(self.bn121(conv1))
        conv2 = self.layer122(actv1)
        bn = self.bn122(conv2)
        
        #最后一层特征图反卷积，使其大小与倒数第二层特征图一致(原图1/16)
        x2_1 = F.relu(self.debn2(self.deconv2(bn)))
        #倒数第二层特征图反卷积，使其大小与倒数第三层特征图一致(原图1/8)
        x1_0 = F.relu(self.debn1(self.deconv1(x2_1)))
        
        #倒数第三层特征图连续3次反卷积，使其大小与原始图像一致(由原图1/8到1/1)
        x0_image = F.relu(self.debn0_1(self.deconv0_1(x1_0)))
        x0_image = F.relu(self.debn0_2(self.deconv0_2(x0_image)))
        x0_image = F.relu(self.debn0_3(self.deconv0_3(x0_image)))
        outputs = self.classifier(x0_image)
        return outputs

    # def forward(self, inputs):
    #     # Layer 1
    #     conv = self.layer1(inputs)
    #     actv = F.relu((conv))
    #     actv = self.attention1(actv)
        
    #     # Layer 2
    #     conv = self.layer2(actv)
    #     actv = F.relu((conv))
    #     actv = self.attention2(actv)
        
    #     # Layer 3
    #     conv1 = self.layer31(actv)
    #     actv1 = F.relu((conv1))
    #     conv2 = self.layer32(actv1)
    #     bn = (conv2)
    #     res = torch.add(actv, bn)
    #     res = self.attention3(res)
        
    #     # Layer 4
    #     conv1 = self.layer41(res)
    #     actv1 = F.relu((conv1))
    #     conv2 = self.layer42(actv1)
    #     bn = (conv2)
    #     res = torch.add(res, bn)
    #     res = self.attention4(res)
        
    #     # Layer 5
    #     conv1 = self.layer51(res)
    #     actv1 = F.relu((conv1))
    #     conv2 = self.layer52(actv1)
    #     bn = (conv2)
    #     res = torch.add(res, bn)
    #     res = self.attention5(res)
        
    #     # Layer 6
    #     conv1 = self.layer61(res)
    #     actv1 = F.relu((conv1))
    #     conv2 = self.layer62(actv1)
    #     bn = (conv2)
    #     res = torch.add(res, bn)
    #     res = self.attention6(res)
        
    #     # Layer 7
    #     conv1 = self.layer71(res)
    #     actv1 = F.relu((conv1))
    #     conv2 = self.layer72(actv1)
    #     bn = (conv2)
    #     res = torch.add(res, bn)
    #     res = self.attention7(res)
        
    #     # Layer 8
    #     convs = self.layer81(res)
    #     convs = (convs)
    #     conv1 = self.layer82(res)
    #     actv1 = F.relu((conv1))
    #     conv2 = self.layer83(actv1)
    #     bn = (conv2)
    #     pool = self.pool1(bn)
    #     res = torch.add(convs, pool)
        
    #     # Layer 9
    #     convs = self.layer91(res)
    #     convs = (convs)
    #     conv1 = self.layer92(res)
    #     actv1 = F.relu((conv1))
    #     conv2 = self.layer93(actv1)
    #     bn = (conv2)
    #     pool = self.pool2(bn)
    #     res = torch.add(convs, pool)
        
    #     # Layer 10
    #     convs = self.layer101(res)
    #     convs = (convs)
    #     conv1 = self.layer102(res)
    #     actv1 = F.relu((conv1))
    #     conv2 = self.layer103(actv1)
    #     bn = (conv2)
    #     pool = self.pool1(bn)
    #     res = torch.add(convs, pool)
        
    #     # Layer 11
    #     convs = self.layer111(res)
    #     convs = (convs)
    #     conv1 = self.layer112(res)
    #     actv1 = F.relu((conv1))
    #     conv2 = self.layer113(actv1)
    #     bn = (conv2)
    #     pool = self.pool1(bn)
    #     res = torch.add(convs, pool)
        
    #     # Layer 12
    #     conv1 = self.layer121(res)
    #     actv1 = F.relu((conv1))
    #     conv2 = self.layer122(actv1)
    #     bn = (conv2)
        
    #     #最后一层特征图反卷积，使其大小与倒数第二层特征图一致(原图1/16)
    #     x2_1 = F.relu(self.deconv2(bn))
    #     #倒数第二层特征图反卷积，使其大小与倒数第三层特征图一致(原图1/8)
    #     x1_0 = F.relu(self.deconv1(x2_1))
        
    #     #倒数第三层特征图连续3次反卷积，使其大小与原始图像一致(由原图1/8到1/1)
    #     x0_image = (F.relu(self.deconv0_1(x1_0)))
    #     x0_image = (F.relu(self.deconv0_2(x0_image)))
    #     x0_image = (F.relu(self.deconv0_3(x0_image)))
    #     outputs = self.classifier(x0_image)
    #     return outputs
