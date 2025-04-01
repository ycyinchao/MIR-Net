import os

import torch.nn.functional as F
from matplotlib import pyplot as plt

from RCAB import *
from backbones.pvtv2 import pvt_v2_b4
# Channel Reduce
class Reduction(nn.Module):
    def __init__(self, in_channel, out_channel, RFB=False):
        super(Reduction, self).__init__()
        # self.dyConv = Dynamic_conv2d(in_channel,out_channel,3,padding = 1)
        if (RFB):
            self.reduce = nn.Sequential(
                RFB_modified(in_channel, out_channel),
            )
        else:
            self.reduce = nn.Sequential(
                BasicConv2d(in_channel, out_channel, 1),
            )

    def forward(self, x):
        return self.reduce(x)


#
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

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

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class SGR(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SGR, self).__init__()
        self.ca = ChannelAttention(channel, reduction)
        self.sa = SpatialAttention()
        self.cb_high = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.cv_low = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.cb_1 = nn.Conv2d(2*channel,channel, 1)


    def forward(self, x_low, x_high):
        x_high = self.cb_high(x_high)
        x_low = F.interpolate(x_low, size=x_high.size()[2:], mode='bilinear')
        x_low = self.cv_low(x_low)
        x = torch.cat((x_high, x_low), dim=1)
        # Spatial Attention
        x_sa = self.sa(x)
        # Channle Attention
        x_ca = self.cb_1(x)
        x_ca = self.ca(x_ca)

        x = x_ca * x_sa * x_high
        x = x_high +x


        return x

# SFC module：Scale-wise Feature Capturing Module
class SFC(nn.Module):
    def __init__(self, in_channel=64, out_channel=64):
        super(SFC, self).__init__()
        # 1x1 卷积层
        self.conv1x1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0)

        # 3x3 卷积层
        self.conv3x3 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)

        # 5x5 卷积层
        self.conv5x5 = nn.Conv2d(in_channel, out_channel, kernel_size=5, padding=2)

        # 空洞卷积，膨胀率分别为2和4
        self.atrous_conv3x3_2 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=2, dilation=2)
        self.atrous_conv3x3_4 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=4, dilation=4)

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_channel, out_channel, kernel_size=1)  # 用于全局池化后的线性变换

        # 最终融合后的卷积层，用于减少通道数
        self.conv_fusion = nn.Conv2d(out_channel * 6, out_channel, kernel_size=1)

    def forward(self, x):
        # 多尺度卷积
        conv1x1_out = self.conv1x1(x)
        conv3x3_out = self.conv3x3(x)
        conv5x5_out = self.conv5x5(x)

        # 空洞卷积
        atrous_conv2_out = self.atrous_conv3x3_2(x)
        atrous_conv4_out = self.atrous_conv3x3_4(x)

        # 全局池化
        global_pool_out = self.global_pool(x)
        global_pool_out = self.fc(global_pool_out)  # 线性变换恢复维度
        global_pool_out = torch.nn.functional.interpolate(global_pool_out, size=x.shape[2:], mode='bilinear',
                                                          align_corners=False)

        # 将不同尺度的特征拼接在一起
        out = torch.cat([conv1x1_out, conv3x3_out, conv5x5_out, atrous_conv2_out, atrous_conv4_out, global_pool_out],
                        dim=1)

        # 通过融合卷积层将通道数恢复
        out = self.conv_fusion(out)

        return out

def vis(fg_enhanced, fg_pool,save_dir='./images'):
    # 创建保存目录 ./images/，如果不存在则创建
    save_dir = save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 创建子图并逐个绘制通道的热力图
    for i in range(64):
        # 为每个通道创建一个新图
        plt.figure()
        plt.imshow(fg_enhanced[0][i].cpu().detach().numpy(), cmap='viridis')
        plt.title(f'Channel {i + 1}:{fg_pool[0][i][0][0].cpu().detach().numpy()}')  # 设置标题
        plt.axis('off')  # 隐藏坐标轴

        # 保存当前通道的图像到 ./images 目录
        save_path = os.path.join(save_dir, f'channel_{i + 1}.png')
        plt.savefig(save_path)
        plt.close()  # 保存完图像后关闭，释放内存

# BFRE Subnetwork：Background-assisted Foreground Region Enhancement
class BFRE(nn.Module):
    def __init__(self, channel=64,isBFRE=True):
        super(BFRE, self).__init__()
        self.isBFRE = isBFRE

        self.cb_1 = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.cb_2 = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.cb_3 = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.cb_4 = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1)

        self.sfc = SFC(in_channel=channel, out_channel=channel)
        self.cbfuse = nn.Sequential(
            BasicConv2d(2*channel, channel, kernel_size=3, stride=1, padding=1),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, fg_high, fg_low, bg_high,predB_low,isVis=False):#up表示高分辨率特征图，down表示低分辨率特征图
        predB_low = F.interpolate(predB_low, size=fg_high.size()[2:], mode='bilinear')
        predB_low = torch.sigmoid(predB_low)
        fg_high_1 = self.cb_1(fg_high)
        fg_low = F.interpolate(fg_low, size=fg_high.size()[2:], mode='bilinear')
        fg_low_1 = self.cb_2(fg_low)

        fg_high_enhanced = fg_high-bg_high.detach()
        fg_low_enhanced = fg_low-bg_high.detach()
        fg_high_pool = self.avg_pool(fg_high_enhanced)
        fg_low_pool = self.avg_pool(fg_low_enhanced)
        # 使用掩码来选择小于0的通道，避免手动遍历每个通道
        high_mask = fg_high_pool < 0  # [b, c,1,1]
        low_mask = fg_low_pool < 0  # [b, c,1,1]
        if isVis:
            vis(fg_high_enhanced, fg_high_pool,save_dir='./images/fg_high/')
            vis(fg_low_enhanced, fg_low_pool,save_dir='./images/fg_low/')

        # 广播掩码，并逐元素相乘
        masked_fg_high_enhanced = fg_high_enhanced * high_mask * torch.exp(-fg_high_pool)
        masked_fg_low_enhanced = fg_low_enhanced * low_mask * torch.exp(-fg_low_pool)

        # 对于满足条件的通道直接进行加和，不再手动逐通道遍历
        result_high_list = masked_fg_high_enhanced.sum(dim=1, keepdim=True)  # 直接在通道维度进行加和
        result_high_list = torch.sigmoid(result_high_list)
        result_low_list = masked_fg_low_enhanced.sum(dim=1, keepdim=True)  # 直接在通道维度进行加和
        result_low_list = torch.sigmoid(result_low_list)

        if not self.isBFRE:
            result_high_list = 1-result_high_list
            result_low_list = 1-result_low_list

        fg_high_2 = self.cb_3(result_high_list * fg_high_1)
        fg_low_2 = self.cb_4(result_low_list * fg_low_1)
        fg_cat = torch.cat((fg_high_2*(1-predB_low), fg_low_2*(1-predB_low)), 1)
        fuse = self.cbfuse(fg_cat)
        out = self.sfc(fuse)

        return out

################################################ Net ###############################################
class Net(nn.Module):
    def __init__(self, cfg, channel=64):
        super(Net, self).__init__()
        self.cfg = cfg
        #  ---- VGG16 Backbone ----
        # self.backbone = eval(vgg16)(pretrained = True)
        # enc_channels=[64, 128, 256, 512, 512]
        #
        #  ---- ConvNext Backbone ----
        # self.backbone = convnext_tiny(pretrained=True)
        # enc_channels=[96, 192, 384,768]
        #
        #   ---- Res2Net50 Backbone ----
        #     self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        #     enc_channels=[256, 512, 1024,2048]
        #
        #   ---- ResNet50 Backbone ----
        # self.resnet = resnet50(pretrained=True)
        # enc_channels=[256, 512, 1024,2048]
        #  ---- PVTv2_B4 Backbone ----

        self.bkbone = pvt_v2_b4()  # [64, 128, 320, 512]
        # 获取预训练的参数
        save_model = torch.load('../weights/pvt_v2_b4.pth')
        # 获取当前模型的参数
        model_dict = self.bkbone.state_dict()
        # 加载部分能用的参数
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        # 更新现有的model_dict
        model_dict.update(state_dict)
        # 加载真正需要的state_dict
        self.bkbone.load_state_dict(model_dict)
        enc_channels = [64, 128, 320, 512]

        self.reduce_1 = Reduction(enc_channels[0], channel, RFB=False)
        self.reduce_2 = Reduction(enc_channels[1], channel, RFB=False)
        self.reduce_3 = Reduction(enc_channels[2], channel, RFB=False)
        self.reduce_4 = Reduction(enc_channels[3], channel, RFB=False)

        self.bfre1 = BFRE(channel)
        self.bfre2 = BFRE(channel)
        self.bfre3 = BFRE(channel)
        self.fbre1 = BFRE(channel=channel,isBFRE=False)
        self.fbre2 = BFRE(channel=channel,isBFRE=False)
        self.fbre3 = BFRE(channel=channel,isBFRE=False)

        self.pre_fg = nn.ModuleList([
            nn.Conv2d(channel,1,1) for _ in range(4)
        ])
        self.pre_bg = nn.ModuleList([
            nn.Conv2d(channel,1,1) for _ in range(4)
        ])

    def forward(self, x,shape = None):
        shape = x.size()[2:] if shape is None else shape
        # Feature Extraction
        #   ---- Res2Net Backbone ----
        #     x = self.resnet.conv1(x)
        #     x = self.resnet.bn1(x)
        #     x = self.resnet.relu(x)
        #     x = self.resnet.maxpool(x)      # bs, 64, 88, 88
        #
        #     x1 = self.resnet.layer1(x)      # bs, 256, 88, 88
        #     x2 = self.resnet.layer2(x1)     # bs, 512, 44, 44
        #     x3 = self.resnet.layer3(x2)     # bs, 1024, 22, 22
        #     x4 = self.resnet.layer4(x3)     # bs, 2048, 11, 11
        #
        #  ---- ConvNext Backbone ----
        # datalist = self.backbone(x)
        # x1, x2, x3, x4 = datalist[0], datalist[1], datalist[2], datalist[3]    # bs, 96,48,24,12

        #  ---- PVTv2_B4 Backbone ----
        x1, x2, x3, x4 = self.bkbone(x)

        # Channel Reduce
        x1_fg = self.reduce_1(x1)
        x2_fg = self.reduce_2(x2)
        x3_fg = self.reduce_3(x3)
        x4_fg = self.reduce_4(x4)

        # x1_bg = self.reduce_5(-x1.clone())
        # x2_bg = self.reduce_6(-x2.clone())
        # x3_bg = self.reduce_7(-x3.clone())
        # x4_bg = self.reduce_8(-x4.clone())
        x1_bg = x1_fg.clone()
        x2_bg = x2_fg.clone()
        x3_bg = x3_fg.clone()
        x4_bg = x4_fg.clone()

        # stage 1
        # SGR in fg
        # sgr_fg = self.sgr1(x4_fg,x1_fg)

        # SGR in bg
        # sgr_bg = self.sgr2(x4_bg,x1_bg)


        pred_bg1 = self.pre_bg[0](x4_bg)
        # FBRE in bg: fg指导bg
        pred_fg1 = self.pre_fg[0](x4_fg)
        bg_1 = self.fbre1(x3_bg, x4_bg,F.interpolate(x4_fg,size=x3_bg.size()[2:],mode='bilinear'),pred_fg1)  # 24*24
        # BFRE in fg
        pred_bg2 = self.pre_bg[1](bg_1)
        fg_1 = self.bfre1(x3_fg, x4_fg, bg_1,pred_bg2)  # B*C*24*24

        # stage 2
        # FBRE in bg: fg指导bg
        pred_fg2 = self.pre_fg[1](fg_1)
        bg_2 = self.fbre2(x2_bg, bg_1, F.interpolate(fg_1, size=x2_bg.size()[2:], mode='bilinear'),pred_fg2)  # 48*48
        # BFRE in fg
        pred_bg3 = self.pre_bg[2](bg_2)
        fg_2 = self.bfre2(x2_fg, fg_1, bg_2,pred_bg3)  # B*C*48*48

        # stage 3
        # FBRE in bg: fg指导bg
        pred_fg3 = self.pre_fg[2](fg_2)
        bg_3 = self.fbre3(x1_bg, bg_2, F.interpolate(fg_2, size=x1_bg.size()[2:], mode='bilinear'),pred_fg3)  # 96*96
        # BFRE in fg
        pred_bg4 = self.pre_bg[3](bg_3)
        fg_3 = self.bfre3(x1_fg, fg_2, bg_3,pred_bg4)  # B*C*96*96




        pred_fg4 = self.pre_fg[3](fg_3)


        pred_fg1 = F.interpolate(pred_fg1, size=shape, mode='bilinear', align_corners=False)
        pred_fg2 = F.interpolate(pred_fg2, size=shape, mode='bilinear', align_corners=False)
        pred_fg3 = F.interpolate(pred_fg3, size=shape, mode='bilinear', align_corners=False)
        pred_fg4 = F.interpolate(pred_fg4, size=shape, mode='bilinear', align_corners=False)

        pred_bg1 = F.interpolate(pred_bg1, size=shape, mode='bilinear', align_corners=False)
        pred_bg2 = F.interpolate(pred_bg2, size=shape, mode='bilinear', align_corners=False)
        pred_bg3 = F.interpolate(pred_bg3, size=shape, mode='bilinear', align_corners=False)
        pred_bg4 = F.interpolate(pred_bg4, size=shape, mode='bilinear', align_corners=False)

        if self.cfg.mode == 'train':
            # pred3 = pred_fg3 - pred_bg3
            # pred2 = pred_fg2 - pred_bg2
            # pred1 = pred_fg1 - pred_bg1
            # pred0 = pred_fg0 - pred_bg0
            # return pred3,None, pred2, pred1, pred0
            pred_fg = [pred_fg4, pred_fg3, pred_fg2, pred_fg1]
            pred_bg = [pred_bg4, pred_bg3, pred_bg2, pred_bg1]
            return pred_fg,pred_bg
        else:
            return pred_fg4 - pred_bg4,None

