import torch.nn as nn
from . import register_model
from .base_model import BaseModel
import torch
import torch.nn.functional as F

@register_model('SubModelDm')
class SubModelDm(BaseModel):
    """
    SubModelDm for demosaicing network
    
    Args:
    """
    
    def __init__(self, config):
        super().__init__(config)

        dm_in_nc = config.get('dm_in_nc', 12)
        dm_nf = config.get('dm_nf', 64)
        dm_out_nc = config.get('dm_out_nc', 12)

        self.dm_conv1_1 = nn.Conv2d(dm_in_nc, dm_nf, kernel_size=3, stride=1, padding=1)
        self.dm_conv1_2 = nn.Conv2d(dm_nf, dm_nf, kernel_size=3, stride=1, padding=1)
        self.dm_pool1 = nn.MaxPool2d(kernel_size=2)

        self.dm_conv2_1 = nn.Conv2d(dm_nf, dm_nf * 2, kernel_size=3, stride=1, padding=1)
        self.dm_conv2_2 = nn.Conv2d(dm_nf * 2, dm_nf * 2, kernel_size=3, stride=1, padding=1)
        self.dm_pool2 = nn.MaxPool2d(kernel_size=2)

        self.dm_conv3_1 = nn.Conv2d(dm_nf * 2, dm_nf * 4, kernel_size=3, stride=1, padding=1)
        self.dm_conv3_2 = nn.Conv2d(dm_nf * 4, dm_nf * 4, kernel_size=3, stride=1, padding=1)
        self.dm_pool3 = nn.MaxPool2d(kernel_size=2)

        self.dm_upv4 = nn.ConvTranspose2d(dm_nf * 4, dm_nf * 2, 2, stride=2)
        self.dm_conv4_1 = nn.Conv2d(dm_nf * 4, dm_nf * 2, kernel_size=3, stride=1, padding=1)
        self.dm_conv4_2 = nn.Conv2d(dm_nf * 2, dm_nf * 2, kernel_size=3, stride=1, padding=1)

        self.dm_upv5 = nn.ConvTranspose2d(dm_nf * 2, dm_nf, 2, stride=2)
        self.dm_conv5_1 = nn.Conv2d(dm_nf * 2, dm_nf, kernel_size=3, stride=1, padding=1)

        self.dm_conv6_1 = nn.Conv2d(dm_in_nc+dm_nf, (dm_in_nc+dm_nf)*2, kernel_size=3, stride=1, padding=1)
        self.dm_conv6_2 = nn.Conv2d((dm_in_nc+dm_nf)*2, (dm_in_nc+dm_nf)*2, kernel_size=3, stride=1, padding=1)

        self.dm_conv7_1 = nn.Conv2d((dm_in_nc+dm_nf)*2, dm_nf, kernel_size=3, stride=1, padding=1)

        self.dm_conv8_1 = nn.Conv2d(dm_nf, dm_out_nc, kernel_size=1, stride=1, padding=0)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        # Green Kernel - 注册为buffer确保设备兼容性
        kernel_g = torch.tensor([[
            [0.0, 0.25, 0.0],
            [0.25, 1.0, 0.25],
            [0.0, 0.25, 0.0]
        ]]).unsqueeze(0)
        kernel_rb = torch.tensor([[
            [0.25, 0.5, 0.25],
            [0.5, 1.0, 0.5],
            [0.25, 0.5, 0.25]
        ]]).unsqueeze(0)
        
        self.register_buffer('kernel_g', kernel_g)
        self.register_buffer('kernel_rb', kernel_rb)
    
    def beyer2rgb(self, bayer):
        batch_size, _, H, W = bayer.shape
        rgb = torch.zeros((batch_size, 3, H, W), dtype=bayer.dtype, device=bayer.device)

        rgb[::, 0, 0::2, 0::2] = bayer[::, 0, 0::2, 0::2]

        rgb[::, 1, 0::2, 1::2] = bayer[::, 0, 0::2, 1::2]
        rgb[::, 1, 1::2, 0::2] = bayer[::, 0, 1::2, 0::2]

        rgb[::, 2, 1::2, 1::2] = bayer[::, 0, 1::2, 1::2]
        
        # 使用注册的buffer进行卷积操作
        rgb[:, 0:1, :, :] = F.conv2d(rgb[:, 0:1, :, :], self.kernel_rb, padding=1)
        rgb[:, 1:2, :, :] = F.conv2d(rgb[:, 1:2, :, :], self.kernel_g, padding=1)
        rgb[:, 2:3, :, :] = F.conv2d(rgb[:, 2:3, :, :], self.kernel_rb, padding=1)
        return rgb
    
    def polar_demosaic(self, mpfa_raw):
        batch_size, _, H, W = mpfa_raw.shape
        mpfa_img = torch.zeros((batch_size, 4, H, W), dtype=mpfa_raw.dtype, device=mpfa_raw.device)

        mpfa_img[::, 0, 0::2, 0::2] = mpfa_raw[::, 0, 0::2, 0::2]   # 90
        mpfa_img[::, 1, 0::2, 1::2] = mpfa_raw[::, 0, 0::2, 1::2]   # 45
        mpfa_img[::, 2, 1::2, 0::2] = mpfa_raw[::, 0, 1::2, 0::2]   # 135
        mpfa_img[::, 3, 1::2, 1::2] = mpfa_raw[::, 0, 1::2, 1::2]   # 0

        # 使用注册的buffer进行卷积操作
        mpfa_img[:, 0:1, :, :] = F.conv2d(mpfa_img[:, 0:1, :, :], self.kernel_rb, padding=1)
        mpfa_img[:, 1:2, :, :] = F.conv2d(mpfa_img[:, 1:2, :, :], self.kernel_rb, padding=1)
        mpfa_img[:, 2:3, :, :] = F.conv2d(mpfa_img[:, 2:3, :, :], self.kernel_rb, padding=1)
        mpfa_img[:, 3:4, :, :] = F.conv2d(mpfa_img[:, 3:4, :, :], self.kernel_rb, padding=1)

        return mpfa_img
        
        

    def forward(self, mid_result_16):

        batch_size, _, H, W = mid_result_16.shape
        device = mid_result_16.device
        dtype = mid_result_16.dtype
        
        bayer_90 = torch.zeros((batch_size, 1, H*2, W*2), dtype=dtype, device=device)
        bayer_45 = torch.zeros((batch_size, 1, H*2, W*2), dtype=dtype, device=device)
        bayer_0 = torch.zeros((batch_size, 1, H*2, W*2), dtype=dtype, device=device)
        bayer_135 = torch.zeros((batch_size, 1, H*2, W*2), dtype=dtype, device=device)

        bayer_90[::, 0, 0::2, 0::2] = mid_result_16[::, 0, ::, ::]
        bayer_90[::, 0, 0::2, 1::2] = mid_result_16[::, 2, ::, ::]
        bayer_90[::, 0, 1::2, 0::2] = mid_result_16[::, 8, ::, ::]
        bayer_90[::, 0, 1::2, 1::2] = mid_result_16[::, 10, ::, ::]

        bayer_45[::, 0, 0::2, 0::2] = mid_result_16[::, 1, ::, ::]
        bayer_45[::, 0, 0::2, 1::2] = mid_result_16[::, 3, ::, ::]
        bayer_45[::, 0, 1::2, 0::2] = mid_result_16[::, 9, ::, ::]
        bayer_45[::, 0, 1::2, 1::2] = mid_result_16[::, 11, ::, ::]

        bayer_0[::, 0, 0::2, 0::2] = mid_result_16[::, 5, ::, ::]
        bayer_0[::, 0, 0::2, 1::2] = mid_result_16[::, 7, ::, ::]
        bayer_0[::, 0, 1::2, 0::2] = mid_result_16[::, 13, ::, ::]
        bayer_0[::, 0, 1::2, 1::2] = mid_result_16[::, 15, ::, ::]

        bayer_135[::, 0, 0::2, 0::2] = mid_result_16[::, 4, ::, ::]
        bayer_135[::, 0, 0::2, 1::2] = mid_result_16[::, 6, ::, ::]
        bayer_135[::, 0, 1::2, 0::2] = mid_result_16[::, 12, ::, ::]
        bayer_135[::, 0, 1::2, 1::2] = mid_result_16[::, 14, ::, ::]

        rgb90 = self.beyer2rgb(bayer_90)
        rgb45 = self.beyer2rgb(bayer_45)
        rgb0 = self.beyer2rgb(bayer_0)
        rgb135 = self.beyer2rgb(bayer_135)

        polar_raw = torch.zeros((batch_size, 3, H*4, W*4), dtype=dtype, device=device)

        polar_raw[::, 0, 0::2, 0::2] = rgb90[::, 0, ::, ::]
        polar_raw[::, 0, 0::2, 1::2] = rgb45[::, 0, ::, ::]
        polar_raw[::, 0, 1::2, 0::2] = rgb135[::, 0, ::, ::]
        polar_raw[::, 0, 1::2, 1::2] = rgb0[::, 0, ::, ::]

        polar_raw[::, 1, 0::2, 0::2] = rgb90[::, 1, ::, ::]
        polar_raw[::, 1, 0::2, 1::2] = rgb45[::, 1, ::, ::]
        polar_raw[::, 1, 1::2, 0::2] = rgb135[::, 1, ::, ::]
        polar_raw[::, 1, 1::2, 1::2] = rgb0[::, 1, ::, ::]

        polar_raw[::, 2, 0::2, 0::2] = rgb90[::, 2, ::, ::]
        polar_raw[::, 2, 0::2, 1::2] = rgb45[::, 2, ::, ::]
        polar_raw[::, 2, 1::2, 0::2] = rgb135[::, 2, ::, ::]
        polar_raw[::, 2, 1::2, 1::2] = rgb0[::, 2, ::, ::]

        mpfa_img_r = self.polar_demosaic(polar_raw[::, 0:1, ::, ::])
        mpfa_img_g = self.polar_demosaic(polar_raw[::, 1:2, ::, ::])
        mpfa_img_b = self.polar_demosaic(polar_raw[::, 2:3, ::, ::])

        polar_rgb_90 = torch.zeros((batch_size, 3, H*4, W*4), dtype=dtype, device=device)
        polar_rgb_45 = torch.zeros((batch_size, 3, H*4, W*4), dtype=dtype, device=device)
        polar_rgb_0 = torch.zeros((batch_size, 3, H*4, W*4), dtype=dtype, device=device)
        polar_rgb_135 = torch.zeros((batch_size, 3, H*4, W*4), dtype=dtype, device=device)

        polar_rgb_90[::, 0, ::, ::] = mpfa_img_r[::, 0, ::, ::]
        polar_rgb_90[::, 1, ::, ::] = mpfa_img_g[::, 0, ::, ::]
        polar_rgb_90[::, 2, ::, ::] = mpfa_img_b[::, 0, ::, ::]

        polar_rgb_45[::, 0, ::, ::] = mpfa_img_r[::, 1, ::, ::]
        polar_rgb_45[::, 1, ::, ::] = mpfa_img_g[::, 1, ::, ::]
        polar_rgb_45[::, 2, ::, ::] = mpfa_img_b[::, 1, ::, ::]

        polar_rgb_135[::, 0, ::, ::] = mpfa_img_r[::, 2, ::, ::]
        polar_rgb_135[::, 1, ::, ::] = mpfa_img_g[::, 2, ::, ::]
        polar_rgb_135[::, 2, ::, ::] = mpfa_img_b[::, 2, ::, ::]

        polar_rgb_0[::, 0, ::, ::] = mpfa_img_r[::, 3, ::, ::]
        polar_rgb_0[::, 1, ::, ::] = mpfa_img_g[::, 3, ::, ::]
        polar_rgb_0[::, 2, ::, ::] = mpfa_img_b[::, 3, ::, ::]

        polar_rgb = torch.cat([polar_rgb_90, polar_rgb_45, polar_rgb_135, polar_rgb_0], 1)

        print("interpolated complete")

        # 使用polar_rgb作为输入进行3层Unet处理
        dm_conv1 = self.relu(self.dm_conv1_1(polar_rgb))
        dm_conv1 = self.relu(self.dm_conv1_2(dm_conv1))
        dm_pool1 = self.dm_pool1(dm_conv1)

        dm_conv2 = self.relu(self.dm_conv2_1(dm_pool1))
        dm_conv2 = self.relu(self.dm_conv2_2(dm_conv2))
        dm_pool2 = self.dm_pool2(dm_conv2)

        dm_conv3 = self.relu(self.dm_conv3_1(dm_pool2))
        dm_conv3 = self.relu(self.dm_conv3_2(dm_conv3))

        dm_up4 = self.dm_upv4(dm_conv3)
        dm_up4 = torch.cat([dm_up4, dm_conv2], 1)
        dm_conv4 = self.relu(self.dm_conv4_1(dm_up4))
        dm_conv4 = self.relu(self.dm_conv4_2(dm_conv4))

        dm_up5 = self.dm_upv5(dm_conv4)
        dm_up5 = torch.cat([dm_up5, dm_conv1], 1)
        dm_conv5 = self.relu(self.dm_conv5_1(dm_up5))

        dm_conv6_input = torch.cat([polar_rgb, dm_conv5], 1)
        dm_conv6 = self.relu(self.dm_conv6_1(dm_conv6_input))
        dm_conv6 = self.relu(self.dm_conv6_2(dm_conv6))

        dm_conv7 = self.relu(self.dm_conv7_1(dm_conv6))

        out = self.dm_conv8_1(dm_conv7)
        
        # 添加全局残差连接
        out = out + polar_rgb

        print("forward out shape: ", out.shape)
        
        return out

    def get_model_info(self):
        pass
