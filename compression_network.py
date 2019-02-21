import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import ConvLSTMCell, Sign


class Compression(nn.Module):
    """
    Code for compression module
    """
    def __init__(self,info):
        super(Compression, self).__init__()
        channels = info[0]
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(64, affine=False)
        self.conv3 = nn.Conv2d(64, channels, kernel_size=3, stride=1, padding=1)

    def forward(self,x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.bn1(out)
        return self.conv3(out)


# class Binizer(nn.Module):
#     def __init__(self):
#         super(Binizer, self).__init__()


class Reconstruction(nn.Module):
    """
    Code for reconstruction module
    """
    def __init__(self,info):
        super(Reconstruction, self).__init__()
        channels = info[0]
        self.interpolate = Interpolate(size=HEIGHT, mode='bilinear')
        self.deconv1 = nn.Conv2d(channels, 64, 3, stride=1, padding=1)
        self.deconv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64, affine=False)
        self.relu = nn.ReLU()

        self.deconv_n = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn_n = nn.BatchNorm2d(64, affine=False)

        self.deconv3 = nn.ConvTranspose2d(64, channels, 3, stride=1, padding=1)
    
    def forward(self, z):
        upscaled_image = self.interpolate(z)
        out = self.relu(self.deconv1(upscaled_image))
        out = self.relu(self.deconv2(out))
        out = self.bn2(out)
        for _ in range(5):
            out = self.relu(self.deconv_n(out))
            out = self.bn_n(out)
        out = self.deconv3(out)
        final = upscaled_image + out
        return final,out,upscaled_image