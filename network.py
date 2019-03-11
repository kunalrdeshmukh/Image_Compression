import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x


class EncoderNet(nn.Module):
    """
    Code for encoder module
    """
    def __init__(self,info):
        super(EncoderNet, self).__init__()
        
        self.channels = info['channels']

        self.conv1 = nn.Conv2d(self.channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(64, affine=False)
        self.conv3 = nn.Conv2d(64,self.channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

        self._initialize_weights()

    def forward(self,x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.bn1(out)
        return self.conv3(out)


    def _initialize_weights(self):
        init.kaiming_normal_(self.conv1.weight,mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.conv2.weight,mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.conv3.weight,mode='fan_out', nonlinearity='relu')
        init.constant_(self.bn1.weight, 1)
        init.constant_(self.bn1.bias, 0)


class DecoderNet(nn.Module):
    """
    Code for decoder module
    """
    def __init__(self,info):
        super(DecoderNet, self).__init__()
        self.channels = info['channels']
        self.size = info['size']

        self.interpolate = Interpolate(size=self.size, mode='bilinear')
        self.deconv1 = nn.Conv2d(self.channels, 64, 3, stride=1, padding=1)
        self.deconv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64, affine=False)
        self.relu = nn.ReLU()

        self.deconv_n = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn_n = nn.BatchNorm2d(64, affine=False)

        self.deconv3 = nn.ConvTranspose2d(64,self.channels, 3, stride=1, padding=1)
    
        self._initialize_weights()

    def forward(self, z):
        upscaled_image = self.interpolate(z)
        out = self.relu(self.deconv1(upscaled_image))
        out = self.relu(self.deconv2(out))
        out = self.bn2(out)
        for _ in range(5):
            out = self.relu(self.deconv_n(out))
            out = self.bn_n(out)
        residual_img = self.deconv3(out)
        final = residual_img #+ upscaled_image 
        return final,residual_img,upscaled_image

    def _initialize_weights(self):
        init.kaiming_normal_(self.deconv1.weight,mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.deconv2.weight,mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.deconv_n.weight,mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.deconv3.weight,mode='fan_out', nonlinearity='relu')
        init.constant_(self.bn2.weight, 1)
        init.constant_(self.bn2.bias, 0)
        init.constant_(self.bn_n.weight, 1)
        init.constant_(self.bn_n.bias, 0)