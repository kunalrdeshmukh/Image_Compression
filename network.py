import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.utils import _pair


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

        # self._initialize_weights()

    def forward(self,x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.bn1(out)
        return self.conv3(out)


    def _initialize_weights(self):
        init.kaiming_normal_(self.conv1.weight,mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.conv2.weight,mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.conv3.weight,mode='fan_out', nonlinearity='relu')
        # init.constant_(self.bn1.weight, 1)
        # init.constant_(self.bn1.bias, 0)


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
    
        # self._initialize_weights()

    def forward(self, z):
        upscaled_image = self.interpolate(z)
        out = self.relu(self.deconv1(upscaled_image))
        out = self.relu(self.deconv2(out))
        out = self.bn2(out)
        strt = out
        for _ in range(5):
            out = self.relu(self.deconv_n(out))
            out = self.bn_n(out)
        out += strt
        interm = out
        for _ in range(5):
            out = self.relu(self.deconv_n(out))
            out = self.bn_n(out)
        out += interm
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

# class ConvLSTMCell(ConvRNNCellBase):
#     def __init__(self,
#                  input_channels,
#                  hidden_channels,
#                  kernel_size=3,
#                  stride=1,
#                  padding=0,
#                  dilation=1,
#                  hidden_kernel_size=1,
#                  bias=True):
#         super(ConvLSTMCell, self).__init__()
#         self.input_channels = input_channels
#         self.hidden_channels = hidden_channels

#         self.kernel_size = _pair(kernel_size)
#         self.stride = _pair(stride)
#         self.padding = _pair(padding)
#         self.dilation = _pair(dilation)

#         self.hidden_kernel_size = _pair(hidden_kernel_size)

#         hidden_padding = _pair(hidden_kernel_size // 2)

#         gate_channels = 4 * self.hidden_channels
#         self.conv_ih = nn.Conv2d(
#             in_channels=self.input_channels,
#             out_channels=gate_channels,
#             kernel_size=self.kernel_size,
#             stride=self.stride,
#             padding=self.padding,
#             dilation=self.dilation,
#             bias=bias)

#         self.conv_hh = nn.Conv2d(
#             in_channels=self.hidden_channels,
#             out_channels=gate_channels,
#             kernel_size=hidden_kernel_size,
#             stride=1,
#             padding=hidden_padding,
#             dilation=1,
#             bias=bias)

#         self.reset_parameters()

#     def reset_parameters(self):
#         self.conv_ih.reset_parameters()
#         self.conv_hh.reset_parameters()

#     def forward(self, input, hidden):
#         hx, cx = hidden
#         gates = self.conv_ih(input) + self.conv_hh(hx)

#         ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

#         ingate = F.sigmoid(ingate)
#         forgetgate = F.sigmoid(forgetgate)
#         cellgate = F.tanh(cellgate)
#         outgate = F.sigmoid(outgate)

#         cy = (forgetgate * cx) + (ingate * cellgate)
#         hy = outgate * F.tanh(cy)

#         return hy, cy

# class Sign(Function):
#     """
#     Variable Rate Image Compression with Recurrent Neural Networks
#     https://arxiv.org/abs/1511.06085
#     """

#     def __init__(self):
#         super(Sign, self).__init__()

#     @staticmethod
#     def forward(ctx, input, is_training=True):
#         # Apply quantization noise while only training
#         if is_training:
#             prob = input.new(input.size()).uniform_()
#             x = input.clone()
#             x[(1 - input) / 2 <= prob] = 1
#             x[(1 - input) / 2 > prob] = -1
#             return x
#         else:
#             return input.sign()

#     @staticmethod
#     def backward(ctx, grad_output):
#       return grad_output, None