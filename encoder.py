from __future__ import print_function
import argparse
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from torch.cuda import is_available 
from network import EncoderNet


import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='Image compression')
parser.add_argument('--input_image', type=str, required=True, help='input image to use')
parser.add_argument('--model', type=str, required=True, help='model file to use')
parser.add_argument('--output_filename', type=str, help='where to save the output image')
parser.add_argument('--channels', type=int, default=3, help='number of channels in an image. Default=3')
parser.add_argument('--imageSize', type=int, default=200, help='image width. Only square images are supported. Default=200.')
opt = parser.parse_args()

print(opt)
img = Image.open(opt.input_image).convert('RGB')
img = img.resize((opt.imageSize, opt.imageSize), Image.ANTIALIAS)

model = EncoderNet(opt.channels)
model = load_state_dict(opt.model)
img_to_tensor = ToTensor()
input = img_to_tensor(img)


out = model(input)
out = out.cpu()
out_img = out[0].detach().numpy()
out_img *= 255.0
out_img = out_img.clip(0, 255)

save_image(out_img,'compressed')

print('output image saved to ', opt.output_filename)