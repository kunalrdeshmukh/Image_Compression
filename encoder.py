from __future__ import print_function
import argparse
from torch import load, from_numpy
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
parser.add_argument('--model_epoch', type=int, default=-1, help='Epoch for which model needs to be loaded. Default=200.')
opt = parser.parse_args()

print(opt)
img = Image.open(opt.input_image).convert('RGB')
img = img.resize((opt.imageSize, opt.imageSize), Image.ANTIALIAS)

if (opt.model_epoch == -1) :
    model = load(opt.model)
else :
    pass
    # TODO : load model for specific epoch
    
img_to_tensor = ToTensor()
input = img_to_tensor(img)

if is_available():
    input = input.cuda()
    model = model.cuda()

input = input.unsqueeze(0)
out = model(input)
out = out.unsqueeze_(0)

out = out.cpu()
out_img = out[0].detach().numpy()
# out_img *= 255.0
# out_img = out_img.clip(0, 255)

out_img = from_numpy(out_img)
save_image(out_img,opt.output_filename)

print('output image saved to ', opt.output_filename)