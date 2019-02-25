from __future__ import print_function
import argparse
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.utils import save_image

import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='Image compression')
parser.add_argument('--input_image', type=str, required=True, help='input image to use')
parser.add_argument('--model', type=str, required=True, help='model file to use')
parser.add_argument('--output_filename', type=str, help='where to save the output image')
opt = parser.parse_args()

print(opt)
img = Image.open(opt.input_image).convert('RGB')

model = torch.load(opt.model)
img_to_tensor = ToTensor()
input = img_to_tensor(img)

if torch.cuda.is_available():
    model = model.cuda()
    input = input.cuda()

out = model(input)
out = out.cpu()
out_img = out[0].detach().numpy()
out_img *= 255.0
out_img = out_img.clip(0, 255)

save_image(out_img,'reconstructed')

print('output image saved to ', opt.output_filename)