from __future__ import print_function
import argparse
from torch import from_numpy, load
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from torch.cuda import is_available 
from network import DecoderNet


import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='Image compression')
parser.add_argument('--input_image', type=str, required=True, help='input image to use')
parser.add_argument('--model', type=str, required=True, help='model file to use')
parser.add_argument('--output_filename', type=str, help='where to save the output image')
parser.add_argument('--model_epoch', type=int, default=-1, help='Epoch for which model needs to be loaded. Default=200.')
parser.add_argument('--image_size', type=int, default=200, help='path to images. Default=200')

opt = parser.parse_args()

print(opt)
img = Image.open(opt.input_image).convert('RGB')

if (opt.model_epoch == -1) :
    model = load(opt.model)
else :
    info['channels'] = opt.channels
    info['size'] = opt.image_size
    model = DecoderNet(info)
    model = model.load_state_dict(load(opt.model))

print("Model Loaded.")
img_to_tensor = ToTensor()
input = img_to_tensor(img)

if is_available():
    model = model.cuda()
    input = input.cuda()

input = input.unsqueeze(0)
final,out,upscaled_image = model(input)
out = out.unsqueeze_(0)

final = final.cpu()
out_img = final[0].detach().numpy()
# out_img *= 255.0
# out_img = out_img.clip(0, 255)

out_img = from_numpy(out_img)
# save_image(out_img,opt.output_filename)
save_image(final[0],opt.output_filename)

print('output image saved to ', opt.output_filename)