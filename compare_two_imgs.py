# import system libraries
import cv2
import math
import numpy as np
import sys

from os import listdir
from os.path import join

#import thirdparty libraries 
from PIL import Image
from skimage.measure import compare_ssim as ssim
from sklearn.metrics import mean_squared_error


# Import user-defined libraries
from dataset import is_image_file


def mse(imageA, imageB):
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	return err

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
    	return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def bits_per_pixel(image_):
	mode_to_bpp = {'1':1, 'L':8, 'P':8, 'RGB':24, 'RGBA':32, 'CMYK':32, 'YCbCr':24, 'I':32, 'F':32}
	im = Image.open(image_)
	mode = im.mode
	return mode_to_bpp.get(mode)


def compare_images(imageA, imageB):
	original = cv2.imread(imageA)
	contrast = cv2.imread(imageB)
	m = mse(original, contrast)
	s = ssim(original, contrast,multichannel=True)
	p = psnr(original, contrast)
	print ("Bits per pixel for "+"first image "+" : "+str(bits_per_pixel(imageA)))
	print ("Bits per pixel for "+"Second image"+" : "+str(bits_per_pixel(imageB)))
	bits_per_pixel(imageB)
	print("MSE: %.2f \nSSIM: %.2f" % (m, s))
	print("PSNR: "+str(p))


print("Images : "+sys.argv[1]+", "+sys.argv[2])

compare_images("images/"+sys.argv[1],"images/"+sys.argv[2])