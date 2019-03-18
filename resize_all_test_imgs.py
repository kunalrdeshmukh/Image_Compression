from PIL import Image
import os, sys

path = "./data/test/"
dirs = os.listdir( path )
size = 96,96

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize(size, Image.ANTIALIAS)
            os.remove(path+item)
            imResize.save(f + '_resized.jpg', 'JPEG', quality=90)

resize()
