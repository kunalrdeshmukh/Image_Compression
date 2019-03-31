import torch.utils.data as data

from os import listdir
from os.path import join
from PIL import Image


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath,channels):
    img = Image.open(filepath).convert('RGB')
    if channels == 3:
        img = Image.open(filepath).convert('RGB')
    else:
        img = Image.open(filepath).convert('L')
    return img


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None,channels = 3):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.input_transform = input_transform
        self.channels = channels

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index],self.channels)
        if self.input_transform:
            input = self.input_transform(input)
        return input 


    def __len__(self):
        return len(self.image_filenames)