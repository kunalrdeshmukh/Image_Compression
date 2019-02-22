from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize, RandomCrop

from dataset import DatasetFromFolder


def input_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])


def target_transform(crop_size):
    return Compose([
        Resize(crop_size),
        ToTensor(),
])


def get_training_set(path,crop_size):
    train_dir = join(path, "train")
    return DatasetFromFolder(train_dir,
                            input_transform=input_transform(crop_size, upscale_factor))


def get_test_set(path,crop_size):
    test_dir = join(path, "test")

    return DatasetFromFolder(test_dir,
                            input_transform=input_transform(crop_size, upscale_factor))