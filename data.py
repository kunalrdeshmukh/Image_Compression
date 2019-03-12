from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize, RandomCrop, Normalize
from torchvision.datasets import STL10
from dataset import DatasetFromFolder


def input_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),Normalize((0.5, 0.5, 0.5), (0.5,  0.5 , 0.5)),
        ToTensor(),
    ])


def target_transform(crop_size):
    return Compose([
        Resize(crop_size),
        ToTensor(),
])


def get_training_set(path,crop_size,dataset):
        if dataset == 'folder':
                train_dir = join(path, "train")
                return DatasetFromFolder(train_dir,
                            input_transform=input_transform(crop_size))
        elif dataset == 'STL10':
                return STL10(root='./data', split='train',download=True, transform=input_transform)


def get_val_set(path,crop_size,dataset):
        if dataset == 'folder':
                test_dir = join(path, "valid")
                return DatasetFromFolder(test_dir,
                            input_transform=input_transform(crop_size))
        elif dataset == 'STL10':
                return STL10(root='./data', split='test',download=True, transform=input_transform)