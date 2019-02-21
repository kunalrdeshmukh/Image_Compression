from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize



def get_training_set(upscale_factor):
    root_dir = download_bsd300()
    train_dir = join(root_dir, "train")
    # crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(train_dir,
                            input_transform=input_transform(crop_size, upscale_factor),
                            target_transform=target_transform(crop_size))


def get_test_set(upscale_factor):
    root_dir = download_bsd300()
    test_dir = join(root_dir, "test")
    # crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(test_dir,
                            input_transform=input_transform(crop_size, upscale_factor),
                            target_transform=target_transform(crop_size))