# Import system libraries
import argparse
import math
import os
import sys
import time


# import 3rd party libraries
import csv
import matplotlib.pyplot as plt
import skimage
import torch
import torchvision
from PIL import Image
from torch import nn , optim
from torch.autograd import Variable
from torchvision import datasets, transforms


# Import user-defined libraries
from dataset import DatasetFromFolder
from network import EncoderNet, DecoderNet



irange = range


parser = argparse.ArgumentParser(description='Image Compression')
parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.001')
parser.add_argument('--beta', type=float, default=0.99, help='beta1 for adam. default=0.99')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
# parser.add_argument('--seed', type=int, default=123, help='random seed to uspe. Default=123')
parser.add_argument('--encoder_net', type=str, default='', help='Path to pre-trained encoder net. Default=3')
parser.add_argument('--decoder_net', type=str, default='', help='path to pre-trained deocder net. Default=3')
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--test_path', default='./Dataset/test', help='path of test images')
parser.add_argument('--channels', type=int, default=3, help='number of channels in an image. Default=3')
parser.add_argument('--dataset', type=str, default='STL10', help='dataset to be used for training and validation. Default=STL10')
parser.add_argument('--data_path', type=str, default='./Dataset/CLIC', help='path to images. Default=CLIC')
parser.add_argument('--image_size', type=int, default=90, help='path to images. Default=100')
parser.add_argument('--loss_function', type=int, default=0, help='Loss function. Default=0')
parser.add_argument('--use_GPU', type=int, default=-1, help='0 for GPU, 1 for CPU . Default=AUTO')
parser.add_argument('--mode', type=str, default='both', help='train / test / both . Default=both')


opt = parser.parse_args()

print (opt)

CUDA = torch.cuda.is_available()
LOG_INTERVAL = 5

if opt.use_GPU == 1:
    CUDA = False

def make_grid(tensor, nrow=8, padding=2,
              normalize=False, range=None, scale_each=False, pad_value=0):
    """Make a grid of images.
    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The Final grid size is (B / nrow, nrow). Default is 8.
        padding (int, optional): amount of padding. Default is 2.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by subtracting the minimum and dividing by the maximum pixel value.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If True, scale each image in the batch of
            images separately rather than the (min, max) over all images.
        pad_value (float, optional): Value for the padded pixels.
    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_
    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.view(1, tensor.size(0), tensor.size(1))
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.view(1, tensor.size(0), tensor.size(1), tensor.size(2))

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.size(0) == 1:
        return tensor.squeeze()

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new(3, height * ymaps + padding, width * xmaps + padding).fill_(pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding)\
                .narrow(2, x * width + padding, width - padding)\
                .copy_(tensor[k])
            k = k + 1
    return grid


def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    """Save a given Tensor into an image file.
    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)



def img_transform(crop_size):
    return transforms.Compose([
        transforms.RandomCrop(crop_size),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5,  0.5 , 0.5)),
        
    ])
  

"""### Loss function"""

def loss_function(final_img,residual_img,upscaled_img,com_img,orig_img):
    if opt.loss_function == 0:
        com_loss = nn.MSELoss()(orig_img, final_img)
        rec_loss = nn.MSELoss()(residual_img,orig_img-upscaled_img)  
        return com_loss + rec_loss
    elif opt.loss_function == 1:
        com_loss = nn.MSELoss()(orig_img, final_img)
        rec_loss = torch.sum(torch.std(residual_img))
        return com_loss + rec_loss


"""### define Train and Test methods"""

def train(epoch,model1,model2,train_loader):
    # model.train()
    optimizer1 = optim.Adam(model1.parameters(), lr=1e-3)
    optimizer2 = optim.Adam(model2.parameters(), lr=1e-3)
    model1.train()
    model2.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        if opt.dataset.upper() == 'STL10':
            data , _ = data
        data = Variable(data)
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        if CUDA:
          data = data.cuda()
          com_img = model1(data)
          final, residual_img, upscaled_image = model2(com_img)
        else :
          data = data.cpu()
          com_img = model1(data)
          final, residual_img, upscaled_image = model2(com_img)
        loss = loss_function(final, residual_img, upscaled_image, com_img, data)
        loss.backward()
        train_loss += loss.item()
        optimizer1.step()
        optimizer2.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    return train_loss / len(train_loader.dataset)

def validation(epoch,model1,model2, test_loader):
  
    model1.eval()
    model2.eval()
    val_loss = 0
    total_psnr = 0 
    torch.no_grad()
    for i, data in enumerate(test_loader):
        if opt.dataset.upper() == 'STL10':
            data , _ = data
        data = Variable(data)
        if CUDA:
            data = data.cuda()
            com_img = model1(data)
            final, residual_img, upscaled_image = model2(com_img)
        else :
            data = data.cpu()
            com_img = model1(data)
            final, residual_img, upscaled_image = model2(com_img)
        
        batch_loss = loss_function(final, residual_img, upscaled_image, com_img, data).item() 
        val_loss += batch_loss
        psnr = 10 * math.log10(1 / batch_loss)
        total_psnr += psnr
    val_loss /= len(test_loader.dataset)
    print('====> val set loss: {:.4f}'.format(val_loss))
    print('====> Avg. PSNR : {:.4f}'.format(total_psnr/len(test_loader.dataset)))
    return val_loss


def save_images(model1,model2,test_loader):
    model1.load_state_dict(torch.load('./Encoder_net.pth'))
    model2.load_state_dict(torch.load('./Decoder_net.pth'))

    model1.eval()
    model2.eval()

    test_loss = 0
    torch.no_grad()
    for i, data in enumerate(test_loader):

            data = Variable(data)

            if CUDA:
                data = data.cuda()
                com_img = model1(data)
                final, residual_img, upscaled_image = model2(com_img)
            else:
                data = data.cpu()
                com_img = model1(data)
                final, residual_img, upscaled_image = model2(com_img)

            test_loss += loss_function(final, residual_img, upscaled_image, com_img, data).item() 

            n = min(data.size(0), 6)

            print("saving the image "+str(i))
            comparison = torch.cat([data[:i],
                final[:i]])
            comparison = comparison.cpu()

            save_image(com_img[0].data,
                            'compressed_' + str(i) +'.png', nrow=n)
            save_image(final[0].data,
                        'final_' + str(i) +'.png', nrow=n)
            save_image(data[0].data,
                        'original_' + str(i) +'.png', nrow=n)


    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def main():
    """### Parameters"""
    CHANNELS = opt.channels
    HEIGHT = opt.image_size
    EPOCHS = opt.nEpochs

    """### Load Dataset"""
    if opt.dataset.upper() == 'STL10':
        trainset = datasets.STL10(root='./data', split='train',download=True, transform=img_transform((opt.image_size,opt.image_size)))

        val_set = datasets.STL10(root='./data', split='test',download=True, transform=img_transform((opt.image_size,opt.image_size)))
    elif opt.dataset.upper() == 'FOLDER':
        trainset = DatasetFromFolder(opt.data_path+'train/', input_transform=img_transform((opt.image_size,opt.image_size)))

        val_set = DatasetFromFolder(opt.data_path+'valid/', input_transform=img_transform((opt.image_size,opt.image_size)))

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=opt.batchSize,
                                        shuffle=True, num_workers=opt.threads)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=opt.batchSize,
                                        shuffle=False, num_workers=opt.threads)

    test_set =  DatasetFromFolder(opt.test_path,
                            input_transform=img_transform((opt.image_size,opt.image_size)))
    test_data_loader = torch.utils.data.DataLoader(dataset=test_set, num_workers=opt.threads, shuffle=False)


    info = {}

    info['channels'] = CHANNELS
    info['size'] = HEIGHT

    if CUDA:
        model1 = EncoderNet(info).cuda()
        model2 = DecoderNet(info).cuda()
    else :
        model1 = EncoderNet(info).cpu()
        model2 = DecoderNet(info).cpu()

    print("GPU available : "+str(CUDA))


    if opt.mode.upper() == 'TEST':  
        print(" Mode selected : test")
        print("run model for Images in test folder.")
        save_images(model1,model2,test_data_loader)
        sys.exit()
    
    """### Program Execution"""
    tr_loss = []
    vl_loss = []
    for epoch in range(1, EPOCHS+1):
        t1 = time.time()
        print("Epoch : "+str(epoch))
        t_loss = train(epoch,model1,model2,train_loader)
        v_loss = validation(epoch,model1,model2,val_loader)
        tr_loss.append(t_loss)
        vl_loss.append(v_loss)
        print("Time required for "+str(epoch)+"th epoch: "+str(time.time() - t1))
        if epoch % 50 == 0:
            torch.save(model1.state_dict(), '%s/Encoder_epoch_%d.pth' % (opt.outf, epoch))
            torch.save(model2.state_dict(), '%s/Decoder_epoch_%d.pth' % (opt.outf, epoch))
    
    print("Plot and save train and validation loss curves")
        # Plot train, test loss curves
    plt.plot(range(EPOCHS),tr_loss , 'r--',label='Training Loss')
    plt.plot(range(EPOCHS), vl_loss, 'b--',label='Validation Loss')
    plt.title('Loss vs Epochs')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.savefig('loss_'+str(time.time())+'.png')

    with open('loss_values.csv', "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows([tr_loss,vl_loss])

    print("save the models")

    torch.save(model1.state_dict(), './Encoder_net.pth')
    torch.save(model2.state_dict(), './Decoder_net.pth')

    if opt.mode.upper() == 'BOTH':  
        print("run model for Images in test folder.")
        save_images(model1,model2,test_data_loader)


if __name__ == '__main__':
    main()