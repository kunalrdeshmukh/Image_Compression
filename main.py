import argparse

from torch.cuda import is_available 
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn
from torch import save,no_grad
from math import log10

from network import EncoderNet, DecoderNet
from data import get_training_set, get_val_set
import matplotlib.pyplot as plt
import time
import csv




parser = argparse.ArgumentParser(description='Image Compression')
parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=8, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.001')
parser.add_argument('--beta', type=float, default=0.99, help='beta1 for adam. default=0.99')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to uspe. Default=123')
parser.add_argument('--encoder_net', type=str, default='', help='Path to pre-trained encoder net. Default=3')
parser.add_argument('--decoder_net', type=str, default='', help='path to pre-trained deocder net. Default=3')
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--channels', type=int, default=3, help='number of channels in an image. Default=3')
parser.add_argument('--dataset', type=str, default='folder', help='dataset to be used for training and validation. Default=folder')
parser.add_argument('--data_path', type=str, default='./Dataset/CLIC', 
                help='path to images. Default=CLIC')
parser.add_argument('--image_size', type=int, default=200, help='path to images. Default=200')

opt = parser.parse_args()

print (opt)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
if opt.dataset == 'folder':
    train_set = get_training_set(opt.data_path,opt.image_size,'folder')
    val_set = get_val_set(opt.data_path,opt.image_size,'folder')
elif opt.dataset == 'stl10' or opt.dataset == 'STL10':
    train_set = get_training_set(opt.data_path,opt.image_size,'STL10')
    val_set = get_val_set(opt.data_path,opt.image_size,'STL10')
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
val_data_loader = DataLoader(dataset=val_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)


cudnn.benchmark = False #TODO : to check with value as True


# Loss function defination
criterion = nn.MSELoss()


def get_encoder_info():
    info = {}
    info['channels'] = opt.channels
    return info


def get_decoder_info():
    info = {}
    info['channels'] = opt.channels
    info['size'] = opt.image_size
    return info



def train(encoder,decoder,CUDA):

    # optimizerE = optim.Adam(encoder.parameters(), lr=opt.lr)
    # optimizerD = optim.Adam(decoder.parameters(), lr=opt.lr)
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=opt.lr)
    # optimizerE = optim.SGD(netD.parameters(), lr=opt.lr)
    # optimizerD = optim.SGD(netG.parameters(), lr=opt.lr)
    train_loss = []
    val_loss = []
    psnr = []
    for epoch in range(1,opt.nEpochs+1):
        print(' ===== Training ===== ')
        epoch_loss = 0
        for i, data in enumerate(training_data_loader, 0):

            # Move data to device
            if CUDA :
                input = data.to("cuda")
            else : 
                input = data.to("cpu")

            # set gradient to zero
            # optimizerE.zero_grad()
            # optimizerD.zero_grad()
            optimizer.zero_grad()

            encoder_output = encoder(input)
            decoder_output, residual_img, upscaled_image  = decoder(encoder_output)

            loss1 = criterion(input, decoder_output)
            # print(loss1.item())

            loss2 = criterion(residual_img,input - upscaled_image)
            # print(loss2.item())

            # loss1.backward()
            (loss1+loss2).backward()

            # optimizerE.step()
            # optimizerD.step()
            optimizer.step()
 

            loss = loss1.item() + loss2.item()
            # loss = loss1.item()

            epoch_loss += loss

            print("===> Epoch[{}]({}/{}): Training Loss: {:.4f}".format(epoch, i, len(training_data_loader), loss))
        
        print("===> Epoch {} Complete: Avg. train Loss: {:.4f}\n".format(epoch, epoch_loss / len(training_data_loader)))
        (avg_mse,avg_psnr) = validation(encoder,decoder,CUDA)
        val_loss.append(avg_mse)
        train_loss.append(epoch_loss)
        psnr.append(avg_psnr)
        # do checkpointing
        if epoch % 20 == 0:  # happens 20 times
            save(encoder.state_dict(), '%s/Encoder_epoch_%d.pth' % (opt.outf, epoch))
            save(decoder.state_dict(), '%s/Decoder_epoch_%d.pth' % (opt.outf, epoch))
    return (train_loss,val_loss,psnr)


def validation(encoder,decoder,CUDA):

    print(' ===== Validation ===== ')

    avg_psnr = 0
    avg_mse = 0
    with no_grad():
        for batch in val_data_loader:
            if CUDA:
                input = batch.to('cuda')
            else:
                input = batch.to('cpu')

            compressed_img = encoder(input)
            final,out,upscaled_imag = decoder(compressed_img)
            mse = criterion(input, final)
            avg_mse += mse.item()
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
    print("===> Avg. MSE: {:.4f} ".format(avg_mse / len(val_data_loader)))
    print("===> Avg. PSNR: {:.4f} dB\n".format(avg_psnr / len(val_data_loader)))
    return (avg_mse,avg_psnr)


    

def main():
    args = parser.parse_args()

    encoder_info = get_encoder_info()
    decoder_info = get_decoder_info()

    CUDA = is_available()
    if CUDA:
        print("|| Using CUDA ||")
        print()
        encoder = EncoderNet(encoder_info).cuda()
        decoder = DecoderNet(decoder_info).cuda()
    else :
        encoder = EncoderNet(encoder_info)
        decoder = DecoderNet(decoder_info)
    
    encoder.apply(init_weights)
    decoder.apply(init_weights)

    if opt.encoder_net != '':
        encoder.load_state_dict(torch.load(opt.encoder_net))
        print("Loaded encoder from file.")
    
    if opt.decoder_net != '':
        decoder.load_state_dict(torch.load(opt.decoder_net))
        print("Loaded decoder from file.")


    print(encoder)
    print(decoder)

    #set loss and optimizer 
    (train_loss,val_loss,psnr) = train(encoder,decoder,CUDA)

    _, _ = validation(encoder,decoder,CUDA)

    # Save Model
    save(encoder,'%s/Encoder_model.pth'%opt.outf )
    save(decoder,'%s/Decoder_model.pth'%opt.outf )
    print("Models saved at "+opt.outf)
    
    # Plot train, test loss curves
    plt.plot(range(opt.nEpochs),train_loss , 'r--',label='Training Loss')
    plt.plot(range(opt.nEpochs), val_loss, 'b--',label='Validation Loss')
    plt.title('Loss vs Epochs')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.savefig('loss_'+str(time.time())+'.png')

    with open('loss_values.csv', "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows([train_loss,val_loss,psnr])

if __name__ == '__main__':
    main()