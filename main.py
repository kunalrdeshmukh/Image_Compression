import argparse

from data import get_training_set, get_test_set
from torch.utils.data import DataLoader
from network import EncoderNet, DecoderNet
import torch.nn as nn
import torch.optim as optim


parser = argparse.ArgumentParser(description='Image Compression')
parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--beta', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--encoder_net', type=str, default='', help='Path to pre-trained encoder net. Default=3')
parser.add_argument('--decoder_net', type=str, default='', help='path to pre-trained deocder net. Default=3')
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--channels', type=int, default=3, help='number of channels in an image. Default=3')
parser.add_argument('--data_path', type=str, default='/Users/kunaldeshmukh/Dropbox/Spring_19/298/Image_compression_network/Dataset/CLIC', 
                help='path to images. Default=CLIC')
parser.add_argument('--image_size', type=int, default=200, help='path to images. Default=200')

opt = parser.parse_args()

print (opt)


print(' ===== Loading datasets ===== ')
train_set = get_training_set(opt.data_path,opt.image_size)
test_set = get_test_set(opt.data_path,opt.image_size)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

cudnn.benchmark = False #TODO : to check with value as True


# weights initilization 

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


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



def train(encoder,decoder,):

    print(' ===== Training ===== ')

    optimizerE = optim.Adam(encoder.parameters(), lr=opt.lr)
    optimizerD = optim.Adam(decoder.parameters(), lr=opt.lr)
    # optimizerE = optim.SGD(netD.parameters(), lr=opt.lr)
    # optimizerD = optim.SGD(netG.parameters(), lr=opt.lr)

    for epoch in range(opt.niter):
        epoch_loss = 0
        for i, data in enumerate(training_data_loader, 0):

            print("len of data : ")
            print(len(Data))

            # Move data to device
            if CUDA :
                input = data[0].to("cuda")
            else : 
                input = data[0].to("cpu")

            # set gradient to zero
            optimizerE.zero_grad()
            optimizerD.zero_grad()

            encoder_output = encoder(input)
            decoder_output, residual_img, upscaled_image  = decoder(input)

            loss1 = criterion(input, decoder_output)
            loss2 = criterion(residual_img,input-upscaled_image)
            (loss1+loss2).backward()

            optimizerE.step()
            optimizerD.step()

            loss += loss1.item() + loss2.item()

            epoch_loss += loss

            print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, i, len(training_data_loader), loss.item()))
        
        print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


        # do checkpointing
        torch.save(encoder.state_dict(), '%s/Encoder_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(decoder.state_dict(), '%s/Decoder_epoch_%d.pth' % (opt.outf, epoch))


def test(encoder,decoder,CUDA):
    avg_psnr = 0
    with torch.no_grad():
        for batch in testing_data_loader:
            if CUDA:
                input = batch[0].to('cuda')
            else:
                input = batch[0].to('cpu')

            compressed_img = encoder(input)
            retrived_img = decoder(compressed)
            mse = criterion(input, retrived_img)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))

    

def main():
    args = parser.parse_args()

    encoder_info = get_encoder_info()
    decoder_info = get_decoder_info()

    CUDA = torch.cuda.is_available()
    if CUDA:
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
    train(encoder,decoder,CUDA)

    test(encoder,decoder,CUDA)




if __name__ == '__main__':
    main()
