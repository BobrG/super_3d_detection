from torch import nn
from torch import tensor
from torch.nn import functional as F
import torch
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
from scipy.misc import imresize
from scipy.ndimage import convolve
import cv2
import h5py
from PIL import Image

# useful functions
def normalize_cleanIm(Z, min_Z=None, max_Z=None):
    if min_Z is None:
        min_Z = float(np.min(Z))
    if max_Z is None:
        max_Z = float(np.max(Z))

    Z = (Z - min_Z)/(max_Z - min_Z)
    return [Z, min_Z, max_Z]

def rgb2ycbcr(im_rgb):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im_rgb.astype(np.float)
    rgb[:,:,[1,2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)

# loss function
class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

        # TODO hack
        #if img.shape[1] % 2 != 0:
        #    img = img[:, :-1]
        #    df = df[:, :-1]
        #if img.shape[2] % 2 != 0:
        #    img = img[:, :, :-1]
        #    df = df[:, :, :-1]
        #img = img[:, :256, :256]
        #df = df[:, :256, :256]


        img_patches =
        df_pathces =
        sample = [img, df]
        if self.transform:
            sample = self.transform(sample)

        return sample


# MSGNet
class ConvPReLu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=5, stride=1, padding=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding)
        self.activation = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class DeconvPReLu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, stride=2, padding=1):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding, output_padding=stride-1)
        self.activation = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

def resample(im, scale):
    im_Dl = Image.fromarray(im)
    old_size = im_Dl.size
    new_size = int(old_size[0] * scale), int(old_size[1] * scale)
    im_Dl = im_Dl.resize(new_size, resample=Image.BICUBIC)
    return np.array(im_Dl)

class MSGNet(nn.Module):
    def __init__(self, upsample=2):
        super(MSGNet, self).__init__()
        # initialize indexes for layers
        self.upsample = upsample
 m = int(np.log2(upsample))
        M = 3*(m + 1)
        j = np.arange(2, 2*m - 1, 2)
        j_ = np.arange(3, 2*m , 2)
        k = np.arange(1, 3*m, 3) # deconv indexes
        k_1 = k + 1 # fusion indexes
        k_2 = k + 2 # post-fusion indexes
        k_3 = np.arange(3*m + 1, M - 1, 3) # post-fusion indexes

        self.feature_extraction_Y = nn.ModuleList()
        self.upsampling_X = nn.ModuleList()
        # Y-branch
        in_, out = 1, 49
        self.feature_extraction_Y.append(ConvPReLu(in_, out, 7, 1, 3))
        in_, out = 49, 32
        self.feature_extraction_Y.append(ConvPReLu(in_, out))
        in_, out = 32, 32
        for i in range(2, 2*m):
            if i in j:
                self.feature_extraction_Y.append(ConvPReLu(in_, out))
            if i in j_:
                self.feature_extraction_Y.append(nn.MaxPool2d(3, 2, padding=1))

        # h(D)-branch
        in_, out = 1, 64
        self.feature_extraction_X = ConvPReLu(in_, out, 5, 1, 2)
        j = 0
        in_, out = 64, 32
        for i in range(1, M):
            if i in k:
                self.upsampling_X.append(DeconvPReLu(in_, out, 5, stride=2, padding=2)) # deconvolution
            if i in k_1:
                self.upsampling_X.append(ConvPReLu(in_*2, out, 5, stride=1, padding=2)) # convolution for concatenation aka fusion
            if (i in k_2) or (i in k_3):
                self.upsampling_X.append(ConvPReLu(in_, out, 5, stride=1, padding=2)) # post-fusion
            in_, out = 32, 32
        in_, out = 32, 1
        self.upsampling_X.append(ConvPReLu(in_, out, 5, 1, 2)) # reconstruction

    def forward(self, rgb, gt):
        print('==== Forward start')
        print(gt.shape)
        # early spectral decomposition
        h = np.ones((3,3))/9
        im_Dl = resample(gt[0,0].cpu().numpy(), 1/self.upsample)
        #im_Dl = imresize(gt[0].cpu().numpy(),  1/self.upsample, 'bicubic')
        [im_Dl, min_D, max_D] = normalize_cleanIm(im_Dl)

        im_Dl_LF = convolve(im_Dl, h, mode='reflect')
        in_D = im_Dl - im_Dl_LF
        print(in_D.shape)
        #TR in_D = np.moveaxis(in_D, 0, 1)
        print(in_D.shape)
        in_D = in_D.reshape(1, 1, in_D.shape[0], in_D.shape[1])
        # Y-channel
        im_Y = np.moveaxis(rgb[0].cpu().numpy(), 0, -1)
        im_I = rgb2ycbcr(im_Y)
        im_Y = im_Y[:,:,1]
        [im_Y, min_Y, max_Y] = normalize_cleanIm(im_Y)
        im_Y_LF = convolve(im_Y, h, mode='reflect')

        [h_Yh, min_, max_] = normalize_cleanIm(im_Y - im_Y_LF)
         #h_Yh = h_Yh[:-1, :-1]
        print(h_Yh.shape)
        #TR h_Yh = np.moveaxis(h_Yh, 0, 1)
        h_Yh = h_Yh.reshape(1, 1, h_Yh.shape[0], h_Yh.shape[1])
        print(h_Yh.shape)

        # forward model
        m = int(np.log2(self.upsample))
        k = np.arange(0, 3*m-1, 3)
        k_1 = k + 1
        # x = imresize(gt[0].cpu().numpy(), 1/self.upsample, 'bicubic')
        # Y-branch
        self.outputs_Y = [torch.cuda.FloatTensor(h_Yh)]
        for layer in self.feature_extraction_Y:
            print('Ok')
            self.outputs_Y.append(layer(self.outputs_Y[-1]))
        # h(D)-branch
        self.outputs_X = []
        self.outputs_X.append(self.feature_extraction_X(torch.cuda.FloatTensor(in_D)).float())
        for i, layer in enumerate(self.upsampling_X):
            print('Ok')
            print(self.outputs_X[-1].shape)
            self.outputs_X.append(layer(self.outputs_X[-1]))

            if i in k:
                y_ind = 2*(m - i // 3)
                print('here')
                print(self.outputs_Y[y_ind].shape, self.outputs_X[-1].shape)
                self.outputs_X.append(torch.cat((self.outputs_Y[y_ind], self.outputs_X[-1]), 1))

        im_Dh = self.outputs_X[-1]
        print(im_Dl_LF.shape, im_Dh.shape)
        #im_Dl_LF = cv2.resize(im_Dl_LF, dsize=gt[0].shape, interpolation=cv2.INTER_CUBIC)
        im_Dl_LF = resample(im_Dl_LF, self.upsample)
        im_Dl_LF = torch.cuda.FloatTensor(im_Dl_LF).float()
        #print(im_Dl_LF[0:-1, 0:-1].shape, im_Dh.shape)
        im_Dh = im_Dh + im_Dl_LF#[0:-1, 0:-1]# post-reconstruction

        #output = im_Dh*(torch.cuda.FloatTensor(max_D) - torch.cuda.FloatTensor(min_D)) + torch.cuda.FloatTensor(min_D)
        output = im_Dh*(max_D - min_D) + min_D
        # output = torch.cuda.FloatTensor(output).float()
        return output

def train(model, train_loader, optimizer, loss, device, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (rgb, df) in enumerate(train_loader):
        rgb, df = rgb.to(device), df.to(device)
        optimizer.zero_grad()
        output = model(rgb, df)
        train_loss = loss(output, df)
        train_loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(rgb), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), train_loss.item()))

def test(model, test_loader, loss, device, epoch):
    # TODO at test time the input depth is low-res
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss(output, target).item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)
    train_data = MSGNet_dataset(root_dir='../datasets/MSGNet_data/')
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    #test =
    #test_loader = DataLoader(test, batch_size=4, shiffle=True)
    model = MSGNet(upsample=2).to(device)
    optimizer = torch.optim.SGD(model.parameters(), 10e-3,
                                momentum=0.9)
    # TODO MSELoss on high-freq
    loss = RMSELoss()
    epochs = 5
    for epoch in range(epochs):
        train(model, train_loader, optimizer, loss, device, epoch)
        #test(model, test_loader, loss, device, epoch)

                                                                           
