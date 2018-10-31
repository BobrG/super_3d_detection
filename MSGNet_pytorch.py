from torch import nn
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

# useful functions
def normalize_cleanIm(Z, min_Z, max_Z):
    if nargin == 1:
        min_Z = min(min(Z))
        max_Z = max(max(Z))

    Z = (Z - min_Z)/(max_Z - min_Z)
    return [Z, min_Z, max_Z] 

def rgb2ycbcr(im_rgb):
    im_rgb = im_rgb.astype(np.float32)
    im_ycrcb = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2YCR_CB)
    im_ycbcr = im_ycrcb[:,:,(0,2,1)].astype(np.float32)
    im_ycbcr[:,:,0] = (im_ycbcr[:,:,0]*(235-16)+16)/255.0 #to [16/255, 235/255]
    im_ycbcr[:,:,1:] = (im_ycbcr[:,:,1:]*(240-16)+16)/255.0 #to [16/255, 240/255]
    return im_ycbcr

# loss function
class RMSELoss(torch.nn.Module):
    def __init__(self):
    	super(RMSELoss,self).__init__()

    def forward(self, x, y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss

# dataset class
class MSGNet_dataset(Dataset):
    """Default MSGNet dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        f = h5py.File(root_dir + 'Df.mat', 'r')
        dataset_shape = f['Df'].shape

    def __len__(self):
        return dataset_shape[1]

    def __getitem__(self, idx):
        # get rgb image
        imgs = h5py.File(root_dir + 'RGB.mat', 'r')
        img = imgs[imgs['RGB'][0][idx]][:]
        
        # get deep map for image
        dfs = h5py.File(root_dir + 'Df.mat', 'r')
        df = dfs[dfs['Df'][0][idx]][:]
        
        sample = [img, dh]
        if self.transform:
            sample = self.transform(sample)

        return sample


# MSGNet 
class ConvPReLu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvPReLu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 5, stride=1, padding=2)
        self.activation = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x
    
class DeconvPReLu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvPReLu, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.activation = nn.PReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


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
        
        self.feature_extraction_Y = []
        self.upsampling_X = []
        # Y-branch
        in_, out = 3, 49
        self.feature_extraction_Y.append(nn.PReLU(nn.Conv2d(in_, out, 7, stride=1, padding=3)))
        in_, out = 49, 32
        self.feature_extraction_Y.append(ConvPReLu(in_, out))
        for i in range(2, 2*m):
            if i in j:
                self.feature_extraction_Y.append(ConvPReLu(in_, out))
            if i in j_:
                self.feature_extraction_Y.append(nn.MaxPool2d(3, 3))
            
        # h(D)-branch   
        in_, out = 3, 64
        self.feature_extraction_X = nn.PReLU(nn.Conv2d(in_, out, 5, stride=1, padding=2))
        j = 0
        for i in range(1, M):
            if i in k:
                self.upsampling_X.append(DeconvPReLu(in_, out, 5, stride=2, padding=2)) # deconvolution 
            if i in k_1:
                self.upsampling_X.append(ConvPReLu(in_*2, out, 5, stride=1, padding=2)) # convolution for concatenation aka fusion
            if (i in k_2) or (i in k_3):
                self.upsampling_X.append(ConvPReLu(in_, out, 5, stride=1, padding=2)) # post-fusion
        in_, out = 32, 1
        self.upsampling_X.append(ConvPReLu(in_, out, 5, stride=1, padding=2)) # reconstruction        
        
    def forward(self, gt, rgb):
    	# early spectral decomposition
        h = np.ones(3,3)/9
        
        im_Dl = imresize(float(gt),  self.upsample, 'bicubic')
        [im_Dl, min_D, max_D] = normalize_cleanIm(im_Dl)
    
        im_Dl_LF = convolve(im_Dl, h, 'reflect')
        in_D = im_Dl - im_Dl_LF
        
        # Y-channel 
        im_I = rgb2ycbcr(rgb)
        im_Y = float(im_I[:,:,1])
            
        im_Y = normalize_cleanIm(im_Y);
        im_Y_LF = convolve(im_Y, h, 'reflect')
        [im_Y, min_Y, max_Y] = normalize_cleanIm(im_Y - im_Y_LF)
    
        im_Y = im_Y[1:-1, 1:-1]
        
        h_Yh = normalize_cleanIm(im_Y - im_Y_LF)
        
        # forward model 
        m = int(np.log2(self.upsample))
        k = np.arange(1, 3*m, 3)
        k_1 = k + 1
        x = imresize(gt, 1/self.upsample, 'bicubic')
        # Y-branch
        self.outputs_Y = [h_Yh]
        for layer in self.feature_extraction_Y:
            self.outputs_Y.append(layer(self.outputs_Y[-1]))
        # h(D)-branch
        self.outputs_X = []
        self.outputs_X.append(self.feature_extraction_X(x))
        k = 0
        for i, layer in enumerate(self.upsampling_X):
            self.outputs_X.append(layer(self.outputs_X[-1]))
            if i in k_1:
                y_ind = 2*(m + i // 3) - 1
                self.outputs_X.append(layer(torch.cat([outputs_Y[y_ind], outputs_X[-1]], 1)))
        
        im_Dh = self.outputs_X[-1]
        x_LF = imresize(x_LF, self.upsample, 'bicubic')
        output = im_Dh + im_D_LF[1:-1, 1:-1]# post-reconstruction
        im_Dh = im_Dh*(max_D - min_D) + min_D
        
        return output

def train(model, train_loader, optimizer, loss, device, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        train_loss = loss(output, target) 
        train_loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), train_loss.item()))

def test(model, test_loader, loss, device, epoch):
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
    train = MSGNet_dataset(root_dir='../datasets/MSGNet_data/')
    train_loader = DataLoader(train, batch_size=4, shuffle=True)
    #test = 
    #test_loader = DataLoader(test, batch_size=4, shiffle=True)
    model = MSGNet(upsample=2).to(device)
    optimizer = torch.optim.SGD(model.parameters(), 10e-3,
                                momentum=0.9)
    loss = RMSELoss()
    for epoch in range(epochs):
        train(model, train_loader, optimizer, loss, device, epoch)
        #test(model, test_loader, loss, device, epoch)


