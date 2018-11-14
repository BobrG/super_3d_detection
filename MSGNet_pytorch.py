from torch import nn
import os
from torch import tensor
from torch.nn import functional as F
import torch
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
from scipy.misc import imresize
from scipy.ndimage import convolve
import h5py
from PIL import Image

from utils.tb import SummaryWriter

# useful functionsy

def h5_loader(path):
    h5f = h5py.File(path, "r")
    rgb = np.array(h5f['rgb'])
    if(rgb.shape[2] != 3):
        rgb = np.transpose(rgb, (1, 2, 0))
    depth = np.array(h5f['depth'])
#     depth_gt = np.array(h5f['depth_gt'])
    return rgb, depth

def add_to_h5(h5, data, calibration=None):
    for k, v in data.items():
        h5.create_dataset('/{}'.format(k), data=v, compression='gzip', dtype=v.dtype)

    if calibration is not None:
        for k, v in calibration.items():
            h5.create_dataset('/calibration/{}'.format(k), data=v)

def make_h5(h5_path, rgb=None, depth=None, disparity=None, calibration=None, **kwargs):
    assert (depth is None) != (disparity is None)
    data = {}

    if rgb is not None:
        data['rgb'] = np.transpose(rgb, (1, 2, 0))

    if depth is not None:
        data['depth'] = depth
    else:
        data['disparity'] = disparity
    with h5py.File(h5_path, 'w') as h5:
        add_to_h5(h5, data, calibration)

def  load_data(dir_in):
    imgs = h5py.File(dir_in + 'RGB.mat', 'r')
    dfs = h5py.File(dir_in + 'Df.mat', 'r')
    return imgs, dfs
def get_paches(dir_in = '',dir_out = '', stride = 22, size_h = 40,size_w  = 40):

    imgs, dfs = load_data(dir_in)

    l = 0
    for idx in range(len(imgs['RGB'][0])):
        img = imgs[imgs['RGB'][0][idx]][:].astype(np.float)
        depth =  dfs[dfs['Df'][0][idx]][:]
        print(idx)
        for i in range(0, img.shape[1] - size_h, stride):
            for j in range(0, img.shape[2] - size_w, stride):
                temp_i = img[:, i:i + size_h, j:j + size_w]
                temp_d = depth[i:i + size_h, j:j + size_w]
                if(np.isnan(temp_i).any() or np.isnan(temp_d).any()):
                    continue
                if((np.array(depth)==0).any()):
                    print('detected patch with 0 in depth')
                    continue
                make_h5(dir_out + str(l)+'.h5',temp_i,temp_d)
                l+=1
    return


def normalize_cleanIm(Z, min_Z=None, max_Z=None):
    if min_Z is None:
        min_Z = float(np.min(Z))
    if max_Z is None:
        max_Z = float(np.max(Z))
    if max_Z != min_Z:
        Z = (Z - min_Z)/(max_Z - min_Z)
    else:
        Z = np.random.random_sample(Z.shape)

    return [Z, min_Z, max_Z]

def rgb2ycbcr(im_rgb):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im_rgb.astype(np.float)
    rgb[:,:,[1,2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)#.astype(np.float)
    np.putmask(rgb, rgb < 0, 0)
    return rgb

def modcrop(imgs, modulo):
    # input image shape (h, w, channels)
    tmpsz = imgs.shape
    sz = [tmpsz[0], tmpsz[1]]
    sz[0] -= sz[0]%modulo
    sz[1] -= sz[1]%modulo

    if len(tmpsz) == 3:
        imgs = imgs[:sz[0], :sz[1], :]
    else:
        imgs = imgs[:sz[0], :sz[1]]
    return imgs

def resample(im, scale):
    im_Dl = Image.fromarray(im)
    old_size = im_Dl.size
    new_size = int(old_size[0] * scale), int(old_size[1] * scale)
    im_Dl = im_Dl.resize(new_size, resample=Image.BICUBIC)
    return np.array(im_Dl)

def rotation_augmentation(x_1, x_2, y):
    x_1, x_2 = np.rot90(x_1, k=1), np.rot90(x_2, k=1)
    y = np.rot90(y, k=1)
    return [x_1, x_2, y]

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

    def __init__(self, root_dir, scale, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dirs = os.listdir(root_dir)
        self.dir = root_dir
        self.transform = transform
        #f = h5py.File(root_dir + 'Df.mat', 'r')
        #self.dataset_shape = f['Df'].shape
        self.scale = scale
        self.stride = [22, 21, 20, 24]

    def __len__(self):

        return len(self.dirs)


    def __getitem__(self, idx):
        """
        sample:
            image - ycbcr image with one channel
            data - small depth map
            target - ground truth sr depth map for loss computation
        """
        # get rgb image
        #imgs = h5py.File(self.root_dir + 'RGB.mat', 'r')
        #img = imgs[imgs['RGB'][0][idx]][:].astype(np.float)
        img, depth = h5_loader(self.dir + self.dirs[idx])
        img = img.astype('float32')
        # img = np.moveaxis(img, 0, -1)
        img = modcrop(img, self.scale)
        img = rgb2ycbcr(img)[:,:,1]
        img = img.reshape(img.shape[0], img.shape[1], 1)
        [rgb, min_rgb, max_rgb] = normalize_cleanIm(img)
        rgb = rgb.astype('float32')
        #if min_rgb == max_rgb:
        #    print('rgb ', self.dir + self.dirs[idx])
        # get deep map for image
        #dfs = h5py.File(self.root_dir + 'Df.mat', 'r')
        #df = dfs[dfs['Df'][0][idx]][:]
        df = depth.astype('float32')
        df = modcrop(df, self.scale)
        h = np.ones((3,3))/9
        [df, min_df, max_df] = normalize_cleanIm(df)
        y = df #- convolve(df, h, mode='reflect')

        #y = y.reshape(y.shape[0], y.shape[1], 1)

        #y = y.astype('float32')
        df = resample(df, 1/self.scale)
        tmp = convolve(df, h, mode='reflect')
        y -= resample(tmp, self.scale)
        y = y.reshape(y.shape[0], y.shape[1], 1)
        y = y.astype('float32')
        df = df.reshape(df.shape[0], df.shape[1], 1)
        x = df

        #[x, min_x, max_x] = normalize_cleanIm(df)
        x = x.astype('float32')
        # transfromation (rot90)
        #if min_x == max_x:
        #    print('depth', self.dir + self.dirs[idx])


        if self.transform:
            [x, rgb, y] = self.transform(x, rgb, y)
        # channels first
        x = np.moveaxis(x, -1, 0)
        y = np.moveaxis(y, -1, 0)
        rgb = np.moveaxis(rgb, -1, 0)
        #x[np.isnan(x)] = 0.0
        sample = {'image': rgb.copy(), 'data': x.copy(), 'target': y.copy()}

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

class MSGNet(nn.Module):
    def __init__(self, upsample):
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

    def forward(self, rgb, depth):
        #print('==== Forward start')
        #print(rgb.shape, depth.shape)
        # early spectral decomposition
        h = np.ones((3,3))/9
        im_Dl_LF = np.zeros(depth.shape)
        # convolving depth map to get low-frequency features
        # normalizing depth map and saving min_depth max_depth for future test
        for i in range(depth.shape[0]):
            im_Dl_LF[i] = convolve(depth[i][0].cpu().numpy(), h, mode='reflect')

        in_D = depth - torch.cuda.FloatTensor(im_Dl_LF)
        # saving low frequency depth map for future test
        self.lf = np.zeros((depth.shape[0], depth.shape[1], depth.shape[2]*self.upsample, depth.shape[3]*self.upsample))
        for i in range(depth.shape[0]):
            self.lf[i] = resample(im_Dl_LF[i][0], self.upsample)

        # Y-channel
        im_Y_LF = np.zeros(rgb.shape)
        # preprocessing
        # convolving to get low frequency image
        for i in range(rgb.shape[0]):
            im_Y_LF[i] = convolve(rgb[i][0], h, mode='reflect')
        h_Yh = rgb - torch.cuda.FloatTensor(im_Y_LF).float()
        # normalizing image
        for i in range(h_Yh.shape[0]):
            h_Yh[i][0] = (h_Yh[i][0] - torch.min(h_Yh[i][0]))/(torch.max(h_Yh[i][0]) - torch.min(h_Yh[i][0]))

        # forward model
        m = int(np.log2(self.upsample))
        k = np.arange(0, 3*m-1, 3)
        k_1 = k + 1
        # Y-branch
        self.outputs_Y = [h_Yh]
        for layer in self.feature_extraction_Y:
            self.outputs_Y.append(layer(self.outputs_Y[-1]))
        # h(D)-branch
        self.outputs_X = []
        self.outputs_X.append(self.feature_extraction_X(in_D))
        for i, layer in enumerate(self.upsampling_X):
            self.outputs_X.append(layer(self.outputs_X[-1]))

            if i in k:
                y_ind = 2*(m - i // 3)
                #print(self.outputs_Y[y_ind].shape, self.outputs_X[-1].shape)
                self.outputs_X.append(torch.cat((self.outputs_Y[y_ind].float(), self.outputs_X[-1]), 1))

        output = self.outputs_X[-1]

        return output

    def get_test_data():
        return self.lf

def train(model, train_loader, optimizer, loss, device, epoch, tb_writer):
    model.train()
    train_loss = 0
    avg_loss = 0
    batches_n = len(train_loader)
    imgrid_w = 10
    imgrid_h = 10
    for batch_idx, batch_sample in enumerate(train_loader):
        rgb, data = batch_sample['image'].to(device), batch_sample['data'].to(device)
        target = batch_sample['target'].to(device)
        # print(rgb.shape)
        output = model(rgb, data)
        # print(output.shape)
        train_loss = loss(output, target)
        train_loss.backward()
        optimizer.step()
        loss_np = train_loss.item()
        avg_loss += loss_np
        optimizer.zero_grad()

        iteration = epoch * batches_n + batch_idx

        if batch_idx % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / batches_n, loss_np))
        tb_writer.add_scalar('Loss', loss_np, iteration)
        tb_writer.add_scalar('Loss_avg', avg_loss/(batch_idx+1), iteration)
        output_np, _, _ = normalize_cleanIm(output.detach().cpu().numpy()[0,0])
        target_np, _, _ = normalize_cleanIm(target.detach().cpu().numpy()[0,0])
        output_np = np.uint8(np.hstack((target_np, output_np)) * 255)[..., None].repeat(3, axis=-1)
        tb_writer.add_image('Target vs Output', output_np, iteration)
    print(avg_loss / batches_n)

def test(model, input_lowres_depth, rgb):
    model.eval()
    h, w = rgb.shape[:2]
    h_low, w_low = input_lowres_depth.shape[:2]
    scale = rgb.shape[1] // input_lowres_depth.shape[1]
    input_lowres_depth[np.isnan(input_lowres_depth)] = 0.0
    lowres, min_, max_ = normalize_cleanIm(input_lowres_depth)
    lowres, rgb = lowres.reshape(1, 1, lowres.shape[0], lowres.shape[1]), rgb.reshape(1, 1, rgb.shape[0], rgb.shape[1])
    print('input scale:', scale)
    with torch.no_grad():
        #for data, target in test_loader:
        lowres, img =torch.FloatTensor(lowres).to(device), torch.tensor(rgb).to(device)
        output = model(img, lowres)
        lf_output = model.lf
#         print(min_,max_)
        output = (lf_output + output.cpu().detach().data.numpy())[0][0]
        output = (output - min_) / (max_ - min_)

    return output
            #test_loss += loss(output, target).item()
    #test_loss /= len(test_loader.dataset)
    #print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)
    train_dir = '../datasets/MSGNet_data/'
    #get_paches(train_dir, train_dir + 'one_sample/', stride=22, size_h = 40, size_w  = 40)
    train_data = MSGNet_dataset(root_dir='../datasets/MSGNet_data/tmp/', scale=2, transform=rotation_augmentation)
    train_loader = DataLoader(train_data, batch_size=2000, shuffle=True)
    #test =
    #test_loader = DataLoader(test, batch_size=4, shiffle=True)
    model = MSGNet(upsample=2).to(device)
    # optimizer = torch.optim.SGD(model.parameters(), 10e-3, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters())
    train_loss = nn.MSELoss()
    #test_loss = RMSELoss()
    tb_writer = SummaryWriter('./.logs/msg-net/train_attempts')
    epochs = 100
    for epoch in range(epochs):
        torch.save(model,'model/checkpoint'+str(epoch))
        train(model, train_loader, optimizer, train_loss, device, epoch, tb_writer)

        #test(model, test_loader, test_loss, device, epoch)
    tb_writer.close()



