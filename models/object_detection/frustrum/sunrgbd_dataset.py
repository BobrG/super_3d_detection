import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from glob import glob
import numpy as np
import cv2
import os

from PIL import Image
import numpy as np
import scipy.io

# def z_to_absolute_coordinates(z, calibration, center_is_in_cell=False):
#     fx = calibration['fx']
#     fy = calibration['fy']
#     cx = calibration['cx']
#     cy = calibration['cy']

#     h,w = z.shape[:2]
#     j,i = np.meshgrid(np.arange(w), np.arange(h), sparse=True)
#     if center_is_in_cell:
#         i = i + .5
#         j = j + .5
#     x = z * (j - cx) / fx
#     y = z * (i - cy) / fy
#     return np.dstack((x, y, z))

def pc_from_rgbd(depthmap, Rtilt, K):
    rows, cols = depthmap.shape
    cx, cy = K[0,2], K[1,2]
    fx, fy = K[0,0], K[1,1]
    
    positive = depthmap > 0.
    finite = np.isfinite(depthmap)
    valid = np.logical_and(positive, finite)

    x = np.where(valid, (c - self.cx) / self.fx, 0)
    x1 = x.reshape(-1)
    y = np.where(valid, (r - self.cy) / self.fy, 0)
    y1 = y.reshape(-1)
    z = np.ones_like(depthmap)
    z1 = z.reshape(-1) 
    to_pixels_in_world = np.dstack((x, y, z))
    to_pixels_in_world /= np.linalg.norm(to_pixels_in_world, axis=-1)[..., None]
    to_pixels_in_world[np.logical_not(valid), 2] = np.nan
    pts_3d_matrix = torch.from_numpy(to_pixels_in_world)*depthmap[..., None] 
    res = np.transpose(np.dot(self.Rot,np.transpose(pts_3d_matrix.reshape(-1, 3))))
    
    return res


class SUNRGBD(Dataset):
    def __init__(self, toolbox_root_path):
        self.toolbox_root_path = toolbox_root_path
        self.meta = scipy.io.loadmat('Metadata/SUNRGBDMeta.mat')['SUNRGBDMeta'][0]
        self.ds_len = self.meta.shape[0]

    def __len__(self):
        return self.ds_len

    def __getitem__(self, idx):
        sample = {}

        depth_path = '..' + self.meta[idx][4][0][16:]
        img_path = '..' + self.meta[idx][5][0][16:]
        sample['rgb'] = cv2.imread(img_path)

        depth = cv2.imread(depth_path, 0).astype(np.uint16)
        depth = np.float32(np.bitwise_or(np.right_shift(depth, 3), np.left_shift(depth, 16-3))) / 1000
        depth[depth == 0] = np.nan
        depth[depth > 8] = np.nan

        sample['K'] = self.meta[idx][3]
        sample['Rtilt'] = self.meta[idx][2] 
        sample['data'] = depth
        sample['gt_bb3d'], sample['gt_bb2d'] = self.meta[idx][1], self.meta[idx][-1]

        return sample