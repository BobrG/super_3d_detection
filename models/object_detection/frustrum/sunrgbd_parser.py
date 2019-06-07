import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from glob import glob
import numpy as np
import scipy.misc as m
import pickle
import h5py
import cv2
import sys
import os
from msg.test import MSGNet, ConvPReLu, DeconvPReLu
import msg.test as test

from PIL import Image
import numpy as np
import scipy.io

from scipy import signal

NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 10
NUM_CLASS = 10

type2class = {'bed':0, 'table':1, 'sofa':2, 'chair':3, 'toilet':4, 'desk':5, 'dresser':6, 'night_stand':7, 'bookshelf':8, 'bathtub':9}
class2type = {type2class[t]:t for t in type2class}
type2onehotclass={'bed':0, 'table':1, 'sofa':2, 'chair':3, 'toilet':4, 'desk':5, 'dresser':6, 'night_stand':7, 'bookshelf':8, 'bathtub':9}
type_mean_size = {'bathtub': np.array([0.765840,1.398258,0.472728]),
                  'bed': np.array([2.114256,1.620300,0.927272]),
                  'bookshelf': np.array([0.404671,1.071108,1.688889]),
                  'chair': np.array([0.591958,0.552978,0.827272]),
                  'desk': np.array([0.695190,1.346299,0.736364]),
                  'dresser': np.array([0.528526,1.002642,1.172878]),
                  'night_stand': np.array([0.500618,0.632163,0.683424]),
                  'sofa': np.array([0.923508,1.867419,0.845495]),
                  'table': np.array([0.791118,1.279516,0.718182]),
                  'toilet': np.array([0.699104,0.454178,0.756250])}

g_mean_size_arr = np.zeros((NUM_SIZE_CLUSTER, 3)) # size clustrs
for i in range(NUM_SIZE_CLUSTER):
    g_mean_size_arr[i,:] = type_mean_size[class2type[i]]


def get_crop_margins(old_size, new_size):
    old_width, old_height = old_size
    new_width, new_height = new_size
    assert old_width >= new_width and old_height >= new_height

    left_crop = (old_width - new_width) // 2
    right_crop = old_width - new_width - left_crop
    top_crop = (old_height - new_height) // 2
    bottom_crop = old_height - new_height - top_crop
    return left_crop, top_crop, right_crop, bottom_crop

def fix_calibration(K, scale=None, crop_after_scale=None):
    K = K.copy()
    # cx, cy = K[0,2], K[1,2]
    # fx, fy = K[0,0], K[1,1]
    if scale is not None and scale != 1:
        K[0,2] /= scale
        K[1,2] /= scale
        # if 'f' in calibration:
        #     calibration['f'] /= scale
        # fx
        K[0,0] /= scale
        # fy
        K[1,1] /= scale
        # if 'baseline' in calibration:
        #     calibration['baseline'] *= scale

    if crop_after_scale is not None:
        left, top = crop_after_scale
        K[0,2] -= left
        K[1,2] -= top

    return K

def resize_and_crop(im, scale=None, method=None, crop=False, div_by=1, imsize=None):
    ''' Downscale only '''
    old_width, old_height = im.size

    if scale is None:
        if imsize is None: # just div_by
            imsize = im.size
        if not isinstance(imsize, (list, tuple)):
            imsize = (imsize, imsize)
    else:
        imsize = im.size[0] / scale, im.size[1] / scale
    
    new_max_width = imsize[0] - imsize[0] % div_by
    new_max_height = imsize[1] - imsize[1] % div_by
    
    scale_width = old_width / new_max_width
    scale_height = old_height / new_max_height
    
    if crop:
        scale = min(scale_width, scale_height)
    else:
        scale = max(scale_width, scale_height)

    new_width = int(old_width / scale)
    new_height = int(old_height / scale)
    
    if isinstance(method, (tuple, list)):
        if method[0] == 'box':
            kernel_width = method[1]
            kernel = np.full((kernel_width, kernel_width), 1 / (kernel_width**2))
            newim = signal.convolve2d(np.array(im), kernel, mode='full')
            newim = Image.fromarray(newim[kernel_width-1::kernel_width, kernel_width-1::kernel_width])
    else:
        newim = im.resize((new_width, new_height), method)

    if not crop:
        new_max_width = new_width - new_width % div_by
        new_max_height = new_height - new_height % div_by

    left_crop, top_crop, right_crop, bottom_crop = get_crop_margins((new_width, new_height), (new_max_width, new_max_height))
    crop_box = (left_crop, top_crop, new_width - right_crop, new_height - bottom_crop)
    newim = newim.crop(crop_box)

    return newim, scale, (left_crop, top_crop)


def resize_and_crop_ar(ar_np, *args, **kwargs):
    im = Image.fromarray(ar_np)
    newim, scale, crop = resize_and_crop(im, *args, **kwargs)
    return np.array(newim), scale, crop


def resize_and_crop_ar_with_inpaint(ar_np, *args, **kwargs):
    ar_filled = ar_np.copy()
    invalids = np.isnan(ar_filled)
    ar_filled[invalids] = 0
    ar_resized, scale, crop = resize_and_crop_ar(ar_filled, *args, **kwargs)

    support = np.ones_like(ar_filled)
    support[invalids] = 0
    support, _, _ = resize_and_crop_ar(support, *args, **kwargs)

    return ar_resized/support, scale, crop

def depth_is_valid(depth):
    positive = depth > 0.
    finite = np.isfinite(depth)
    return np.logical_and(positive, finite)

def get_pc(depthmap, Rtilt, K):
        rows, cols = depthmap.shape
        cx, cy = K[0,2], K[1,2]
        fx, fy = K[0,0], K[1,1]
        c, r = np.meshgrid(np.arange(1, cols+1), np.arange(1, rows+1), sparse=True)

        valid = depthmap > 0.

        z = np.where(valid, depthmap, np.nan)
        x = (c - cx) / fx * z
        y = (r - cy) / fy * z
        
        to_pixels_in_world = np.dstack((x, z, -1*y))
        pts_3d_matrix = to_pixels_in_world
        if Rtilt is not None:
            res = np.transpose(np.matmul(Rtilt, np.transpose(np.reshape(pts_3d_matrix, (-1,3)))))
        else:
            res = np.reshape(pts_3d_matrix, (-1,3))

        return res

def upsample_and_inpaint(depth, factor, method):
    # make coordinate grid
    lr_h, lr_w = depth.shape[:2]
    hr_h, hr_w = int(lr_h * factor), int(lr_w * factor)
    lr_h_grid_step = 1/lr_h
    lr_w_grid_step = 1/lr_w
    hr_h_grid_step = 1/hr_h
    hr_w_grid_step = 1/hr_w

    hr_i, hr_j = np.mgrid[hr_h_grid_step/2:1:1/hr_h, hr_w_grid_step/2:1:1/hr_w]
    lr_i, lr_j = np.mgrid[lr_h_grid_step/2:1:1/lr_h, lr_w_grid_step/2:1:1/lr_w]

    # add boundary points
    lr_i = np.hstack((lr_i[:,:1], lr_i, lr_i[:,:1]))
    lr_i = np.vstack((lr_i[:1, :] * 0 - lr_h_grid_step/2, lr_i, lr_i[:1, :] * 0 + 1 + lr_h_grid_step/2))
    hr_i = np.hstack((hr_i[:,:1], hr_i, hr_i[:,:1]))
    hr_i = np.vstack((hr_i[:1, :] * 0 - hr_h_grid_step/2, hr_i, hr_i[:1, :] * 0 + 1 + hr_h_grid_step/2))

    lr_j = np.vstack((lr_j[:1,:], lr_j, lr_j[:1,:]))
    lr_j = np.hstack((lr_j[:, :1] * 0 - lr_w_grid_step/2, lr_j, lr_j[:, :1] * 0 + 1 + lr_w_grid_step/2))
    hr_j = np.vstack((hr_j[:1,:], hr_j, hr_j[:1,:]))
    hr_j = np.hstack((hr_j[:, :1] * 0 - hr_w_grid_step/2, hr_j, hr_j[:, :1] * 0 + 1 + hr_w_grid_step/2))

    depth = np.vstack((depth[:1,:], depth, depth[-1:,:]))
    depth = np.hstack((depth[:, :1], depth, depth[:, -1:]))

    # interpolate
    points = np.dstack((lr_i.reshape(-1), lr_j.reshape(-1)))[0]
    values = depth.reshape(-1)

    mask_values = np.zeros_like(values)
    valids = depth_is_valid(values)
    mask_values[valids] = 1
    mask = scipy.interpolate.griddata(points, mask_values, (hr_i, hr_j), method='linear')

    points = points[valids]
    values = values[valids]
    interpolated_values = scipy.interpolate.griddata(points, values, (hr_i, hr_j), method=method)
    interpolated_values[mask <= 0] = np.nan
    return interpolated_values[1:-1, 1:-1]


def angle2class(angle, num_class):
    ''' Convert continuous angle to discrete class
        [optinal] also small regression number from  
        class center angle to current angle.
      ) 
        angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
        return is class of int32 of 0,1,...,N-1 and a number such that
            class*(2pi/N) + number = angle
    '''
    angle = angle%(2*np.pi)
    assert(angle>=0 and angle<=2*np.pi)
    angle_per_class = 2*np.pi/float(num_class)
    shifted_angle = (angle+angle_per_class/2)%(2*np.pi)
    class_id = int(shifted_angle/angle_per_class)
    residual_angle = shifted_angle - (class_id*angle_per_class+angle_per_class/2)
    return class_id, residual_angle

def class2angle(pred_cls, residual, num_class, to_label_format=True):
    ''' Inverse function to angle2class '''
    angle_per_class = 2*np.pi/float(num_class)
    angle_center = pred_cls * angle_per_class
    angle = angle_center + residual
    if to_label_format and angle>np.pi:
        angle = angle - 2*np.pi
    return angle

def size2class(size, type_name):
    ''' Convert 3D box size (l,w,h) to size class and size residual '''
    size_class = type2class[type_name]
    size_residual = size - type_mean_size[type_name]
    return size_class, size_residual

def class2size(pred_cls, residual):
    ''' Inverse function to size2class '''
    mean_size = type_mean_size[class2type[pred_cls]]
    return mean_size + residual

# --------------------------------- Calibration ---------------------------------

def flip_axis_to_camera(pc):
    ''' 
        Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
        Input and output are both (N,3) array
    '''
    pc2 = np.copy(pc)
    pc2[:,[0,1,2]] = pc2[:,[0,2,1]] # cam X,Y,Z = depth X,-Z,Y
    pc2[:,1] *= -1
    return pc2

def flip_axis_to_depth(pc):
    pc2 = np.copy(pc)
    pc2[:,[0,1,2]] = pc2[:,[0,2,1]] # depth X,Y,Z = cam X,Z,-Y
    pc2[:,2] *= -1
    return pc2

def project_image_to_camera(uv_depth, K):
    n = uv_depth.shape[0]
    c_u, c_v = K[0,2], K[1,2]
    f_u, f_v = K[0,0], K[1,1]

    x = ((uv_depth[:,0] - c_u)*uv_depth[:,2])/f_u
    y = ((uv_depth[:,1] - c_v)*uv_depth[:,2])/f_v
    pts_3d_camera = np.zeros((n,3))
    pts_3d_camera[:,0] = x
    pts_3d_camera[:,1] = y
    pts_3d_camera[:,2] = uv_depth[:,2]

    return pts_3d_camera

def project_upright_depth_to_camera(pc, rtilt):
    ''' project point cloud from depth coord to camera coordinate
        Input: (N,3) Output: (N,3)
    '''
    # Project upright depth to depth coordinate
    pc2 = np.dot(np.transpose(rtilt), np.transpose(pc[:,0:3])) # (3,n)
    return flip_axis_to_camera(np.transpose(pc2))

def project_upright_depth_to_upright_camera(pc):
    return flip_axis_to_camera(pc)

def project_upright_depth_to_image(pc, rtilt, K):
    ''' Input: (N,3) Output: (N,2) UV and (N,) depth '''
    pc2 = project_upright_depth_to_camera(pc, rtilt)
    uv = np.dot(pc2, np.transpose(K)) # (n,3)
    uv[:,0] /= uv[:,2]
    uv[:,1] /= uv[:,2]
    return uv[:,0:2], pc2[:,2] 
    
def project_image_to_upright_camera(uv_depth, Rtilt, K):
    pts_3d_camera = project_image_to_camera(uv_depth, K)
    pts_3d_depth = flip_axis_to_depth(pts_3d_camera)
    pts_3d_upright_depth = np.transpose(np.dot(Rtilt, np.transpose(pts_3d_depth)))
    return project_upright_depth_to_upright_camera(pts_3d_upright_depth)

def rotate_pc_along_y(pc, rot_angle):
    ''' Input ps is NxC points with first 3 channels as XYZ
        z is facing forward, x is left ward, y is downward
    '''
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval],[sinval, cosval]])
    pc[:,[0,2]] = np.dot(pc[:,[0,2]], np.transpose(rotmat))
    return pc

# 

def get_center_view_rot_angle(angle):
        return np.pi/2.0 + angle

def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

# original
def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def z_to_disparity(z, K, baseline=.2):
    fx = K[0,0]
    return baseline * fx / z

def disparity_to_z(disparity, K, baseline=.2):
    return z_to_disparity(disparity, K, baseline)

def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc, box3d)
    return pc[box3d_roi_inds,:], box3d_roi_inds

# new, time non efficient
#def in_hull(p, hull):
#    tolerance=1e-12
#    return [(np.dot(eq[:-1], p) + eq[-1] <= tolerance)
#           for eq in hull.equations]
#
#def extract_pc_in_box3d(pc, box3d):
#    ''' pc: (N,3), box3d: (8,3) '''
#    from scipy.spatial import ConvexHull
#    hull = ConvexHull(box3d)
#    box3d_roi_inds = np.asarray([all(in_hull(i, hull)) for i in pc])
#    return pc[box3d_roi_inds,:], box3d_roi_inds

def compute_box_3d(obj, angle, rtilt, K):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in image coord.
            corners_3d: (8,3) array in in upright depth coord.
    '''
    centroid = obj['centroid']
    # compute rotational matrix around yaw axis
    R = rotz(-1*angle)
    #b,a,c = dimension
    #print R, a,b,c
    
    # 3d bounding box dimensions
    l = obj['coeffs'][1] # along heading arrow
    w = obj['coeffs'][0] # perpendicular to heading arrow
    h = obj['coeffs'][-1]
    
    # rotate and translate 3d bounding box
    x_corners = [-l,l,l,-l,-l,l,l,-l]
    y_corners = [w,w,-w,-w,w,w,-w,-w]
    z_corners = [h,h,h,h,-h,-h,-h,-h]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    
    corners_3d[0,:] += centroid[0]
    corners_3d[1,:] += centroid[1]
    corners_3d[2,:] += centroid[2]
    
    # project the 3d bounding box into the image plane
    corners_2d, _ = project_upright_depth_to_image(np.transpose(corners_3d), rtilt, K)
    corners_3d = np.transpose(corners_3d)
    
    return corners_2d, flip_axis_to_camera(corners_3d)

class SunrgbdParser(Dataset):
    def __init__(self, toolbox_root_path, path, rotate_to_center, val_indexes='val_indexes.txt', scale=None, upscale_with=None):
        self.toolbox_root_path = toolbox_root_path
        self.detaset_path = path
        self.meta = scipy.io.loadmat(toolbox_root_path + '/Metadata/SUNRGBDMeta.mat')['SUNRGBDMeta'][0]
        self.rotate_to_center = rotate_to_center
        self.image_sizes = {'kv2': [530, 730], 'kv1': [427, 561], 'realsense': [531, 681], 'xtion': [441, 591]}
        self.num_objects = {'default': 32360, 'scale_4_bicub': 31976, 'scale_4_msg': 32579, 'scale_8_bicub': 31906 , 'scale_8_msg': 32635}     
        if val_indexes is not None:
            self.val_idx = []
            with open(val_indexes) as f:
                for line in f:
                   self.val_idx.append(line[:-1])
        
        self.new_val_idx = []
        bad_idx = []
        with open('/home/gbobrovskih/bad_indexes.txt') as f:
            for line in f:
                bad_idx.append(int(line))        
        bad_idx = set(bad_idx)
        idx = set([i for i in range(1,10334)])
        self.good_idx = list(idx.difference(bad_idx))
        
        self.ds_len = {'kv2':(299 + 3485)+1, 'kv1':(554 + 1449)+1, 'realsense':(548 + 439 + 21 + 151)+1, 'xtion':(3090 + 299)+1}#self.meta.shape[0]
        self.key_to_num = {'kv2': 0, 'kv1': -3, 'realsense': -2, 'xtion': -1}
        self.scale = scale
        self.method = upscale_with       
        if scale is not None:
            def downscale(depth, K, rgb=None):
                h, w = depth.shape
                depth = np.pad(depth, ((0, scale*self.div_by - h%(scale*self.div_by)), (0, scale*self.div_by - w%(scale*self.div_by))), mode='constant', constant_values=np.nan)
                if rgb is not None:
                    rgb = np.pad(rgb, ((0, scale*self.div_by - h%(scale*self.div_by)), (0, scale*self.div_by - w%(scale*self.div_by)), (0,0)), mode='constant', constant_values=np.nan) 
                depth, actual_scale, crop_after_scale = resize_and_crop_ar_with_inpaint(depth, scale=scale, method=['box', scale], div_by=self.div_by)
                
                if upscale_with is None:
                    assert actual_scale == scale, 'actual_scale {}, scale {}'.format(actual_scale, scale)
                    assert crop_after_scale[0] == 0 and crop_after_scale[1] == 0
                if rgb is not None:
                    return depth, np.asarray(fix_calibration(K, scale=actual_scale, crop_after_scale=crop_after_scale)), rgb
                return depth, np.asarray(fix_calibration(K, scale=actual_scale, crop_after_scale=crop_after_scale))

            if upscale_with is None:
                self.div_by = 1
                upscale = lambda depth, rgb, K: (depth, K)
            elif upscale_with == 'msg':
                sys.path = ['', '/home/gbobrovskih/miniconda3/lib/python36.zip', '/home/gbobrovskih/miniconda3/lib/python3.6', '/home/gbobrovskih/miniconda3/lib/python3.6/lib-dynload', '/home/gbobrovskih/miniconda3/lib/python3.6/site-packages']
                checkpoints = {4: '/home/gbobrovskih/msg/trained_models/x4/Lap1andSin2_Nearest_03_checkpoint11',
                               8: '/home/gbobrovskih/msg/trained_models/x8/Lap1andSin2_Nearest_1.5_checkpoint204'}
                msgnet = test.Test(model=torch.load(checkpoints[scale]))
                def upscale(depth, rgb, K):
                    lr_disparity = z_to_disparity(depth, K)
                    torch.save(disparity_to_z(lr_disparity, K), 'lr_disparity.txt')
                    min_d = np.nanmin(lr_disparity)
                    max_d = np.nanmax(lr_disparity)
                    sr_disparity = msgnet.test(input_lowres_depth=lr_disparity, rgb=rgb, scale=self.scale, min_=min_d, max_=max_d)
                    return disparity_to_z(sr_disparity, K), np.asarray(fix_calibration(K, scale=1/scale))

                self.div_by = 8
            elif upscale_with == 'bicub':
                def upscale(depth, rgb, K):
                    depth = cv2.resize(depth, (depth.shape[1]*scale, depth.shape[0]*scale), interpolation=cv2.INTER_CUBIC)
                    return depth, np.asarray(fix_calibration(K, scale=1/scale))

                self.div_by = 1

            def resample(depth, rgb, K):
                if upscale_with is not None:
                    h, w = depth.shape 
                depth, K, rgb = downscale(depth, K, rgb)
                depth, K = upscale(depth, rgb, K)
                if upscale_with is not None:
                    depth = depth[:h, :w]
                return depth, K

            self.resample_depth = resample

    def save_data(self):
        sample={}
        file = h5py.File(self.detaset_path + '/SUNRGBD' + '/scale_' + str(self.scale) + '_' + str(self.method) +'.h5','w')
        image_dataset = {}
        depth_dataset = {}
        pc_dataset = {}
        for k in self.image_sizes.keys():
            image_dataset[k] = file.create_dataset('image_' + k, (self.ds_len[k], self.image_sizes[k][0], self.image_sizes[k][1], 3)) # better to use uint8
            depth_dataset[k] = file.create_dataset('depth_' + k, (self.ds_len[k], self.image_sizes[k][0], self.image_sizes[k][1]))
            pc_dataset[k] = file.create_dataset('point_cloud_' + k, (self.ds_len[k], self.image_sizes[k][0]*self.image_sizes[k][1], 3))
        count_idx = {'kv2':0, 'kv1':0, 'realsense':0, 'xtion':0}
        if self.scale and self.method is not None:
            key = 'scale_' + str(self.scale) + '_' + str(self.method)
        elif self.scale is not None:
            key = 'scale_' + str(self.scale)
        else:
            key = 'default'
        idxs = file.create_dataset('idx', (self.num_objects[key], 1)) # use int64
        frustum_indexes = file.create_dataset('frustum', (self.num_objects[key], 2048)) # same
        frustum_to_pc = file.create_dataset('indexes_pc', (self.num_objects[key], 1)) # same
        idx_to_scaner = file.create_dataset('indexes_scaner', (self.num_objects[key], 1), dtype=h5py.special_dtype(vlen=str))
        box_id = file.create_dataset('box_id', (self.num_objects[key], 1)) # same
        gt_bb2d_hdf5 = file.create_dataset('gt_bb2d', (self.num_objects[key], 4))
        gt_bb3d_hdf5 = file.create_dataset('gt_bb3d', (self.num_objects[key], 8, 3))
        box3d_center_hdf5 = file.create_dataset('box3d_center', (self.num_objects[key], 3))
        obj_type_hdf5 = file.create_dataset('obj_type', (self.num_objects[key], 1), dtype=h5py.special_dtype(vlen=str))
        frustum_angle_hdf5 = file.create_dataset('frustum_angle', (self.num_objects[key], 1))
        angle_cls_hdf5 = file.create_dataset('angle_cls', (self.num_objects[key], 1))
        angle_resid_hdf5 = file.create_dataset('angle_resid', (self.num_objects[key], 3))
        size_cls_hdf5 = file.create_dataset('size_cls', (self.num_objects[key], 1))
        size_resid_hdf5 = file.create_dataset('size_resid', (self.num_objects[key], 3))
        label_hdf5 = file.create_dataset('label', (self.num_objects[key], 2048))
        count_ind = 0
        one_more_count = 0
        for idx in self.good_idx:#range(self.meta.shape[0]):
        #--------------------------------- collecting rgb + depth data ---------------------------------
            depth_path = self.detaset_path + self.meta[idx][4][0][16:]
            img_path = self.detaset_path + self.meta[idx][5][0][16:]
            im_data = cv2.imread(img_path)
        
            if len(self.meta[idx][1]) > 0 :
                num_boxes = len(self.meta[idx][1][0])
            else:
                break
                continue
            sample['image'] = im_data
            depth = np.asarray(Image.open(depth_path), dtype=np.uint16)
            depth = np.float32(np.bitwise_or(np.right_shift(depth, 3), np.left_shift(depth, 16-3))) / 1000
            depth[depth == 0] = np.nan
            sample['K'] = np.reshape(self.meta[idx][3], (3,3), order='F').astype(np.float32)

            if self.scale is not None:
                depth, sample['K_scaled'] = self.resample_depth(depth, rgb=im_data, K=sample['K'])


            depth[depth > 8] = 8
            sample['data'] = depth
            
        # --------------------------------- point cloud creation ---------------------------------

            sample['Rtilt'] = np.reshape(self.meta[idx][2], (3,3), order='F').astype(np.float32)
            if self.scale is not None:
                point_cloud = get_pc(depth, sample['Rtilt'], sample['K_scaled']) # x, z, -y
            else:
                point_cloud = get_pc(depth, sample['Rtilt'], sample['K']) # x, z, -y
            
            key = img_path.split('/')[4]
            
            image_dataset[key][count_idx[key]] = sample['image']
            depth_dataset[key][count_idx[key]] = depth
            pc_dataset[key][count_idx[key]] = point_cloud
            count_idx[key] += 1

        # ---------------------------------- collecting target data ---------------------------------
        # gt_bb3d metadata construction
        # dtype=[('basis', 'O'), ('coeffs', 'O'), ('centroid', 'O'), ('classname', 'O'), ('labelname', 'O'), 
        # ('sequenceName', 'O'), ('orientation', 'O'), ('gtBb2D', 'O'), ('label', 'O')

            target_list = []
            pc_upright_camera = project_upright_depth_to_upright_camera(point_cloud[:,0:3]) # x, y, z
            num_boxes_left = num_boxes
            count = 0
            print('__________________________________________________________')
            print('file ', idx)
            for i in range(num_boxes):
                tmp = {}
                dat = self.meta[idx][1][0][i]
                coeffs = abs(dat[1][0]).astype(np.float32) # w, l, h
                if np.any(coeffs <= 0.00001):            
                    print('broken bounding box')
                    num_boxes_left -= 1
                    count += 1
                    continue
                else:
                    tmp['coeffs'] = coeffs

                tmp['centroid'] = dat[2][0] # in original code authors do not save this data
                tmp['gt_bb2d'] = []
                tmp['gt_bb3d'] = []            
                if len(dat[-2]) == 0:
                    print('no gt 2d bb')
                    num_boxes_left -= 1
                    count += 1
                    continue
                else:
                    xmin,ymin,xmax,ymax = dat[-2][0].astype(np.float32)
                    xmax += xmin
                    ymax += ymin
                    tmp['gt_bb2d'] = torch.from_numpy(np.asarray([xmin,ymin,xmax,ymax]))
                
                if dat[3][0] not in type2class.keys():
                    print(dat[3][0], 'unknown class')
                    num_boxes_left -= 1
                    count += 1
                    continue
                else:
                    obj_type = dat[3][0]
            
                tmp['object_type'] = obj_type
                heading_angle = -1 * np.arctan2(dat[-3][0][1], dat[-3][0][0])
                box2d_center = np.array([(xmin+xmax)/2.0, (ymin+ymax)/2.0])
                uvdepth = np.zeros((1,3))
                uvdepth[0,0:2] = box2d_center
                uvdepth[0,2] = 20 # some random depth
                box2d_center_upright_camera = project_image_to_upright_camera(uvdepth, 
                                                                          sample['Rtilt'],
                                                                          sample['K'])
                frustum_angle = -1 * np.arctan2(box2d_center_upright_camera[0,2],
                                            box2d_center_upright_camera[0,0]) # angle as to positive x-axis as in the Zoox paper
                tmp['frustum_angle'] = frustum_angle
                rot_angle = get_center_view_rot_angle(frustum_angle)

                # Get 3D box corners
                box3d_pts_2d, box3d_pts_3d = compute_box_3d(tmp, heading_angle, sample['Rtilt'], sample['K']) 
                tmp['gt_bb3d'] = box3d_pts_3d.astype(np.float32)

                # Heading correction
                if self.rotate_to_center:
                    heading_angle -= rot_angle
                tmp['angle_class'], tmp['angle_residual'] = angle2class(heading_angle, NUM_HEADING_BIN)

                # Get center point and size of 3D box

                if self.rotate_to_center:
                    box3d_center = (box3d_pts_3d[0,:] + box3d_pts_3d[6,:])/2.0
                    tmp['box3d_center'] = rotate_pc_along_y(np.expand_dims(box3d_center,0),
                                       rot_angle).squeeze()
                else:
                    tmp['box3d_center'] = (box3d_pts_3d[0,:] + box3d_pts_3d[6,:])/2.0
           
                box3d_size = np.array([2*tmp['coeffs'][1],2*tmp['coeffs'][0],2*tmp['coeffs'][-1]])
                tmp['size_class'], tmp['size_residual'] = size2class(box3d_size, obj_type)

                # labeling points for gt boxes aka segmentation

                pc_image_coord, _ = project_upright_depth_to_image(point_cloud, sample['Rtilt'], sample['K']) # x, y, z
                box_inds = (pc_image_coord[:,0]<xmax) & (pc_image_coord[:,0]>=xmin) & (pc_image_coord[:,1]<ymax) & (pc_image_coord[:,1]>=ymin)
            
                pc_in_box_fov = pc_upright_camera[box_inds,:] # x, y, z
                #print(xmax, xmin, ymax, ymin)
                if len(pc_in_box_fov) == 0:
                    print('no object in box')
                    num_boxes_left -= 1
                    count += 1
                    continue
                _, inds = extract_pc_in_box3d(pc_in_box_fov, box3d_pts_3d)
                label = np.zeros((pc_in_box_fov.shape[0])).astype(np.float32)
                label[inds] = 1.0
                num_point = pc_in_box_fov.shape[0]
                if num_point != 2048:
                    choice = np.random.choice(pc_in_box_fov.shape[0], 2048, replace = num_point < 2048)
                    pc_in_box_fov = pc_in_box_fov[choice,:]
                    label = label[choice]
                # Reject object with too few points
            
                if np.sum(label) < 5:
                    print(obj_type, 'rejecting object with too few points')
                    num_boxes_left -= 1
                    count += 1
                    continue
                else:
                    tmp['label'] = label
                print('i', i)
                print('count', count)
                print(str(one_more_count) + '_box' + str(i - count))
                if str(one_more_count) + '_box' + str(i - count) in self.val_idx:
                    print('here') 
                    print(str(one_more_count) + '_box' + str(i - count))
                    self.new_val_idx.append(str(one_more_count) + '_box' + str(i - count))
                    frustum_indexes[count_ind] = choice
                    idx_to_scaner[count_ind] = key#self.key_to_num[key]
                    frustum_to_pc[count_ind] = count_idx[key]
                    box_id[count_ind] = i
                    idxs[count_ind] = idx
                    gt_bb3d_hdf5[count_ind] = np.asarray(tmp['gt_bb3d'])
                    gt_bb2d_hdf5[count_ind] = np.asarray(tmp['gt_bb2d'])
                    box3d_center_hdf5[count_ind] = np.asarray(box3d_center)
                    obj_type_hdf5[count_ind] = tmp['object_type']
                    frustum_angle_hdf5[count_ind] = tmp['frustum_angle']
                    angle_cls_hdf5[count_ind] = tmp['angle_class']
                    angle_resid_hdf5[count_ind] = tmp['angle_residual']
                    size_cls_hdf5[count_ind] = tmp['size_class']
                    size_resid_hdf5[count_ind] = tmp['size_residual']
                    label_hdf5[count_ind] = tmp['label']
                    print(count_ind)
                    count_ind += 1
            one_more_count += 1 
            print('number of boxes left {}/{}'.format(num_boxes_left, num_boxes))
        file.close() 

        file = open('new_val_indexes.txt', 'w')
        print('writting to file')
        for val_file in self.new_val_idx:
            file.write("%s\n" % val_file)
        file.close()
        # ------------------------------------ END ------------------------------------

class DatasetStarter(Dataset):
    def __init__(self, dataset, first_batch_i):
        self.dataset = dataset
        self.first_batch_i = first_batch_i
        
    def __len__(self):
        return len(self.dataset) - self.first_batch_i
    
    def __getitem__(self, idx):
        return self.dataset[self.first_batch_i + idx]


    
if __name__ == "__main__":
    dataset = SunrgbdParser(toolbox_root_path='/home/gbobrovskih/SUNRGBDtoolbox', path='/home/gbobrovskih', rotate_to_center=True)  
    dataset.save_data()
