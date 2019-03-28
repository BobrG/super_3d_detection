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

NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 10
NUM_CLASS = 10

type2class={'bed':0, 'table':1, 'sofa':2, 'chair':3, 'toilet':4, 'desk':5, 'dresser':6, 'night_stand':7, 'bookshelf':8, 'bathtub':9}
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

def get_pc(depthmap, Rtilt, K):
        rows, cols = depthmap.shape
        cx, cy = K[0,2], K[1,2]
        fx, fy = K[0,0], K[1,1]
        c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
        # c = torch.FloatTensor(c)
        # r = torch.FloatTensor(r)
        c = c + .5
        r = r + .5
        
        positive = depthmap > 0.
        finite = np.isfinite(depthmap)
        valid = np.logical_and(positive, finite)

        x = np.where(valid, (c - cx) / fx, 0)
        y = np.where(valid, (r - cy) / fy, 0)
        z = np.ones_like(depthmap)
        
        to_pixels_in_world = np.dstack((x, y, z))
        to_pixels_in_world /= np.linalg.norm(to_pixels_in_world, axis=-1)[..., None]
        to_pixels_in_world[np.logical_not(valid), 2] = np.nan
        pts_3d_matrix = torch.from_numpy(to_pixels_in_world)*depthmap[..., None] 
        res = flip_axis_to_depth(pts_3d_matrix)

        res = np.transpose(np.dot(np.transpose(Rtilt),np.transpose(pts_3d_matrix.reshape(-1, 3))))
        
        return res

def get_center_view_rot_angle(angle):
        return np.pi/2.0 + angle

def correct_pc(pc, rotate_to_center, npoints):
    if rotate_to_center:
        # Input ps is NxC points with first 3 channels as XYZ
        # z is facing forward, x is left ward, y is downward
        new_pc = rotate_pc_along_y(np.copy(pc),
                                        get_center_view_rot_angle(frustum_angle))
    else:
        new_pc = np.copy(pc)
    # Resample point cloud
    choice = np.random.choice(new_pc.shape[0], npoints, replace=True)

    return new_pc[choice, :]

def angle2class(angle, num_class):
    ''' Convert continuous angle to discrete class
        [optinal] also small regression number from  
        class center angle to current angle.
       
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

def rotate_pc_along_y(pc, rot_angle):
    ''' Input ps is NxC points with first 3 channels as XYZ
        z is facing forward, x is left ward, y is downward
    '''
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval],[sinval, cosval]])
    pc[:,[0,2]] = np.dot(pc[:,[0,2]], np.transpose(rotmat))
    return pc

def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:,0:3], box3d)
    return pc[box3d_roi_inds,:], box3d_roi_inds

def compute_box_3d(obj, angle):
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
    corners_2d, _ = project_upright_depth_to_image(np.transpose(corners_3d))
    corners_3d = np.transpose(corners_3d)
    
    return corners_2d, flip_axis_to_camera(corners_3d)

def get_center_view_box3d_center(box3d):
    box3d_center = (box3d[0,:] + box3d[index][6,:])/2.0
    return rotate_pc_along_y(np.expand_dims(box3d_center,0), get_center_view_rot_angle(index)).squeeze()

class SUNRGBD(Dataset):
    def __init__(self, toolbox_root_path, npoints, rotate_to_center):
        self.toolbox_root_path = toolbox_root_path
        self.meta = scipy.io.loadmat('Metadata/SUNRGBDMeta.mat')['SUNRGBDMeta'][0]
        self.ds_len = self.meta.shape[0]
        self.rotate_to_center = rotate_to_center
        self.npoints = npoints

    def __len__(self):
        return self.ds_len

    def __getitem__(self, idx):
        sample = {}

        # --------------------------------- collecting rgb + depth data ---------------------------------

        depth_path = '..' + self.meta[idx][4][0][16:]
        img_path = '..' + self.meta[idx][5][0][16:]
        sample['rgb'] = cv2.imread(img_path)

        depth = cv2.imread(depth_path, 0).astype(np.uint16)
        depth = np.float32(np.bitwise_or(np.right_shift(depth, 3), np.left_shift(depth, 16-3))) / 1000
        depth[depth == 0] = np.nan
        depth[depth > 8] = np.nan
        sample['data'] = depth

        # --------------------------------- point cloud creation ---------------------------------

        sample['K'] = self.meta[idx][3]
        sample['Rtilt'] = self.meta[idx][2] 
        point_cloud = get_pc(depth, sample['Rtilt'], sample['K'])
        
        # ---------------------------------- collecting target data ---------------------------------
        # gt_bb3d metadata construction
        # dtype=[('basis', 'O'), ('coeffs', 'O'), ('centroid', 'O'), ('classname', 'O'), ('labelname', 'O'), 
        # ('sequenceName', 'O'), ('orientation', 'O'), ('gtBb2D', 'O'), ('label', 'O')

        sample['target'] = []
        pc_upright_camera = self.project_upright_depth_to_upright_camera(point_cloud[:,0:3])

        for dat in self.meta[idx][1][0]:
            tmp = {}

            tmp['coeffs'] = dat[1] # w, l, h
            tmp['centroid'] = dat[3] # in original code authors do not save this data
            # tmp['basis'] = dat[0]
            # tmp['orientation'] = dat[-3]
            tmp['gt_bb2d'] = dat[-2] # xmin,ymin,xmax,ymax
            obj_type = dat[4]
            heading_angle = -1 * np.arctan2(dat[-3][1], dat[-3][0])
            
            # Get frustum angle (according to center pixel in 2D BOX)
            # TODO: understand why authors recreate a point cloud from random depth
            
            box2d_center = np.array([(xmin+xmax)/2.0, (ymin+ymax)/2.0])
            uvdepth = np.zeros((1,3))
            uvdepth[0,0:2] = box2d_center
            uvdepth[0,2] = 20 # some random depth
            box2d_center_upright_camera = self.project_image_to_upright_camera(uvdepth, 
                                                                               sample['Rtilt'],
                                                                               sample['K'])
            frustum_angle = -1 * np.arctan2(box2d_center_upright_camera[0,2],
                                            box2d_center_upright_camera[0,0]) # angle as to positive x-axis as in the Zoox paper
            tmp['frustum_angle'] = frustum_angle
            rot_angle = get_center_view_rot_angle(frustum_angle)

            # Get 3D box corners

            box3d_pts_2d, box3d_pts_3d = compute_box_3d(tmp, heading_angle) 
            tmp['gt_bb3d'] = box3d_pts_3d

            # Heading correction
            if self.rotate_to_center:
                heading_angle -= - rot_angle
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

            box_inds = (point_cloud[:,0]<xmax) & (point_cloud[:,0]>=xmin) & (point_cloud[:,1]<ymax) & (point_cloud[:,1]>=ymin)
            pc_in_box_fov = pc_upright_camera[box_fov_inds,:]

            _, inds = extract_pc_in_box3d(pc_in_box_fov, box3d_pts_3d)
            label = np.zeros((pc_in_box_fov.shape[0]))
            label[inds] = 1
            num_point = pc_in_box_fov.shape[0]

            if num_point > 2048:
                choice = np.random.choice(pc_in_box_fov.shape[0], 2048, replace=False)
                pc_in_box_fov = pc_in_box_fov[choice,:]
                label = label[choice]
            # Reject object with too few points
            if np.sum(label) < 5:
                continue
            tmp['label'] = label

            sample['target'].append(tmp)

        # ------------------------------------ END ------------------------------------

        return sample

    # --------------------------------- Calibration ---------------------------------

    def flip_axis_to_camera(self, pc):
        ''' 
            Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
            Input and output are both (N,3) array
        '''
        pc2 = np.copy(pc)
        pc2[:,[0,1,2]] = pc2[:,[0,2,1]] # cam X,Y,Z = depth X,-Z,Y
        pc2[:,1] *= -1
        return pc2

    def flip_axis_to_depth(self, pc):
        pc2 = np.copy(pc)
        pc2[:,[0,1,2]] = pc2[:,[0,2,1]] # depth X,Y,Z = cam X,Z,-Y
        pc2[:,2] *= -1
        return pc2

    def project_image_to_camera(self, uv_depth, K):
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

    def project_upright_depth_to_upright_camera(self, pc):
        return self.flip_axis_to_camera(pc)

    def project_upright_depth_to_image(pc, rtilt):
        ''' 
            project point cloud from depth coord to camera coordinate
                Input: (N,3) Output: (N,3)
        '''
        # Project upright depth to depth coordinate
        pc2 = np.dot(np.transpose(rtilt), np.transpose(pc[:,0:3])) # (3,n)
        return self.flip_axis_to_camera(np.transpose(pc2)) 
        
    def project_image_to_upright_camera(self, uv_depth, Rtilt, K):
        pts_3d_camera = self.project_image_to_camera(uv_depth, K)
        pts_3d_depth = self.flip_axis_to_depth(pts_3d_camera)
        pts_3d_upright_depth = np.transpose(np.dot(Rtilt, np.transpose(pts_3d_depth)))
        return self.project_upright_depth_to_upright_camera(pts_3d_upright_depth)
