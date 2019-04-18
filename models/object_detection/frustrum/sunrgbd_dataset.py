import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from glob import glob
import numpy as np
import scipy.misc as m
import cv2
import os

from PIL import Image
import numpy as np
import scipy.io


original_print = print
def print(*args, **kwargs):
    pass

def new_print(*args, **kwargs):
    original_print(*args, **kwargs)


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

def get_pc(depthmap, Rtilt, K):
        print(depthmap)
        rows, cols = depthmap.shape
        cx, cy = K[0,2], K[1,2]
        fx, fy = K[0,0], K[1,1]
        print(cx, cy, fx, fy)
        c, r = np.meshgrid(np.arange(1, cols+1), np.arange(1, rows+1), sparse=True)

        valid = depthmap > 0.

        z = np.where(valid, depthmap, np.nan)
        x = (c - cx) / fx * z
        y = (r - cy) / fy * z
        
        to_pixels_in_world = np.dstack((x, z, -1*y))
        # print('to_pixels_in_world', to_pixels_in_world.shape)
        # to_pixels_in_world /= np.linalg.norm(to_pixels_in_world, axis=-1)[..., None]
        # print(to_pixels_in_world.shape)
        # to_pixels_in_world[np.logical_not(valid), 2] = np.nan
        pts_3d_matrix = to_pixels_in_world#torch.from_numpy(to_pixels_in_world)*torch.FloatTensor(depthmap[..., None])
        # print(len(np.where(~np.isnan(to_pixels_in_world))[0]))
        # print(len(np.where(~np.isnan(pts_3d_matrix))[0]))
        # print('pts_3d_matrix', pts_3d_matrix[tuple(np.argwhere(~np.isnan(pts_3d_matrix)))])
        res = np.transpose(np.matmul(Rtilt, np.transpose(np.reshape(pts_3d_matrix, (-1,3)))))
        #res = np.reshape(pts_3d_matrix, (-1,3))
        #res = torch.t(torch.mm(torch.FloatTensor(Rtilt), torch.t(torch.reshape(pts_3d_matrix, (-1,3)))))
        #res = torch.reshape(pts_3d_matrix, (-1,3))
        print(np.where(~np.isnan(res)))     
        return res

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
    # pc2[:, 0] = (pc2[:, 0]/ pc2[:, -1])*K[0,0] + K[0, 2]
    # pc2[:, 1] = (pc2[:, 0]/ pc2[:, -1])*K[1,1] + K[1, 2]
    uv[:,0] /= uv[:,2]
    uv[:,1] /= uv[:,2]
    return uv[:,0:2], pc2[:,2] 
    #return pc2[:,0:2], pc2[:, -1]
    
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
    #new_print('corners_3d', corners_3d)
    corners_3d[0,:] += centroid[0]
    corners_3d[1,:] += centroid[1]
    corners_3d[2,:] += centroid[2]
    #new_print('corners_3d', corners_3d)
    # project the 3d bounding box into the image plane
    corners_2d, _ = project_upright_depth_to_image(np.transpose(corners_3d), rtilt, K)
    corners_3d = np.transpose(corners_3d)
    
    return corners_2d, flip_axis_to_camera(corners_3d)

# def get_center_view_box3d_center(box3d):
#     box3d_center = (box3d[0,:] + box3d[index][6,:])/2.0
#     return rotate_pc_along_y(np.expand_dims(box3d_center,0), get_center_view_rot_angle(index)).squeeze()

# for faster rcnn

def im_list_to_blob(ims):
    """Convert a list of images into a network input.
    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in range(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    
    return blob

def _get_image_blob(im, im_size):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  #im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  scale = (min(im_size), )
  max_size = max(im_size)
  
  processed_ims = []
  im_scale_factors = []

  for target_size in scale: # The scale is the pixel size of an image's shortest side
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size: # Max pixel size of the longest side of a scaled input image
      im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

class SUNRGBD(Dataset):
    def __init__(self, toolbox_root_path, npoints, rotate_to_center, im_sz=(480, 640)):
        self.toolbox_root_path = toolbox_root_path
        self.meta = scipy.io.loadmat(toolbox_root_path + '/Metadata/SUNRGBDMeta.mat')['SUNRGBDMeta'][0]
        self.ds_len = self.meta.shape[0]
        self.rotate_to_center = rotate_to_center
        self.npoints = npoints
        self.im_sz = im_sz

    def __len__(self):
        return self.ds_len

    def __getitem__(self, idx):
        sample = {}
        # --------------------------------- collecting rgb + depth data ---------------------------------

        depth_path = '..' + self.meta[idx][4][0][16:]
        img_path = '..' + self.meta[idx][5][0][16:]
        print(img_path)
        im = cv2.imread(img_path)[:, :, ::-1] # rgb -> bgr
        im = m.imresize(im, (self.im_sz[0], self.im_sz[1]))
        im_blob, im_scales = _get_image_blob(im, self.im_sz)
        im_data = torch.from_numpy(im_blob).permute(0, 3, 1, 2)
        im_info = torch.from_numpy(np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32))
        
        sample['image'] = im_data
        sample['im_info'] = im_info
        if len(self.meta[idx][1]) > 0 :
            sample['num_boxes'] = len(self.meta[idx][1][0])
        else:
            sample['num_boxes'] = 0

        #depth = cv2.imread(depth_path, 0).astype(np.uint16)
        depth = np.asarray(Image.open(depth_path), dtype=np.uint16)
        depth = np.float32(np.bitwise_or(np.right_shift(depth, 3), np.left_shift(depth, 16-3))) / 1000
        depth[depth == 0] = np.nan
        depth[depth > 8] = 8
        sample['data'] = m.imresize(depth, (self.im_sz[0], self.im_sz[1]))

        # --------------------------------- point cloud creation ---------------------------------
        sample['K'] = np.reshape(self.meta[idx][3], (3,3), order='F').astype(np.float32)
        sample['Rtilt'] = np.reshape(self.meta[idx][2], (3,3), order='F').astype(np.float32)
        point_cloud = get_pc(depth, sample['Rtilt'], sample['K'])
        
        # ---------------------------------- collecting target data ---------------------------------
        # gt_bb3d metadata construction
        # dtype=[('basis', 'O'), ('coeffs', 'O'), ('centroid', 'O'), ('classname', 'O'), ('labelname', 'O'), 
        # ('sequenceName', 'O'), ('orientation', 'O'), ('gtBb2D', 'O'), ('label', 'O')

        sample['target'] = []
        tmp = {'coeffs':[], 'centroid':0.0, 'gt_bb2d':[], 'gt_bb3d':[], 'object_type':'none',
              'frustum_angle':0.0, 'angle_class':0.0, 'angle_residual':0.0,
              'size_class':0.0, 'size_residual':0.0, 'label':[]}
        pc_upright_camera = project_upright_depth_to_upright_camera(point_cloud[:,0:3])

        if sample['num_boxes'] == 0:
            # sample['target'] = [{'gt_bb2d': [], 'gt_bb3d': []}]
            return sample

        for i in range(sample['num_boxes']):
            dat = self.meta[idx][1][0][i]
            
            coeffs = abs(dat[1][0]).astype(np.float32) # w, l, h
            if np.any(coeffs <= 0.00001):            
                print('broken bounding box')
                continue
            else:
                tmp['coeffs'] = coeffs

            tmp['centroid'] = dat[2][0] # in original code authors do not save this data
            tmp['gt_bb2d'] = []
            tmp['gt_bb3d'] = []            
            # tmp['basis'] = dat[0]
            # tmp['orientation'] = dat[-3]
            if len(dat[-2]) == 0:
                print('no gt 2d bb')
                continue
            else:
                print(dat[-2])
                xmin,ymin,xmax,ymax = dat[-2][0].astype(np.float32)
                xmax += xmin
                ymax += ymin
                #new_print(xmin,ymin,xmax,ymax)
                tmp['gt_bb2d'] = [xmin,ymin,xmax,ymax]

            if len(dat[3]) == 0:
                print('no class')
                tmp['gt_bb2d'] = []
                continue

            if dat[3][0] not in type2class.keys():
                print('unknown class')
                tmp['gt_bb2d'] = []
                continue
            else:
                obj_type = dat[3][0]
            
            tmp['object_type'] = obj_type
            print(obj_type)
            print('xmin,ymin,xmax,ymax', xmin,ymin,xmax,ymax)
            heading_angle = -1 * np.arctan2(dat[-3][0][1], dat[-3][0][0])
            
            # Get frustum angle (according to center pixel in 2D BOX)
            # TODO: understand why authors recreate a point cloud from random depth
            # TODO: decide where to compute frustum angle
            
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

            pc_image_coord, _ = project_upright_depth_to_image(point_cloud, sample['Rtilt'], sample['K'])
            
            # bb2d_depth = depth[xmin:xmax, ymin:ymax]
            # print('depth shape', depth.shape)
            # print('bb2d_depth shape', bb2d_depth.shape)
            # pc_in_box_fov = get_pc(bb2d_depth, sample['Rtilt'], sample['K'])
            box_inds = (pc_image_coord[:,0]<xmax) & (pc_image_coord[:,0]>=xmin) & (pc_image_coord[:,1]<ymax) & (pc_image_coord[:,1]>=ymin)
            pc_in_box_fov = pc_upright_camera[box_inds,:]
            #new_print('pc_in_box_fov',pc_in_box_fov)
            #new_print('box3d_pts_3d', box3d_pts_3d)
            if len(pc_in_box_fov) == 0:
                tmp['gt_bb3d'] = []
                continue
            print('box_inds {}; trues N {}'.format(box_inds, box_inds.sum()))
            print('pc in box fov', pc_in_box_fov)
            print('box3d pts 3d ', box3d_pts_3d)
            print('not nans in pc in box fov', len(np.where(~np.isnan(pc_in_box_fov))[0]))
            _, inds = extract_pc_in_box3d(pc_in_box_fov, box3d_pts_3d)
            label = np.zeros((pc_in_box_fov.shape[0])).astype(np.float32)
            print('labels shape', label.shape)
            print('inds:', inds[inds == True])
            print('label sum', np.sum(label))
            label[inds] = 1.0
            num_point = pc_in_box_fov.shape[0]
            print('num points:', num_point)
            if num_point != 2048:
                choice = np.random.choice(pc_in_box_fov.shape[0], 2048, replace = num_point < 2048)
                print('choice:', choice)
                pc_in_box_fov = pc_in_box_fov[choice,:]
                label = label[choice]
            # Reject object with too few points
            print(label)
            if np.sum(label) < 5:
                print(obj_type, 'rejecting object with too few points')
                continue
            else:
                print(obj_type, 'has enough points')
                tmp['label'] = label
                        
            sample['target'].append(tmp)
            tmp = {}
        
        # ------------------------------------ END ------------------------------------

        return sample

class DatasetStarter(Dataset):
    def __init__(self, dataset, first_batch_i):
        self.dataset = dataset
        self.first_batch_i = first_batch_i
        
    def __len__(self):
        return len(self.dataset) - self.first_batch_i
    
    def __getitem__(self, idx):
        return self.dataset[self.first_batch_i + idx]


    
if __name__ == "__main__":
    dataset_0 = SUNRGBD(toolbox_root_path='/home/gbobrovskih/datasets/SUNRGBD/SUNRGBDtoolbox', npoints=10000, rotate_to_center=True)
    dataset = DatasetStarter(dataset_0, 1800)
    batch_size=4
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    #loader = list(enumerate(dataloader))
    try:
        for i_batch, sample_batched in enumerate(dataloader):
            #print(i_batch, sample_batched['image'].size())
            #print(sample_batched['target'])
            new_print('batch', i_batch)
            for sample_i, target in enumerate(sample_batched['target']):
                if len(target) == 0:
                    new_print('Broken empty target in batch {}, sample {}'.format(i_batch, sample_i))
                    continue
                if len(target['gt_bb3d']) == 0:
                    new_print('OK empty target in batch {}, sample {}'.format(i_batch, sample_i))
                    continue
    except ValueError as e:
        new_print('ValueError in batch {}, sample {}'.format(i_batch, sample_i))
        raise e
    new_print('Finished')
