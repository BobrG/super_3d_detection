import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import cv2
import os

'''
Calibration
    Initialized with:
        : intr_path - path to .txt file with intrinsic parameters of camera;
        : extr_path - path to .txt file with extrinsic parameters of camera;
    Function ...

'''

class Calibration():
    def __init__(self, intr_path, extr_path, name=None):
        line = [tmp.rstrip() for tmp in open(intr_path)][0]
        lines = [tmp.rstrip() for tmp in open(extr_path)]
        intr = np.array([float(x) for x in line.split(' ')])
        extr = np.array([[float(x) for x in i.split(' ')] for i in lines])
        self.Rot = np.reshape(extr[:3, :3], (3,3), order='F')
        self.t = np.reshape(extr[:3, -1], (3,1), order='F')
        self.K = np.reshape(intr[2:], (3,3), order='F')
        self.width, self.height = intr[0], intr[1]
        self.fx, self.fy = intr[2], intr[3]
        self.cx, self.cy = intr[4], intr[5]
        self.name = name

    def flip_axis_to_camera(self, pc):
        ''' Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
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

    def project_upright_depth_to_camera(self, pc):
        ''' project point cloud from depth coord to camera coordinate
            Input: (N,3) Output: (N,3)
        '''
        # Project upright depth to depth coordinate
        pc2 = np.dot(np.transpose(self.Rot), np.transpose(pc[:,0:3])) # (3,n)
        return self.flip_axis_to_camera(np.transpose(pc2))

    def pc_from_depth(self, depthmap):
        rows, cols = depthmap.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
        c = c + .5
        r = r + .5
        
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
        res = np.transpose(np.dot(self.Rot, pts_3d_matrix.reshape(-1, 3)))

        return res

'''
Matterport3D Loader
    The main dataset resides in the "data" directory. It contains several subdirectory for every house, 
    which is named by a unique string (e.g., "1pXnuDYAj8r"). Within each house directory, there are separate 
    directories for different types of data.
        : matterport_color_images = color images provided by matterport,
          tone-mapped color images in jpg format;
          
        : matterport_depth_images = depth images provided by matterport,
          reprojected depth images aligned with the color images,
          every depth image is a 16 bit PNG containing the pixel's distance
          in the z-direction from the camera center;

        : matterport_camera_intrinsics = camera intrinsics provided by matterport,
          intrinsic parameters for every camera, stored as ASCII 
          in the following format: width height fx fy cx cy k1 k2 p1 p2 k3,
          the Matterport camera axis convention is: x-axis: right y-axis: down z-axis: "look";

        : matterport_camera_poses = camera extrinsics provided by matterport,
          camera pose files contain a 4x4 matrix that transforms column vectors
          from camera to global coordinates: 3x3 rotation matrix in the left-block,
          3x1 translation column-vector in the right, extra row of (0,0,0,1) added to the bottom;

    in addition
        : house_segmentations = manually specified semantic segmentations of houses into regions,
          a manually specified decomposition of a house into levels, room-like regions,
          and objects with semantic labels;

        : region_segmentations = manually specified semantic segmentations of regions into objects;
          a  set of manually specified segment, object instance, and semantic category labels for
          walls, floors, ceilings, doors, windows, and "furniture-sized" objects for each region
          of each house;

    file name constraction is as following
        : <panorama_uuid>_<imgtype><camera_index>_<yaw_index>.<extension>
          where <panorama_uuid> is a unique string, <camera_index> is [0-5], and <yaw_index> is [0-2],
          <imgtype> is 'd' for depth images, 'i' for tone-mapped color images, 
          "intrinsics" for camera intrinsics files;
    
for more details: https://github.com/niessner/Matterport/blob/master/data_organization.md
'''

class Matterport3DLoader(Dataset):
    def __init__(self, root_dir, house_id='17DRP5sb8fy', transforms=None):
        self.house_id = house_id
        self.dir = os.path.join(root_dir, self.house_id)
        self.depth_path = os.path.join(self.dir, 'matterport_depth_images')
        self.img_path = os.path.join(self.dir, 'matterport_color_images')
        self.intr_path = os.path.join(self.dir, 'matterport_camera_intrinsics')
        self.extr_path = os.path.join(self.dir, 'matterport_camera_poses')
        self.seg_path = os.path.join(self.dir, 'house_segmentations')
        self.img_files = np.asarray([i for i in os.listdir(self.img_path)])
        self.depth_files = np.asarray([i for i in os.listdir(self.depth_path)]) 
        self.extr_files = np.asarray([i for i in os.listdir(self.extr_path)])
        self.intr_files = np.asarray([i for i in os.listdir(self.intr_path)])
        self.transform = transforms
        self.calib = []

    def __len__(self):
        return len(self.img_files)

    def get_calib(self):
        return self.calib
    
    def __getitem__(self, idx):
        name = self.img_files[idx].split('_')[0]
        camera_yaw = self.img_files[idx][len(name) + 2:-4]
        img = cv2.imread(os.path.join(self.img_path, self.img_files[idx]))
        depth = cv2.imread(os.path.join(self.depth_path, name + '_d' + camera_yaw + '.png'), 0).astype(float)
        self.calib.append(Calibration(os.path.join(self.intr_path, name + '_intrinsics_' + camera_yaw[0] + '.txt'),
                            os.path.join(self.extr_path, name + '_pose_' + camera_yaw + '.txt'), name=self.img_files[idx]))
        # f = open(os.path.join(self.seg_path, self.house_id + '.house'))
        # flag = 0
        # bb_3d = []
        # name = self.img_files[idx].split('.')[0].split('_')[0]
        # for line in f:
        #     if ('I' in line and name in line):
        #         flag = 1
        #         bb_3d.append(line.split(' ')) 
        #     elif (name not in line and flag == 1):
        #         break

        # rgbd = np.concatenate((img, depth), axis=2)
        # bounding box gt ???
        # bb_gt = 
        # super resolution gt ???
        sample = {'image': img, 'data': depth, 'name': self.img_files[idx]} #'bb_gt': bb_gt, 'calibration': calib}

        if self.transform:
            sample = self.transform(sample)

        return sample

if __name__ == "__main__":
    dataset = Matterport3DLoader(root_dir='/Users/bobrg/Downloads')
    # for i in range(len(dataset)):
    #     sample = dataset[i]

    #     print(i, sample['idx'], sample['image'].shape, sample['data'].shape)
    batch_size=4
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(),
          sample_batched['data'].size(), sample_batched['name'])
        print(dataloader.dataset.get_calib()[batch_size*i_batch:])
        for i, _ in enumerate(dataloader.dataset.get_calib()[batch_size*i_batch:]):
            print(_.pc_from_depth(sample_batched['data'][i]))