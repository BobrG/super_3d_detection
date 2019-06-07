import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models 
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sunrgbd_dataset import get_pc, SUNRGBD
from utils_1 import masking, correct_pc, extract_objects, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, NUM_CLASS, g_mean_size_arr
from components_1 import T_Net, Box_Estimation_Net, Seg_PointNet
#from faster_rcnn_pytorch.lib.model.faster_rcnn.vgg16 import vgg16
#from faster_rcnn_pytorch.lib.model.faster_rcnn.resnet import resnet

# from easy_ssd.model import SSD300, MultiBoxLoss

import cv2

class object_detection_model(nn.Module):
    def __init__(self, num_points, seg_net, tnet, box_est, min_score=0.2, max_overlap=0.5, top_k=200):
        super(object_detection_model, self).__init__()
        self.num_points = int(num_points)
        #self.cnn2d = cnn2d # pass initialized model class
        self.min_score = min_score
        self.max_overlap = max_overlap
        self.top_k = top_k
        self.seg_net = nn.ModuleList(seg_net(self.num_points))
        self.tnet = nn.ModuleList(tnet(self.num_points))
        self.box_est = nn.ModuleList(box_est(self.num_points, NUM_SIZE_CLUSTER, NUM_HEADING_BIN))

    def forward(self, point_cloud, sample=None, one_hot=False):
        output = {}
        batch_size = point_cloud.shape[0]

        if one_hot:
            onehot_vec = make_oh_vector()
        else:
            onehot_vec = None 

        # 3D Instance Segmentation PointNet
        # input batch x num_points x 3
        # output 
        mlp_output = self.seg_net[0](point_cloud) # mlp1, torch.Size([1, 64, num_points])
        global_feature = self.seg_net[2](self.seg_net[1](mlp_output)).squeeze(2) # mlp2 -> maxpool, torch.Size([B, 1024])
        global_feature_repeated = global_feature.unsqueeze(-1).repeat(1, 1, self.num_points) # torch.Size([B, 1024, num_points])

        if onehot_vec is not None:
            output_scores = self.seg_net[-1](
                                                torch.cat([onehot_vec, mlp_output, global_feature_repeated], 1) # !!! 
                                            ) 
        else:
            output_scores = self.seg_net[-1](
                                                torch.cat([mlp_output, global_feature_repeated], 1) # torch.Size([B, 1088, num_points])
                                            )
        # concatenation -> mlp3
        output['mask_logits'] = output_scores # num_points x (1088 + k)

        # Masking
        # select masked points and translate to masked points' centroid
        ### NOMASK mask, object_point_cloud_xyz, mask_xyz_mean = masking(point_cloud.permute(0, 2, 1), output_scores.permute(0, 2, 1))
        object_point_cloud_xyz, mask_xyz_mean, mask = masking(point_cloud, output_scores)
        ### NOMASK output['mask'] = mask

        # T-Net 
        # input batch x m_points x 4
        # output 
        output_mlp = self.tnet[0](object_point_cloud_xyz)
        #
        output_mlp = output_mlp * mask
        #
        m = output_mlp.shape[-1]
        max_pool = nn.MaxPool1d(m)
        if onehot_vec is not None:
            global_feature = torch.cat([max_pool(output_mlp).squeeze(2), onehot_vec], 1)
        else:
            global_feature = max_pool(output_mlp).squeeze(2)
        
        residual_center = self.tnet[-1](global_feature)
        
        # coordinate translation
#        print(residual_center.shape, mask_xyz_mean[0].shape)
        center_1 = residual_center + mask_xyz_mean.squeeze(-1) # Bx3 C_mask + delta_C_tnet
        object_point_cloud_xyz_new = object_point_cloud_xyz - residual_center[..., None] # B, C, num_points
        output['center_1'] = center_1

        # Amodel Box Estimation PointNet
        output_mlp = self.box_est[0](object_point_cloud_xyz_new)
        output_mlp = output_mlp * mask
        if onehot_vec is not None:
            global_features = torch.cat([self.box_est[1](output_mlp).squeeze(2), onehot_vec], 1) # !!!
        else:
            global_features = self.box_est[1](output_mlp).squeeze(2)

        box_parameters = self.box_est[-1](global_features)

        # Parse output to 3D box parameters
        center = box_parameters[:,:3]
#        output['box3d_center'] = center
        output['center_prediction'] = center + center_1 # Bx3 (C_mask + delta_C_tnet) + delta_C_boxnet

        heading_scores = box_parameters[:, 3:NUM_HEADING_BIN + 3]
        heading_residuals_normalized = box_parameters[:, (3+NUM_HEADING_BIN):(3+NUM_HEADING_BIN) + NUM_HEADING_BIN]
        output['heading_scores'] = heading_scores # BxNUM_HEADING_BIN
        output['heading_residuals_normalized'] = heading_residuals_normalized # BxNUM_HEADING_BIN (-1 to 1)
        output['heading_residuals'] = heading_residuals_normalized * (np.pi/NUM_HEADING_BIN) # BxNUM_HEADING_BIN

        size_scores = box_parameters[:, (3+NUM_HEADING_BIN*2):(3+NUM_HEADING_BIN*2) + NUM_SIZE_CLUSTER] # BxNUM_SIZE_CLUSTER
        size_residuals_normalized = box_parameters[:, (3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER):(3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER) + NUM_SIZE_CLUSTER*3]
        size_residuals_normalized = size_residuals_normalized.view(batch_size, NUM_SIZE_CLUSTER, 3) # BxNUM_SIZE_CLUSTERx3
        output['size_scores'] = size_scores
        output['size_residuals_normalized'] = size_residuals_normalized
        g_mean_size_arr_pt = torch.FloatTensor(g_mean_size_arr).to(size_residuals_normalized.device)
        output['size_residuals'] = size_residuals_normalized * g_mean_size_arr_pt

#        print(output['box3d_center'].shape, center_1.shape)
        #output['center_prediction'] = output['box3d_center'] + center_1 # Bx3 (C_mask + delta_C_tnet) + delta_C_boxnet
        
        return output

if __name__ == "__main__":
    num_points = 10e4
    net = object_detection_model(num_points, vgg16(n_class=NUM_CLASS, backbone='vgg16').create_architecture(), Seg_PointNet, T_Net, Box_Estimation_Net)
    net.train()
    # sample = {}
    # sample['image'] = torch.FloatTensor(cv2.imread('/Users/bobrg/Downloads/sunrgbd/image/NYU1001.jpg')).permute(2, 0, 1).unsqueeze(0)
    # sample['data'] = torch.FloatTensor(cv2.imread('/Users/bobrg/Downloads/sunrgbd/depth/NYU1000.png', 0)).unsqueeze(0)
    # print(sample['image'].shape)
    # intr = np.reshape(
    #     np.array([float(x) for x in open('/Users/bobrg/Downloads/sunrgbd/intrinsics.txt').read().split(' ')[:-1]]),
    #     (3,3), order='F')
    # extr = np.array([[float(x) for x in i.split(' ')] for i in open('/Users/bobrg/Downloads/sunrgbd/20150118235401.txt').read().splitlines()])[:, :3]
    # sample['K'] = torch.FloatTensor(intr).unsqueeze(0)
    # sample['Rtilt'] = torch.FloatTensor(extr).unsqueeze(0)
    dataset = SUNRGBD(toolbox_root_path='/home/gbobrovskih/datasets/SUNRGBD/SUNRGBDtoolbox', npoints=10000, rotate_to_center=True)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size())
        net(sample_batched['image'], sample_batched['data'], sample_batched)
