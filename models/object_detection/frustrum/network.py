import torch
from torch import nn
from torch.nn import functional as F
from tochvision import models 
import numpy as np
from utils import masking, NUM_HEADING_BIN, NUM_SIZE_CLUSTER
from components import T_Net, Box_Estimation_Net, Detector2D, Seg_PointNet

class object_detection_model(nn.Module):
    def __init__(self, num_points, cnn2d, seg_net, tnet, box_est):
        self.num_points = num_points
        self.cnn2d = nn.ModuleList(cnn2d)
        self.seg_net = nn.ModuleList(seg_net(self.num_points))
        self.tnet = nn.ModuleList(tnet(self.num_points))
        self.box_est = nn.ModuleList(box_est(self.num_points, NUM_SIZE_CLUSTER, NUM_HEADING_BIN))

    def forward(self, rgb, data):
        output = {}
        batch_size = data.shape[0]
        # 2D detection
        out_cnn = self.cnn2d(rgb)
        onehot_vec = #?
        point_cloud = region2frustrum(out_cnn, data)

        # 3D Instance Segmentation PointNet
        mlp_output = self.seg_net[0](frustrum) # mlp1
        global_feature = self.seg_net[2](self.seg_net[1](mlp_output)).squeeze(2) # mlp2 -> maxpool
        output_scores = self.seg_net[-1](torch.cat([onehot_vec, mlp_output, global_feature], 1)) # concatenation -> mlp3
        output['mask_logits'] = output_scores

        # Masking
        # select masked points and translate to masked points' centroid
        mask, object_point_cloud_xyz, mask_xyz_mean = masking(point_cloud, output_scores)
        output['mask'] = mask

        # T-Net 
        output_mlp = self.tnet[0](object_point_cloud_xyz)
        global_feature = torch.cat([self.tnet[1](output_mlp).squeeze(2), onehot_vec], 1)
        residual_center = self.tnet[-1](global_feature)

        # coordinate translation
        center_1 = residual_center + mask_xyz_mean # Bx3 C_mask + delta_C_tnet
        object_point_cloud_xyz_new = object_point_cloud_xyz - residual_center.unsqueeze(1)
        output['center_1'] = center_1

        # Amodel Box Estimation PointNet
        output_mlp = self.box_est[0](object_point_cloud_xyz_new)
        global_features = torch.cat([self.box_est[1](output_mlp).squeeze(2), onehot_vec], 1)
        box_parameters = self.box_est[-1](global_features)

        # Parse output to 3D box parameters
        center = output[:,:3]
        output['center_boxnet'] = center

        heading_scores = output[:, 3:NUM_HEADING_BIN]
        heading_residuals_normalized = output[:, 3+NUM_HEADING_BIN:NUM_HEADING_BIN]
        output['heading_scores'] = heading_scores # BxNUM_HEADING_BIN
        output['heading_residuals_normalized'] = heading_residuals_normalized # BxNUM_HEADING_BIN (-1 to 1)
        output['heading_residuals'] = heading_residuals_normalized * (np.pi/NUM_HEADING_BIN) # BxNUM_HEADING_BIN

        size_scores = box_parameters[:, 3+NUM_HEADING_BIN*2:NUM_SIZE_CLUSTER] # BxNUM_SIZE_CLUSTER
        size_residuals_normalized = output[:, 3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER:NUM_SIZE_CLUSTER*3]
        size_residuals_normalized = size_residuals_normalized.view(batch_size, NUM_SIZE_CLUSTER, 3) # BxNUM_SIZE_CLUSTERx3
        output['size_scores'] = size_scores
        output['size_residuals_normalized'] = size_residuals_normalized
        output['size_residuals'] = size_residuals_normalized * torch.from_numpy(g_mean_size_arr).unsqueeze(0)

        output['center_prediction'] = output['center_boxnet'] + center_1 # Bx3 (C_mask + delta_C_tnet) + delta_C_boxnet
        
        return output
