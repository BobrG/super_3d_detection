import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models 
import numpy as np
from utils import masking, NUM_HEADING_BIN, NUM_SIZE_CLUSTER
from components import T_Net, Box_Estimation_Net, Seg_PointNet
from easy_detection_2d.model import SSD300, MultiBoxLoss

class object_detection_model(nn.Module):
    def __init__(self, num_points, cnn2d, seg_net, tnet, box_est):
        super(object_detection_model, self).__init__()
        self.num_points = num_points
        self.cnn2d = cnn2d # pass initialized model class
        self.seg_net = nn.ModuleList(seg_net(self.num_points))
        self.tnet = nn.ModuleList(tnet(self.num_points))
        self.box_est = nn.ModuleList(box_est(self.num_points, NUM_SIZE_CLUSTER, NUM_HEADING_BIN))

    def forward(self, rgb, data, sample=None, one_hot=False):
        output = {}
        batch_size = data.shape[0]
        # 2D detection: 
        # input batch x channels x height x width
        # output 
        # im_info = torch.from_numpy(sample['im_info']).float()
        # bb_gt = torch.from_numpy(sample['bb_gt']).float()
        # num_boxes = torch.from_numpy(sample['num_boxes']).float()

        predicted_locs, predicted_scores = self.cnn2d(rgb)

        output['2d_loss'] = rpn_loss_cls.mean() + rpn_loss_box.mean() \
           + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
        
        if one_hot:
            onehot_vec = make_oh_vector()
        else:
            onehot_vec = None 

        point_cloud = torch.zeros((batch_size, self.num_points, 4)) # batch x num_points x 4
        for i in range(batch_size):
            point_cloud[i] = sample[i]['calibration'].pc_from_rgbd(data[i])

        # 3D Instance Segmentation PointNet
        # input batch x num_points x 4
        # output 
        mlp_output = self.seg_net[0](point_cloud) # mlp1
        global_feature = self.seg_net[2](self.seg_net[1](mlp_output)).squeeze(2) # mlp2 -> maxpool
        global_feature_repeated = global_feature.repeat(1, 1, self.num_points)
        if onehot_vec is not None:
            output_scores = self.seg_net[-1](
                                                torch.cat([onehot_vec, mlp_output, global_feature_repeated], 1) # !!! 
                                            ) 
        else:
            output_scores = self.seg_net[-1](
                                                torch.cat([mlp_output, global_feature_repeated], 1) # !!! 
                                            ) 
        # concatenation -> mlp3
        output['mask_logits'] = output_scores # num_points x (1088 + k) 

        # Masking
        # select masked points and translate to masked points' centroid
        mask, object_point_cloud_xyz, mask_xyz_mean = masking(point_cloud, output_scores)
        output['mask'] = mask

        # T-Net 
        # input batch x m_points x 4
        # output 
        output_mlp = self.tnet[0](object_point_cloud_xyz)
        if onehot_vec is not None:
            global_feature = torch.cat([self.tnet[1](output_mlp).squeeze(2), onehot_vec], 1) # !!!
        else:
            global_feature = self.tnet[1](output_mlp).squeeze(2)

        residual_center = self.tnet[-1](global_feature)

        # coordinate translation
        center_1 = residual_center + mask_xyz_mean # Bx3 C_mask + delta_C_tnet
        object_point_cloud_xyz_new = object_point_cloud_xyz - residual_center.unsqueeze(1)
        output['center_1'] = center_1

        # Amodel Box Estimation PointNet
        output_mlp = self.box_est[0](object_point_cloud_xyz_new)
        if onehot_vec is not None:
            global_features = torch.cat([self.box_est[1](output_mlp).squeeze(2), onehot_vec], 1) # !!!
        else:
            global_features = self.box_est[1](output_mlp).squeeze(2)

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

if __name__ == "__main__":
    num_points = 10e4

    net = object_detection_model(num_points, SSD300(n_classes=5), Seg_PointNet, T_Net, Box_Estimation_Net)
    print(net)
    net.train()