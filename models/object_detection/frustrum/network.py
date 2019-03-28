import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models 
import numpy as np
from utils import masking, get_pc, correct_pc, extract_objects, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, NUM_CLASS, g_mean_size_arr
from components import T_Net, Box_Estimation_Net, Seg_PointNet
from easy_detection_2d.model import SSD300, MultiBoxLoss
import cv2

class object_detection_model(nn.Module):
    def __init__(self, num_points, cnn2d, seg_net, tnet, box_est, min_score=0.2, max_overlap=0.5, top_k=5):
        super(object_detection_model, self).__init__()
        self.num_points = int(num_points)
        self.cnn2d = cnn2d # pass initialized model class
        self.min_score = min_score
        self.max_overlap = max_overlap
        self.top_k = top_k
        self.seg_net = nn.ModuleList(seg_net(self.num_points))
        self.tnet = nn.ModuleList(tnet(self.num_points))
        self.box_est = nn.ModuleList(box_est(self.num_points, NUM_SIZE_CLUSTER, NUM_HEADING_BIN))

    def forward(self, rgb, data, sample=None, one_hot=False):
        output = {}
        batch_size = data.shape[0]

        # 2D detection: 
        # input batch x channels x height x width
        predicted_locs, predicted_scores = self.cnn2d(rgb) # (B, 8732, 4), (B, 8732, n_classes)
        output['pred_bb'] = predicted_locs
        output['pred_scores'] = preducted_scores
    
        if one_hot:
            onehot_vec = make_oh_vector()
        else:
            onehot_vec = None 

        point_cloud = torch.zeros((int(batch_size), 3, int(self.num_points))) # batch x num_points x 3 !!! check how to make 4 channels
        objects_pc = []

        # extracting objects from point clounds
        for i in range(batch_size):
            tmp = get_pc(data[i], Rtilt=sample['Rtilt'][i], K=sample['K'][i]).permute(1, 0)
            det_boxes, det_labels, det_scores = self.cnn2d.detect_objects(
                                                                      predicted_locs[i], predicted_scores[i],
                                                                      min_score=self.min_score, 
                                                                      max_overlap=self.max_overlap, 
                                                                      top_k=self.top_k
                                                                     ) # (n_objects, 4), (n_objects), (n_objects)

            # objects = torch.zeros(())
            objects = extract_objects(tmp[i], det_boxes, sample['rotate_to_center'], self.num_points)
            objects_pc.append(objects)
            
        # tensor with cropped objects 
        objects_pc_tensor = torch.cat(objects_pc, 0)

        # 3D Instance Segmentation PointNet
        # input batch x num_points x 3
        # output 
        mlp_output = self.seg_net[0](objects_pc_tensor) # mlp1, torch.Size([1, 64, num_points])
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
        mask, object_point_cloud_xyz, mask_xyz_mean = masking(point_cloud.permute(0, 2, 1), output_scores.permute(0, 2, 1))
        output['mask'] = mask

        # T-Net 
        # input batch x m_points x 4
        # output 
        output_mlp = self.tnet[0](object_point_cloud_xyz.permute(0, 2, 1))
        m = output_mlp.shape[-1]
        max_pool = nn.MaxPool1d(m)
        if onehot_vec is not None:
            global_feature = torch.cat([max_pool(output_mlp).squeeze(2), onehot_vec], 1)
        else:
            global_feature = max_pool(output_mlp).squeeze(2)
        
        residual_center = self.tnet[-1](global_feature)
        
        # coordinate translation
        center_1 = residual_center + mask_xyz_mean # Bx3 C_mask + delta_C_tnet
        object_point_cloud_xyz_new = object_point_cloud_xyz - residual_center.unsqueeze(1) # B, num_points, C
        output['center_1'] = center_1

        # Amodel Box Estimation PointNet
        output_mlp = self.box_est[0](object_point_cloud_xyz_new.permute(0, 2, 1))
        if onehot_vec is not None:
            global_features = torch.cat([self.box_est[1](output_mlp).squeeze(2), onehot_vec], 1) # !!!
        else:
            global_features = self.box_est[1](output_mlp).squeeze(2)

        box_parameters = self.box_est[-1](global_features)

        # Parse output to 3D box parameters
        center = box_parameters[:,:3]
        output['center_boxnet'] = center

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
        output['size_residuals'] = size_residuals_normalized * torch.FloatTensor(g_mean_size_arr).unsqueeze(0)

        output['center_prediction'] = output['center_boxnet'] + center_1 # Bx3 (C_mask + delta_C_tnet) + delta_C_boxnet
        
        return output

if __name__ == "__main__":
    num_points = 10e4
    net = object_detection_model(num_points, SSD300(n_classes=NUM_CLASS), Seg_PointNet, T_Net, Box_Estimation_Net)
    net.train()
    sample = {}
    sample['image'] = torch.FloatTensor(cv2.imread('/Users/bobrg/Downloads/sunrgbd/image/NYU1000.jpg')).permute(2, 0, 1).unsqueeze(0)
    sample['depth'] = torch.FloatTensor(cv2.imread('/Users/bobrg/Downloads/sunrgbd/depth/NYU1000.png', 0)).unsqueeze(0)
    intr = np.reshape(
        np.array([float(x) for x in open('/Users/bobrg/Downloads/sunrgbd/intrinsics.txt').read().split(' ')[:-1]]),
        (3,3), order='F')
    extr = np.array([[float(x) for x in i.split(' ')] for i in open('/Users/bobrg/Downloads/sunrgbd/20150118235401.txt').read().splitlines()])[:, :3]
    sample['K'] = torch.FloatTensor(intr).unsqueeze(0)
    sample['Rtilt'] = torch.FloatTensor(extr).unsqueeze(0)
    net(sample['image'], sample['depth'], sample)