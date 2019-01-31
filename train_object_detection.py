import torch
from torch import nn
from torch.nn import functional as F
from tochvision import models 
import numpy as np
from components import T_Net, Box_Estimation_Net, Detector2D, Seg_PointNet

NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 8 # one cluster for each type
NUM_OBJECT_POINT = 512

def mask_to_indices(mask):
    indices = torch.zeros((mask.shape[0], npoints, 2), dtype=torch.int32)
    for i in range(mask.shape[0]):
        pos_indices = torch.where(mask[i,:]>0.5, mask[i, :], mask[i, :])[0]
        # skip cases when pos_indices is empty
        if len(pos_indices) > 0: 
            if len(pos_indices) > npoints:
                choice = np.random.choice(len(pos_indices), npoints, replace=False)
            else:
                choice = np.random.choice(len(pos_indices), npoints-len(pos_indices), replace=True)
                choice = np.concatenate((np.arange(len(pos_indices)), choice))
            choice = torch.from_numpy(np.random.shuffle(choice))
            for j in range(C):
                indices[i,:,j] = torch.gather(pos_indices, 0, choice)
        # indices[i,:,0] = i
    return indices

def gather_object_pc(point_cloud, mask, npoints):
    ''' Gather object point clouds according to predicted masks.
    Input:
        point_cloud: TF tensor in shape (B,N,C)
        mask: TF tensor in shape (B,N) of 0 (not pick) or 1 (pick)
        npoints: int scalar, maximum number of points to keep (default: 512)
    Output:
        object_pc: TF tensor in shape (B,npoint,C)
        indices: TF int tensor in shape (B,npoint,2)
    '''
    C = point_cloud.shape[-1]
    indices = mask_to_indices(mask, C)  
    object_pc = torch.gather(point_cloud, 1,  indices)
    return object_pc, indices

def get_box3d_corners_helper(centers, headings, sizes):
    """ Input: (N,3), (N,), (N,3), Output: (N,8,3) """
    N = centers.shape[0]
    l = sizes[:, 0] # (N,1)
    w = sizes[:, 1] # (N,1)
    h = sizes[:, 2] # (N,1)
    #print l,w,h
    x_corners = torch.cat([l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2], axis=1) # (N,8)
    y_corners = torch.cat([h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2], axis=1) # (N,8)
    z_corners = torch.cat([w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2], axis=1) # (N,8)
    corners = torch.cat([x_corners.unsqueeze(1), y_corners.unqueeze(1), z_corners.unsqueeze(1)], axis=1) # (N,3,8)
    #print x_corners, y_corners, z_corners
    c = torch.cos(headings)
    s = torch.sin(headings)
    ones = torch.ones([N], dtype=torch.float32)
    zeros = torch.zeros([N], dtype=torch.float32)
    row1 = torch.stack([c,zeros,s], dim=1) # (N,3)
    row2 = torch.stack([zeros,ones,zeros], dim=1)
    row3 = torch.stack([-s,zeros,c], dim=1)
    R = torch.cat([row1.unsqueeze(1), row2.unsqueeze(1), row3.unsqueeze(1)], axis=1) # (N,3,3)
    #print row1, row2, row3, R, N
    corners_3d = torch.matmul(R, corners) # (N,3,8)
    corners_3d += centers.unsqueeze(2).repeat(1,1,8) # (N,3,8)
    corners_3d = torch.transpose(corners_3d, 2, 1) # (N,8,3)
    return corners_3d

def get_box3d_corners(center, heading_residuals, size_residuals):
    """ 
    Inputs:
        center: (B,3)
        heading_residuals: (B,NH)
        size_residuals: (B,NS,3)
    Outputs:
        box3d_corners: (B,NH,NS,8,3) tensor
    """
    batch_size = center.shape[0]
    heading_bin_centers = torch.from_numpy(0,2*np.pi,2*np.pi/NUM_HEADING_BIN) # (NH,)
    headings = heading_residuals + heading_bin_centers.unsqueeze(0) # (B,NH)
    
    mean_sizes = tensor.from_numpy(g_mean_size_arr).unsqueeze(0) + size_residuals # (B,NS,1)
    sizes = mean_sizes + size_residuals # (B,NS,3)
    sizes = sizes.unsqueeze(1).repeat(1,NUM_HEADING_BIN,1,1) # (B,NH,NS,3)
    headings = headings.unsqueeze(-1).repeat(1,1,NUM_SIZE_CLUSTER) # (B,NH,NS)
    centers = center.unsqueeze(1).unsqueeze(1).repeat(1,NUM_HEADING_BIN, NUM_SIZE_CLUSTER,1) # (B,NH,NS,3)

    N = batch_size*NUM_HEADING_BIN*NUM_SIZE_CLUSTER
    corners_3d = get_box3d_corners_helper(centers.view(N,3), headings.view(N), sizes.view(N,3))

    return tf.reshape(corners_3d, [batch_size, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, 8, 3])


def masking(point_cloud, segmentation):
    mask = segmentation[:, :, 0] < segmentation[:, :, 1] # BxNx1
    mask_count = torch.sum(mask, 1).repeat(1, 1, 3) # Bx1x3
    batch_size = point_cloud.size()[0]
    num_point = point_cloud.size()[1]

    point_cloud_xyz = point_cloud[:, :, :3] #torch.narrow(point_cloud, 2, 0, 3) # BxNx3
    mask_xyz_mean = torch.sum(mask.repeat(1, 1, 3)*point_cloud_xyz, 1)/torch.max(mask_count,1) # Bx1x3
    mask = mask.squeeze(2) # BxN
    # Translate to masked points' centroid
    point_cloud_xyz_stage1 = point_cloud_xyz - \
        mask_xyz_mean.repeat(1,num_point,1)

    num_channels = point_cloud_xyz_stage.size()[-1]
    
    object_point_cloud, _ = gather_object_pc(point_cloud_stage1, mask, NUM_OBJECT_POINT)
    object_point_cloud = object_point_cloud.view(batch_size, NUM_OBJECT_POINT, num_channels)

    return mask, object_point_cloud, mask_xyz_mean

def object_detection_model(nn.Module):
    def __init__(self, num_points, cnn2d, seg_net, tnet, box_est):
        
        self.num_points = num_points
        self.k = onehot_length
        self.cnn2d = nn.ModuleList(cnn2d)
        self.seg_net = nn.ModuleList(seg_net(self.num_points))
        self.tnet = nn.ModuleList(tnet(self.num_points))
        self.box_est = nn.ModuleList(box_est(self.num_points, NUM_SIZE_CLUSTER, NUM_HEADING_BIN))
        
        return end_points

    def forward(self, x, onehot_vec):
        output = {}
        # 2D detection

        # 3D Instance Segmentation PointNet
        for layer in self.seg_net[-2]:
            output_mlp = layer(x)
        global_feature = self.seg_net[-2](output_mlp).squeeze(2)
        output_scores = self.seg_net[-1](torch.cat([onehot_vec, output_mlp, global_feature], 1))
        output['mask_logits'] = output_scores

        # Masking
        # select masked points and translate to masked points' centroid
        mask, object_point_cloud, mask_xyz_mean = masking(point_cloud, output_scores)
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
        global_feature = torch.cat([self.box_est[1](output_mlp).squeeze(2), onehot_vec], 1)
        box_parameters = self.box_est[-1](global_features)

        # Parse output to 3D box parameters
        center = torch.narrow(box_parameters, 1, 0, 3)
        output['center_boxnet'] = center

        heading_scores = torch.narrow(box_parameters, 1, 3, NUM_HEADING_BIN)
        heading_residuals_normalized = torch.narrow(box_parameters, 1, 3+NUM_HEADING_BIN, NUM_HEADING_BIN)
        output['heading_scores'] = heading_scores # BxNUM_HEADING_BIN
        output['heading_residuals_normalized'] = heading_residuals_normalized # BxNUM_HEADING_BIN (-1 to 1)
        output['heading_residuals'] = heading_residuals_normalized * (np.pi/NUM_HEADING_BIN) # BxNUM_HEADING_BIN

        size_scores = torch.narrow(box_parameters, 1, 3+NUM_HEADING_BIN*2, NUM_SIZE_CLUSTER]) # BxNUM_SIZE_CLUSTER
        size_residuals_normalized = torch.narrow(box_parameters, 1, 3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER, NUM_SIZE_CLUSTER*3)
        batch_size = box_parameters.size()[0]
        size_residuals_normalized = size_residuals_normalized.view(batch_size, NUM_SIZE_CLUSTER, 3) # BxNUM_SIZE_CLUSTERx3
        output['size_scores'] = size_scores
        output['size_residuals_normalized'] = size_residuals_normalized
        output['size_residuals'] = size_residuals_normalized * \
                                   #tf.expand_dims(tf.constant(g_mean_size_arr, dtype=tf.float32), 0)


        output['center_prediction'] = output['center_boxnet'] + center_1 # Bx3 (C_mask + delta_C_tnet) + delta_C_boxnet
        heading_scores =
        return output
    
def huber_loss(error, delta):
    abs_error = torch.abs(error)
    quadratic = torch.min(abs_error, delta)
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic**2 + delta * linear
    return torch.mean(losses)

def loss(mask_label, center_label, \
         heading_class_label, heading_residual_label, \
         size_class_label, size_residual_label, \
         pipelin_output, corner_loss_weight):
    ''' Loss function for 3D object detection pipeline.
    Input:
        mask_label:  shape (B,N)
        center_label:  shape (B,3)
        heading_class_label:  shape (B,) 
        heading_residual_label: shape (B,) 
        size_class_label: shape (B,)
        size_residual_label: shape (B,)
        end_points: dict, outputs from our model
        corner_loss_weight: float scalar
        box_loss_weight: float scalar
    Output:
        total_loss: scalar tensor
            the total_loss is also added to the losses collection
    '''
    seg_loss = torch.mean(torch.nn.Softmax())
    tnet_loss = huber_loss()
    center_loss = huber_loss()
    hcls_loss = torch.mean(torch.nn.Softmax())
    hreg_loss = huber_loss()
    heading_angle_loss = hcls_loss + hreg_loss
    box_size_loss = scls_loss + sreg_loss
    corner_loss = 
    loss = seg_loss + l*(tnet_loss + center_loss + heading_angle_loss + box_size_loss + g*corner_loss)

    return loss 

def train(model, train_loader, optimizer, loss, device, epoch):
    model.train()
    batches_n = len(train_loader)
    for batch_idx, batch_sample in enumerate(train_loader):
        rgb, data = batch_sample['image'].to(device), batch_sample['data'].to(device)
        target = batch_sample['target'].to(device)
        output = model(rgb, data)
        train_loss = loss(output, target)
        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        iteration = epoch * batches_n + batch_idx

        if batch_idx % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / batches_n, loss_np))
