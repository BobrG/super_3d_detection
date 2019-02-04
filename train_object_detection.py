import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from tochvision import models 
import numpy as np
from components import T_Net, Box_Estimation_Net, Detector2D, Seg_PointNet

NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 8 # one cluster for each type
NUM_OBJECT_POINT = 512

# TODO: rewrite for matterport or sunrgbd
# g_type2class={'Car':0, 'Van':1, 'Truck':2, 'Pedestrian':3,
#               'Person_sitting':4, 'Cyclist':5, 'Tram':6, 'Misc':7}
# g_class2type = {g_type2class[t]:t for t in g_type2class}
# g_type2onehotclass = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
# g_type_mean_size = {'Car': np.array([3.88311640418,1.62856739989,1.52563191462]),
#                     'Van': np.array([5.06763659,1.9007158,2.20532825]),
#                     'Truck': np.array([10.13586957,2.58549199,3.2520595]),
#                     'Pedestrian': np.array([0.84422524,0.66068622,1.76255119]),
#                     'Person_sitting': np.array([0.80057803,0.5983815,1.27450867]),
#                     'Cyclist': np.array([1.76282397,0.59706367,1.73698127]),
#                     'Tram': np.array([16.17150617,2.53246914,3.53079012]),
#                     'Misc': np.array([3.64300781,1.54298177,1.92320313])}
# g_mean_size_arr = np.zeros((NUM_SIZE_CLUSTER, 3)) # size clustrs
# for i in range(NUM_SIZE_CLUSTER):
#     g_mean_size_arr[i,:] = g_type_mean_size[g_class2type[i]]

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
    corners = torch.cat([x_corners.unsqueeze(1), y_corners.unsqueeze(1), z_corners.unsqueeze(1)], axis=1) # (N,3,8)
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
        batch_size = output.shape[0]
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
    
def huber_loss(error, delta):
    abs_error = torch.abs(error)
    quadratic = torch.min(abs_error, delta)
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic**2 + delta * linear
    return torch.mean(losses)

def loss(mask_label, center_label, \
         heading_class_label, heading_residual_label, \
         size_class_label, size_residual_label, \
         pipeline_output, corner_loss_weight):
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
    # 2dNet loss

    net2d_loss = #?

    # 3D Segmentation loss
    mask_loss = torch.mean(F.nll_loss(F.softmax(\
        pipeline_output['mask_logits'], mask_label)))

    # Center regression losses
    center_dist = torch.norm(center_label - end_points['center'], p=2, dim=-1)
    center_loss = huber_loss(center_dist, delta=2.0)
    
    stage1_center_dist = torch.norm(center_label - \
        pipeline_output['stage1_center'], dim=-1)
    stage1_center_loss = huber_loss(stage1_center_dist, delta=1.0)
    
    # Heading loss
    heading_class_loss = torch.mean( \
        F.nll_loss(F.softmax( \
        pipeline_output['heading_scores'], heading_class_label)))

    hcls_onehot = torch.nn.functional.one_hot(heading_class_label,
        depth=NUM_HEADING_BIN) # BxNUM_HEADING_BIN
    heading_residual_normalized_label = \
        heading_residual_label / (np.pi/NUM_HEADING_BIN)
    heading_residual_normalized_loss = huber_loss(torch.sum( \
        pipeline_output['heading_residuals_normalized']*hcls_onehot.float()), 1) - \
        heading_residual_normalized_label, delta=1.0)
    
    # Size loss
    size_class_loss = torch.mean( /
            F.nll_loss(F.softmax(pipeline_output['size_scores']), size_class_label))
    
    scls_onehot = torch.nn.functional.one_hot(size_class_label,
        depth=NUM_SIZE_CLUSTER) # BxNUM_SIZE_CLUSTER
    scls_onehot_tiled = scls_onehot.float().unsqueeze(-1).repeat(1,1,3) # BxNUM_SIZE_CLUSTERx3
    predicted_size_residual_normalized = torch.sum( \
        pipeline_output['size_residuals_normalized']*scls_onehot_tiled, 1) # Bx3

    mean_size_arr_expand = torch.from_numpy(g_mean_size_arr).unsqueeze(0) # 1xNUM_SIZE_CLUSTERx3
    mean_size_label = torch.sum( \
        scls_onehot_tiled * mean_size_arr_expand, 1) # Bx3
    size_residual_label_normalized = size_residual_label / mean_size_label
    size_normalized_dist = torch.norm( \
        size_residual_label_normalized - predicted_size_residual_normalized, p=2,
        dim=-1)
    size_residual_normalized_loss = huber_loss(size_normalized_dist, delta=1.0)

    # Corner loss
    # We select the predicted corners corresponding to the 
    # GT heading bin and size cluster.
    corners_3d = get_box3d_corners(pipeline_output['center'],
        pipeline_output['heading_residuals'],
        pipeline_output['size_residuals']) # (B,NH,NS,8,3)
    gt_mask = hcls_onehot.unsqueeze(2).repeat(1,1,NUM_SIZE_CLUSTER) * \
        scls_onehot.unsqueeze(1).repeat(1,NUM_HEADING_BIN,1) # (B,NH,NS)
    corners_3d_pred = torch.sum( \
        gt_mask.unsqueeze(-1).unsqueeze(-1).float() * corners_3d,
        dim=(1,2))# (B,8,3)

    heading_bin_centers = torch.from_numpy(np.arange(0,2*np.pi,2*np.pi/NUM_HEADING_BIN)) # (NH,)
    heading_label = heading_residual_label.unsqueeze(1) + heading_bin_centers.unsqueeze(0) # (B,NH)
    heading_label = torch.sum(hcls_onehot.float()*heading_label, dim=1)
    mean_sizes = torch.from_numpy(g_mean_size_arr).unsqueeze(0) # (1,NS,3)
    size_label = mean_sizes + size_residual_label.unsqueeze(1) # (1,NS,3) + (B,1,3) = (B,NS,3)
    size_label = torch.sum(scls_onehot.float().unsqueeze(-1)*size_label, dim=1) # (B,3)
    corners_3d_gt = get_box3d_corners_helper( \
        center_label, heading_label, size_label) # (B,8,3)
    corners_3d_gt_flip = get_box3d_corners_helper( \
        center_label, heading_label+np.pi, size_label) # (B,8,3)

    corners_dist = torch.min(torch.norm(corners_3d_pred - corners_3d_gt, p=2, dim=-1),
        torch.norm(corners_3d_pred - corners_3d_gt_flip, p=2, dim=-1))

    corners_loss = huber_loss(corners_dist, delta=1.0) 

    total_loss = net2d_loss + mask_loss + box_loss_weight * (center_loss + \
        heading_class_loss + size_class_loss + \
        heading_residual_normalized_loss*20 + \
        size_residual_normalized_loss*20 + \
        stage1_center_loss + \
        corner_loss_weight*corners_loss)

    return total_loss 

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
