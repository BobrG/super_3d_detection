import torch
import numpy as np
from torch.nn import functional as F
from sunrgbd_dataset import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, NUM_CLASS, g_mean_size_arr, rotate_pc_along_y, get_center_view_rot_angle 

def make_oh_vector():
    pass

# TODO: check mask_to_indices and gather_object_pc

# --------------------------------- DATASET MANIPULATIONS ---------------------------------

def correct_pc(pc, rotate_to_center, npoints, angle):

    if rotate_to_center:
        # Input ps is NxC points with first 3 channels as XYZ
        # z is facing forward, x is left ward, y is downward
        new_pc = rotate_pc_along_y(pc, get_center_view_rot_angle(angle))
    else:
        new_pc = pc
    # Resample point cloud
    # TODO: add resample which expands amount of points
    choice = np.random.choice(new_pc.shape[0], npoints, replace=True)

    return new_pc[choice, :]

def extract_objects(pc, coords, npoints, sample):
    res = torch.zeros((coords.shape[0], npoints, 3))
    # for sunrgbd:
    # same shit as in sunrgbd_datase string 170
    pc_ = pc.permute(0, 2, 1)
    pc_[:, -1] *= -1.0 
    K = sample['K']
    Rtilt = sample['Rtilt']

    for box in coords:
        xmin,ymin,xmax,ymax = box
        box_inds = (pc_[:,0]<xmax) & (pc_[:,0]>=xmin) & (pc_[:,1]<ymax) & (pc_[:,1]>=ymin)
        box2d_center = np.array([(xmin+xmax)/2.0, (ymin+ymax)/2.0])
        pc_cut = pc_[box_inds, :]

        uvdepth = torch.zeros((1,3))
        uvdepth[0,0:2] = box2d_center
        uvdepth[0,2] = 20 # some random depth

        n = uvdepth.shape[0]
        c_u, c_v = K[0,2], K[1,2]
        f_u, f_v = K[0,0], K[1,1]

        x = ((uvdepth[:,0] - c_u)*uvdepth[:,2])/f_u
        y = ((uvdepth[:,1] - c_v)*uvdepth[:,2])/f_v
        pts_3d_camera = torch.zeros((n,3))
        pts_3d_camera[:,0] = x
        pts_3d_camera[:,1] = uvdepth[:,2]
        pts_3d_camera[:,2] = -1*y
        pts_3d_camera = torch.t(
                                torch.mm(torch.t(Rtilt),
                                torch.t(pts_3d_matrix[:, 0:3]))
                               ).permute(0, 2, 1)
        pts_3d_camera[:, -1] *= -1

        box2d_center_upright_camera = pts_3d_camera
        
        frustum_angle = -1 * torch.atan2(box2d_center_upright_camera[0,2],
                                    box2d_center_upright_camera[0,0]) # angle as to positive x-axis as in the Zoox paper
        res[i] = correct_pc(pc_cut, sample['rotate_to_center'], npoints, frustum_angle)

    return res

# --------------------------------- NN DATA MANIPULATIONS ---------------------------------

def mask_to_indices(mask, npoints):
    indices = torch.zeros((mask.shape[0], npoints), dtype=torch.long)
    for i in range(mask.shape[0]):
        pos_indices = torch.where(mask[i,:]>0.5, mask[i, :], mask[i, :])[0]
        # skip cases when pos_indices is empty
        if len(pos_indices.size()) > 0: 
            if len(pos_indices) > npoints:
                choice = np.random.choice(len(pos_indices), npoints, replace=False)
            else:
                choice = np.random.choice(len(pos_indices), npoints-len(pos_indices), replace=True)
                choice = np.concatenate((np.arange(len(pos_indices)), choice))
            choice = torch.from_numpy(np.random.shuffle(choice))
            indices[i] = choice #torch.gather(pos_indices, 0, choice)
    
    return indices

def gather_object_pc(point_cloud, mask, npoints=512): 
    ''' Gather object point clouds according to predicted masks.
    Input:
        point_cloud: TF tensor in shape (B,N,C)
        mask: TF tensor in shape (B,N) of 0 (not pick) or 1 (pick)
        npoints: int scalar, maximum number of points to keep (default: 512)
    Output:
        object_pc: TF tensor in shape (B,npoint,C)
        indices: TF int tensor in shape (B,npoint)
    '''
    # C = point_cloud.shape[-1]
    indices = mask_to_indices(mask, npoints)
    object_pc = torch.cat([point_cloud[i, j, :] for i, j in enumerate(indices)])
    # object_pc = torch.gather(point_cloud, 0,  indices)
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
    batch_size = point_cloud.size()[0]
    num_point = point_cloud.size()[1]

    mask = (segmentation[:, :, 0] < segmentation[:, :, 1]).type(torch.FloatTensor).unsqueeze(2) # BxNx1
    mask_count = torch.sum(mask, 1).repeat(1, 1, 3) # Bx1x3
    point_cloud_xyz = point_cloud[:, :, :3] #torch.narrow(point_cloud, 2, 0, 3) # BxNx3
    max_ = torch.max(mask_count,1)[0]

    if len(max_[max_ == 0.0].size()) == 0:
        mask_xyz_mean = torch.sum(mask.repeat(1, 1, 3)*point_cloud_xyz, 1)/max_ # Bx1x3
    else:
        max_[max_ == 0.0] = 1.0
        mask_xyz_mean = torch.sum(mask.repeat(1, 1, 3)*point_cloud_xyz, 1)/max_
    
    mask = mask.squeeze(2) # BxN
    
    # Translate to masked points' centroid
    point_cloud_xyz_stage1 = point_cloud_xyz - \
        mask_xyz_mean.repeat(1, num_point, 1)
    
    # num_channels = point_cloud_xyz_stage1.size()[-1]
    
    # object_point_cloud, _ = gather_object_pc(point_cloud_xyz_stage1, mask)
    # print(object_point_cloud.shape)
    # object_point_cloud = object_point_cloud.view(batch_size, num_point, num_channels)

    return mask, point_cloud_xyz_stage1, mask_xyz_mean # object_point_cloud, mask_xyz_mean

# --------------------------------- LOSS FUNCTIONS ---------------------------------

def huber_loss(error, delta):
    abs_error = torch.abs(error)
    quadratic = torch.min(abs_error, delta)
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic**2 + delta * linear
    return torch.mean(losses)

def loss(pipeline_output, target, corner_loss_weight=0.001):
    ''' Loss function for 3D object detection pipeline.
    Input:
        pipeline_output: dict, outputs from our model
       
        sample['target']:
            label:  shape (B,N)
            box3d_center:  shape (B,3)
            angle_class:  shape (B,) 
            angle_residual: shape (B,) 
            size_class: shape (B,)
            size_residual: shape (B,)
            corner_loss_weight: float scalar
    Output:
        total_loss: scalar tensor
            the total_loss is also added to the losses collection
    '''
    # 3D Segmentation loss
    mask_loss = torch.mean(F.nll_loss(F.softmax(\
        pipeline_output['mask_logits'], target['label'])))

    # Center regression losses
    center_dist = torch.norm(target['box3d_centre'] - end_points['center'], p=2, dim=-1)
    center_loss = huber_loss(center_dist, delta=2.0)
    
    stage1_center_dist = torch.norm(target['box3d_centre'] - \
        pipeline_output['stage1_center'], dim=-1)
    stage1_center_loss = huber_loss(stage1_center_dist, delta=1.0)
    
    # Heading loss
    heading_class_loss = torch.mean( \
        F.nll_loss(F.softmax( \
        pipeline_output['heading_scores'], target['angle_class'])))

    hcls_onehot = torch.nn.functional.one_hot(target['angle_class'],
        depth=NUM_HEADING_BIN) # BxNUM_HEADING_BIN
    heading_residual_normalized_label = \
        target['angle_residual'] / (np.pi/NUM_HEADING_BIN)
    heading_residual_normalized_loss = huber_loss(torch.sum( \
        pipeline_output['heading_residuals_normalized']*hcls_onehot.float(), 1) - \
        heading_residual_normalized_label, delta=1.0)
    
    # Size loss
    size_class_loss = torch.mean(F.nll_loss(F.softmax(pipeline_output['size_scores']), target['size_class']))
    
    scls_onehot = torch.nn.functional.one_hot(target['size_class'],
        depth=NUM_SIZE_CLUSTER) # BxNUM_SIZE_CLUSTER
    scls_onehot_tiled = scls_onehot.float().unsqueeze(-1).repeat(1,1,3) # BxNUM_SIZE_CLUSTERx3
    predicted_size_residual_normalized = torch.sum( \
        pipeline_output['size_residuals_normalized']*scls_onehot_tiled, 1) # Bx3

    mean_size_arr_expand = torch.from_numpy(g_mean_size_arr).unsqueeze(0) # 1xNUM_SIZE_CLUSTERx3
    mean_size_label = torch.sum( \
        scls_onehot_tiled * mean_size_arr_expand, 1) # Bx3
    target['size_residual']_normalized = target['size_residual'] / mean_size_label
    size_normalized_dist = torch.norm( \
        target['size_residual']_normalized - predicted_size_residual_normalized, p=2,
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
    heading_label = target['angle_residual'].unsqueeze(1) + heading_bin_centers.unsqueeze(0) # (B,NH)
    heading_label = torch.sum(hcls_onehot.float()*heading_label, dim=1)
    mean_sizes = torch.from_numpy(g_mean_size_arr).unsqueeze(0) # (1,NS,3)
    size_label = mean_sizes + target['size_residual'].unsqueeze(1) # (1,NS,3) + (B,1,3) = (B,NS,3)
    size_label = torch.sum(scls_onehot.float().unsqueeze(-1)*size_label, dim=1) # (B,3)
    corners_3d_gt = get_box3d_corners_helper( \
        target['box3d_centre'], heading_label, size_label) # (B,8,3)
    corners_3d_gt_flip = get_box3d_corners_helper( \
        target['box3d_centre'], heading_label+np.pi, size_label) # (B,8,3)

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
