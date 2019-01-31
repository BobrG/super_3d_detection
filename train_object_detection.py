from components import T_Net, Box_Estimation_Net, Detector2D, Seg_PointNet

NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 8 # one cluster for each type
NUM_OBJECT_POINT = 512

def masking(point_cloud, segmentation):
    mask = torch.index_select(segmentation, 2, 0) < /
           torch.index_select(segmentation, 2, 1) # BxNx1
    mask_count = torch.sum(mask, 1).repeat(1, 1, 3) # Bx1x3
    batch_size = point_cloud.size()[0]
    num_point = point_cloud.size()[1]

    point_cloud_xyz = torch.narrow(point_cloud, 0, 2, 3) # BxNx3
    mask_xyz_mean = torch.sum(mask.repeat(1, 1, 3)*point_cloud_xyz, 1)/torch.max(mask_count,1) # Bx1x3
    mask = mask.squeeze(2) # BxN
    # Translate to masked points' centroid
    point_cloud_xyz_stage1 = point_cloud_xyz - \
        mask_xyz_mean.repeat(1,num_point,1)

    num_channels = point_cloud_xyz_stage.size()[-1]
    
    return mask, object_point_cloud, mask_xyz_mean

def object_detection_model(nn.Module):
    def __init__(self, num_points, cnn2d, seg_net, tnet, box_est):
        
        self.num_points = num_points
        self.k = onehot_length
        self.cnn2d = cnn2d
        self.seg_net = seg_net(self.num_points, )
        self.tnet = tnet(self.num_points)
        self.box_est = box_est(self.num_points, NUM_SIZE_CLUSTER, NUM_HEADING_BIN)
        
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
        stage1_center = residual_center + mask_xyz_mean # Bx3 C_mask + delta_C_tnet
        object_point_cloud_xyz_new = object_point_cloud_xyz - residual_center.unsqueeze(1)
        output['stage1_center'] = stage1_center

        # Amodel Box Estimation PointNet
        output_mlp = self.box_est[0](object_point_cloud_xyz_new)
        global_feature = torch.cat([self.box_est[1](output_mlp).squeeze(2), onehot_vec], 1)
        box_parameters = self.box_est[-1](global_features)

        # Parse output to 3D box parameters
         = parse_output_to_tensors(box_parameters, output)
        output['center'] = output['center_boxnet'] + stage1_center # Bx3 (C_mask + delta_C_tnet) + delta_C_boxnet

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
