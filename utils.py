import torch
import numpy as np
from torch.nn import functional as F
from sunrgbd_dataset import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, NUM_CLASS, g_mean_size_arr, rotate_pc_along_y, get_center_view_rot_angle 
from scipy.spatial import ConvexHull

def make_oh_vector():
    pass

# TODO: check mask_to_indices and gather_object_pc

# --------------------------------- DATASET MANIPULATIONS ---------------------------------

def get_pc(depthmap, Rtilt, K):
        rows, cols = depthmap.shape
        cx, cy = K[0,2], K[1,2]
        fx, fy = K[0,0], K[1,1]
        
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
        return res

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
            indices.numpy()[i] = np.random.shuffle(choice) #torch.gather(pos_indices, 0, choice)
    
    return indices

def gather_object_pc(point_cloud, mask, npoints=512): 
    ''' Gather object point clouds according to predicted masks.
    Input:
        point_cloud: TF tensor in shape (B,C,N)
        mask: TF tensor in shape (B,N) of 0 (not pick) or 1 (pick)
        npoints: int scalar, maximum number of points to keep (default: 512)
    Output:
        object_pc: TF tensor in shape (B,C,npoint)
        indices: TF int tensor in shape (B,npoint)
    '''
    # C = point_cloud.shape[-1]
    
    indices = mask_to_indices(mask, npoints)
    object_pc = torch.stack([point_cloud[i, :, js] for i, js in enumerate(indices)])
    # object_pc = torch.gather(point_cloud, 0,  indices)
    
    return object_pc, indices

def masking(point_cloud, segmentation):
    batch_size = point_cloud.size()[0]
    num_point = point_cloud.size()[-1]
    mask = torch.empty_like(point_cloud[:, :1])
    mask[:] = segmentation[:, :1] < segmentation[:, 1:] # Bx1xN
    mask_count = torch.sum(mask, -1, keepdim=True) # Bx1x1
    mask_count[mask_count == 0] = 1
    point_cloud_xyz = point_cloud[:, :3]# Bx3xN
    mask_xyz_mean = torch.sum(mask*point_cloud_xyz, -1, keepdim=True) / torch.max(mask_count) # Bx3x1
    

    # Translate to masked points' centroid
    point_cloud_xyz_stage1 = point_cloud_xyz - mask_xyz_mean

    # num_channels = point_cloud_xyz_stage1.size()[-1]

    object_point_cloud = point_cloud_xyz_stage1#, _ = gather_object_pc(point_cloud_xyz_stage1, mask.squeeze(1), npoints=2048)
    # print(object_point_cloud.shape)
    ### NOMASK return mask, object_point_cloud, mask_xyz_mean
    return object_point_cloud, mask_xyz_mean, mask

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def polygon_clip(subjectPolygon, clipPolygon):
    """ Clip a polygon with another polygon.
    Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python
    Args:
     subjectPolygon: a list of (x,y) 2d points, any polygon.
     clipPolygon: a list of (x,y) 2d points, has to be *convex*
    Note:
     **points have to be counter-clockwise ordered**
    Return:
     a list of (x,y) vertex point for the intersection polygon.
    """
    def inside(p):
        return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
 
    def computeIntersection():
        dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
        dp = [ s[0] - e[0], s[1] - e[1] ]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0] 
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]

    outputList = subjectPolygon
    cp1 = clipPolygon[-1]

    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]

        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
        if len(outputList) == 0:
            return None
    return(outputList)


def poly_area(x,y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0

def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c

def box3d_iou(corners1, corners2):
    ''' Compute 3D bounding box IoU.
    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU
    todo (rqi): add more description on corner points' orders.
    '''
    # corner points are in counter clockwise order
    rect1 = [(corners1[i,0], corners1[i,2]) for i in range(3,-1,-1)]
    rect2 = [(corners2[i,0], corners2[i,2]) for i in range(3,-1,-1)] 
    area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
    area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area/(area1+area2-inter_area)
    ymax = min(corners1[0,1], corners2[0,1])
    ymin = max(corners1[4,1], corners2[4,1])
    inter_vol = inter_area * max(0.0, ymax-ymin)
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d

def get_iou(bb1, bb2):
    """ Compute IoU of two bounding boxes.
        ** Define your bod IoU function HERE **
    """
    #pass
    iou3d, iou2d = box3d_iou(bb1, bb2)
    return iou3d

def eval_det_cls(pred, gt, ovthresh=0.25, use_07_metric=False):
    """ Generic functions to compute precision/recall for object detection
        for a single class.
        Input:
            pred: map of {img_id: [(bbox, score)]} where bbox is numpy array
            gt: map of {img_id: [bbox]}
            ovthresh: scalar, iou threshold
            use_07_metric: bool, if True use VOC07 11 point method
        Output:
            rec: numpy array of length nd
            prec: numpy array of length nd
            ap: scalar, average precision
    """

    # construct gt objects
    class_recs = {} # {img_id: {'bbox': bbox list, 'det': matched list}}
    npos = 0
    for img_id in gt.keys():
        bbox = np.array(gt[img_id])
        det = [False] * len(bbox)
        npos += len(bbox)
        class_recs[img_id] = {'bbox': bbox, 'det': det}
    # pad empty list to all other imgids
    for img_id in pred.keys():
        if img_id not in gt:
            class_recs[img_id] = {'bbox': np.array([]), 'det': []}

    # construct dets
    image_ids = []
    confidence = []
    BB = []
    for img_id in pred.keys():
        for box,score in pred[img_id]:
            image_ids.append(img_id)
            confidence.append(score)
            BB.append(box)
    confidence = np.array(confidence)
    BB = np.array(BB) # (nd,4 or 8,3)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, ...]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        if d%100==0: 
            print(d)
        R = class_recs[image_ids[d]]
        bb = BB[d,:].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            for j in range(BBGT.shape[0]):
                iou = get_iou(bb, BBGT[j,...]) 
                if iou > ovmax:
                    ovmax = iou
                    jmax = j

        #print d, ovmax
        if ovmax > ovthresh:
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    print('NPOS: ', npos)
    print('ND:', nd)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap

def eval_det(pred_all, gt_all, ovthresh=0.25, use_07_metric=False):
    """ Generic functions to compute precision/recall for object detection
        for multiple classes.
        Input:
            pred_all: map of {img_id: [(classname, bbox, score)]}
            gt_all: map of {img_id: [(classname, bbox)]}
            ovthresh: scalar, iou threshold
            use_07_metric: bool, if true use VOC07 11 point method
        Output:
            rec: {classname: rec}
            prec: {classname: prec_all}
            ap: {classname: scalar}
    """
    pred = {} # map {classname: pred}
    gt = {} # map {classname: gt}
    for img_id in pred_all.keys():
        for classname, bbox, score in pred_all[img_id]:
            if classname not in pred: pred[classname] = {}
            if img_id not in pred[classname]:
                pred[classname][img_id] = []
            if classname not in gt: gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            pred[classname][img_id].append((bbox,score))
    for img_id in gt_all.keys():
        for classname, bbox in gt_all[img_id]:
            if classname not in gt: gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            gt[classname][img_id].append(bbox)

    rec = {}
    prec = {}
    ap = {}
    for classname in gt.keys():
        print('Computing AP for class: ', classname)
        rec[classname], prec[classname], ap[classname] = eval_det_cls(pred[classname], gt[classname], ovthresh, use_07_metric)
        print(classname, ap[classname])
    
    return rec, prec, ap 

# --------------------------------- LOSS FUNCTIONS ---------------------------------

class FrustumLoss(torch.nn.Module):
    def __init__(self, device='cpu'):
        super(FrustumLoss, self).__init__()
        self.device = device
        self.g_mean_size_arr = torch.nn.Parameter(torch.from_numpy(g_mean_size_arr).type(torch.float32).to(device), requires_grad=False)
        self.heading_bin_centers = torch.nn.Parameter(torch.arange(0,2*np.pi,2*np.pi/NUM_HEADING_BIN, device=device), requires_grad=False) # (NH,)
        self.loss_func_cls = torch.nn.CrossEntropyLoss(reduction='mean')
        self.loss_func_reg = torch.nn.SmoothL1Loss(reduction='mean')
        self.loss_func_reg_nomean = torch.nn.SmoothL1Loss(reduction='none')
        self.center_huber_delta = 2.0
        self.stage1_center_huber_delta = 1.0
        self.h_huber_delta = 1.0
        self.s_huber_delta = 1.0
        self.corners_huber_delta = 1.0

    def _huber(self, input_, delta):
        return delta * delta * self.loss_func_reg(input_ / delta, torch.zeros_like(input_))

    def _huber_nomean(self, input_, delta):
        return delta * delta * self.loss_func_reg_nomean(input_ / delta, torch.zeros_like(input_)) 

    def forward(self, pipeline_output, target, tb_writer=None, iteration=None, mode='train', corner_loss_weight=0.001,  box_loss_weight=1.0):
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
        mask_loss = self.loss_func_cls(pipeline_output['mask_logits'], target['label'])
        # Center regression losses
        center_dist = torch.norm(target['box3d_center'] - pipeline_output['center_prediction'], p=2, dim=-1)
        
        center_loss = self._huber(center_dist, self.center_huber_delta)
        
        stage1_center_dist = torch.norm(target['box3d_center'] - \
            pipeline_output['center_1'], dim=-1)
        stage1_center_loss = self._huber(stage1_center_dist, self.stage1_center_huber_delta)
        
        # Heading loss
        heading_class_loss = self.loss_func_cls(pipeline_output['heading_scores'], target['angle_class'])
        hcls_onehot = self.one_hot(target['angle_class'], NUM_HEADING_BIN) # BxNUM_HEADING_BIN

        heading_residual_normalized_label = \
            target['angle_residual'] / (np.pi/NUM_HEADING_BIN)
        heading_residual_normalized_loss = self._huber(torch.sum( \
            pipeline_output['heading_residuals_normalized']*hcls_onehot, 1) - \
            heading_residual_normalized_label, self.h_huber_delta)
        
        # Size loss
        size_class_loss = self.loss_func_cls(pipeline_output['size_scores'], target['size_class'].type(torch.LongTensor).to(target['size_class'].device))
        
        scls_onehot = self.one_hot(target['size_class'], NUM_SIZE_CLUSTER) # BxNUM_SIZE_CLUSTER
        scls_onehot_tiled = scls_onehot.unsqueeze(-1) # BxNUM_SIZE_CLUSTERx1
        predicted_size_residual_normalized = torch.sum( \
            pipeline_output['size_residuals_normalized']*scls_onehot_tiled, 1) # Bx3

        mean_size_label = torch.sum( \
            scls_onehot_tiled * self.g_mean_size_arr, 1) # Bx3
        size_residual_normalized = target['size_residual'] / mean_size_label
        
        size_normalized_dist = torch.norm( \
            size_residual_normalized - predicted_size_residual_normalized, p=2,
            dim=-1)
        size_residual_normalized_loss = self._huber(size_normalized_dist, self.s_huber_delta)

        # Corner loss
        # We select the predicted corners corresponding to the 
        # GT heading bin and size cluster.
        corners_3d = self.get_box3d_corners(pipeline_output['center_prediction'],
            pipeline_output['heading_residuals'],
            pipeline_output['size_residuals']) # (B,NH,NS,8,3)
        gt_mask = hcls_onehot.unsqueeze(2) * scls_onehot.unsqueeze(1) # (B,NH,NS)
        corners_3d_pred = torch.sum( \
            gt_mask.unsqueeze(-1).unsqueeze(-1) * corners_3d,
            dim=(1,2))# (B,8,3)
        heading_label = target['angle_residual'].unsqueeze(1) + self.heading_bin_centers # (B,NH)
        heading_label = torch.sum(hcls_onehot*heading_label, dim=1)
        size_label = self.g_mean_size_arr + target['size_residual'].unsqueeze(1) # (1,NS,3) + (B,1,3) = (B,NS,3)
        size_label = torch.sum(scls_onehot.unsqueeze(-1)*size_label, dim=1) # (B,3)
        corners_3d_gt = self.get_box3d_corners_helper( \
            target['box3d_center'], heading_label, size_label) # (B,8,3)
        corners_3d_gt_flip = self.get_box3d_corners_helper( \
            target['box3d_center'], heading_label+np.pi, size_label) # (B,8,3)

        corners_loss = torch.min(
          self._huber_nomean(torch.norm(corners_3d_pred - corners_3d_gt, p=2, dim=-1), self.corners_huber_delta).mean(dim=-1),
          self._huber_nomean(torch.norm(corners_3d_pred - corners_3d_gt_flip, p=2, dim=-1), self.corners_huber_delta).mean(dim=-1)).mean()
        if tb_writer is not None:
            tb_writer.add_scalar(mode+'mask loss', mask_loss.item(), iteration)
            tb_writer.add_scalar(mode+'center loss', box_loss_weight * center_loss.item(), iteration)
            tb_writer.add_scalar(mode+'heading loss', box_loss_weight * heading_class_loss.item(), iteration)
            tb_writer.add_scalar(mode+'size loss', box_loss_weight * size_class_loss.item(), iteration)
            tb_writer.add_scalar(mode+'heading residual normalized loss', box_loss_weight * 20 * heading_residual_normalized_loss.item(), iteration)
            tb_writer.add_scalar(mode+'size residual normalized loss', box_loss_weight * 20 * size_residual_normalized_loss.item(), iteration)
            tb_writer.add_scalar(mode+'stage1 center loss', box_loss_weight * stage1_center_loss.item(), iteration)
            tb_writer.add_scalar(mode+'corners loss', box_loss_weight * corner_loss_weight * corners_loss.item(), iteration)

        total_loss = mask_loss + box_loss_weight * (center_loss + \
            heading_class_loss + size_class_loss + \
            heading_residual_normalized_loss*20 + \
            size_residual_normalized_loss*20 + \
            stage1_center_loss + \
            corner_loss_weight*corners_loss)

        return total_loss 
    
    
    def get_box3d_corners_helper(self, centers, headings, sizes):
        """ Input: (N,3), (N,), (N,3), Output: (N,8,3) """
        N = centers.shape[0]
        l = sizes[:, 0].reshape(-1,1) # (N,1)
        w = sizes[:, 1].reshape(-1,1) # (N,1)
        h = sizes[:, 2].reshape(-1,1) # (N,1)

        #print l,w,h
        x_corners = torch.cat([l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2], dim=1) # (N,8)
        y_corners = torch.cat([h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2], dim=1) # (N,8)
        z_corners = torch.cat([w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2], dim=1) # (N,8)
        corners = torch.cat([x_corners.unsqueeze(1), y_corners.unsqueeze(1), z_corners.unsqueeze(1)], dim=1) # (N,3,8)
        #print x_corners, y_corners, z_corners
        c = torch.cos(headings)
        s = torch.sin(headings)
        ones = torch.ones_like(c)
        zeros = torch.zeros_like(c)
        row1 = torch.stack([c,zeros,s], dim=1) # (N,3)
        row2 = torch.stack([zeros,ones,zeros], dim=1)
        row3 = torch.stack([-s,zeros,c], dim=1)
        R = torch.cat([row1.unsqueeze(1), row2.unsqueeze(1), row3.unsqueeze(1)], dim=1) # (N,3,3)
        #print row1, row2, row3, R, N
        corners_3d = torch.matmul(R, corners) # (N,3,8)
        corners_3d += centers.unsqueeze(2).repeat(1,1,8) # (N,3,8)
        corners_3d = torch.transpose(corners_3d, 2, 1) # (N,8,3)
        return corners_3d
    
    def get_box3d_corners(self, center, heading_residuals, size_residuals):
        """ 
        Inputs:
            center: (B,3)
            heading_residuals: (B,NH)
            size_residuals: (B,NS,3)
        Outputs:
            box3d_corners: (B,NH,NS,8,3) tensor
        """
        batch_size = center.shape[0]
        headings = heading_residuals + self.heading_bin_centers # (B,NH)
        
        mean_sizes = self.g_mean_size_arr + size_residuals # (B,NS,1)
        sizes = mean_sizes + size_residuals # (B,NS,3)
        sizes = sizes.unsqueeze(1).repeat(1,NUM_HEADING_BIN,1,1) # (B,NH,NS,3)
        headings = headings.unsqueeze(-1).repeat(1,1,NUM_SIZE_CLUSTER) # (B,NH,NS)
        centers = center.unsqueeze(1).unsqueeze(1).repeat(1,NUM_HEADING_BIN, NUM_SIZE_CLUSTER,1) # (B,NH,NS,3)

        N = batch_size*NUM_HEADING_BIN*NUM_SIZE_CLUSTER
        corners_3d = self.get_box3d_corners_helper(centers.view(N,3), headings.view(N), sizes.view(N,3))

        return torch.reshape(corners_3d, (batch_size, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, 8, 3))

    def one_hot(self, data, num):
        vec = torch.zeros(data.shape[0], num, device=self.device)
        for i in range(vec.shape[0]):
            vec[i][data[i]] = 1
        return vec

