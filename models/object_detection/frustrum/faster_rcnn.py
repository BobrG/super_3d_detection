import torch
from torch import nn
from torch.nn import functional as F
from tochvision import models
from rpn_fpn import RPN_FPN
from roi_pooling import RoIPooling
from proposal_target_layer import ProposalTargetLayer

class FC(nn.Module):
    def __init__(self, in_features, out_features, relu=True):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class FasterRCNN(nn.Module):
    def __init__(self, classes, rpn=RPN_FPN): #class_agnostic):
        super(FasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        #self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.rcnn_rpn = rpn()
        self.rcnn_proposal_target = ProposalTargetLayer(self.n_classes)

        POOLING_SIZE = 7
        self.RCNN_roi_pool = RoIPooling(POOLING_SIZE, POOLING_SIZE, 1.0/16.0)

        self.fc6 = FC(512 * 7 * 7, 4096)
        self.fc7 = FC(4096, 4096)
        self.score_fc = FC(4096, self.n_classes, relu=False)
        self.bbox_fc = FC(4096, self.n_classes * 4, relu=False)

    
    def _PyramidRoI_Feat(self, feat_maps, rois, im_info):
        ''' roi pool on pyramid feature maps'''
        # do roi pooling based on predicted rois
        img_area = im_info[0][0] * im_info[0][1]
        h = rois.data[:, 4] - rois.data[:, 2] + 1
        w = rois.data[:, 3] - rois.data[:, 1] + 1
        roi_level = torch.log(torch.sqrt(h * w) / 224.0)
        roi_level = torch.round(roi_level + 4)
        roi_level[roi_level < 2] = 2
        roi_level[roi_level > 5] = 5
        # roi pooling
        roi_pool_feats = []
        box_to_levels = []
        for i, l in enumerate(range(2, 6)):
            if (roi_level == l).sum() == 0:
                continue
            idx_l = (roi_level == l).nonzero().squeeze()
            box_to_levels.append(idx_l)
            scale = feat_maps[i].size(2) / im_info[0][0]
            feat = self.rcnn_roi_pool(feat_maps[i], rois[idx_l], scale)
            roi_pool_feats.append(feat)
        roi_pool_feat = torch.cat(roi_pool_feats, 0)
        box_to_level = torch.cat(box_to_levels, 0)
        idx_sorted, order = torch.sort(box_to_level)
        roi_pool_feat = roi_pool_feat[order]
            
        return roi_pool_feat

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        rois, rpn_loss_cls, rpn_loss_bbox, feature_maps = self.rcnn_rpn(im_data, im_info, gt_boxes, num_boxes)
        roi_data = self.rcnn_proposal_target(rois, gt_boxes, num_boxes)
        rois, rois_label, gt_assign, rois_target, rois_inside_ws, rois_outside_ws = roi_data
        rois = rois.view(-1, 5)
        rois_label = rois_label.view(-1).long()
        gt_assign = gt_assign.view(-1).long()
        pos_id = rois_label.nonzero().squeeze()
        gt_assign_pos = gt_assign[pos_id]
        rois_label_pos = rois_label[pos_id]
        rois_label_pos_ids = pos_id

        rois_pos = rois[pos_id]
        rois_target = rois_target.view(-1, rois_target.size(2))
        rois_inside_ws = rois_inside_ws.view(-1, rois_inside_ws.size(2))
        rois_outside_ws = rois_outside_ws.view(-1, rois_outside_ws.size(2))

        # pooling features based on rois, output 7x7 map
        roi_pool_feat = self.PyramidRoI_Feat(feature_maps, rois, im_info)

        # feed pooled features to top model
        x = roi_pool_feat.view(roi_pool_feat.size()[0], -1)
        x = self.fc6(x)
        x = F.dropout(x, training=self.training)
        x = self.fc7(x)
        x = F.dropout(x, training=self.training)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(x)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.long().view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # loss (cross entropy) for object classification
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
            # loss (l1-norm) for bounding box regression
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        rois = rois.view(batch_size, -1, rois.size(1))
        cls_prob = cls_prob.view(batch_size, -1, cls_prob.size(1))
        bbox_pred = bbox_pred.view(batch_size, -1, bbox_pred.size(1))

        if self.training:
            rois_label = rois_label.view(batch_size, -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label
        