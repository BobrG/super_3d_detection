import torch
import torch.nn as nn
import numpy as np
import math
from utils import generate_anchors, generate_anchors_all_pyramids, bbox_transform_inv, clip_boxes, clip_boxes_batch, nms

class proposal_layer_fpn(nn.Module):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def __init__(self, feat_strides, scales, ratios):
    """
    feat_strides: the downsampling ratios of feature maps to the original input image
    scales: the scales to the basic_anchor (basic anchor is [16, 16])
    ratios: upsampling scale of anchor boxes (could be a 2d array if ratios are variable for different feture map)
    """
        super(proposal_layer_fpn, self).__init__()
        self.anchor_ratios = ratios
        self.feat_strides = feat_strides
        self.fpn_scales = scales 
        self.fpn_ratios = ratios
        self.fpn_anchor_stride = 1
        self.post_nms_topN = 2000 # Number of top scoring boxes to keep after applying NMS to RPN proposals

    def forward(self, input):
    """
    Parameters
    ----------
    input - list contains:
        cls_prob_alls: (BS , H , W , Ax2) outputs of RPN, prob of bg or fg
        bbox_pred_alls: (BS , H , W , Ax4), rgs boxes output of RPN
        im_info: a list of [image_height, image_width, scale_ratios]
        rpn_shapes: width and height of feature map
    ----------
    Returns
    ----------
    rpn_rois : (1 x H x W x A, 5) e.g. [0, x1, y1, x2, y2]
    # Algorithm:
    #
    # for each (H, W) location i
    # generate A anchor boxes centered on cell i
    # apply predicted bbox deltas at cell i to each of the A anchors
    # clip predicted boxes to image
    # remove predicted boxes with either height or width < threshold
    # sort all (proposal, score) pairs by score from highest to lowest
    # take top pre_nms_topN proposals before NMS
    # apply NMS with threshold 0.7 to remaining proposals
    # take after_nms_topN proposals after NMS
    # return the top proposals (-> RoIs top, scores top)
    """

    scores = input[0][:, :, 1]  # batch_size x num_rois x 1
    bbox_deltas = input[1]      # batch_size x num_rois x 4
    im_info = input[2]

    anchors = torch.from_numpy(generate_anchors_all_pyramids(self.fpn_scales, self.anchor_ratios, 
                feat_shapes, self.feat_strides, self.fpn_anchor_stride)).type_as(scores)
    num_anchors = anchors.size(0)
     
    anchors = anchors.view(1, num_anchors, 4).expand(batch_size, num_anchors, 4)

    # Convert anchors into proposals via bbox transformations
    proposals = bbox_transform_inv(anchors, bbox_deltas, batch_size)

    # 2. clip predicted boxes to image
    proposals = clip_boxes(proposals, im_info, batch_size)
    # keep_idx = self._filter_boxes(proposals, min_size).squeeze().long().nonzero().squeeze()
                
    scores_keep = scores
    proposals_keep = proposals

    _, order = torch.sort(scores_keep, 1, True)

    output = scores.new(batch_size, self.post_nms_topN, 5).zero_()

    for i in range(batch_size):
        # # 3. remove predicted boxes with either height or width < threshold
        # # (NOTE: convert min_size to input image scale stored in im_info[2])
        proposals_single = proposals_keep[i]
        scores_single = scores_keep[i]

        # # 4. sort all (proposal, score) pairs by score from highest to lowest
        # # 5. take top pre_nms_topN (e.g. 6000)
        order_single = order[i]

        if pre_nms_topN > 0 and pre_nms_topN < scores_keep.numel():
            order_single = order_single[:pre_nms_topN]

        proposals_single = proposals_single[order_single, :]
        scores_single = scores_single[order_single].view(-1,1)

        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)

        keep_idx_i = nms(torch.cat((proposals_single, scores_single), 1), nms_thresh)
        keep_idx_i = keep_idx_i.long().view(-1)

        if self.post_nms_topN > 0:
            keep_idx_i = keep_idx_i[:self.post_nms_topN]
        proposals_single = proposals_single[keep_idx_i, :]
        scores_single = scores_single[keep_idx_i, :]

        # padding 0 at the end.
        num_proposal = proposals_single.size(0)
        output[i,:,0] = i
        output[i,:num_proposal,1:] = proposals_single

    return output

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _filter_boxes(self, boxes, min_size):
        """Remove all boxes with any side smaller than min_size."""
        ws = boxes[:, :, 2] - boxes[:, :, 0] + 1
        hs = boxes[:, :, 3] - boxes[:, :, 1] + 1
        keep = ((ws >= min_size) & (hs >= min_size))
        return keep
