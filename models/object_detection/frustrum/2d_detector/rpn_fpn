import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from tochvision import models 
from proposal_layer import proposal_layer_fpn

# We take the reduced VGG base network architecture from SSD;
# vgg16 encoder cfg: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'];
# all 'M' are replaced with Conv2d(kernel=3x3, stride=2);    
# add the feature pyramid layers from conv3_3, conv4_3, conv5_3, and fc7 and additional conv8;
def vgg16():
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'N', 'M', 512, 512, 512, 'N', 'M', 512, 512, 512, 'N', 'M']
    encoder = models.vgg16(pretrained=True).features
    layers = []
    for i, layer in enumerate(cfg):
        if cfg == 'M':
            layers.append(nn.Conv2d(layer[i-1], layer[-1], 3, stride=2))
        elif cfg == 'N':
            layers.append(normalization(40))
        else:
            layers.append(encoder[i])
    # original vgg16
    # layers.append(nn.Linear(512 * 7 * 7, 4096)) # fc6
    # layers.append(nn.ReLU(True))
    # layers.append(nn.Dropout())
    # layers.append(nn.Linear(4096, 4096)) 
    # layers.append(nn.ReLU(True))
    # layers.append(nn.Dropout())
        
    layers.append(nn.Conv2d(512, 1024, kernel_size=3)) # fc6 padding=6, dilation=6
    layers.append(nn.Conv2d(1024, 1024, kernel_size=1)) # fc7
    layers.append(nn.Conv2d(1024, 1024, kernel_size=2, stride=2)) # conv8

    return layers

def _upsample_add(x, y):
    '''Upsample and add two feature maps.
    Args:
      x: tensor top feature map to be upsampled.
      y: tensor lateral feature map.
    Returns:
      tensor added feature map.
    Note in PyTorch, when input size is odd, the upsampled feature map
    with `F.upsample(..., scale_factor=2, mode='nearest')`
    maybe not equal to the lateral feature map size.
    e.g.
    original input size: [N,_,15,15] ->
    conv2d feature map size: [N,_,8,8] ->
    upsampled feature map size: [N,_,16,16]
    So we choose bilinear upsample which supports arbitrary output sizes.
    '''
    _,_,H,W = y.size()
    return F.upsample(x, size=(H,W), mode='bilinear') + y

def fpl():
    layers = []
    # Top layer
    layers.append(nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0))  # Reduce channels

    # Lateral layers 
    layers.append(nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)) # lateral 1
    layers.append(nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)) # lateral 2
    layers.append(nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)) # lateral 3
    layers.append(nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)) # lateral 4

    # Smooth layers
    layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)) # smooth 1
    layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)) # smooth 2
    layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)) # smooth 3
    layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)) # smooth 4

    return layers

class RPN_FPN(nn.Module):
    def __init__(self, base=vgg16, fpl=fpl):
        super().__init__()
        self.num_layers_base = len(base)
        self.num_layers_fpl = len(fpl)
        self.num_features = 5 # conv3_3, conv4_3, conv5_3, fc7 and conv8
        
        # feature extractor (bottom-up pathway)
        self.vgg = nn.ModuleList(base)
        
        # feature pyramid layers
        self.fpl = nn.ModuleList(fpl)
        
        # one scale per feature map
        self.anchor_scales = [16, 32, 64, 128, 256]
        # all ratio are used for each feature map, exept conv3_3, for which we ignore 1/3 and 3
        self.anchor_ratios = np.asarray([[1/3, 1/2, 1, 2, 3] for i in range(self.num_features - 1)])
        self.anchor_ratios.append([1/2, 1, 2])
        self.nc_score_out = [1 * len(self.anchor_ratios[i]) * 2 for i in range(self.num_features)] # 2(bg/fg) * 3 (anchor ratios) * 1 (anchor scale)
        self.nc_bbox_out = [1 * len(self.anchor_ratios[i]) * 4 for i in range(self.num_features)] # 4(coords) * 3 (anchors) * 1 (anchor scale)
        
        # predictor head
        self.conv = nn.Conv2d(256, 256, 3, 1, 1)
        self.score_conv = nn.ModuleList([nn.Conv2d(256, self.nc_score_out[i], 1, 1, 0) for i in range(self.num_features)])
        self.bbox_conv = nn.ModuleList([nn.Conv2d(256, self.nc_bbox_out, 1, 1, 0) for i in range(self.num_features)])
        
        # region proposal layer
        self.proposal_layer = proposal_layer_fpn(feat_strides=[4, 8, 16, 32, 64],
                                                 scales=self.anchor_scales,
                                                 ratios=self.anchor_ratios)

    def reshape_layer(self, x, d):
        input_shape = x.size()
        # x = x.permute(0, 3, 1, 2)
        # b c w h
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        # x = x.permute(0, 2, 3, 1)
        return x

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        output_base = [im_data]

        # vgg part
        for i, layer in enumerate(self.vgg):
            output_base.append(layer(output_base[i]))
        # features from conv3_3, conv4_3, conv5_3, fc7, conv8
        feature_maps = torch.tensor([output_base[-1], output_base[-2], output_base[18], output_base[13], output_base[8]])
        
        # feature pyramid layers
        output_fpl = [feature_maps[0]]
        
        for i, layer in enumerate(self.fpl):
            if i < self.num_features: # from where smooth layers start
                output_fpl.append(_upsample_add(feature_maps[i+1], layer(output_fpl[i]))) # lateral
            else:
                output_fpl.append(layer(output_fpl[i % self.num_features])) # smooth

        # prediction head
        bbox_pred = []
        score_pred = []
        cls_scores_pred = []
        cls_probs_pred = []
        rpn_shapes = []
        # multiple outputs from feature pyramid network 
        # pass on input to predictors head consistently
        for i, out in enumerate(output_fpl[self.num_features:]):
            batch_size = out.size(0)
            # return feature map after convrelu layer
            out_conv = F.relu(self.conv(out), inplace=True)
            # get rpn offsets to the anchor boxes
            bbox_pred.append(self.bbox_conv[i](out_conv).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4))
            # get rpn classification score
            cls_score = self.score_conv[i](out_conv)

            cls_score_reshape = self.reshape_layer(cls_score, 2)
            cls_prob_reshape = F.softmax(cls_score_reshape)
            cls_prob = self.reshape_layer(cls_prob, self.nc_score_out)
            score_pred.append(cls_prob_reshape)

            rpn_shapes.append([cls_score.size()[2], cls_score.size()[3]]) # width and height
            cls_scores_pred.append(cls_score.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2))
            cls_probs_pred.append(cls_prob.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2))
        # no explanation to this right now...
        # cls_score_alls = torch.cat(cls_scores_pred, 1)
        cls_prob_alls = torch.cat(cls_probs_pred, 1)
        bbox_pred_alls = torch.cat(bbox_pred, 1)
        # n_pred = cls_score_alls.size(1)

        # proposal layer
        rois = self.RPN_proposal([cls_prob_alls.data, bbox_pred_alls.data, im_info, rpn_shapes])
        
        # generating training labels and build the rpn loss
        # if self.training:
        #     assert gt_boxes is not None
        #     rpn_data = self.anchor_target_layer(rpn_cls_score, gt_boxes, gt_ishard, dontcare_areas,
        #                                         im_info, self._feat_stride, self.anchor_scales)
        #     self.cross_entropy, self.loss_box = self.build_loss(rpn_cls_score_reshape, rpn_bbox_pred, rpn_data)

        return rois, self.rpn_loss_cls, self.rpn_loss_box, feature_maps
