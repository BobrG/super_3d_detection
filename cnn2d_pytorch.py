from torch import nn
from torch.nn import functional as F
from tochvision import models 
from utils import proposal_layer

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
        layres.append(nn.Conv2d(1024, 1024, kernel_size=1)) # fc7
        layers.append(nn.Conv2d(1024, 1024, kernel_size=2, stride=2)) # conv8

        return layers

def _upsample_add(self, x, y):
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
# Fast-RCNN predicts final 2D detection bounding boxes from the region proposals
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

# # Base Detector class
# def Detector2D(nn.Module):
#     """
#     Args:
#         extractor (nn.Module): A module that takes a BCHW image
#             array and returns feature maps.
#         rpn (nn.Module): A module that takes feature maps produced
#             by extractor and propose class bounding boxes.
#         head (nn.Module): A module that takes a BCHW variable,
#             RoIs and batch indices for RoIs. This returns class
#             dependent localization paramters and class scores.
#         loc_normalize_mean (tuple of four floats): Mean values of
#             localization estimates.
#         loc_normalize_std (tupler of four floats): Standard deviation
#             of localization estimates.
#     """
#     def __init__(self, extractor, rpn, head):

def RPN(nn.Module):
    def __init__(self, base, fpl):
        super().__init__()
        anchor_sizes = [16, 32, 64, 128, 256]
        self.num_layers_base = len(base)
        self.num_layers_fpl = len(fpl)
        # feature extractor (bottom-up pathway)
        self.vgg = nn.ModuleList(base)
        # feature pyramid layers
        self.fpl = nn.ModuleList(fpl)
        # predictor head
        self.anchor_scales = []
        self.conv = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.score_con = nn.Conv2d(mid_channels, len(self.anchor_scales) * 2, 1, 1, 0)
        self.bbox_conv = nn.Conv2d(mid_channels, len(self.anchor_scales) * 4, 1, 1, 0)
        
    def forward(self, x):
        output_base = [x]
        # vgg part
        for i, layer in enumerate(self.vgg):
            output_base.append(layer(output_base[i]))
        # features from conv3_3, conv4_3, conv5_3, fc7, conv8
        output = [output_base[8], output_base[13], output_base[18], output_base[-2], output_base[-1]]
        
        # feature pyramid layers
        output_fpl = [self.fpl[0](output[-1])]
        separator_idx = round(self.num_layers_fpl // 2)# from where smooth layers start
        for i, layer in enumerate(self.fpl, self.num_layers_base + 1):
            if i < separator_idx:
                output_fpl.append(_upsample_add(output[i], layer(output_fpl[i]))) # lateral
            else:
                output_fpl.append(layer(output_fpl[i % separator])) # smooth

        # prediction head
        box_pred = []
        score_pred = []
        for out in output_fpl[separator_idx:]:
            out_conv = self.conv(out)
            box_pred.append(self.bbox_conv(out_conv))
            rpn_cls_score = self.score_conv(out_conv)
            rpn_cls_score_reshape = self.reshape_layer(rpn_cls_score, 2)
            rpn_cls_prob = F.softmax(rpn_cls_score_reshape)
            rpn_cls_prob_reshape = self.reshape_layer(rpn_cls_prob, len(self.anchor_scales)*3*2)
            score_pred.append(rpn_cls_prob_reshape)

        # proposal layer
        rois = proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, im_info, self.anchor_scales).view(-1, 5)

        # generating training labels and build the rpn loss
        # if self.training:
        #     assert gt_boxes is not None
        #     rpn_data = self.anchor_target_layer(rpn_cls_score, gt_boxes, gt_ishard, dontcare_areas,
        #                                         im_info, self._feat_stride, self.anchor_scales)
        #     self.cross_entropy, self.loss_box = self.build_loss(rpn_cls_score_reshape, rpn_bbox_pred, rpn_data)

        return output, rois

        def reshape_layer(x, d):
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


class FasterRCNN(nn.Module):
    n_classes = # number of classes
    classes = # classes list 
    PIXEL_MEANS = # ???
    SCALES = # ??? 
    MAX_SIZE = # ??? 

    def __init__(self, classes=None, debug=False):
        super(FasterRCNN, self).__init__()

        if classes is not None:
            self.classes = classes
            self.n_classes = len(classes)

        self.rpn = RPN()
        self.roi_pool = RoIPool(7, 7, 1.0/16)
        self.fc6 = FC(512 * 7 * 7, 4096)
        self.fc7 = FC(4096, 4096)
        self.score_fc = FC(4096, self.n_classes, relu=False)
        self.bbox_fc = FC(4096, self.n_classes * 4, relu=False)

        # loss
        self.cross_entropy = None
        self.loss_box = None
    
    def loss(self):
        return self.cross_entropy + self.loss_box * 10

    def forward(self, x):
        features, rois = self.rpn(x)

        if self.training:
            roi_data = self.proposal_target_layer(rois, gt_boxes, gt_ishard, dontcare_areas, self.n_classes)
            rois = roi_data[0]

        # roi pool
        pooled_features = self.roi_pool(features, rois)
        x = pooled_features.view(pooled_features.size()[0], -1)
        x = self.fc6(x)
        x = F.dropout(x, training=self.training)
        x = self.fc7(x)
        x = F.dropout(x, training=self.training)

        cls_score = self.score_fc(x)
        cls_prob = F.softmax(cls_score)
        bbox_pred = self.bbox_fc(x)

        if self.training:
            self.cross_entropy, self.loss_box = self.build_loss(cls_score, bbox_pred, roi_data)

        return cls_prob, bbox_pred, rois

    