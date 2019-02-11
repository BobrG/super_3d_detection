# DRAFT
# 2D Detection Net for Frustrum PointNet pipeline.
## Feature Pyramid Network integrated with Faster R-CNN.

Authors of Frustrum PointNet model propose a modified network architecture for object detectors. 
The architecture combines the encoder-decoder structure (here: Feature Pyramid Network) to generate region proposals from multiple feature maps and Fast R-CNN to predict final 2D detection bounding boxes from the region proposals. 
Here an explanation of current implementation of both parts of combination is presented.

### Part 1: Feature Pyramid Region Proposal Network.
This network could also be divided on two branches - base and fpl. A reduced VGG16 architecture from SSD is used as base and all max pooling are changed to convolution layers with 3 × 3 kernel size and stride of 2. Also extra conv8 layer is added and it halves the fc7 feature map size.

The second branch - fpl - starts at the first one's output. It uses nearest neighbour upsampling to re-increase the resolution back to the original one. It does not contain convolutions. All layers have 256 channels. 

There are connections between the layers of the first and second branch. These connections are simply 1x1 convolutions followed by an addition (similar to residual connections). Only layers with similar height and width are connected.
Feature maps from outputs of some base network's layers are passed through the proposed connections. Here feature pyramid layers are added from conv3_3, conv4_3, conv5_3, fc7 and conv8. 

Here is presented a FPN architecture: 

  ![Feature Pyramid Network Classic](https://habrastorage.org/webt/2v/e9/fg/2ve9fgmxb0yuzkrn0df5lznzycu.jpeg)
  
After feature pyramid layers all the outputs are passed to prediction head. Prediction head architecture is presented on picture below:

  ![prediction head](https://habrastorage.org/webt/8c/km/ga/8ckmgaq4dnz5xy64awuhd159d8m.png)
  
**here add more description of inputs and outputs**

**Integration with Faster R-CNN**

While usually an RPN is applied to a single feature map of one scale, in their case it is applied to many feature maps of varying scales. In current implementation scale factors are 16, 32, 64, 128 and 256 (for conv3_3 - conv8 respectively). 
The Region Proposal Network uses the same parameters for all scales.

Anchor boxes are used for pyramid of features and boxes have different aspect ratios, not different scales (as scales are already covered by their feature map heights/widths). Ground truth bounding boxes are associated with the best matching anchor box (i.e. one box among all scales). Anchor boxes are generated between fpl and prediction head. 

### Part 2: RCNN part.

  ![Pipeline](https://habrastorage.org/webt/tc/h6/v8/tch6v8beukqhvgdicxxliga9hji.png)
  
  
