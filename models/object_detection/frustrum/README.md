# Frustrum PointNet 

Current directory contains PyTorch implementation of *Frustrum PointNet v1*.

## Model Arhitecture 

Whole pipeline visualisation:

![Pipeline](https://habrastorage.org/webt/tu/yg/ns/tuygnstcq-upgaxqduvwaio5y8k.jpeg)

Structural parts visualisation:

![Structural parts](https://habrastorage.org/webt/3o/pk/t2/3opkt2kckzwqjie7xanxzdudo7k.png)

This directory is constructed as follows:
  * 2d_detector subdirectory contains CNN from the pipeline.
  * components.py contains implementations of Segmentation PointNet, T-Net and 3D Amodal Box Estimation PointNet.
  * network.py contains implementation of whole pipeline.
  * utils.py contains functions of masking, region2frustrum, loss and others.
  * train.py and test.py contains learning and evaluation.

References:

  * Frustum PointNets for 3D Object Detection from RGB-D Data, 2017 <br /> 
  Qi, Charles R and Liu, Wei and Wu, Chenxia and Su, Hao and Guibas, Leonidas J <br /> 
  [github](https://github.com/charlesq34/frustum-pointnets)
