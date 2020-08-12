# CVPR 2020: The Top Object Detection Papers

转自 medium,原链接:

https://heartbeat.fritz.ai/cvpr-2020-the-top-object-detection-papers-f920a6e41233

[![Derrick Mwiti](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-08-02-031542.jpg)](https://heartbeat.fritz.ai/@mwitiderrick?source=post_page-----f920a6e41233----------------------)

[Derrick Mwiti](https://heartbeat.fritz.ai/@mwitiderrick?source=post_page-----f920a6e41233----------------------)Following

![Image for post](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-08-02-031546.jpg)

The recently-concluded CVPR 2020 had quite a large number of contributions in pushing [object detection](https://www.fritz.ai/object-detection/) forward. In this piece, we’ll look at a couple of the especially impressive papers.

------

# A Hierarchical Graph Network for 3D Object Detection on Point Clouds

This paper proposes a graph convolution-based (GConv) hierarchical graph network (HGNet) for 3D object detection. It processes raw point clouds directly to predict 3D bounding boxes. HGNet is able to capture the relationship of the points and uses multi-level semantics for object detection.

[CVPR 2020 Open Access RepositoryA Hierarchical Graph Network for 3D Object Detection on Point Clouds Jintai Chen, Biwen Lei, Qingyu Song, Haochao Ying…openaccess.thecvf.com](http://openaccess.thecvf.com/content_CVPR_2020/html/Chen_A_Hierarchical_Graph_Network_for_3D_Object_Detection_on_Point_CVPR_2020_paper.html)

HGNet consists of three main components:

- a GConv based U-shape network (GU-net)
- a Proposal Generator
- a Proposal Reasoning Module (ProRe Module) — that uses a fully-connected graph to reason on the proposals

![Image for post](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-08-02-031859.png)

The authors present a shape-attentive GConv (SA-GConv) to capture the local shape features. This is done by modeling the relative geometric positions to describe object shapes.

![Image for post](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-08-02-031547.png)

The SA-GConv based U-shape network captures the multi-level features. They are then mapped onto an identical feature space by a voting module and used to generate proposals. In the next step, a GConv based Proposal Reasoning Module uses the proposals to predict bounding boxes.

Here are some of the performance results obtained on the [SUN RGB-D V1](https://rgbd.cs.princeton.edu/) dataset.

![Image for post](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-08-02-031548.png)

------

# HVNet: Hybrid Voxel Network for LiDAR Based 3D Object Detection

In this paper, the authors present the Hybrid Voxel Network (HVNet), a one-stage network for point cloud-based 3D object detection for autonomous driving.

![Image for post](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-08-02-031552.png)

[CVPR 2020 Open Access RepositoryHVNet: Hybrid Voxel Network for LiDAR Based 3D Object Detection Maosheng Ye, Shuangjie Xu, Tongyi Cao ; The IEEE/CVF…openaccess.thecvf.com](http://openaccess.thecvf.com/content_CVPR_2020/html/Ye_HVNet_Hybrid_Voxel_Network_for_LiDAR_Based_3D_Object_Detection_CVPR_2020_paper.html)

The voxel feature encoding (VFE) method used in this paper contains three steps:

- Voxelization — assigning of a point cloud to a 2D voxel grid
- Voxel Feature Extraction — computation of a grid-dependent point-wise feature that’s fed to a PointNet style feature encoder
- Projection — aggregation of the point-wise feature to the voxel-level feature and projection to their original grid. This forms a pseudo-image feature map

![Image for post](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-08-02-031938.png)

The size of the voxel is very important in VFE methods. Smaller voxel sizes capture finer geometry features. They’re also better at object localization, but take longer at inference. Faster inference speeds can be obtained using a coarser voxel, since it leads to a smaller feature map. Its performance is inferior, however.

The authors propose the Hybrid Voxel Network (HVNet) to enable the utilization of fine-grained voxel features. It’s made up of three steps:

- Multi-Scale Voxelization — the creation of a set of feature voxel scales and the assignment of each to multiple voxels.
- Hybrid Voxel Feature Extraction —computing of a voxel dependent feature for each scale and feeding it into the attentive feature encoder (AVFE). Features from each voxel scale are concatenated point-wise.
- Dynamic Feature Projection — Projecting the feature back to the pseudo-image by creating a set of multi-scale project voxels.

![Image for post](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-08-02-031619.png)

Here are the results obtained on the KITTI dataset.

![Image for post](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-08-02-031558.png)

------

> State-of-the-art object detection models can also work in real-time on mobile devices. [And Fritz AI Studio allows you to build, test, and deploy custom object detection models to iOS and Android. Start building for free](https://www.fritz.ai/product/studio.html?utm_campaign=object-detection-cvpr&utm_source=heartbeat).

# Point-GNN: Graph Neural Network for 3D Object Detection in a Point Cloud

Authors of this paper present a graph neural network — Point-GNN — to detect objects from a [LiDAR](https://en.wikipedia.org/wiki/National_lidar_dataset) point cloud. The network predicts the category and shape of the object that each vertex in the graph belongs to. Point-GNN has an auto-regression mechanism that detects multiple objects in a single shot.

[CVPR 2020 Open Access RepositoryPoint-GNN: Graph Neural Network for 3D Object Detection in a Point Cloud Weijing Shi, Raj Rajkumar ; The IEEE/CVF…openaccess.thecvf.com](http://openaccess.thecvf.com/content_CVPR_2020/html/Shi_Point-GNN_Graph_Neural_Network_for_3D_Object_Detection_in_a_CVPR_2020_paper.html)

The proposed method has three components:

- graph construction: a voxel downsampled point cloud is used for graph construction
- a graph neural network of *T* iterations
- bounding box merging and scoring


![Image for post](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-08-02-031603.png)

Here’re the results obtained on the KITTI dataset:

![Image for post](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-08-02-031617.png)

The code is available here:

[WeijingShi/Point-GNNThis repository contains a reference implementation of our Point-GNN: Graph Neural Network for 3D Object Detection in a…github.com](https://github.com/WeijingShi/Point-GNN)

------

# Camouflaged Object Detection

This paper addresses the challenge of detecting objects that are embedded in their surroundings — camouflaged object detection (COD). The authors also present a new dataset called COD10K. It contains 10,000 images covering camouflaged objects in many natural scenes. It has 78 object categories. The images are annotated with category labels, bounding boxes, instance-level, and matting-level labels.

![Image for post](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-08-02-031636.png)

[CVPR 2020 Open Access RepositoryCamouflaged Object Detection Deng-Ping Fan, Ge-Peng Ji, Guolei Sun, Ming-Ming Cheng, Jianbing Shen, Ling Shao ; The…openaccess.thecvf.com](http://openaccess.thecvf.com/content_CVPR_2020/html/Fan_Camouflaged_Object_Detection_CVPR_2020_paper.html)

![Image for post](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-08-02-032046.png)

The authors develop a COD framework called a Search Identification Network (SINet). The code is available here:

[DengPingFan/SINetThis repository includes detailed introduction, strong baseline (Search & Identification Net, SINet), and one-key…github.com](https://github.com/DengPingFan/SINet/)

The network has two main modules:

- the search module (SM) for searching for a camouflaged object
- the identification module (IM) for detecting the object

![Image for post](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-08-02-031615.png)

Here are the results obtained on various datasets:

![Image for post](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-08-02-031627.png)

------

## Few-Shot Object Detection with Attention-RPN and Multi-Relation Detector

This paper proposes a few-shot object detection network whose objective is to detect objects of unseen categories that have a few annotated examples.

[CVPR 2020 Open Access RepositoryFew-Shot Object Detection With Attention-RPN and Multi-Relation Detector Qi Fan, Wei Zhuo, Chi-Keung Tang, Yu-Wing Tai…openaccess.thecvf.com](http://openaccess.thecvf.com/content_CVPR_2020/html/Fan_Few-Shot_Object_Detection_With_Attention-RPN_and_Multi-Relation_Detector_CVPR_2020_paper.html)

Their method includes an attention-RPN, multi-relation detector, and a contrastive training strategy. The method takes advantage of the similarity between the few-shot support set and query set to identify new objects, while also reducing false identification. The authors also contribute a new dataset that contains 1000 categories with objects that have high-quality annotations.

[fanq15/Few-Shot-Object-Detection-DatasetThe original code is released in fanq15/FSOD-code! (13/5/2020) Please forget the detectron2 based code. I will directly…github.com](https://github.com/fanq15/Few-Shot-Object-Detection-Dataset)

The network architecture consists of a weight-shared framework that has multiple branches—one branch is the query set, while the rest are for the support set. The query branch of the weight-shared framework is a Faster R-CNN network.



![Image for post](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-08-02-031635.png)

The authors introduce an attention-RPN and detector with multi-relation modules to produce accurate parsing between support and the potential boxes in the query.



![Image for post](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-08-02-31640.png)

Here are some results obtained on the ImageNet dataset.



![Image for post](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-08-02-031640.png)

Here are some observations obtained on a number of datasets.

![Image for post](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-08-02-031626.png)

------

> Deploying object detection models to mobile can lead to engaging user experiences and lower costs. [Subscribe to the Fritz AI Newsletter to learn more about what’s possible with mobile machine learning](https://www.fritz.ai/newsletter?utm_campaign=fritzai-newsletter-values4&utm_source=heartbeat).

------

# D2Det: Towards High-Quality Object Detection and Instance Segmentation

Authors of this paper propose D2Det, a method that addresses both precise localization and accurate classification. They introduce a dense local regression that predicts multiple dense box offsets for an object proposal. This enables them to achieve precise localization.

[CVPR 2020 Open Access RepositoryD2Det: Towards High Quality Object Detection and Instance Segmentation Jiale Cao, Hisham Cholakkal, Rao Muhammad Anwer…openaccess.thecvf.com](http://openaccess.thecvf.com/content_CVPR_2020/html/Cao_D2Det_Towards_High_Quality_Object_Detection_and_Instance_Segmentation_CVPR_2020_paper.html)

The authors also introduce a discriminative RoI pooling scheme in order to achieve accurate classification. The pooling scheme samples from several sub-regions of a proposal and performs adaptive weighting to get discriminating features.

The code is available at:

[JialeCao001/D2DetThis code is an official implementation of "D2Det: Towards High Quality Object Detection and Instance Segmentation…github.com](https://github.com/JialeCao001/D2Det)

The method is based on the standard Faster R-CNN framework. In this method, the traditional box offset regression of Faster R-CNN is replaced by the proposed dense local regression. In the method, classification is enhanced by the discriminative RoI pooling.



![Image for post](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-08-02-031641.png)

In the two-stage method, a region proposal network (RPN) is used in the first stage, while separate classification and regression branches are put into effect in the second stage. The classification branch is based on discriminative pooling. The local regression branch’s objective is exact localization of an object.



![Image for post](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-08-02-031629.png)

Here are the results obtained on the MS COCO dataset:

![Image for post](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-08-02-031633.png)

------

## Final Thought

When it comes to object detection and a whole host of other computer vision tasks, CVPR 2020 offered plenty more. Here’s the open source repo of all the conference papers, in case you’d like to explore further.

[CVPR 2020 Open Access RepositoryUnsupervised Learning of Probably Symmetric Deformable 3D Objects From Images in the Wild Footprints and Free Space…openaccess.thecvf.com](http://openaccess.thecvf.com/CVPR2020.py)

------

*Editor’s Note:*[ ***Heartbeat\***](http://heartbeat.fritz.ai/) *is a contributor-driven online publication and community dedicated to exploring the emerging intersection of mobile app development and machine learning. We’re committed to supporting and inspiring developers and engineers from all walks of life.*

*Editorially independent, Heartbeat is sponsored and published by*[ ***Fritz AI\***](http://fritz.ai/)*, the machine learning platform that helps developers teach devices to see, hear, sense, and think. We pay our contributors, and we don’t sell ads.*

*If you’d like to contribute, head on over to our*[ ***call for contributors\***](https://heartbeat.fritz.ai/call-for-contributors-october-2018-update-fee7f5b80f3e)*. You can also sign up to receive our weekly newsletters (*[***Deep Learning Weekly\***](https://www.deeplearningweekly.com/) *and the* [***Fritz AI Newsletter\***](https://www.fritz.ai/newsletter/?utm_campaign=fritzai-newsletter&utm_source=heartbeat-statement)*), join us on*[ ](https://join.slack.com/t/fritz-ai-community/shared_invite/enQtNTY5NDM2MTQwMTgwLWU4ZDEwNTAxYWE2YjIxZDllMTcxMWE4MGFhNDk5Y2QwNTcxYzEyNWZmZWEwMzE4NTFkOWY2NTM0OGQwYjM5Y2U)[***Slack\***](http://fritz.ai/slack)*, and follow Fritz AI on*[ ***Twitter\***](https://twitter.com/fritzlabs) *for all the latest in mobile machine learning.*