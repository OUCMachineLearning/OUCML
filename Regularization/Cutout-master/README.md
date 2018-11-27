# Cutout

This repository contains the code for the paper [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/abs/1708.04552). 

## Introduction

Cutout is a simple regularization method for convolutional neural networks which consists of masking out random sections of input images during training. This technique simulates occluded examples and encourages the model to take more minor features into consideration when making decisions, rather than relying on the presence of a few major features.  
  
![Cutout applied to CIFAR-10](https://github.com/uoguelph-mlrg/Cutout/blob/master/images/cutout_on_cifar10.jpg "Cutout applied to CIFAR-10")

Bibtex:  
```
@article{devries2017cutout,  
  title={Improved Regularization of Convolutional Neural Networks with Cutout},  
  author={DeVries, Terrance and Taylor, Graham W},  
  journal={arXiv preprint arXiv:1708.04552},  
  year={2017}  
}
```

## Results and Usage   
### Dependencies  
[PyTorch v0.4.0](http://pytorch.org/)  
[tqdm](https://pypi.python.org/pypi/tqdm)

### ResNet18  
Test error (%, flip/translation augmentation, mean/std normalization, mean of 5 runs) 

| **Network** | **CIFAR-10** | **CIFAR-100** |
| ----------- | ------------ | ------------- |
| ResNet18    | 4.72         | 22.46         |
| ResNet18 + cutout | 3.99   | 21.96         |  

To train ResNet18 on CIFAR10 with data augmentation and cutout:    
`python train.py --dataset cifar10 --model resnet18 --data_augmentation --cutout --length 16`

To train ResNet18 on CIFAR100 with data augmentation and cutout:  
`python train.py --dataset cifar100 --model resnet18 --data_augmentation --cutout --length 8`

### WideResNet
WideResNet model implementation from https://github.com/xternalz/WideResNet-pytorch  

Test error (%, flip/translation augmentation, mean/std normalization, mean of 5 runs)  

| **Network** | **CIFAR-10** | **CIFAR-100** | **SVHN** |
| ----------- | ------------ | ------------- | -------- |
| WideResNet  | 3.87         | 18.8          | 1.60     |
| WideResNet + cutout | 3.08 | 18.41         | **1.30** |

To train WideResNet 28-10 on CIFAR10 with data augmentation and cutout:    
`python train.py --dataset cifar10 --model wideresnet --data_augmentation --cutout --length 16`

To train WideResNet 28-10 on CIFAR100 with data augmentation and cutout:  
`python train.py --dataset cifar100 --model wideresnet --data_augmentation --cutout --length 8`

To train WideResNet 16-8 on SVHN with cutout:  
`python train.py --dataset svhn --model wideresnet --learning_rate 0.01 --epochs 160 --cutout --length 20`

### Shake-shake Regularization Network
Shake-shake regularization model implementation from https://github.com/xgastaldi/shake-shake

Test error (%, flip/translation augmentation, mean/std normalization, mean of 3 runs)  

| **Network** | **CIFAR-10** | **CIFAR-100** |
| ----------- | ------------ | ------------- |
| Shake-shake | 2.86         | 15.58         |
| Shake-shake + cutout | **2.56** | **15.20** |

See README in [shake-shake](https://github.com/uoguelph-mlrg/Cutout/tree/master/shake-shake) folder for usage instructions.
