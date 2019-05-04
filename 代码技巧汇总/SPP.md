## [全连接层的好伴侣：空间金字塔池化（SPP） || 5分钟看懂CV顶刊论文](https://zhuanlan.zhihu.com/p/64510297)ß

> 原论文地址：
> [Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1406.4729)

## **本文结构**

1. **SSPNet的动机**
2. **SPP Layer的实现**
3. **SPP有啥好处**
4. **SPPNet用于图像分类**
5. **SPPNet用于物体检测**

## **一、SSPNet的动机**

一般而言，对于一个CNN模型，可以将其分为两个部分：

1. 前面包含卷积层、激活函数层、池化层的特征提取网络，下称CNN_Pre
2. 后面的全连接网络，下称CNN_Post

许多CNN模型都对输入的图片大小有要求，实际上CNN_Pre对输入的图片没有要求，可以简单认为其将图片缩小了固定的倍数，而CNN_Post对输入的维度有要求，简而言之，限制输入CNN模型的图片尺寸是为了迁就CNN_Post。



而本文的立意就在于，找到一种合适的方式，无论CNN_Pre输出的feature maps尺寸是怎样，都能输出固定的维度传给CNN_Post。如下图，而给出的方案即SPP：空间金字塔池化

![img](https://ws1.sinaimg.cn/large/006tNc79ly1g2pa0okplqj30go0ah0ul.jpg)



## **二、SPP Layer的实现**

> Pytorch 实现参考：
> [https://github.com/GitHberChen/Pytorch-Implement-of-Papers/blob/master/SPP_layer.py](https://link.zhihu.com/?target=https%3A//github.com/GitHberChen/Pytorch-Implement-of-Papers/blob/master/SPP_layer.py)

```python3
import math
import torch
from torch import nn
# https://github.com/yueruchen/sppnet-pytorch/blob/master/spp_layer.py


def spatial_pyramid_pool(self, previous_conv, num_sample, previous_conv_size, out_pool_size):
    '''
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer

    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    '''
    # print(previous_conv.size())
    for i in range(len(out_pool_size)):
        # print(previous_conv_size)
        h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
        w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
        h_pad = (h_wid * out_pool_size[i] - previous_conv_size[0] + 1) / 2
        w_pad = (w_wid * out_pool_size[i] - previous_conv_size[1] + 1) / 2
        maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
        x = maxpool(previous_conv)
        if (i == 0):
            spp = x.view(num_sample, -1)
            # print("spp size:",spp.size())
        else:
            # print("size:",spp.size())
            spp = torch.cat((spp, x.view(num_sample, -1)), 1)
    return spp
```



SPP的本质就是多层maxpool，只不过为了对于不同尺寸大小 ![a\times a](https://www.zhihu.com/equation?tex=a\times+a) 的featur map 生成固定大小 ![n\times n](https://www.zhihu.com/equation?tex=n\times+n) 的的输出，那么 ![pool_{n\times n}](https://www.zhihu.com/equation?tex=pool_{n\times+n}) 的滑窗win大小，以及步长str都要作自适应的调整：

![win = ceil(https://www.zhihu.com/equation?tex=win+%3D+ceil(a%2Fn)+\\+str%3Dfloor(a%2Fn)) \\ str=floor(a/n)](https://www.zhihu.com/equation?tex=win+%3D+ceil%28a%2Fn%29+%5C%5C+str%3Dfloor%28a%2Fn%29)

ceil、floor分别表示上取整、下取整。

然后多个不同固定输出尺寸的 ![Pool](https://www.zhihu.com/equation?tex=Pool) 组合在一起就构成了SPP Layer，在论文中就用了 ![pool_{5\times 5}](https://www.zhihu.com/equation?tex=pool_{5\times+5})、![pool_{4\times 4}](https://www.zhihu.com/equation?tex=pool_{4\times+4})、![pool_{3\times 3}](https://www.zhihu.com/equation?tex=pool_{3\times+3})、![pool_{2\times 2}](https://www.zhihu.com/equation?tex=pool_{2\times+2})、![pool_{1\times 1}](https://www.zhihu.com/equation?tex=pool_{1\times+1})的组合，对于尺寸为 ![(https://www.zhihu.com/equation?tex=(c%2Ca%2Ca))](https://www.zhihu.com/equation?tex=%28c%2Ca%2Ca%29) 的feature maps，该组合的输出为 ![(https://www.zhihu.com/equation?tex=(c%2C50))](https://www.zhihu.com/equation?tex=%28c%2C50%29)

![img](https://ws2.sinaimg.cn/large/006tNc79ly1g2pa0s91w9j30go0e2764.jpg)









## **三、SPP有啥好处？**

对于不同尺寸的CNN_Pre输出能够输出固定大小的向量当然是其最大的好处，除此之外SPP的优点还有：

1. 可以提取不同尺寸的空间特征信息，可以提升模型对于空间布局和物体变性的鲁棒性。
2. 可以避免将图片resize、crop成固定大小输入模型的弊端。

resize、crop有啥弊端呢？一方面是resize会导致图片中的物体尺寸变形，比如下面这个劳拉；另一方面，crop会导致图片不同位置的信息出现频率不均衡，例如图片中间的部分会比角落的部分会被CNN看到更多次。

![img](https://ws1.sinaimg.cn/large/006tNc79ly1g2pa0pkbfsj30go09emyp.jpg)

![img](https://ws2.sinaimg.cn/large/006tNc79ly1g2pa0sre82j30go0hidie.jpg)



## **四、SPPNet用于图像分类**

将SPP层添加到原有的CNN模型中可以提高其模型表现，有趣的是，最大的提升出现在表现最好的网络Overfeat-7上，提升为top-1 的准确率提升1.65%，由于SPP层输出不变的特性，训练时可以采用不同尺寸的图片，如果在 ![224\times 224](https://www.zhihu.com/equation?tex=224\times+224) 的基础上增加 ![180\times 180](https://www.zhihu.com/equation?tex=180\times+180) 的尺寸进行训练，top-1 准确率会进一步提升0.68%，但如果训练时平均地从180~224选择尺寸进行训练，Overfeat-7的top-1/top-5 error 为 30.06%/10.96%，不如![224\times 224](https://www.zhihu.com/equation?tex=224\times+224)+![180\times 180](https://www.zhihu.com/equation?tex=180\times+180)下的训练结果，原因可能是测试用图是224分辨率的，而平均地从180~224选择尺寸稀释了224的比例。

![img](https://ws4.sinaimg.cn/large/006tNc79ly1g2pa0raovtj30go06at9m.jpg)

另外，作者还用ZFNet使用no SPP ![36\times 256](https://www.zhihu.com/equation?tex=36\times+256) 的全连接和![pool_{4\times 4}](https://www.zhihu.com/equation?tex=pool_{4\times+4})、![pool_{3\times 3}](https://www.zhihu.com/equation?tex=pool_{3\times+3})、![pool_{2\times 2}](https://www.zhihu.com/equation?tex=pool_{2\times+2})、![pool_{1\times 1}](https://www.zhihu.com/equation?tex=pool_{1\times+1})组合的SPP输出 ![30\times 256](https://www.zhihu.com/equation?tex=30\times+256) 的维度给全连接层进行对比，有SPP层的ZFNet表现更好，证明了SPP-ZFNet并非靠更多的参数取胜，而是靠其本身的特性取胜。



## **五、SPPNet用于物体检测**

SPPNet用于物体检测只需对图像（每种尺寸下）做一次特征提取，比当时的 R-CNN 快了很多，具体算法流程如下：

1. 使用selective search 生成2000个Proposal区域。
2. 将图片resize 成 ![s=min(https://www.zhihu.com/equation?tex=s%3Dmin(w%2Ch)\in\{480%2C576%2C688%2C864%2C1200\})\in\{480,576,688,864,1200\}](https://www.zhihu.com/equation?tex=s%3Dmin%28w%2Ch%29%5Cin%5C%7B480%2C576%2C688%2C864%2C1200%5C%7D) ，每种尺寸使用CNN做一次特征提取。
3. 对于每个Proposal找到Proposal区域大小最接近 ![224\times 224](https://www.zhihu.com/equation?tex=224\times+224) 的那个尺寸，找到feature maps上对应的区域，坐上角坐标 ![x_{lt}=floor(https://www.zhihu.com/equation?tex=x_{lt}%3Dfloor(x%2FS)%2B1)+1](https://www.zhihu.com/equation?tex=x_%7Blt%7D%3Dfloor%28x%2FS%29%2B1) ，右下角坐标 ![x _{rb}=ceil(https://www.zhihu.com/equation?tex=x+_{rb}%3Dceil(x%2FS)-1)-1](https://www.zhihu.com/equation?tex=x+_%7Brb%7D%3Dceil%28x%2FS%29-1) ，其中 ![x](https://www.zhihu.com/equation?tex=x) 为Proposal于图像上的像素坐标，S为feature maps 的最终Stride大小，floor、ceil分别为下取整、上取整。找到对应区域后，使用 ![pool_{6\times 6}](https://www.zhihu.com/equation?tex=pool_{6\times+6}) 、![pool_{3\times 3}](https://www.zhihu.com/equation?tex=pool_{3\times+3})、![pool_{2\times 2}](https://www.zhihu.com/equation?tex=pool_{2\times+2})、![pool_{1\times 1}](https://www.zhihu.com/equation?tex=pool_{1\times+1})对该区域的feature map 进行特征提取，得到 ![50\times Channel](https://www.zhihu.com/equation?tex=50\times+Channel) 维的特征向量。
4. 每一个类别都使用训练好的二值分类SVM对这个特征向量进行判定，or 使用全连接层+(class_num+1)路softmax进行类别判定。
5. 边框回归，NMS。

具体的SVM、全连接层微调训练细节可见原论文的4.1章，此不赘述。



SPPNet用于物体检测的效果还是不错的，在取得相当于R-CNN的准度的同时，速度最高提升了102x，具体对比如下列图表：

![img](https://ws2.sinaimg.cn/large/006tNc79ly1g2pa0qjzatj30go0o4jv9.jpg)

![img](https://ws2.sinaimg.cn/large/006tNc79ly1g2pa0ozf8oj30go09emyp.jpg)



最后，有没有觉得SPP和Fast R-CNN的ROI Pooling有些类似？没错，他们是一脉相承的。