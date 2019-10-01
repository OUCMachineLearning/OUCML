## [【论文解读】图像/视频去噪中的*Deformable Kernels*](https://zhuanlan.zhihu.com/p/76791923)

[![CV路上一名研究僧](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-10-01-110355.jpg)](https://www.zhihu.com/people/z-bingo)

[CV路上一名研究僧](https://www.zhihu.com/people/z-bingo)



东南大学 电子与通信工程硕士在读

## 1. 简介

这是一篇源自商汤(SenseTime)的论文，文章题目"Learning Deformable Kernels for Image and Video Denoising"。与KPN、MKPN([传送门](https://zhuanlan.zhihu.com/p/73349632))相似，也是一种基于核预测的去噪模型，整体上，网络结构也比较相似，不同点在于，预测出的Kernels对于周围像素点的权重，以及周围像素点的选择方式。**本文源码已开源至Github，欢迎Start/Fork~**

https://github.com/z-bingo/Deformable-Kernels-For-Video-Denoisinggithub.com



从BM3D这类经典的传统图像去噪方式起，不同方法之间的最大区别就在于如何用一种有效的方式选择合适的像素点、以及如何定义这些像素点的权重，进而对这些像素点加权平均可得到近似“干净”的图像，也就达到了图像去噪的目的。

就KPN和MKPN而言，网络的输出是per-pixel的自适应卷积核，KPN仅预测单一尺寸的卷积核；而MKPN可以同时预测多个尺寸的卷积核，进而适应含噪图像中物体的不同尺度，达到既可以较好地保留细节、又可以平滑平坦区域的目的。这篇文章提出的DKPN（暂且称之为DKPN），亦是如此，预测出per-pixel的自适应卷积核，但是对于可变形卷积来说，不仅包含weights和bias两部分，还会包含offsets，用于指示“周围像素”是指哪些像素，打破了传统卷积运算对于方形领域的定义。通过这种方式，可以在采样点有限的情况下，尽可能地增大kernels的感受野，有利于高效地利用图像中的信息。Deformable Kernels和传统的Kernels异同可通过下图体现：

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-10-01-110358.jpg)

​																				Deformable Kernels

## 2. Deformable Convolution

本节通过尽可能简短、简洁的语言来介绍2D以及3D可变形卷积的原理，以及工作方式。

### 2.1 2D Deformable Convolution

对于一个已知的图像X，在去噪任务中可认为是含噪声的图像，那么，卷积核在该图像上卷积可表示为

![[公式]](https://www.zhihu.com/equation?tex=Y(y%2Cx)%3D\sum_{n%3D1}^{N}X(y%2By_n%2C+x%2Bx_n)F(y%2Cx%2Cn))

其中， ![[公式]](https://www.zhihu.com/equation?tex=(y%2Cx)) 表示在图像中的坐标， ![[公式]](https://www.zhihu.com/equation?tex=N) 表示卷积核的个数， ![[公式]](https://www.zhihu.com/equation?tex=(y_n%2Cx_n)) 表示卷积核的采样偏移点，对于传统的方形领域，由 ![[公式]](https://www.zhihu.com/equation?tex=(y_n%2Cx_n)) 构成的集合为（以 ![[公式]](https://www.zhihu.com/equation?tex=3\times+3) 的卷积核为例）：

![[公式]](https://www.zhihu.com/equation?tex=\{(-1%2C-1)%2C+(-1%2C+0)%2C+(-1%2C+1)%2C+(0%2C+-1)%2C+(0%2C0)%2C+(0%2C1)%2C+(1%2C-1)%2C(1%2C0)%2C(1%2C1)\})

基于传统卷积理论，可变形卷积在采样偏移点上额外叠加了一个可以学习的offsets参数，将规则采样问题变为了一个不规则采样问题，若将offests表示为 ![[公式]](https://www.zhihu.com/equation?tex=(\tilde{y}_n%2C+\tilde{x}_n)) ,2D可变性卷积可表示为：

![[公式]](https://www.zhihu.com/equation?tex=Y(y%2Cx)%3D\sum_{n%3D1}^{N}X(y%2By_n%2B\tilde{y}_n%2C++x%2Bx_n+%2B+\tilde{x}_n)F(y%2Cx%2Cn))

在此，有一个问题需要注意，由于offsets是学习得到的偏移量，因此，其一般不会是整数，而是小数，意味着要采样的点不处于规则的像素上。此时，就要通过双线性插值等插值算法根据规则的像素点进行插值，得到想要坐标点的像素值。

### 2.2 3D Deformable Convolution

与2D可变形卷积相似，3D可变性卷积就是3D卷积的拓展，在3D卷积的基础上添加三个可学习的offsets。传统3D卷积一般可用于Volume数据的处理中，对应于去噪领域，即多帧图像去噪或视频去噪，此时网络的输入可以看做 ![[公式]](https://www.zhihu.com/equation?tex=b\times+c\times+d\times+h\times+w) ，分别表示batch_size，输入图像的数量或视频的帧数，颜色通道数以及长和宽，3D卷积会作用于颜色通道、长和宽三个维度。那么，对于3D可变性卷积来说，除了长和宽两个维度的offsets之外，第三个offsets就成了帧与帧之间的偏移，这样就会更有利与网络在不同的帧之间提取有用的信息，对于充分利用连续的视频帧之间的冗余信息是非常有效的。3D可变形卷积可表示为：

![[公式]](https://www.zhihu.com/equation?tex=Y(y%2Cx)%3D\sum_{n%3D1}^{N}X(y%2By_n%2B\tilde{y}_n%2C+x%2Bx_n%2B\tilde{x}_n%2Ct_n%2B\tilde{t}_n)+F(y%2Cx%2Cn))

对于非规则的采样点，在三维空间中通常通过三线性插值来实现。

## 3. DKPN网络结构

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-10-01-110402.jpg)DKPN网络结构图

上图是DKPN网络结构图，其主干网络也是一个基于U-Net的Encoder-Decoder结构，由于可变性卷积多了额外的offsets参数需学习，与KPN不同，U-Net的输出不再是自适应卷积核的权重，可是offsets；offsets经过Sampler采样后，与输入的多帧含噪声图像concat到一起，再经过几个卷积层后可得到自适应卷积核的weights。此时，将刚刚采样得到的图像与weigths相乘便可得到去噪后图像。

需要注意的是，当去噪任务为单帧图像去噪时，每个像素点有两个需要需要的offset，而视频去噪时有三个offsets需要学习。

------

KPN中，作者采用了退火项作为Loss函数的一部分，单独为每帧输入图像预测去噪后的图像，这是为了防止网络很快收敛至一个局部极小值，使得参考帧之外的图像帧在输出去噪图像中几乎不起作用。相似的思想在DKPN中也得到了使用，对于输出的N个采样点，DKPN会将其分为s个组，每个组相互独立地预测去噪后的干净图像，并作为loss函数的退火项，这样就可以有效防止网络很快收敛至局部最小值。将输入图像序列经过Sampler后的图像分为s组，每组平均包含 ![[公式]](https://www.zhihu.com/equation?tex=N%2Fs+) 个采样点，分别表示为 ![[公式]](https://www.zhihu.com/equation?tex=\mathcal{N}_1%2C+\mathcal{N}_2%2C+\ldots%2C\mathcal{N}_s) ，那么，相互独立的去噪后图像可以表示为：

![[公式]](https://www.zhihu.com/equation?tex=Y_i%3Ds\sum_{j\in+\mathcal{N}_i}+X(y%2By_j%2C+x%2Bx_j%2C+t_j)F(y%2Cx%2Cj))

loss函数可以表示为：

![[公式]](https://www.zhihu.com/equation?tex=\ell%3Dl(Y%2CY_{gt})%2B\eta+\gamma^p+l(Y_i%2C+Y_{gt}))

## 4. 实验结果

文中部分实验结果如下：

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-10-01-110356.jpg)



![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-10-01-110404.jpg)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-10-01-110359.jpg)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-10-01-110359.jpg)

## 5.一些思考

为了避免网络迅速收敛至一个局部最优解，对于得到的Deformable Kernels采样点，会将其分为s组，每组负责预测一个去噪后的图像尽可能与ground truth相似...同时，这种做法也会增强网络的泛化能力。

那么，如何分组或许也是一个比较重点的地方，博主想到了以下几种方法：

(1) 规则分组，对于N个采样点，均匀地划分为s组，这也是最常规的方式无需仔细讨论；

(2) 随机分组，对N个采样点随机分为s组，而不采用规则的分组，这样可以使得每个采样点都能发挥近似相等的地位，能否进一步提高泛化能力？

(3) 其实，分组的做法可以衍生至dropout层的原理，为了防止过拟合，按照一定概率断开一些链接。那么，也可以像dropout层一样，而不是采用分组的方式，每次按照概率p随机取一些采样点预测去噪后的图像，这样能够提高泛化能力？

欢迎大家与我一起讨论~

## Reference

[1] [Learning Deformable Kernels for Image and Video Denoising](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1904.06903.pdf)