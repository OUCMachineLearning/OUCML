# 超分辨率的损失函数总结

MSE，是图像空间的**内容**“相似”，而在图像上普遍存在区域，其属于某个类别（老虎皮，草，渔网等），如果出现纹理或者网格，那么优化MSE很容易将这个区域磨平，即平滑。【直接对应评价指标MSE和MAE，以及PSNR，但是PSNR高不见得好，图像可能不自然。见文献2的图：PSNR/SSIM与感知质量(视觉效果)的一致性】

![img](https://ws2.sinaimg.cn/large/006tNc79ly1g2vdtyj7v2j30go04u753.jpg)PSNR/SSIM与感知质量(视觉效果)的一致性

L1，可忍受异常值，相较于MSE和L2是没有那么平滑一些的。

Perceptual loss，是特征空间的**类别/纹理**“相似”。毕竟有学者认为深度卷积网络用于图像分类，利用的是**物体的纹理差异**。

多尺度(MS)-SSIM，那就是图像空间的**结构**“相似”。在文献[1]中，就找到了MS-SSIM+L1的混合损失适合于图像复原。

图像SR问题有两个主要方向，见文献[2]的图。其一为了实现更好的图像重建效果，即灰度/RGB复原；其二为了实现更好的视觉质量，即看起来“自然”。有没有方法是兼顾两者的呢？似乎。。。

![img](https://ws4.sinaimg.cn/large/006tNc79ly1g2vdty9cboj30go0bi3zb.jpg)图像SR的两个方向

[1] Loss Functions for Image Restoration with Neural Networks IEEE TCI 2017 [[[paper](http://link.zhihu.com/?target=http%3A//ieeexplore.ieee.org/document/7797130/)]] [[[code](http://link.zhihu.com/?target=https%3A//github.com/NVlabs/PL4NN)]]

[2] 2018 PIRM Challenge on Perceptual Image Super-resolution [[[paper](http://link.zhihu.com/?target=https%3A//arxiv.org/abs/1809.07517)]]

