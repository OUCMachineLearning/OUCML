## [Perceptual GAN for Small Object Detection阅读笔记](https://zhuanlan.zhihu.com/p/83659247)

作者：唐青

链接：https://zhuanlan.zhihu.com/p/83659247

来源：知乎

著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

Perceptual Generative Adversarial Networks for Small Object Detectionopenaccess.thecvf.com



## 1 摘要及介绍概述

基于小目标，提出了Perceptual GAN网络来生成小目标的超分表达(super-resolved representation)，Perceptual GAN利用大小目标的结构相关性来增强小目标的表达(represnetation)，使其与其对应大目标的表达相似。

Perceptual GAN分为两个子网络，生成器(generator network) 和一个感知判别器( perceptual discriminator network)。原始的GAN的判别器用于判别生成器生成的图片为fake or ture，本文主要修改了判别器部分，使判别器还能从生成的超分图形中的检测获益量来计算损失值(perceptual loss)。

使用的数据库为Tsinghua-Tencent 100K(交通标志检测) 和 the Caltech benchmark(行人检测)  。

## 2 Perceptual GANs

Perceptual GANs 对生成器和判别器都进行了修改，使生成器能生成小目标的超分表达，判别器拥有两个loss(dversarial loss and perceptual loss )。

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-10-09-35260.png)Figure 1.　公式

文中3.1中的公式(本文中figure 1)表示本文目标是要**最小化G和最大化D**，即使用Generator生成小目标的超分图(与Groundtruth越近越好)，使用Discriminator判别生成的超分图与Groundtruth(使Discriminator约混淆越好)，**让Discriminator不能判别生成的超分图是原图true还是生成图fake，从而达到生成图形与原图的无限接近**。

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-10-09-35261.png)Figure 2.　公式

文中使用大目标 ![[公式]](https://www.zhihu.com/equation?tex=F_{l}) 作为Groundtruth，希望使用Generator将小目标![[公式]](https://www.zhihu.com/equation?tex=F_{s})生成为大目标![[公式]](https://www.zhihu.com/equation?tex=F_{l})。基于此，Figure1公式变为Figure2公式。

**判别器**输出为两个分枝：

- 对抗分枝：用于区分生成的超分表达与大目标groudtruth图片。训练办法和一般GAN中判别器类似，loss公式为章节3.1公式(3)。
- 感知分枝：用于justify生成的表达对检测精度的获益量。感知分枝先需提前使用大目标进行训练得到pre-trained模型，使其能产生相对高的检测精度。感知分枝的输出为一个多任务输出(classification+bounding-box regression)。

训练时**轮次训练生成器与判别器**

把所有的实例平均分为两类：大目标和小目标。首先使用大目标训练判别器的感知分枝，基于已学习过的感知分枝，使用小目标来训练生成器和使用大小目标一起来训练判别器的对抗分枝。交替地执行生成器和判别器网络的对抗分枝的训练过程，直到最终达到一个平衡点，即：可以为小目标生成类似大目标的超分特征，并且具有较高的检测精度。

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-10-09-035259.png)Figure 3. 整个Perceptual GAN 结构图。

**2.1 Conditional Generator Network Architecture生成器网络结构**

虚线左边是生成器结构，上面为一般ConvNet用于特征提取，下面加了一组残差block，目的是输出一张图，让其对上面网络输出的小物体特征图进行补充(Eltwise-Sum)，组合生成Super-Resolved Feature。

**2.2 Discriminator Network Architecture 判别器网络结构**

虚线右边是判别器结构，其上下两部分别为对抗分枝与感知分枝。对抗分枝结构与一般classification(fake/true)网络类似，感知分枝从全连接开始又分为另外两个分枝，分别输出classification和Bounding box。判别器网络的损失为上下两枝的损失各自乘以权重后的和，本文中这两权重都设为1。

Figure 4为判别器对抗损失，最小化 ![[公式]](https://www.zhihu.com/equation?tex=L_{dis\_a}) ，即，使 ![[公式]](https://www.zhihu.com/equation?tex=D_{\Theta_{a}}(G_{\Theta_{g}}(F_{s}))) 接近1。

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-10-09-035301.png)Figure 4. 公式

判别器感知损失和一般detection结构类似。