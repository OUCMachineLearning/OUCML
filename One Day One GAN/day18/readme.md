# Self-Supervised Generative Adversarial Networks

Github:<https://github.com/zhangqianhui/Self-Supervised-GANs>

Arrive:<https://arxiv.org/pdf/1811.11212.pdf>

## Abstract

cGAN处于自然图像合成的最前沿。这种模型的主要缺点是标记数据的必要性。在这项工作中，我们利用两种流行的无监督学习技术（对抗性训练和自我监督）来缩小有条件和无条件GAN之间的差距。自监督的作用是鼓励判别器学习有意义的特征表示，这些表示在训练期间不会被遗忘。

## 前言

**什么是自监督学习？为什么需要自监督学习？**

简言之，自监督学习是一种特殊目的的无监督学习。不同于传统的AutoEncoder等方法，仅仅以重构输入为目的，而是希望通过surrogate task学习到和高层语义信息相关联的特征。LeCun 认为自监督学习是有望突破有监督学习与强化学习现状、学习世界模型的一个潜在研究方向。自监督学习将输入和输出当成一个完整的整体，它通过挖掘输入数据本身提供的弱标注信息，基于输入数据的某些部分预测其它部分。在达到预测目标的过程中，模型可以学习到数据本身的语义特征表示，这些特征表示可以进一步被用于其他任务当中。当前自监督学习的发展主要体现在视频、图像处理领域。例如，在空间层面上包括图像补全、图像语义分割、灰度图像着色等，在时间层面上包括视频帧预测、自动驾驶等。

## A Key Issue:Discriminator Forgetting

传统的GAN的核心公式为：

$$
\begin{aligned} V(G, D)=& \mathbb{E}_{x \sim P_{\text { data }}(x)}\left[\log P_{D}(S=1 | x)\right] \\ &+\mathbb{E}_{x \sim P_{G}(x)}\left[\log \left(1-P_{D}(S=0 | x)\right)\right] \end{aligned}
$$
每当生成器$G$的参数发生变化时，判别器也会发生变化，这意味着判别器是非平稳的在线学习

在非凸函数的在线学习中，神经网络已被证明会忘记先前的任务，在GAN的背景下，学习不同级别的细节，结构和纹理可以被认为是不同的任务。因此，训练中不稳定的一个原因是，只要当前表示对分类有用，就不会激励判别器维持有用的数据表示。

作者在论文中设计实验来证明了特征的遗忘，如下图(a)所示：

![image-20190413132827777](https://ws2.sinaimg.cn/large/006tNc79ly1g20xyue2ioj30ma0aw42j.jpg)

分类器被训练用1 vs all的方式在CIFAR10数据集的十个类别上进行训练，在每个任务上训练1k次，10k以后回到原点，根据图像我们可以发现，每次任务切换时，分类器精度都会大幅下降。在10k次迭代之后，任务循环重复，并且精度与第一个循环相同，即没有任何有用的信息在任务中传递。(b)中加入的自监督，我们可以发现其准确率是在逐渐提升的。

GAN的情况与此类似，如下图：

![image-20190413132843616](https://ws4.sinaimg.cn/large/006tNc79ly1g20xz4jw0dj30ly0fogny.jpg)

在训练期间，无条件GAN的准确率增加，然后减少，表明有关分类的信息被获取并随后被遗忘。这种遗忘与训练不稳定性有关。添加自我监督可以防止在鉴别器表示中遗忘这些类。

## The Self-Supervised GAN

在判别器遗忘的主要挑战的推动下，我们的目标是为鉴别器注入一种机制，允许学习有用的表示，而不依赖于当前生成器的质量。 为此，我们利用自监督的表示学习方法的最新进展。 自监督背后的主要思想是在预测任务上训练模型，如预测图像块的旋转角度或相对位置，然后从结果网络中提取相关表示。

作者应用了基于图像旋转的最先进的自监督方法， 在该方法中，旋转图像，并且旋转角度变为人造标签。 然后，自监督的任务是预测图像的旋转角度。直观地，这种损失促使分类器学习有用的图像表示以检测旋转角度，并转移到图像分类任务。

![image-20190413132855392](https://ws1.sinaimg.cn/large/006tNc79ly1g20xzbs2uqj30n20cen2t.jpg)

加入自监督以后的损失函数为：

$$
\begin{array}{l}{L_{G}=-V(G, D)-\alpha \mathbb{E}_{\boldsymbol{x} \sim P_{G}} \mathbb{E}_{\boldsymbol{r} \sim \mathcal{R}}\left[\log Q_{D}\left(R=r | \boldsymbol{x}^{r}\right)\right]} \\ {L_{D}=V(G, D)-\beta \mathbb{E}_{\boldsymbol{x} \sim P_{\mathrm{data}}} \mathbb{E}_{r \sim \mathcal{R}}\left[\log Q_{D}\left(R=r | \boldsymbol{x}^{r}\right)\right]}\end{array}
$$

$$
\begin{array}{l}{\text { where } V(G, D) \text { is the value function from Equation } 1, r \in \mathcal{R}} \\ {\text { is a rotation selected from a set of possible rotations. In this }} \\ {\text { work we use } \mathcal{R}=\left\{0^{\circ}, 90^{\circ}, 180^{\circ}, 270^{\circ}\right\} \text { as in Gidaris et al. }} \\ {[26] . \text { Image } \boldsymbol{x} \text { rotated by } r \text { degrees is denoted as } \boldsymbol{x}^{r}, \text { and }} \\ {Q\left(R | \boldsymbol{x}^{r}\right) \text { is the discriminator's predictive distribution over }}\end{array}
$$

生成器和判别器在真实与假预测损失方面是对抗的，然而，它们在旋转任务方面是协作的。生成器不是有条件的，而是仅生成“直立”图像，随后将其旋转并馈送到判别器。另一方面，训练判别器仅基于真实数据检测旋转角度。换句话说，判别器的参数仅基于真实数据上的旋转损失而更新。鼓励生成器生成可旋转检测的图像，因为它们与用于旋转分类的真实图像共享特征。

## Experiments

![image-20190413134721167](https://ws2.sinaimg.cn/large/006tNc79ly1g20yii4qyaj30x50dk77z.jpg)

上图显示了通常用于定量评估两种自我监督的GAN变体以及条件和非条件GAN模型的图像质量的FID分数。图表显示具有self-modulated Batch Normalization（sBN）的SS-GAN能够在ImageNet和CIFAR-10上执行与条件GAN一样好的操作，上图和右图显示了无条件GAN如何执行特别是在ImageNet生成任务上。

![image-20190413133509545](https://ws3.sinaimg.cn/large/006tNc79ly1g20y5tjnmoj30tm0vcq8o.jpg)

作者还比较了在其他常见的数据集上 SSGAN 和 CGAN 的一些指标上的差距.



参考:

<https://zhuanlan.zhihu.com/p/30265894>

<https://cloud.tencent.com/developer/article/1356966>

<https://towardsdatascience.com/self-supervised-gans-2aec1eadaccd>