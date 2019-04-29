# One Day One GAN

Hi ,my name is Chen Yang ,I am a sophomore in ocean university of China .I do some scientific research in my spare time. Based on the current hot direction of artificial intelligence, I hope to share my research progress with **Generative adversarial network**

嗨，我的名字是陈扬，我是中国海洋大学的二年级学生。我在业余时间做了一些科学研究。基于当前人工智能的热点方向，我希望与**生成对抗网络分享我的研究进展**

## 前言

**ODOG**,顾名思义就我我希望能每天抽出一个小时的时间来讲讲到目前为止,GAN的前沿发展和研究,笔者观察了很多深度学习的应用,特别是在图像这一方面,GAN已经在扮演着越来越重要的角色,我们经常可以看到老黄的NVIDIA做了各种各样的application,而且其中涉及到了大量GAN的理论及其实现,再者笔者个人也觉得目前国内缺少GAN在pytorch,keras,tensorflow等主流的框架下的实现教学.

我的老师曾经对我说过:"**深度学习是一块未知的新大陆,它是一个大的黑箱系统,而GAN则是黑箱中的黑箱,谁要是能打开这个盒子,将会引领一个新的时代**"

## WGAN

WGAN,这篇文章我相信对于非科班出身或者说数学知识相对较弱的朋友来说具有一定的难度,但是为什么我把这篇文章放到了第一期的 ODOG 来讲呢?很大程度上来说是因为我认为 WGAN 中的原理虽然看起来比较的复杂,但是其代码实现相对来说不是说很困难,而且在 WGAN 中我们提出了 Wasserstein 距离,这个 W 距离很好程度的弥补了原始 GAN 中的 KL 散度的不足,我希望能够抛开具体大段的原理公式推导,把这一简洁有效的方法教给大家,至少能保证在实验结果上能够有一些提升.

- 彻底解决GAN训练不稳定的问题，不再需要小心平衡生成器和判别器的训练程度
- 基本解决了collapse mode的问题，确保了生成样本的多样性 
- 训练过程中终于有一个像交叉熵、准确率这样的数值来指示训练的进程，这个数
- 值越小代表GAN训练得越好，代表生成器产生的图像质量越高
- 以上一切好处不需要精心设计的网络架构，最简单的多层全连接网络就可以做到

#### 回顾

在讲 WGAN 之前,我还是要来回顾一下原始的 GAN的核心公式:

![image_1ct4sn8kqg8ftika3b1nmj6i1j.png-43kB](https://ws3.sinaimg.cn/large/006tKfTcly1g1lv49i8yzj30w805it98.jpg)

 我们来看**Gennerator和Discriminator的KL散度和JS散度解释**

**Generator中θ的极大似然估计(是求真实分布和生成器分布的KL散度的最小值)**

![image-20190331122346341](https://ws1.sinaimg.cn/large/006tKfTcly1g1lv4a78lfj30gz0cnq3l.jpg)

![image-20190331122401672](https://ws1.sinaimg.cn/large/006tKfTcly1g1lv4aofulj30ln0ggt9q.jpg)

我们可以把原始GAN定义的生成器$loss$等价变换为最小化真实分布$P_r$与生成分布$P_g$之间的$JS$散度。我们越训练判别器，它就越接近最优，最小化生成器的$loss$也就会越近似于最小化$P_r$和$P_g$之间的JS散度。



而在 WGAN 中,我们对这个 KL 散度进行了改进,该文章围绕 Wasserstein 距离对两个分布p(X),q(X)做出如下定义:

![image-20190428121136284](https://ws2.sinaimg.cn/large/006tNc79ly1g2i81gba70j318g04omyp.jpg)

![image-20190331123200062](https://ws4.sinaimg.cn/large/006tKfTcly1g1lva1f13kj30f70onae4.jpg)

(该部分转载于大神苏剑林:<https://kexue.fm/archives/6280>)

### 话锋一转-代码实现

我们从现在看这个可能还是觉得有些困难,包括在实现 WGAN 的过程中涉及到了 loss 函数的反向传播需要满足$Lipschitz连续$等等问题,我也是找了很多的博客等等,直接了当地总结了如何在 GAN的基础上实现 WGAN:

![image-20190331124000273](https://ws1.sinaimg.cn/large/006tKfTcly1g1lvid8bfjj310k0lbqtl.jpg)

上文说过，WGAN与原始GAN相比，只改了四点：

- 判别器最后一层去掉sigmoid
- 生成器和判别器的loss不取log
- 每次更新判别器的参数之后把它们的绝对值截断到不超过一个固定常数c(满足利普西斯连续)
- 不要用基于动量的优化算法（包括momentum和Adam），推荐RMSProp，SGD也行

我们一个一个来看是怎么做的:

**判别器最后一层去掉sigmoid,生成器和判别器的loss不取log**

```python
def wasserstein_loss(self, y_true, y_pred):
    return K.mean(y_true * y_pred)

self.critic = self.build_critic()
self.critic.compile(loss=self.wasserstein_loss,
    optimizer=optimizer,
    metrics=['accuracy'])

        self.critic = self.build_critic()
self.critic.compile(loss=self.wasserstein_loss,
    optimizer=optimizer,
    metrics=['accuracy'])
```



**每次更新判别器的参数之后把它们的绝对值截断到不超过一个固定常数c(满足利普西斯连续)**

```python
# Clip critic weights
for l in self.critic.layers:
    weights = l.get_weights()
    weights = [np.clip(w, -self.clip_value, 
                       self.clip_value) for w in weights]
    l.set_weights(weights)
```

**不要用基于动量的优化算法**

```python
 optimizer = RMSprop(lr=0.00005)
```

![image-20190331125317070](https://ws4.sinaimg.cn/large/006tKfTcly1g1lvw744dij30vs0u0qv6.jpg)

作者通过对比验证了 WGAN 的鲁棒性.

## 总结

WGAN前作分析了Ian Goodfellow提出的原始GAN两种形式各自的问题，第一种形式等价在最优判别器下等价于最小化生成分布与真实分布之间的JS散度，由于随机生成分布很难与真实分布有不可忽略的重叠以及JS散度的突变特性，使得生成器面临梯度消失的问题；第二种形式在最优判别器下等价于既要最小化生成分布与真实分布直接的KL散度，又要最大化其JS散度，相互矛盾，导致梯度不稳定，而且KL散度的不对称性使得生成器宁可丧失多样性也不愿丧失准确性，导致collapse mode现象。

WGAN前作针对分布重叠问题提出了一个过渡解决方案，通过对生成样本和真实样本加噪声使得两个分布产生重叠，理论上可以解决训练不稳定的问题，可以放心训练判别器到接近最优，但是未能提供一个指示训练进程的可靠指标，也未做实验验证。

WGAN本作引入了Wasserstein距离，由于它相对KL散度与JS散度具有优越的平滑特性，理论上可以解决梯度消失问题。接着通过数学变换将Wasserstein距离写成可求解的形式，利用一个参数数值范围受限的判别器神经网络来最大化这个形式，就可以近似Wasserstein距离。在此近似最优判别器下优化生成器使得Wasserstein距离缩小，就能有效拉近生成分布与真实分布。WGAN既解决了训练不稳定的问题，也提供了一个可靠的训练进程指标，而且该指标确实与生成样本的质量高度相关。作者对WGAN进行了实验验证。