## [就是不GAN——生成式矩(Moment)匹配网络GMMN](https://zhuanlan.zhihu.com/p/40697096)

[![龙腾](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-27-084853.jpg)](https://www.zhihu.com/people/long-teng-39)

[龙腾](https://www.zhihu.com/people/long-teng-39)

![就是不GAN——生成式矩(https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-27-084843.jpg)匹配网络GMMN](https://pic2.zhimg.com/50/v2-b1a7f1bbe484d794e831564315951d8d_hd.jpg)

在GAN发展热火朝天的当下，生成模型的其他方向显得有些势单力薄。GAN的一个薄弱环节在于Min-max Game问题的不易优化，该问题在[WGAN](https://zhuanlan.zhihu.com/p/25071913)中得到了较好的解决。WGAN核心思想即把判别器的Loss替换为衡量真实数据分布 ![[公式]](https://www.zhihu.com/equation?tex=P_r(x)) 与生成数据分布 ![[公式]](https://www.zhihu.com/equation?tex=P_g(x)) 的分布差异——Wasserstein距离。

事实上，这种思路并非首先由WGAN提出，在2015年的ICML上，就有人用优化MMD(**maximum mean discrepancy,一种度量两个数据集的差异的度量**)的方式提出了不同于GAN的生成式模型——Generative Moment Matching Networks。该模型使用一个(多元均匀分布上的)随机采样Sample作为输入，将经过若干非线性层之后的输出作为生成的样本。



本文的贡献有二：1.提出了基于MMD优化的GMMN，2.针对GMMN可能存在的问题(高维数据难以表现)给出了一个基于Auto-Encoder的改进方案GMMN-AE。其网络结构如下图所示：

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-27-084851.jpg)

GMMN的贡献在于：使用了MMD作为损失函数易于优化，避免了难解的Min-max问题。MMD衡量两组样本 ![[公式]](https://www.zhihu.com/equation?tex=\{x_i\}) 和 ![[公式]](https://www.zhihu.com/equation?tex=\{y_i\}) 在**某个空间S**中的分布差异。如果**某个空间S**中包含了 ![[公式]](https://www.zhihu.com/equation?tex=\{x_i\}) 和 ![[公式]](https://www.zhihu.com/equation?tex=\{y_i\}) 的分布信息。那么最小化MMD就等价于拟合两个分布。

本文使用MMD的平方作为目标函数，如下式(1)所示：

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-27-084844.jpg)

(这里的 ![[公式]](https://www.zhihu.com/equation?tex=\phi) 是一个**再生核变换**，将样本从原空间映射到**某个空间S，**我们选择的函数 ![[公式]](https://www.zhihu.com/equation?tex=\phi) 决定了**目标空间S。**再生核变换的理论性质参见视频课程：[再生核变换与核技巧](http://link.zhihu.com/?target=https%3A//www.camdemy.com/media/1662))

使用MMD的平方而非MMD是因为我们需要利用核技巧。（**核技巧：**机器学习中常用的技术，核技巧使得我们我们不用显式地知道变换后的样本 ![[公式]](https://www.zhihu.com/equation?tex=\mathbf{z}+%3D+\phi+({\mathbf{x}}))的形式，甚至不需要知道 ![[公式]](https://www.zhihu.com/equation?tex=\phi) 的形式。因而 ![[公式]](https://www.zhihu.com/equation?tex=\mathbf{z}) 具有很高灵活性。使用核技巧只需要知道目标空间中样本的内积![[公式]](https://www.zhihu.com/equation?tex=K(\mathbf{x_i}%2C\mathbf{x_j})%3D\mathbf{z_i}^T\mathbf{z_j}+%3D+\phi+({\mathbf{x_i}})^T+\phi+({\mathbf{x_j}})) ，而无需知道目标空间的其他性质。）



基于核技巧，(1)式展开为：

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-27-84854.jpg)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-27-84848.jpg)

考虑 ![[公式]](https://www.zhihu.com/equation?tex=\phi) 是恒等变换的情况 ![[公式]](https://www.zhihu.com/equation?tex=\phi(\mathbf{x})%3D\mathbf{x}) 不难发现(1)式计算的是样本均值。当 ![[公式]](https://www.zhihu.com/equation?tex=\phi) 是二次变换时，(1)式则计算了 ![[公式]](https://www.zhihu.com/equation?tex=\{x_i\}) 和 ![[公式]](https://www.zhihu.com/equation?tex=\{y_i\}) 的**二阶矩。**当 ![[公式]](https://www.zhihu.com/equation?tex=\phi) 包含各次项的时候，(1)式就计算了**各阶矩，**此时，最小化(1)式就相当于匹配 ![[公式]](https://www.zhihu.com/equation?tex=\{x_i\}) 和 ![[公式]](https://www.zhihu.com/equation?tex=\{y_i\}) 的各阶矩，因而等效于**匹配**了两个分布。这就是为什么本文被称作**矩匹配**网络。



本文使用 的 ![[公式]](https://www.zhihu.com/equation?tex=%5Cphi) 是高斯变换：

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-27-084845.jpg)

高斯函数能够展开为无穷级数，因此可以满足(1)式**计算各阶矩**的需求。

标记生成样本为 ![[公式]](https://www.zhihu.com/equation?tex=\{\mathbf{x}^s_i\}) ，数据样本为 ![[公式]](https://www.zhihu.com/equation?tex=\{\mathbf{x}^d_i\}) ，我们容易计算(1)对生成样本的第 ![[公式]](https://www.zhihu.com/equation?tex=p) 维的梯度

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-27-084847.jpg)

利用链式法则，梯度可以一直沿着网络后向传播。因此网络是可训练的。

------

GMMN存在一个难点：直接生成高维数据较为困难，因为：

1. 高维的图像实际上存在低维流形。
2. 高维数据不易处理

因此，本文提出GMMN+AE进行改进。该模型将整个训练过程打破为两部分，AE负责生成高维数据的低维流形，即中间表示层representation，然后在此基础上进行图像重构。而GMMN生成的内容是**representation，**而非图像，这个过程如图1.(b)所示。

实际应用中，GMMN+AE的训练分成三步完成：

1.逐层预训练AE

2.Finetune AE

3.训练GMMN

在后面的实验部分我们可以看到加入AE极大的提升了GMMN的表现。

------

本文还指出了一些GMMN常见的训练技巧，包括：

1. 对AE的各个Encoder层施加Drop-out。
2. 使用集成方法来集成不同 ![[公式]](https://www.zhihu.com/equation?tex=\sigma) 的高斯函数(实验中MNIST取 ![[公式]](https://www.zhihu.com/equation?tex=\sigma%3D) [2,5,10,20,40,80]六个核平均，TFD取[5,10,20,40,80,160]六个核平均)：

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-27-084852.jpg)

\3. 使用MMD的平方根来监督训练，以获得自适应学习率的效果(MMD较小时梯度不消失),

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-27-084846.jpg)

\4. Mini-batch训练：**核技巧的弱点是：**需要构造核矩阵(即目标空间的中样本的两两之间的内积)，因此，核矩阵的规模是n^2。所以不能处理大规模数据集。GMMN使用Mini-batch的方式来规避核矩阵过大的问题。其理由是——一个Batch的数据能够近似数据整体的分布。实际实现中使用的Batch-size为1000。

------



**样本生成过程：**在GMMN中，从多元均匀分布采样一个样本作为网络输入，进行一轮前向传播，得到的输出即为生成图像样本。在GMMN-A中，从均匀分布采样一个样本作为网络输入，进行一轮前向传播得到的是图像的representation。该representation再经过AE的解码层生成图像。

------

实验：生成模型的研究中，常用的衡量标准是：使用生成分布和真实分布的log-likelihood判定生成模型效果，本文对比方法包括:

- 深度信念网络DBN
- Contractive stacked auto-encoder
- Deep Generative Stochastic Network
- GAN
- GMMN
- GMMN+AE

使用的数据集为：MNIST 和 TFD（多伦多人脸）

其定量的结果为：

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-27-084850.jpg)

定性的结果如下图：(a)-(d)生成的结果 (e)-(f)生成数据以及生成的图像在真实数据中的最近邻

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-27-84845.jpg)

生成模型的一个风险是，仅仅记住了数据集而非真正的学会了数据分布，为此作者给出的两点证据：

- 上图(e)-(h)可以看出生成数据和真实数据集中的最近邻有略微差异。
- 用均匀分布中的5个点生成的5张图像（下图红框）；再以均匀分布上，这5个点之间的其他位置采样到的点（采样方式为线性插值）作为输入生成的图像。

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-27-084854.jpg)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-27-084849.jpg)



------

本文的启示：

1. 本文对Related Work的探讨非常全面，分析了其他生成模型的特点和不足：

- Boltzman系列的模型，主要问题是MCMC过程难以优化
- 信念网络系列模型，主要问题是对隐变量顺序有所假设，与真实世界不符
- 基于Auto Encoder重构概率分布的模型，也需要MCMC优化。
- 变分网络系列，需要额外信息。

\2. 本文的写作很值得学习，作者在各个章节都反复强调了MMD对于Mini-max的优点，使得各个Section相对独立，降低了审稿人阅读的难度。

\3. 本文的末尾探讨了许多基于MMD的可能的改进方向。

\4. 本文的[开源代码](http://link.zhihu.com/?target=http%3A//github.com/yujiali/gmmn)和[Tensorflow版本](http://link.zhihu.com/?target=https%3A//github.com/siddharth-agrawal/)

\4. 本文的不足：Batch_size 1000的时候才能做到一个batch能够近似数据分布，这是对于MNIST和TFD这样的小数据集而言的，对于大规模数据集仍然是一个问题。