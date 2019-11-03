作者：cuicuicui





今年ICCV论文中关于GAN的论文有40多篇...简直疯狂，关于图像间转换的文章也有10多篇，因此打算分两篇文章分别分析一下这些文章，上篇介绍图像间转换的GAN，下篇介绍现有GAN的发展。

[http://openaccess.thecvf.com/ICCV2019.pyopenaccess.thecvf.com](https://link.zhihu.com/?target=http%3A//openaccess.thecvf.com/ICCV2019.py)

image to image translation这块的研究主要集中在如何把CNN中比较成熟的一些方法(如NAS, 终身学习)迁移到GAN中，令人眼前一亮的工作应该还是NVIDIA Lab提出的工作FUNIT。目前为止，许多引领潮流的image to image translation模型都是NVIDIA的，如UNIT, MUNIT这些。

## **一、FUNIT：Few-Shot Unsupervised Image-to-Image Translation**

[https://arxiv.org/pdf/1905.01723.pdfarxiv.org](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1905.01723.pdf)

源代码在：

[Few-Shot Unsupervised Image-to-Image Translationnvlabs.github.io![图标](https://tva1.sinaimg.cn/large/006y8mN6ly1g8l234niqyj305003cwed.jpg)](https://link.zhihu.com/?target=https%3A//nvlabs.github.io/FUNIT)

文章一开始就指出，现有的image to image translation 尽管很成功，但在训练时需要很多来自source domain和target domain的图片。这篇文章的目标就是解决这种few-shot的问题，针对于此，本文设计了一种新的网络架构FUNIT，该网络可以利用测试时用少数图片学到source class到target class的映射。Image to image translation 发展到19年有两个问题：

1. **如果训练时图片不够多，则生成的图片效果很差**

**2.生成模型的泛化能力不强。比如，尽管猫和老虎很像，但是训练一个狗图像转换成猫图像的模型，这个模型是很难迁移到狗图像转化成老虎图像。这说明现有的模型缺乏few-shot能力**

什么是few-shot呢？这个指的是联想能力，文章给出了一个例子，当一个人第一次看到一个站着的老虎时，很容易就能想象出该老虎躺着的样子，因为这个人看过其他动物躺下的姿态。记得NIPS 18的时候有一篇文章说过这个问题：

[http://papers.nips.cc/paper/7480-one-shot-unsupervised-cross-domain-translation.pdfpapers.nips.cc](https://link.zhihu.com/?target=http%3A//papers.nips.cc/paper/7480-one-shot-unsupervised-cross-domain-translation.pdf)

NIPS 18年那篇文章说的是one-shot，也就是说训练集中的source domain只有一种类别，而target domain有多个类别。而这篇文章说的是few-shot，当然也有zero-shot，比如cycleGAN. 它们之间的区别当然就是一开始能够用来训练的数据量的多少。作者也指出，FUNIT与one-shot 那篇文章的区别有两点：FUNIT的source domain有多个类别，target domain中有少数几个类别；

之前的工作如UNIT,CoupleGAN,CycleGAN只能生成原有的类别的图像，并不能生成属于新类别的图像。要把image to image的问题看成few-shot的问题，需要一个重要的假设：**假设人的这种few-shot的能力来自于以往的看到的事物。**为了实现这个假设，本文使用的数据集包含很多不同类物体的图片。

**FUNIT的主要架构**

FUNIT的主要任务是根据target domain的几张图，把source domain的一张图映射到target domain，其中target domain的图片种类是之前训练的时候没见过的类别。这篇文章提出的网络架构由一个条件图像生成器 ![[公式]](https://www.zhihu.com/equation?tex=G) 和多任务的判别器 ![[公式]](https://www.zhihu.com/equation?tex=D) 组成，特别地，这篇文章提出的生成器的输入与传统的I2I的生成器有很大的差别，传统的I2I生成器是以一张输入图片为条件，而这篇文章提出的生成器则是输入一张来自 ![[公式]](https://www.zhihu.com/equation?tex=c_x) 类的图片 ![[公式]](https://www.zhihu.com/equation?tex=x) 和K张来自 ![[公式]](https://www.zhihu.com/equation?tex=c_y) 类的图片 ![[公式]](https://www.zhihu.com/equation?tex=\{y_1%2Cy_2%2C...%2Cy_k\}) 

总的来说，假设 ![[公式]](https://www.zhihu.com/equation?tex=S%2CT) 分别表示source domain和target domain的类别的集合。在训练时，选自 ![[公式]](https://www.zhihu.com/equation?tex=S) 中的两个类别 ![[公式]](https://www.zhihu.com/equation?tex=c_x%2Cc_y) 中的任意两张图片，测试的时候则选择 ![[公式]](https://www.zhihu.com/equation?tex=T) 中没见过的类别 ![[公式]](https://www.zhihu.com/equation?tex=c) 中的少数图片作为生成器的条件，生成器就可以把source domain中的某张图片映射到 ![[公式]](https://www.zhihu.com/equation?tex=c) 类别中，也就是生成属于![[公式]](https://www.zhihu.com/equation?tex=c) 类别的图片。

**生成器的设计**

网络的生成器G由内容编码器 ![[公式]](https://www.zhihu.com/equation?tex=E_x)，类别编码器 ![[公式]](https://www.zhihu.com/equation?tex=E_y) 和解码器 ![[公式]](https://www.zhihu.com/equation?tex=F_x) 组成。![[公式]](https://www.zhihu.com/equation?tex=E_x%2CE_y) 都是由残差块组成。

![[公式]](https://www.zhihu.com/equation?tex=\bar{x}+%3D+F_x(z_x%2Cz_y)%3DF_x(E_x(x)%2CE_y(\{y_1%2C...%2Cy_k\}))) 

其中 ![[公式]](https://www.zhihu.com/equation?tex=E_x) 把输入的图片 ![[公式]](https://www.zhihu.com/equation?tex=x) 映射成内容隐编码(content latent code) ![[公式]](https://www.zhihu.com/equation?tex=z_x). 类别编码器则把 ![[公式]](https://www.zhihu.com/equation?tex=k) 张图片 ![[公式]](https://www.zhihu.com/equation?tex=\{y_1%2Cy_2%2C...%2Cy_k\}) 分别映射成中间隐变量(intermediante latent vector)最后取平均得到类别隐编码 ![[公式]](https://www.zhihu.com/equation?tex=z_y). 解码器 ![[公式]](https://www.zhihu.com/equation?tex=F_x) 则是有AdaIN 残差块组成，即残差块+AdaIN的normalization方法，具体来说，就是AdaIN把每个通道的激活值标准化(nomalization)成均值为0，方差为1，接着用学到的仿射变换(affine transformation，该仿射变换由scales和biases组成)放缩标准化后激活值，**由于仿射变换是空间不变的因此可以用来获取全局外观信息，**仿射变换的参数则是用两个全连接层计算类别编码 ![[公式]](https://www.zhihu.com/equation?tex=z_y) 得到。设计这种结构的生成器，是为了用内容编码器提取类不变的表示(class-invariant latent representaion)，如图片中物体的姿态，以及用类编码器提取类改变的表示(class-spesific latent representation)。这种设计思路就特别像MUNIT，DIRT中提到的分别提取内容编码和风格编码。

**判别器的设计**

这篇文章的判别器的设计思路与分类器类似。 ![[公式]](https://www.zhihu.com/equation?tex=D) 会判断每一类图片的真假，因此会输出 ![[公式]](https://www.zhihu.com/equation?tex=|S|) 个数。![[公式]](https://www.zhihu.com/equation?tex=|S|) 代表 ![[公式]](https://www.zhihu.com/equation?tex=S) 中source domain类 ![[公式]](https://www.zhihu.com/equation?tex=c_x) 的个数，生成器生成的属于 ![[公式]](https://www.zhihu.com/equation?tex=c_x) 的图片会判断为假，真实的 ![[公式]](https://www.zhihu.com/equation?tex=c_x) 会判断为真。

**损失函数的设计**

损失函数包括：判别损失 ![[公式]](https://www.zhihu.com/equation?tex=L_{GAN}) ，内容重建损失 ![[公式]](https://www.zhihu.com/equation?tex=L_R) 以及特征匹配损失 ![[公式]](https://www.zhihu.com/equation?tex=L_F) ，其中判别损失为：

![img](https://tva1.sinaimg.cn/large/006y8mN6ly1g8l2363ptkj309k01xjrb.jpg)

其中![[公式]](https://www.zhihu.com/equation?tex=D) 的上标表示sample的图片所属于的类别，因为D会产生很多类别的损失，计算的损失的时候只计算对应类别的那个数。

内容重建损失为：

![img](https://tva1.sinaimg.cn/large/006y8mN6ly1g8l236myyjj307i010wec.jpg)

这个其实是L1损失，把相同的图片输入到生成器对应的内容图片和类别图片中，希望这种情况下生成的图片与原图越相似越好。

最后一个是特征匹配损失：

![img](https://tva1.sinaimg.cn/large/006y8mN6ly1g8l23b7v4mj3077013wec.jpg)

因为判别器 ![[公式]](https://www.zhihu.com/equation?tex=D) 的设计思路与分类器类似，因此 ![[公式]](https://www.zhihu.com/equation?tex=D) 是可以提取图片特征的，去掉 ![[公式]](https://www.zhihu.com/equation?tex=D) 的最后输出层，加入特征提取器 ![[公式]](https://www.zhihu.com/equation?tex=D_f) 就可以提取生成图片 ![[公式]](https://www.zhihu.com/equation?tex=\bar{x}) 与target class的图片 ![[公式]](https://www.zhihu.com/equation?tex=\{y_1%2Cy_2%2C...%2Cy_k\}) 的特征，因为生成图片![[公式]](https://www.zhihu.com/equation?tex=\bar{x})也属于target类，因此有：

![img](https://pic4.zhimg.com/v2-65aa234b5df0d97030edb2b4ca440ce7_b.png)

**实验部分**

训练网络时使用RMSProp方法，判别损失则用GAN的hinge形式，k在训练时选择1，测试时选择1，5，10，16，20。数据集主要来自

1. **动物脸(Animal Faces)**，作者从ImageNet中手动选取了117574张图片，共分成119个source 类和30个target 类
2. **鸟类图片(Birds)**，包含444个source类和111个target类

> Horn G V , Branson S , Farrell R , et al. Building a bird recognition app and large scale dataset with citizen scientists: The fine print in fine-grained dataset collection[C]// 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2015.

\3. **花类图片(Flowers)，**包含85个source类和17个target类

> M.-E. Nilsback and A. Zisserman. Automated flower classification over a large number of classes. In Indian Conference on Computer Vision, Graphics & Image Processing, 2008.

**4. 食物类图片(Foods)，**包含224个source类和32个target类

> Y. Kawano and K. Yanai. Automatic expansion of a food image dataset leveraging existing categories with domain adaptation. In European Conference on Computer Vision (ECCV) Workshop, 2014. 6

实验结果如图所示：

![img](https://tva1.sinaimg.cn/large/006y8mN6ly1g8l239u0g1j30go0jwn3x.jpg)FUNIT生成的图片

![img](https://tva1.sinaimg.cn/large/006y8mN6ly1g8l23dk8x8j30go0eqwja.jpg)

![img](https://tva1.sinaimg.cn/large/006y8mN6ly1g8l23ouu96j30go04y75q.jpg)

![img](https://tva1.sinaimg.cn/large/006y8mN6ly1g8l2373gdpj30go0evq7k.jpg)

![img](https://pic3.zhimg.com/v2-a1d988ec23b194ec68029d22b41e75ca_b.jpg)

![img](https://tva1.sinaimg.cn/large/006y8mN6ly1g8l239emn0j30go0eyq87.jpg)

![img](https://tva1.sinaimg.cn/large/006y8mN6ly1g8l237ii4sj30go04vmyo.jpg)



**对比方法(Baseline)**

这篇文章的作者根据是否可以在训练中使用target类中的图片，把对比方法分成公平对比和不公平对比两类。由于现有的方法并没有针对few-shot image to image translation问题的，因此作者扩展了starGAN方法，不公平的对比方法则包括CycleGAN,UNIT,MUNIT：

![img](https://pic4.zhimg.com/v2-9983d1b0c17163aa1c450a8bdc9f799f_b.jpg)

**评价标准**

这篇文章使用的评价标准包括：**转换准确性**，**图片内容不变性**，**IS**，**FID**以及user study。此外也分析了k的大小、超参数对生成图片的影响，编码插值：

![img](https://pic1.zhimg.com/v2-7849988f6563a7b43238939f20b62b64_b.jpg)

也进行了消融实验。

**失败的例子**

![img](https://tva1.sinaimg.cn/large/006y8mN6ly1g8l23ipkihj30go09ldi4.jpg)

![img](https://pic3.zhimg.com/v2-5a1dc6ccae5b788305adc8facd5b4ae6_b.jpg)

当source class和target class的外形差距太大的时候，本文提出的方法失效，这种情况下FUNIT生成的图片更像是给输入的图片换了种颜色。个人认为，出现这种情况的原因是AdaIN可能并没有真正学到形态的变化，可能需要加入一些保持物体形态的信息。

## **二、Co-Evolutionary Compression for Unpaired Image Translation**

[https://arxiv.org/pdf/1907.10804.pdfarxiv.org](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1907.10804.pdf)

这篇工作来自华为的**诺亚方舟实验室**。文章指出目前有很多方法可以压缩DNN分类器的参数，但是随着GAN在Image to image translation 中的成果应用，并没有针对于GAN的压缩方法。本文提出一种协同计算的方法，用来降低image to image 算法运行时所需要的内存和FLOPS(每秒浮点运算次数)。

作者的动机说在手机上做image to image translation需要很大的内存和计算消耗。作者也举了例子，在手机上部署cycleGAN并生成一张图片需要消耗43MB的内存和 ![[公式]](https://www.zhihu.com/equation?tex=10^{10})的浮点数计算，当用ResNet或者MobileNet做生成器时，这两个消耗会更大。尽管已经涌现出了很多压缩CNN模型的方法，但是这些方法大多针对图像分类或者物体检测任务，并不适合于GAN，本文则提出一种协同计算的方法，主要思路如图：

![img](https://tva1.sinaimg.cn/large/006y8mN6ly1g8l23ashe8j30go0a4jt2.jpg)

生成器中的滤波器被表示成了二进制字符串，通过迭代不断消除多余的filter。

部分实验结果如图：

![img](https://pic2.zhimg.com/v2-815c6c58c3269f6f72953eb46020c7d9_b.jpg)

通过压缩方法，减少了模型的尺寸同时也在一定程度上保证了生成图像的质量。同时本也跟对CNN做剪枝(pruning)的方法做了比较：

![img](https://tva1.sinaimg.cn/large/006y8mN6ly1g8l23nikpfj30go0bptc6.jpg)

如果对于Image to image 算法落地感兴趣的同学，可以读一下这篇方法。

## 三、Lifelong GAN: Continual Learning for Conditional Image Generation

[https://arxiv.org/pdf/1907.10107.pdfarxiv.org](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1907.10107.pdf)

这篇文章可以看成是cGAN的扩展，把终身学习(Lifelong learning，或者持续学习，continual learning)的概念引入到GAN中去解决translation的问题。

**简介**

DNN有一个致命的缺点：**存在遗忘现象**。这就说在一个task1上训练的模型通过迁移学习运用到task2上后，能在task2上取得很好的效果，但是重新训练后的模型在task1的效果会大大降低。以edges转换成shoes和segmentation转换成facades为例：

![img](https://pic3.zhimg.com/v2-f0e0141f0078f91904f77233e11193ca_b.jpg)

训练好edges→shoes的模型后，生成器可以把边缘图转化鞋子，但是如果在这个模型上fine-tune重新训练segmentation→facades的模型，重新训练的模型就很难用到edges→shoes这个任务上。如上图左上角，鞋子的边缘图生成的是很糊的图像。这篇文章提出的模型则可以同时用到多个image to image的translation上，研究生成器的终身学习能力，现有的基于**记忆回放(memory replay)**的方法只适合于label-to-image这类任务，并不适合于image to image translation. 本文提出的Lifelong GAN采用**知识升华(knowledge distillation)**把前一个网络学到的知识迁移到新的网络中，进而使得网络有了终身学习或持续学习的能力。知识升华的概念来自于Hinton老爷子：

> Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. In NeurIPS workshop on Deep Learning and Representation Learning, 2015



文章一开始就提出，人都是活到老学到老，而且脑子是越用越灵活。但跟人不一样，终身学习对机器来说一种都是个挑战，现在的DNN都会遇到一种叫做灾难性遗忘的问题(catastrophic forgetting)，现有的解决方法是把当前任务用到的图像数据与之前所有task用到的图像数据一起训练新的模型，但这种方法并不是万全之策，因为我们不可能穷举所有类型的图像数据，而且硬件条件也不允许。这篇文章的贡献可以概括为：

1. 提出一种可以持续学习cGAN模型，并且条件可以是图片、标签
2. 通过多组实验验证了模型的有效性

**模型简介**

这篇文章提出的方法主要基于BicycleGAN，主要思路如图：

![img](https://tva1.sinaimg.cn/large/006y8mN6ly1g8l23f4eqaj30go08pmyi.jpg)

给定第 ![[公式]](https://www.zhihu.com/equation?tex=t) 次任务的训练数据去训练模型 ![[公式]](https://www.zhihu.com/equation?tex=M_t)，为了防止模型遗忘前一个任务 ![[公式]](https://www.zhihu.com/equation?tex=t-1) 的信息，模型 ![[公式]](https://www.zhihu.com/equation?tex=M_t) 采用知识升华的方法去升华来自 ![[公式]](https://www.zhihu.com/equation?tex=M_{t-1}) 的信息，具体知识升华是指模型 ![[公式]](https://www.zhihu.com/equation?tex=M_t) 和 ![[公式]](https://www.zhihu.com/equation?tex=M_{t-1}) 在给定辅助数据(auxiliary data)的条件下可以产生相似的结果。辅助数据由本文定义的两种操作Montage(剪辑)和Swap(交换)得到。

具体来说，BicyleGAN中包含两个cycle: cVAE-GAN和cLR-GAN，其中cVAE-GAN完成的任务是，用编码器把领域 ![[公式]](https://www.zhihu.com/equation?tex=\mathbb{B}) 的图像 ![[公式]](https://www.zhihu.com/equation?tex=B) 编码成 ![[公式]](https://www.zhihu.com/equation?tex=z) ，生成器则把 ![[公式]](https://www.zhihu.com/equation?tex=z) 和领域 ![[公式]](https://www.zhihu.com/equation?tex=\mathbb{A}) 的图像 ![[公式]](https://www.zhihu.com/equation?tex=A) 转换成领域 ![[公式]](https://www.zhihu.com/equation?tex=\mathbb{B}) 的图像![[公式]](https://www.zhihu.com/equation?tex=\tilde{B})，期望![[公式]](https://www.zhihu.com/equation?tex=B)与 ![[公式]](https://www.zhihu.com/equation?tex=\tilde{B}) 越接近越好；cLR-GAN则是把生成图像 ![[公式]](https://www.zhihu.com/equation?tex=\tilde{B}) 重新编码成 ![[公式]](https://www.zhihu.com/equation?tex=\tilde{z}) ,期望 ![[公式]](https://www.zhihu.com/equation?tex=\tilde{z}) 与 ![[公式]](https://www.zhihu.com/equation?tex=z) 越接近越好。本文在此基础上分别对这两个cycle做知识升华。

对于cVAE-GAN来说，期望任务 ![[公式]](https://www.zhihu.com/equation?tex=t) 不忘记任务 ![[公式]](https://www.zhihu.com/equation?tex=t-1) 的知识，因此任务 ![[公式]](https://www.zhihu.com/equation?tex=t) 产生的编码应该与任务 ![[公式]](https://www.zhihu.com/equation?tex=t-1) 接近，而生成图像也应该接近，进而有：

![img](https://tva1.sinaimg.cn/large/006y8mN6ly1g8l23cnk0rj309n02daa3.jpg)

类似的，对于cLR-GAN，也希望任务 ![[公式]](https://www.zhihu.com/equation?tex=t) 生成的图像，以及任务 ![[公式]](https://www.zhihu.com/equation?tex=t) 重建的编码与任务 ![[公式]](https://www.zhihu.com/equation?tex=t-1) 相似，因此有：

![img](https://pic4.zhimg.com/v2-72e329a2e3dc1174ee51dde967d26a97_b.jpg)

至于这两个损失项的具体形式，这篇文章提出使用L1损失以防止模糊现象。

**辅助数据**

上边的两个式子其实是存在相互矛盾的目标的，因为任务 ![[公式]](https://www.zhihu.com/equation?tex=t) 和任务 ![[公式]](https://www.zhihu.com/equation?tex=t-1) 有各自的目标函数，期望生成的图像也是不同的。为了解决这种矛盾的情况，文章提出使用辅助数据，这些辅助数据并不需要标注，使用剪辑和交换的方法生成辅助数据

**剪辑(montage)：**从当前任务的输入图像的数据集中随机sample一些图像块，然后把它们剪辑在一起

**交换(swap)：**把输入的条件图像和ground truth图像交换，就是让编码器对原来的条件编码，原来的ground truth则作为生成器的条件



**实验部分**

具体训练时，使用的图像分辨率为128×128，对比的方法是现有的用于持续学习的方法，包括：**记忆回放的方法(MR)、序列微调(SFT)、联合学习(JL)**。定量评价的方法为LPIPS。实验结果如下：

![img](https://tva1.sinaimg.cn/large/006y8mN6ly1g8l23mgo8hj30go073408.jpg)

个人感觉这篇文章是对BicycleGAN做维度上的扩充，让BicyclcGAN适用于不同的任务，而且辅助数据那块好像信服力不是特别强，也没有对主流的几种不同的模型做扩展。

## 四、Face-to-Parameter Translation for Game Character Auto-Creation

来自网易伏羲AI Lab团队，原文在：

[http://openaccess.thecvf.com/content_ICCV_2019/papers/Shi_Face-to-Parameter_Translation_for_Game_Character_Auto-Creation_ICCV_2019_paper.pdfopenaccess.thecvf.com](https://link.zhihu.com/?target=http%3A//openaccess.thecvf.com/content_ICCV_2019/papers/Shi_Face-to-Parameter_Translation_for_Game_Character_Auto-Creation_ICCV_2019_paper.pdf)

文章指出，人们在游戏中，尤其是角色扮演类游戏，都喜欢捏脸功能，这篇文章提出一种把输入图片转化成三维游戏人物的方法。如把刘亦菲姐姐转化成对应的三维模型：

![img](https://tva1.sinaimg.cn/large/006y8mN6ly1g8l23aapryj30hs0botaj.jpg)

这篇文章提出用最优化的方法来建模。损失函数由**判别损失+脸部内容损失**(facial content loss)构成。生成网络则更细化成**模仿器(imitator)**，模仿器用来游戏引擎建模的过程。除了刘亦菲姐姐，还有baby，热巴这些，结果如下：

![img](https://tva1.sinaimg.cn/large/006y8mN6ly1g8l23c64h4j30hs0iy78k.jpg)

![img](https://tva1.sinaimg.cn/large/006y8mN6ly1g8l23e0o59j30hs07a0u7.jpg)





## 五、Learning Fixed Points in Generative Adversarial Networks: From Image-to-Image Translation to Disease Detection and Localization

[http://openaccess.thecvf.com/content_ICCV_2019/papers/Siddiquee_Learning_Fixed_Points_in_Generative_Adversarial_Networks_From_Image-to-Image_Translation_ICCV_2019_paper.pdfopenaccess.thecvf.com](https://link.zhihu.com/?target=http%3A//openaccess.thecvf.com/content_ICCV_2019/papers/Siddiquee_Learning_Fixed_Points_in_Generative_Adversarial_Networks_From_Image-to-Image_Translation_ICCV_2019_paper.pdf)

源代码在：

[mahfuzmohammad/Fixed-Point-GANgithub.com![图标](https://tva1.sinaimg.cn/large/006y8mN6ly1g8l23lzoy9j303c03c3ya.jpg)](https://link.zhihu.com/?target=https%3A//github.com/mahfuzmohammad/Fixed-Point-GAN)

这篇文章提出Fixed-Point GAN，是对StarGAN的改进，在此基础上作者把GAN更进一步运用到疾病检测中。

![img](https://tva1.sinaimg.cn/large/006y8mN6ly1g8l23q0eiwj30hs08ftbi.jpg)

如图，左边的StarGAN中生成的人脸存在额外生成的小胡子(第一行红色框)，fuse与输入不一致(第二、三、四的红色框)以及多余的头发(最后一行的红色框)。文章指出，随着Image to image translation中GAN研究的深入，人们就会疑问，能不能用GAN去除图片中的某个物体，同时保留图片的其他部分？比如用GAN去除人的眼镜？更进一步，能不能用GAN把人生病的照片转换成健康状态，达到虚拟治愈(virtual healing)的效果？针对这些问题，这篇文章给出了自己的解决方法，主要贡献：

1.引入了一种新的概念：固定点的转换(fixed-point translation)

2.提出了一种训练固定点转换的新方法：监督相同领域的转换，正则化(regularization)cross-domain的转换

3.这篇文章提出的方法在医疗和自然图像转换方面表现都很好

4.应用于疾病诊断并帮助定位患病的位置

5.由于现有的IIT方法，并且在疾病检测中优于弱监督的方法

## **六、Interactive Sketch & Fill: Multiclass Sketch-to-Image Translation**

[http://openaccess.thecvf.com/content_ICCV_2019/papers/Ghosh_Interactive_Sketch__Fill_Multiclass_Sketch-to-Image_Translation_ICCV_2019_paper.pdfopenaccess.thecvf.com](https://link.zhihu.com/?target=http%3A//openaccess.thecvf.com/content_ICCV_2019/papers/Ghosh_Interactive_Sketch__Fill_Multiclass_Sketch-to-Image_Translation_ICCV_2019_paper.pdf)

来自牛津、Adobe和UC Berkeley，这篇文章介绍了一种用GAN做sketch-to-image的translation，具体转换时，用户先画出个大概的轮廓，网络会推测出几种合理的结果，根据网络的推荐，用户可以编辑之前的sketch，网络再给出更好的结果。本文提出一种基于选通的方法使得(gating-based method)使得一个网络可以运用于多种类别图像的生成。

![img](https://tva1.sinaimg.cn/large/006y8mN6ly1g8l23gf6z6j30hs05k3zq.jpg)

网络结构如图：

![img](https://tva1.sinaimg.cn/large/006y8mN6ly1g8l23iaossj30hs07ajrz.jpg)

生成器分成两个部分，首先填充用户画的不完整的sketch，然后把完整的sketch输入到第二个生成器生真实图像。判别器也分成两部分，一个判断图像是否完整，另一个判断生成图像是否真实。部分结果如下：

![img](https://tva1.sinaimg.cn/large/006y8mN6ly1g8l23oe2lfj30hs0iewin.jpg)

![img](https://pic3.zhimg.com/v2-2d6d55be54c4b63c5da38fcc9797934a_b.jpg)

## **七、Deep CG2Real: Synthetic-to-Real Translation via Image Disentanglement**

[http://openaccess.thecvf.com/content_ICCV_2019/papers/Bi_Deep_CG2Real_Synthetic-to-Real_Translation_via_Image_Disentanglement_ICCV_2019_paper.pdfopenaccess.thecvf.com](https://link.zhihu.com/?target=http%3A//openaccess.thecvf.com/content_ICCV_2019/papers/Bi_Deep_CG2Real_Synthetic-to-Real_Translation_via_Image_Disentanglement_ICCV_2019_paper.pdf)

来自UC San Diego, Adobe. 主要是把OpenGL渲染的图像转换成真实图片。

![img](https://tva1.sinaimg.cn/large/006y8mN6ly1g8l23jtbl9j30hs0f50vw.jpg)

相比于相比于原来的渲染图像，这篇文章提出方法生成的图像在光照和纹理方面表现更好，也许这种思想可以用到服装图像的生成中。这篇文章提出的方法是弱监督。网络结构如下：

![img](https://tva1.sinaimg.cn/large/006y8mN6ly1g8l23bpiyjj30hs06s75f.jpg)

部分实验结果如下：

![img](https://tva1.sinaimg.cn/large/006y8mN6ly1g8l23d3wz0j30hs0dzdjc.jpg)

## 八、Sym-Parameterized Dynamic Inference for Mixed-Domain Image Translation

[http://openaccess.thecvf.com/content_ICCV_2019/papers/Chang_Sym-Parameterized_Dynamic_Inference_for_Mixed-Domain_Image_Translation_ICCV_2019_paper.pdfopenaccess.thecvf.com](https://link.zhihu.com/?target=http%3A//openaccess.thecvf.com/content_ICCV_2019/papers/Chang_Sym-Parameterized_Dynamic_Inference_for_Mixed-Domain_Image_Translation_ICCV_2019_paper.pdf)

来自三星和首尔大学，主要做艺术图像与真实图像的translation

![img](https://tva1.sinaimg.cn/large/006y8mN6ly1g8l23haj6fj30hs09a771.jpg)

 文章指出在现有的IIT中，如果不用target domain的数据去训练网络，则无法产生该类的图片。这篇文章提出，把“多领域”(multi-domain)的概念从数据集方面扩展到损失函数方面，学习不同领域共有的特征(combined characteristics)并动态地推断出混合领域图像的转换。具体说，先用符号参数(Sym-parameter)学各种领域的混合损失， 接着提出一种基于符号参数的生成网络，两部分网络如下：

![img](https://tva1.sinaimg.cn/large/006y8mN6ly1g8l23nx8iwj30hs07f0tl.jpg)

部分生成结果如下：

![img](https://tva1.sinaimg.cn/large/006y8mN6ly1g8l23rqllij30hs08hjuf.jpg)

![img](https://tva1.sinaimg.cn/large/006y8mN6ly1g8l23fg4w1j30hs08l40y.jpg)

这篇文章提出方法所生成的图像可以在不同领域间连续转换。

![img](https://tva1.sinaimg.cn/large/006y8mN6ly1g8l23fxhl3j30hs06m76j.jpg)

## 九、RelGAN: Multi-Domain Image-to-Image Translation via Relative Attributes

[http://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_RelGAN_Multi-Domain_Image-to-Image_Translation_via_Relative_Attributes_ICCV_2019_paper.pdfopenaccess.thecvf.com](https://link.zhihu.com/?target=http%3A//openaccess.thecvf.com/content_ICCV_2019/papers/Wu_RelGAN_Multi-Domain_Image-to-Image_Translation_via_Relative_Attributes_ICCV_2019_paper.pdf)

![img](https://tva1.sinaimg.cn/large/006y8mN6ly1g8l23s6j3jj30hs0gltb4.jpg)

来自台大+HTC+Stanford，主要改进starGAN。文章指出现有的多领域IIT中有两个缺陷：首先假设生成图像的视觉属性都是二维的，因此并不能fine-grained的控制，其次，现有方法需要制定整个target domain的所有属性，但其实有些属性是不变的，并没有用。为了解决这个问题，这篇文章提出RelGAN, 主要贡献是提出一种”相对属性“（relative attributes），用来描述那些选中的想要改变的属性，网络结构如下：

![img](https://tva1.sinaimg.cn/large/006y8mN6ly1g8l23l6lmpj30hs07gwfm.jpg)

部分实现结果如下：

![img](https://tva1.sinaimg.cn/large/006y8mN6ly1g8l23quwvzj30hs05ggnb.jpg)

对比结果：

![img](https://tva1.sinaimg.cn/large/006y8mN6ly1g8l23hsiitj30hs05jmym.jpg)



十、Attribute-Driven Spontaneous Motion in Unpaired Image Translation

来自港中文Jia Jia ya团队+腾讯优图。文章指出现有IIT在几何变换方面有缺陷，本提出一种自发运动估计模块（spontaneous motion estimation module）和修复模块去学一种源Domain和target domain间属性驱动的变换。下图是为人脸添加笑容：

![img](https://tva1.sinaimg.cn/large/006y8mN6ly1g8l23gujr1j30dl0afq4v.jpg)

部分实验结果如下：

![img](https://tva1.sinaimg.cn/large/006y8mN6ly1g8l23ejv1nj30hs0fs78g.jpg)

![img](https://tva1.sinaimg.cn/large/006y8mN6ly1g8l23t4ti7j30hs09hac6.jpg)

十一、Guided Image-to-Image Translation with Bi-Directional Feature Transformation

这篇文章改进TextureGAN。针对



十二、

四、Towards Multi-pose Guided Virtual Try-on Network

[https://arxiv.org/pdf/1902.11026.pdfarxiv.org](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1902.11026.pdf)

这篇文章来自我中大的梁小丹老师团队，之后补充详细的文章解读。

FW-GAN: Flow-navigated Warping GAN for Video Virtual Try-on

[http://openaccess.thecvf.com/content_ICCV_2019/papers/Dong_FW-GAN_Flow-Navigated_Warping_GAN_for_Video_Virtual_Try-On_ICCV_2019_paper.pdfopenaccess.thecvf.com](https://link.zhihu.com/?target=http%3A//openaccess.thecvf.com/content_ICCV_2019/papers/Dong_FW-GAN_Flow-Navigated_Warping_GAN_for_Video_Virtual_Try-On_ICCV_2019_paper.pdf)

同样来自中大梁小丹团队



五、Controllable Artistic Text Style Transfer via Shape-Matching GAN

[https://arxiv.org/pdf/1905.01354.pdfarxiv.org](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1905.01354.pdf)

六、COCO-GAN: Generation by Parts via Conditional Coordinating

[https://arxiv.org/pdf/1904.00284.pdfarxiv.org](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1904.00284.pdf)







其他介绍GAN，不是Image to image translation的文章包括：

六、AutoGAN: Neural Architecture Search for Generative Adversarial Networks

[https://arxiv.org/pdf/1908.03835.pdfarxiv.org](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1908.03835.pdf)

源代码在：

[TAMU-VITA/AutoGANgithub.com![图标](https://tva1.sinaimg.cn/large/006y8mN6ly1g8l23sizgkj303c03c744.jpg)](https://link.zhihu.com/?target=https%3A//github.com/TAMU-VITA/AutoGAN)

这篇文章主要讲了在图像分类和分割领域中成功应用的自动调参方法NAS(Neural Architecture Search)用到GAN中，这篇文章把这种类型的GAN命名为AutoGAN。

补充中....未完待续....