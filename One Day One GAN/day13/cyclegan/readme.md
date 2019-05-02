# One Day One GAN

Hi ,my name is Chen Yang ,I am a sophomore in ocean university of China .I do some scientific research in my spare time. Based on the current hot direction of artificial intelligence, I hope to share my research progress with **Generative adversarial network**

嗨，我的名字是陈扬，我是中国海洋大学的二年级学生。我在业余时间做了一些科学研究。基于当前人工智能的热点方向，我希望与**生成对抗网络分享我的研究进展**

## 前言

**ODOG**,顾名思义就我我希望能每天抽出一个小时的时间来讲讲到目前为止,GAN的前沿发展和研究,笔者观察了很多深度学习的应用,特别是在图像这一方面,GAN已经在扮演着越来越重要的角色,我们经常可以看到老黄的NVIDIA做了各种各样的application,而且其中涉及到了大量GAN的理论及其实现,再者笔者个人也觉得目前国内缺少GAN在pytorch,keras,tensorflow等主流的框架下的实现教学.

我的老师曾经对我说过:"**深度学习是一块未知的新大陆,它是一个大的黑箱系统,而GAN则是黑箱中的黑箱,谁要是能打开这个盒子,将会引领一个新的时代**"

## cycleGAN

大名鼎鼎的 cycleGAN 就不用我多介绍了吧,那个大家每一次一讲到深度学习,总有人会放出来的那张图片,马变斑马的那个,就是出自于大名鼎鼎的 cycleGAN 了

![horse2zebra](/Users/Macbook/Downloads/horse2zebra.gif)

![68747470733a2f2f6a756e79616e7a2e6769746875622e696f2f4379636c6547414e2f696d616765732f7465617365725f686967685f7265732e6a7067](https://ws2.sinaimg.cn/large/006tNc79ly1g2n89418lmj31le0rlqi1.jpg)

大概的思想我简要的说一下吧

![img](http://media.paperweekly.site/LUOHAO_1513256736.8965938.png?imageView2/2/w/800/q/70|imageslim)

就是我输入的图片是宋冬野的《斑马》,他经过一个简单的卷积神经网络 $G_{AB}$后,从 集合 A 映射到了集合 B,也是就《董小姐》里面的宋冬野爱上的那匹野马.

你以为这就结束了吗?

如果就此结束,

岂不是变成了讲 Pix2Pix 了

emmmmmmm,~~可惜我先写 cycleGAN 了~~,可惜我不是宋冬野啊,就算我是宋冬野,D 也不是董小姐啊,就算 D 是董小姐,我也不是宋冬野啊,就算我是宋冬野,D 是董小姐,可是我家里没有草原,这让我感到绝望,董小姐.

所以我为什么即兴写了这么多乱七八糟的东西呢,一是我确实遇见过一个很像董小姐的她,而是在图像风格迁移的实际训练中,我们很难找到能够一一对应的数据集,比如这样的:

![image-20190502201615505](https://ws3.sinaimg.cn/large/006tNc79ly1g2n8j5oayjj30gl0a5tck.jpg)

我们大部分情况下做的有实际意义的数据集,都是 Unpaid 的.

所以我们在把斑马变成野马的过程中,我们希望能够尽可能的减少信息的损失.

所以,马卡斯·扬要把野马用$G_{BA}$变回斑马,再别讲变回去的斑马和宋冬野原来的斑马用 L1loss做对比,因为 L1loss 相比于L2 Loss保边缘（L2 Loss基于高斯先验，L1 Loss基于拉普拉斯先验）https://blog.csdn.net/m0_38045485/article/details/82147817

最后,判别器还是和寻常的一样,判别是A 还是 B(其实我觉得应该可以考虑read_A,real_B,fake_A,fake_B)的

好了,正经的来看一下损失函数:

![img](http://media.paperweekly.site/LUOHAO_1513259309.1659012.png?imageView2/2/w/800/q/70|imageslim)

X→Y的判别器损失为，字母换了一下，和上面的单向GAN是一样的： 

$L_{GAN}(G,D_Y,X,Y)=𝔼y∼p_{data}(y)[logD_Y(y)]+𝔼x∼p_{data}(x)[log(1−D_Y(G(x)))]$

同理Y→X的判别器损失为 :

$L_{GAN}(G,D_X,X,Y)=𝔼x∼p_{data}(x)[logD_X(x)]+𝔼y∼p_{data}(y)[log(1−D_X(F(y)))]$

合在一起就是:
$$
\begin{aligned} \mathcal{L}_{\mathrm{cyc}}(G, F) &=\mathbb{E}_{x \sim p_{\text { tata }}(x)}\left[\|F(G(x))-x\|_{1}\right] \\ &+\mathbb{E}_{y \sim p_{\text { data }}(y)}\left[\|G(F(y))-y\|_{1}\right] \end{aligned}
$$
