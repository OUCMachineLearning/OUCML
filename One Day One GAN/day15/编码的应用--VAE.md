# 编码的应用--VAE 

## 什么是编码?

> **编码**是[信息](https://zh.wikipedia.org/wiki/信息)从一种[形式](https://zh.wikipedia.org/wiki/形式)或[格式](https://zh.wikipedia.org/wiki/格式)转换为另一种形式的过程。**解码**，是编码的逆过程。
>
> 对于特定的上下文，**编码**有一些更具体的[意义](https://zh.wikipedia.org/w/index.php?title=意义&action=edit&redlink=1)。
>
> - **编码**（Encoding）在[认知](https://zh.wikipedia.org/wiki/认知)上是解释传入的刺激的一种基本知觉的过程。技术上来说，这是一个复杂的、多阶段的转换过程，从较为客观的感觉输入（例如光、声）到主观上有意义的体验。

## 本质:数据降维

## 新的问题

在之前,老师给我们讲解了霍夫曼编码的应用,使用[变长编码表](https://baike.baidu.com/item/变长编码表/17490831)对源符号（如文件中的一个字母）进行编码，其中[变长编码表](https://baike.baidu.com/item/变长编码表/17490831)是通过一种评估来源符号出现机率的方法得到的，出现机率高的字母使用较短的编码，反之出现机率低的则使用较长的编码，这便使编码之后的字符串的平均长度、[期望值](https://baike.baidu.com/item/期望值/8664642)降低，从而达到[无损压缩](https://baike.baidu.com/item/无损压缩/2817566)数据的目的。

![âéå¤«æ¼ç¼ç âçå¾çæç´¢ç»æ](https://ws4.sinaimg.cn/large/006tNc79ly1g2yuz2994zj30kf0bi0tc.jpg)

---

# 可是,我们日常生活中,视觉信息>>听觉信息

## 图像压缩的传统方法:

Gif，PNG，JPG,TIF,TIFF,WEBP…….

---

换个思路,如果我们能够抽离出图像的更高维信息,比如将图像表示成语义信息?

![img](https://ws3.sinaimg.cn/large/006tNc79ly1g2yyg9wur8j30km06iq6u.jpg)

---

# STGAN (CVPR 2019)

![img](https://ws2.sinaimg.cn/large/006tNc79ly1g2yxsx2fahj31ng0dt43u.jpg)

Overall architecture of our STGAN. Taking the image above as an example, in the difference attribute vector ![$\mathbf{att}_\mathit{diff}$](https://ws1.sinaimg.cn/large/006tNc79ly1g2yxr8jworg301b00i08x.gif), ![$Young$](https://ws4.sinaimg.cn/large/006tNc79ly1g2yxr8ubn9g301i00g09a.gif) is set to 1, ![$Mouth\ Open$](https://ws1.sinaimg.cn/large/006tNc79ly1g2yxra89j7g302v00h0fa.gif)

 is set to -1, and others are set to zeros. The outputs of ![$\mathit{D_{att}}$](https://ws1.sinaimg.cn/large/006tNc79ly1g2yxr9bb04g300s00f063.gif) and ![$\mathit{D_{adv}}$](https://ws4.sinaimg.cn/large/006tNc79ly1g2yxr9sjt8g300x00f06e.gif) are the scalar ![$\mathit{D_{adv}}(https://ws3.sinaimg.cn/large/006tNc79ly1g2yxrasqjvg303z00k0kn.gif)$](https://camo.githubusercontent.com/9f34088445b7aa9a8fe6dc9989dd01bfe053067d/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f5c6d61746869747b445f7b6164767d7d285c6d61746869747b477d285c6d61746862667b787d2c5c6d61746862667b6174747d5f5c6d61746869747b646966667d2929) and the vector ![$\mathit{D_{att}}(https://ws2.sinaimg.cn/large/006tNc79ly1g2yxrb6hqag303v00k0kn.gif)$](https://camo.githubusercontent.com/1ee0abad4b308f092ab0b24104c45c90a547246e/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f5c6d61746869747b445f7b6174747d7d285c6d61746869747b477d285c6d61746862667b787d2c5c6d61746862667b6174747d5f5c6d61746869747b646966667d2929), respectively

---

假如我们能够把图像的高维信息提取出来表层上可以被人们予以理解的信息的话,通过这个 ENcode--DEcode 的结构,我们就可以实现图像翻译的工作.

![image-20190512220545840](https://ws3.sinaimg.cn/large/006tNc79ly1g2yvw1fdx4j31pi0muwzf.jpg)

---

类似的工作:

![img](https://ws4.sinaimg.cn/large/006tNc79ly1g2yxx9lfg3j316x0j5k24.jpg)

感兴趣:[http://www.twistedwg.com/2019/04/29/STGAN.html](http://www.twistedwg.com/2019/04/29/STGAN.html)

---

# 我们要做的 demo

可是我并不打算在这里讲解或者说演示怎么复杂的 demo,我为大家准备了一个最简单的也是其中最核心的原理部分—**自编码器**



---

# VAE:

变分自动编码器，而其中利用到的自编码器(autoencoder)是机器学习的一个基础知识。为了更好的理解VAE先要把autoencoder部分又进 行一下消化。

自编码器是通过对输入X进行编码后得到一个低维的向量y，然后根据这个向量还原出输入X，。通过对比X与X，的误差，再利用神经网络去训练使得误差逐渐 减小从而达到非监督学习的目的，图1展示autoencoder过程。

![img](https://ws4.sinaimg.cn/large/006tNc79ly1g2yyhz8o35j30i9061dg2.jpg)

---

这样训练好的模型，就可以提取输入的特征从而更好的还原输入。我也是根据这中思想参考了一些知识写了一个可视化的autoencoder的小代码，在minist 通过matplotlib显示原始数据与autoencoder训练后的数据的图片的对比来观察autoencoder的过程，再提取训练后的encoder的特征分类通过rainbow 显示分类的结果，这里也是用到了matplotlib中的3D显示效果，代码在我的[主页](https://github.com/TwistedW/Tensorflow-noting/blob/master/tensorflow_autoencoder_show.py) 下面为实验结果：

![exp1](https://ws2.sinaimg.cn/large/006tNc79ly1g2yyifna71j30b504mjrb.jpg)

![exp2](https://ws4.sinaimg.cn/large/006tNc79ly1g2yyip1ja7j30ge0cwmyh.jpg)详情:https://zhuanlan.zhihu.com/p/34998569

---

![v2-784891edddff506ea1670c81767e993c_hd](https://ws2.sinaimg.cn/large/006tNc79ly1g2yypzc050j30k00a0q36.jpg)

![image-20190512234612097](https://ws2.sinaimg.cn/large/006tNc79ly1g2yyshisqpj312o054gmg.jpg)

p(Z)p(Z)是标准正态分布，这就意味着，隐参量符合了正态分布，这样就可以放心地从这个分布中随机采样，保证了生成能力。

