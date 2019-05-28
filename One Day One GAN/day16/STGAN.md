## CVPR 2019 | STGAN: 人脸高精度属性编辑模型

> [AttGAN](https://arxiv.org/abs/1711.10678)和[StarGAN](http://openaccess.thecvf.com/content_cvpr_2018/papers/Choi_StarGAN_Unified_Generative_CVPR_2018_paper.pdf)在人脸属性编辑上取得了很大的成功，但当人脸属性之间相互交集或者目标人脸属性比较复杂时，这两种方式对于控制属性的标签上的精细化就显示了些许的不足。[STGAN](https://arxiv.org/abs/1904.09709)是一个建立在[AttGAN](https://arxiv.org/abs/1711.10678)基础上的人脸属性编辑模型，通过差分属性标签下选择性传输单元的跳跃连接实现了人脸高精度属性的编辑。

前言--ATTGAN

![image-20190527094105959](http://ww3.sinaimg.cn/large/006tNc79ly1g3fmnu46soj31480ranbv.jpg)

判别器D:

属性分类限制

![img](http://ww2.sinaimg.cn/large/006tNc79ly1g3fmqlv6ewj30dk0370sp.jpg)

classification也要训练的，和auto-encoder一起训练，介样练：

![img](http://ww3.sinaimg.cn/large/006tNc79ly1g3fmqm8pkuj30d7035q2v.jpg)

---

生成器:

重建误差

![img](http://ww4.sinaimg.cn/large/006tNc79ly1g3fmr2s097j30cv01mmwz.jpg)

对抗误差

![img](http://ww4.sinaimg.cn/large/006tNc79ly1g3fmr2akxej30da036749.jpg)

总误差（分为两步）



![img](http://ww1.sinaimg.cn/large/006tNc79ly1g3fmr10695j30cl01jglg.jpg)

![img](http://ww2.sinaimg.cn/large/006tNc79ly1g3fmr1qy6ej309i01k3yb.jpg)

ATTGAN 模型都是架构在编码器 - 解码器上，同时将源图像和目标属性向量作为输入，[AttGAN](https://arxiv.org/abs/1711.10678)不是对潜在表示施加约束，而是对生成的图像应用属性分类约束，以保证所需属性的正确变化，同时引入重建学习以保留属性排除细节。[StarGAN](http://openaccess.thecvf.com/content_cvpr_2018/papers/Choi_StarGAN_Unified_Generative_CVPR_2018_paper.pdf)只用一个generator网络，处理多个domain之间互相generate图像的问题，这是比[AttGAN](https://arxiv.org/abs/1711.10678)更深一步的人脸属性迁移。

----

不过,我们今天要介绍的是 **STAGN**:

![img](http://ww1.sinaimg.cn/large/006tNc79ly1g3fmv3r311j30qn0b8whv.jpg)

STGAN 用的的是一个类 UNET 的网络结构做生成器,在 skip-connection 的时候使用了一个叫 STU(提出了选择性传输单元) 的操作,来自适应地选择和修改编码器特征，其进一步与解码器特征连接以增强图像质量和属性操纵能力,相当于给生成器加上了硬解耦的作用.

---

## 选择性传输单元

在介绍选择性传输单元之前，我们先把文章对目标属性和源属性的标签处理交代一下。StarGAN 和AttGAN 都将目标属性向量AtttAttt和源图像x作为输入到生成器。实际上，使用完整目标属性向量是多余的，可能对编辑结果有害。如果目标属性向量AtttAttt与源AtttAttt完全相同，此时，理论上输入只需要对图像进行重构即可，但StarGAN和AttGAN 可能会错误地操作一些未更改的属性，比如把原本就是金色头发变得更加的金色。

对于任意图像属性编辑，而不是完整目标属性向量，只应考虑要更改的属性以保留源图像的更多信息。因此，将差异属性向量定义为目标和源属性向量之间的差异是合适的：
$$
Att_{diff}=Att_t−Att_S
$$

---

比如男性有胡子戴眼镜的源图编辑到男性无胡子戴眼镜秃头目标图，这里面仅仅是添加了秃头这一属性，减少了胡子这一属性，其它的可以保持不变。AttdiffAttdiff可以为指导图像属性编辑提供更有价值的信息，包括是否需要编辑属性，以及属性应该改变的方向。然后可以利用该信息来设计合适的模型，以将编码器特征与解码器特征进行变换和连接，并且在不牺牲属性操纵精度的情况下提高图像重建质量。

选择性传输单元（STU）来选择性地转换编码器特征，使其与解码器特征兼容并互补，而不是通过skip connection直接将编码器与解码器特征连接起来。这个变换需要适应变化的属性，并且在不同的编码器层之间保持一致，作者修改[GRU](https://arxiv.org/abs/1412.3555)[3]的结构以构建用于将信息从内层传递到外层的STU。

---

我们来看一下STU的结构：

![img](http://ww3.sinaimg.cn/large/006tNc79ly1g3fnhemzfcj309g09uaap.jpg)

---

$f^l_{enc}$为编码器第l层输出，$s^{l+1}$为数据编码在$l+1$层的隐藏状态，隐藏状态$ŝ^{l+1}$则是结合了$Att^{diff}$得到的：                
$$
ŝ^{l+1}=W_{t∗T}[s^{l+1},Att_{diff}]
$$
其中[⋅,⋅][⋅,⋅]表示为concatenation操作，∗T∗T为转置卷积，然后，STU采用GRU的数学模型来更新隐藏状态slsl和转换后的编码器特征fltftl:

$$
r^l=σ(W_r∗[f^l_{enc},ŝ^{l+1}]),z^l=σ(W_z∗[f^l_{enc},ŝ^{l+1}]),s^l=r^l∘ŝ^{l+1}
$$

$$
f̂^l_t=tanh(W_h∗[f^l_{enc},s^l]),f^l_t=(1−z^l)∘ŝ^{l+1}+z^l∘f̂^l_t
$$

其中∗表示卷积运算，∘表示逐项乘积，σ(⋅)表示sigmoid函数。复位门$r^l$和更新门$z^l$的引入允许以选择性方式控制隐藏状态，差异属性向量和编码器特征。输出$f_t^l$提供了一种自适应的编码器特征传输方法及其与隐藏状态的组合。

选择性传输单元（STU）说白了就是在GRU的结构上实现的，差分标签控制下的编码特征的选择。

```python
if use_stu:
            self.stu = nn.ModuleList()
            for i in reversed(range(self.n_layers - 1 - self.shortcut_layers,\
                                    self.n_layers - 1)):
                self.stu.append(ConvGRUCell(self.n_attrs, conv_dim * 2 ** i, \
                                	conv_dim * 2 ** i, stu_kernel_size))
```

---

# 模型结构

有了上述的分析，我们再看模型的结构则是比较容易理解了：

![img](http://ww1.sinaimg.cn/large/006tNc79ly1g3h6qdvfv9j30qn0b8whv.jpg)

整个模型比较简单，在编码器和解码器过程中，加入$STU$选择单元，从而获得人脸属性编辑后的输出。编码器的输入端包括源图x和差分属性标签$Att_{diff}$。对于判别器，也是判别生成器输出真假和对应的属性标签。对抗损失采用WGAN-GP来实现生成优化，对应着$L_{D_{adv}}$,$L_{G_{adv}}$。对于属性标签和生成器的属性优化通过源真实样本和标签优化判别器，再通过判别器去判别目标生成的属性结果来优化生成器：

![image-20190528180335754](http://ww3.sinaimg.cn/large/006tNc79ly1g3h6sxijsmj30qh0ac3zh.jpg)

----

![image-20190528180316991](http://ww1.sinaimg.cn/large/006tNc79ly1g3h6slsg52j30qh0ac3zh.jpg)

---

# 实验

CelebA数据集包含裁剪到178×218的202,599个对齐的面部图像，每个图像有40个带/不带属性标签。图像分为训练集，验证集和测试集，文章从验证集中获取1,000张图像以评估训练过程，使用验证集的其余部分和训练集来训练STGAN模型，并利用测试集进行性能评估。实验考虑13种属性，包括秃头，爆炸，黑发，金发，棕色头发，浓密眉毛，眼镜，男性，嘴微微开口，小胡子，无胡子，苍白皮肤和年轻，实验中，每个图像的中心170×170区域被裁剪并通过双三次插值调整为128×128。

---

![img](http://ww3.sinaimg.cn/large/006tNc79ly1g3h6udndn8j30qm0dmdts.jpg)

定量评估上，文章从两个方面评估属性编辑的性能，即图像质量和属性生成准确性。图像质量上，保持目标属性向量与源图像属性相同，得到了PSNR / SSIM结果：

![img](http://ww1.sinaimg.cn/large/006tNc79ly1g3h6ujxoorj30do030dfw.jpg)

吐槽:PSNR 和 SSIM 在这里当做评价指标简直是瞎扯淡…...

---

![STGAN8](http://ww4.sinaimg.cn/large/006tNc79ly1g3h6vpw0zhj30fj0ao74y.jpg)

这个指标还行…...

---

![STGAN9](http://ww3.sinaimg.cn/large/006tNc79ly1g3h6wcqf92j30fh0dowog.jpg)

实验在用户的选择测试上也取得了最佳效果，Ablation Study实验上也证实了模型的每一部分的优势和必要。最后放一张STGAN在图像季节转换的实验效果：

---

# 总结

文章研究了选择性传输视角下任意图像属性编辑的问题，并通过在编码器 - 解码器网络中结合差分属性向量和选择性传输单元（STU）来提出STGAN模型。通过将差异属性向量而不是目标属性向量作为模型输入，STGAN可以专注于编辑要改变的属性，这极大地提高了图像重建质量，增强了属性的灵活转换。

---

