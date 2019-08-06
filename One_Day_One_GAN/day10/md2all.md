---
author: '微信公众号：颜家大少'
description: '一个Markdown在线转换工具，让Markdown内容，不需作任何调整就能同时在微信公众号、博客园、掘金、csdn等平台正确显示当前预览的效果'
email: '3056432@qq.com'
title: Md2All export document
viewport: 'width=device-width,height=device-height,initial-scale=1,user-scalable=0'
---

<div id="export_content">

<div id="output_wrapper_id" class="output_wrapper">

[infoGAN]{} {#hinfogan}
-----------

[[[互信息](#h)]{.toc_left}]{.toc_item}[[[实现细节](#h-1)]{.toc_left}]{.toc_item}

论文地址：[InfoGAN: Interpretable Representation Learning by Information
Maximizing Generative Adversarial Nets](http://arxiv.org/abs/1606.03657)

源码地址：[InfoGAN in
TensorFlow](https://github.com/JonathanRaiman/tensorflow-infogan)

生成对抗网络（Generative Adversarial
Nets）是一类新兴的生成模型，由两部分组成：一部分是判别模型（discriminator）[[$D\left( \cdot \right)$]{.katex-mathml}[[[]{.strut
style="height:1em;vertical-align:-0.25em;"}[D]{.mord .mathit
style="margin-right:0.02778em;"}[(]{.mopen}[⋅]{.mpunct}[]{.mspace
style="margin-right:0.16666666666666666em;"}[)]{.mclose}]{.base}]{.katex-html
aria-hidden="true"}]{.katex}，用来判别输入数据是真实数据还是生成出来的数据；另一部分是是生成模型（generator）[[$G\left( \cdot \right)$]{.katex-mathml}[[[]{.strut
style="height:1em;vertical-align:-0.25em;"}[G]{.mord
.mathit}[(]{.mopen}[⋅]{.mpunct}[]{.mspace
style="margin-right:0.16666666666666666em;"}[)]{.mclose}]{.base}]{.katex-html
aria-hidden="true"}]{.katex}，由输入的噪声生成目标数据。GAN
的优化问题可以表示为：

![image-20190406235920213](https://ws2.sinaimg.cn/large/006tNc79ly1g1tcvbjv7kj30u002wmxe.jpg "image-20190406235920213")
image-20190406235920213
其中[[$Pdata$]{.katex-mathml}[[[]{.strut
style="height:0.69444em;vertical-align:0em;"}[P]{.mord .mathit
style="margin-right:0.13889em;"}[d]{.mord .mathit}[a]{.mord
.mathit}[t]{.mord .mathit}[a]{.mord .mathit}]{.base}]{.katex-html
aria-hidden="true"}]{.katex}
是生成样本，[[$noise$]{.katex-mathml}[[[]{.strut
style="height:0.65952em;vertical-align:0em;"}[n]{.mord .mathit}[o]{.mord
.mathit}[i]{.mord .mathit}[s]{.mord .mathit}[e]{.mord
.mathit}]{.base}]{.katex-html
aria-hidden="true"}]{.katex}是随机噪声。而对于带标签的数据，通常用潜码（latent
code）[[$c$]{.katex-mathml}[[[]{.strut
style="height:0.43056em;vertical-align:0em;"}[c]{.mord
.mathit}]{.base}]{.katex-html aria-hidden="true"}]{.katex}
来表示这一标签，作为生成模型的一个输入，这样我们有：

![image-20190407000040736](https://ws2.sinaimg.cn/large/006tNc79ly1g1tcwgby27j30u002vjrm.jpg "image-20190407000040736")
image-20190407000040736
然而当我们遇到存在潜在的类别差别而没有标签数据，要使 GAN
能够在这类数据上拥有更好表现，我们就需要一类能够无监督地辨别出这类潜在标签的数据，InfoGAN
就给出了一个较好的解决方案----利用互信息来对c进行约束，这是因为如果c对于生成数据G(z,c)具有可解释性，那么c和G(z,c)应该具有高度相关性，即互信息大，而如果是无约束的话，那么它们之间没有特定的关系，即互信息接近于0。因此我们希望c与G(z,c)的互信息I(c;G(z,c))越大越好，因此，模型的目标函数也变为：

![image-20190407005056917](https://ws1.sinaimg.cn/large/006tNc79ly1g1tecr513vj30ve04cq3i.jpg "image-20190407005056917")
image-20190407005056917
但是在I(c;G(z,c))的计算中，真实的P(c|x)并不清楚，因此在具体的优化过程中，作者采用了变分推断的思想，引入了变分分布Q(c|x)来逼近P(c|x)，它是基于最优互信息下界的轮流迭代实现最终的求解，于是InfoGAN的目标函数变为：

![image-20190407004821018](https://ws1.sinaimg.cn/large/006tNc79ly1g1tea1p7swj310u08o0ub.jpg "image-20190407004821018")
image-20190407004821018
![image-20190407004904021](https://ws3.sinaimg.cn/large/006tNc79ly1g32f81r1znj310g0akjtu.jpg "image-20190407004904021")
image-20190407004904021
把[[$L_{1}\left( G,Q \right)$]{.katex-mathml}[[[]{.strut
style="height:1em;vertical-align:-0.25em;"}[[L]{.mord
.mathit}[[[[[[]{.pstrut style="height:2.7em;"}[[1]{.mord
.mtight}]{.sizing .reset-size6 .size3
.mtight}]{style="top:-2.5500000000000003em;margin-left:0em;margin-right:0.05em;"}]{.vlist
style="height:0.30110799999999993em;"}[​]{.vlist-s}]{.vlist-r}[[[]{}]{.vlist
style="height:0.15em;"}]{.vlist-r}]{.vlist-t
.vlist-t2}]{.msupsub}]{.mord}[(]{.mopen}[G]{.mord
.mathit}[,]{.mpunct}[]{.mspace
style="margin-right:0.16666666666666666em;"}[Q]{.mord
.mathit}[)]{.mclose}]{.base}]{.katex-html
aria-hidden="true"}]{.katex}作为互信息的下界带入(3)式,使KL
散度最小化,用蒙特卡罗模拟（Monte Carlo
simulation）去逼近[[$L_{I}\left( G,Q \right)$]{.katex-mathml}[[[]{.strut
style="height:1em;vertical-align:-0.25em;"}[[L]{.mord
.mathit}[[[[[[]{.pstrut style="height:2.7em;"}[[I]{.mord .mathit .mtight
style="margin-right:0.07847em;"}]{.sizing .reset-size6 .size3
.mtight}]{style="top:-2.5500000000000003em;margin-left:0em;margin-right:0.05em;"}]{.vlist
style="height:0.32833099999999993em;"}[​]{.vlist-s}]{.vlist-r}[[[]{}]{.vlist
style="height:0.15em;"}]{.vlist-r}]{.vlist-t
.vlist-t2}]{.msupsub}]{.mord}[(]{.mopen}[G]{.mord
.mathit}[,]{.mpunct}[]{.mspace
style="margin-right:0.16666666666666666em;"}[Q]{.mord
.mathit}[)]{.mclose}]{.base}]{.katex-html aria-hidden="true"}]{.katex}

==&gt;[[$\left. E_{x}\left\lbrack D_{KL}\left( P\left( \cdot \mid x \right)\parallel Q\left( \cdot \mid x \right) \right) \right\rbrack\rightarrow 0 \right.$]{.katex-mathml}[[[]{.strut
style="height:1em;vertical-align:-0.25em;"}[[E]{.mord .mathit
style="margin-right:0.05764em;"}[[[[[[]{.pstrut
style="height:2.7em;"}[[x]{.mord .mathit .mtight}]{.sizing .reset-size6
.size3
.mtight}]{style="top:-2.5500000000000003em;margin-left:-0.05764em;margin-right:0.05em;"}]{.vlist
style="height:0.151392em;"}[​]{.vlist-s}]{.vlist-r}[[[]{}]{.vlist
style="height:0.15em;"}]{.vlist-r}]{.vlist-t
.vlist-t2}]{.msupsub}]{.mord}[\[]{.mopen}[[D]{.mord .mathit
style="margin-right:0.02778em;"}[[[[[[]{.pstrut
style="height:2.7em;"}[[[K]{.mord .mathit .mtight
style="margin-right:0.07153em;"}[L]{.mord .mathit .mtight}]{.mord
.mtight}]{.sizing .reset-size6 .size3
.mtight}]{style="top:-2.5500000000000003em;margin-left:-0.02778em;margin-right:0.05em;"}]{.vlist
style="height:0.32833099999999993em;"}[​]{.vlist-s}]{.vlist-r}[[[]{}]{.vlist
style="height:0.15em;"}]{.vlist-r}]{.vlist-t
.vlist-t2}]{.msupsub}]{.mord}[(]{.mopen}[P]{.mord .mathit
style="margin-right:0.13889em;"}[(]{.mopen}[⋅]{.mpunct}[]{.mspace
style="margin-right:0.16666666666666666em;"}[∣]{.mord}[x]{.mord
.mathit}[)]{.mclose}[∥]{.mord}[Q]{.mord
.mathit}[(]{.mopen}[⋅]{.mpunct}[]{.mspace
style="margin-right:0.16666666666666666em;"}[∣]{.mord}[x]{.mord
.mathit}[)]{.mclose}[)]{.mclose}[\]]{.mclose}[]{.mspace
style="margin-right:0.2777777777777778em;"}[→]{.mrel}[]{.mspace
style="margin-right:0.2777777777777778em;"}]{.base}[[]{.strut
style="height:0.64444em;vertical-align:0em;"}[0]{.mord}]{.base}]{.katex-html
aria-hidden="true"}]{.katex}

==&gt;[[$L_{I}\left( G,Q \right) = H\left( c \right)$]{.katex-mathml}[[[]{.strut
style="height:1em;vertical-align:-0.25em;"}[[L]{.mord
.mathit}[[[[[[]{.pstrut style="height:2.7em;"}[[I]{.mord .mathit .mtight
style="margin-right:0.07847em;"}]{.sizing .reset-size6 .size3
.mtight}]{style="top:-2.5500000000000003em;margin-left:0em;margin-right:0.05em;"}]{.vlist
style="height:0.32833099999999993em;"}[​]{.vlist-s}]{.vlist-r}[[[]{}]{.vlist
style="height:0.15em;"}]{.vlist-r}]{.vlist-t
.vlist-t2}]{.msupsub}]{.mord}[(]{.mopen}[G]{.mord
.mathit}[,]{.mpunct}[]{.mspace
style="margin-right:0.16666666666666666em;"}[Q]{.mord
.mathit}[)]{.mclose}[]{.mspace
style="margin-right:0.2777777777777778em;"}[=]{.mrel}[]{.mspace
style="margin-right:0.2777777777777778em;"}]{.base}[[]{.strut
style="height:1em;vertical-align:-0.25em;"}[H]{.mord .mathit
style="margin-right:0.08125em;"}[(]{.mopen}[c]{.mord
.mathit}[)]{.mclose}]{.base}]{.katex-html aria-hidden="true"}]{.katex}

所以:

![image-20190407005215167](https://ws3.sinaimg.cn/large/006tNc79ly1g1tee3vq10j30y804kwf5.jpg "image-20190407005215167")
image-20190407005215167
那么这里整个网络的架构图就是:

![image-20190407004702678](https://ws3.sinaimg.cn/large/006tNc79ly1g1te8p78d8j30xh0u0agr.jpg "image-20190407004702678")
image-20190407004702678

### [互信息]{} {#h}

简单的来说:互信息指的是两个随机变量之间的关联程度，即给定一个随机变量后，另一个随机变量不确定性的削弱程度，因而互信息取值最小为0，意味着给定一个随机变量对确定一另一个随机变量没有关系，最大取值为随机变量的熵，意味着给定一个随机变量，能完全消除另一个随机变量的不确定性

在[概率论](https://zh.wikipedia.org/wiki/%E6%A6%82%E7%8E%87%E8%AE%BA)和[信息论](https://zh.wikipedia.org/wiki/%E4%BF%A1%E6%81%AF%E8%AE%BA)中，两个[随机变量](https://zh.wikipedia.org/wiki/%E9%9A%8F%E6%9C%BA%E5%8F%98%E9%87%8F)的**互信息**（Mutual
Information，简称MI）或**转移信息**（transinformation）是变量间相互依赖性的量度。不同于相关系数，互信息并不局限于实值随机变量，它更加一般且决定着联合分布
p(X,Y) 和分解的边缘分布的乘积 p(X)p(Y)
的相似程度。互信息是[点间互信息](https://zh.wikipedia.org/w/index.php?title=%E7%82%B9%E9%97%B4%E4%BA%92%E4%BF%A1%E6%81%AF&action=edit&redlink=1)（PMI）的期望值。互信息最常用的[单位](https://zh.wikipedia.org/wiki/%E8%AE%A1%E9%87%8F%E5%8D%95%E4%BD%8D)是[bit](https://zh.wikipedia.org/wiki/%E4%BD%8D%E5%85%83)。

![](https://ws2.sinaimg.cn/large/006tNc79ly1g1tcyo35msj316m0u0gpj.jpg){width="300"}

一般地，两个离散随机变量 X 和 Y 的互信息可以定义为：

\
\
[[$I\left( X;Y \right) = \sum\limits_{y \in Y}\sum\limits_{x \in X}p\left( x,y \right)\log\left( \frac{p\left( x,y \right)}{p\left( x \right)\, p\left( y \right)} \right),\,$*{y\\in
Y}\\sum* {x\\in X}p(x,y)\\log {\\left({\\frac
{p(x,y)}{p(x)\\,p(y)}}\\right)},\\,!}]{.katex-mathml}[[[]{.strut
style="height:2.880444em;vertical-align:-1.430444em;"}[[I]{.mord .mathit
style="margin-right:0.07847em;"}[(]{.mopen}[X]{.mord .mathit
style="margin-right:0.07847em;"}[;]{.mpunct}[]{.mspace
style="margin-right:0.16666666666666666em;"}[Y]{.mord .mathit
style="margin-right:0.22222em;"}[)]{.mclose}[]{.mspace
style="margin-right:0.2777777777777778em;"}[=]{.mrel}[]{.mspace
style="margin-right:0.2777777777777778em;"}[[[[[[]{.pstrut
style="height:3.05em;"}[[[y]{.mord .mathit .mtight
style="margin-right:0.03588em;"}[∈]{.mrel .mtight}[Y]{.mord .mathit
.mtight style="margin-right:0.22222em;"}]{.mord .mtight}]{.sizing
.reset-size6 .size3
.mtight}]{style="top:-1.8556639999999998em;margin-left:0em;"}[[]{.pstrut
style="height:3.05em;"}[[∑]{.mop .op-symbol
.large-op}]{}]{style="top:-3.0500049999999996em;"}]{.vlist
style="height:1.050005em;"}[​]{.vlist-s}]{.vlist-r}[[[]{}]{.vlist
style="height:1.430444em;"}]{.vlist-r}]{.vlist-t .vlist-t2}]{.mop
.op-limits}[]{.mspace
style="margin-right:0.16666666666666666em;"}[[[[[[]{.pstrut
style="height:3.05em;"}[[[x]{.mord .mathit .mtight}[∈]{.mrel
.mtight}[X]{.mord .mathit .mtight
style="margin-right:0.07847em;"}]{.mord .mtight}]{.sizing .reset-size6
.size3
.mtight}]{style="top:-1.8556639999999998em;margin-left:0em;"}[[]{.pstrut
style="height:3.05em;"}[[∑]{.mop .op-symbol
.large-op}]{}]{style="top:-3.0500049999999996em;"}]{.vlist
style="height:1.050005em;"}[​]{.vlist-s}]{.vlist-r}[[[]{}]{.vlist
style="height:1.321706em;"}]{.vlist-r}]{.vlist-t .vlist-t2}]{.mop
.op-limits}[]{.mspace
style="margin-right:0.16666666666666666em;"}[p]{.mord
.mathit}[(]{.mopen}[x]{.mord .mathit}[,]{.mpunct}[]{.mspace
style="margin-right:0.16666666666666666em;"}[y]{.mord .mathit
style="margin-right:0.03588em;"}[)]{.mclose}[]{.mspace
style="margin-right:0.16666666666666666em;"}[lo[g]{style="margin-right:0.01389em;"}]{.mop}[]{.mspace
style="margin-right:0.16666666666666666em;"}[[[[(]{.delimsizing
.size3}]{.mopen .delimcenter style="top:0em;"}[[[]{.mopen
.nulldelimiter}[[[[[[]{.pstrut style="height:3em;"}[[p]{.mord
.mathit}[(]{.mopen}[x]{.mord .mathit}[)]{.mclose}[]{.mspace
style="margin-right:0.16666666666666666em;"}[p]{.mord
.mathit}[(]{.mopen}[y]{.mord .mathit
style="margin-right:0.03588em;"}[)]{.mclose}]{.mord}]{style="top:-2.314em;"}[[]{.pstrut
style="height:3em;"}[]{.frac-line
style="border-bottom-width:0.04em;"}]{style="top:-3.23em;"}[[]{.pstrut
style="height:3em;"}[[p]{.mord .mathit}[(]{.mopen}[x]{.mord
.mathit}[,]{.mpunct}[]{.mspace
style="margin-right:0.16666666666666666em;"}[y]{.mord .mathit
style="margin-right:0.03588em;"}[)]{.mclose}]{.mord}]{style="top:-3.677em;"}]{.vlist
style="height:1.427em;"}[​]{.vlist-s}]{.vlist-r}[[[]{}]{.vlist
style="height:0.936em;"}]{.vlist-r}]{.vlist-t
.vlist-t2}]{.mfrac}[]{.mclose
.nulldelimiter}]{.mord}]{.mord}[[)]{.delimsizing .size3}]{.mclose
.delimcenter style="top:0em;"}]{.minner}]{.mord}[,]{.mpunct}[]{.mspace
style="margin-right:0.16666666666666666em;"}[]{.mspace
style="margin-right:-0.16666666666666666em;"}]{.mord}]{.base}]{.katex-html
aria-hidden="true"}]{.katex}\
\

其中 p(x,y) 是 X 和 Y
的联合概率分布函数，而[[${p\left( x \right)}p\left( x \right){p\left( y \right)}p\left( y \right)$]{.katex-mathml}[[[]{.strut
style="height:1em;vertical-align:-0.25em;"}[[p]{.mord
.mathit}[(]{.mopen}[x]{.mord .mathit}[)]{.mclose}]{.mord}[p]{.mord
.mathit}[(]{.mopen}[x]{.mord .mathit}[)]{.mclose}[和]{.mord
.cjk_fallback}[[p]{.mord .mathit}[(]{.mopen}[y]{.mord .mathit
style="margin-right:0.03588em;"}[)]{.mclose}]{.mord}[p]{.mord
.mathit}[(]{.mopen}[y]{.mord .mathit
style="margin-right:0.03588em;"}[)]{.mclose}]{.base}]{.katex-html
aria-hidden="true"}]{.katex}分别是 X 和 Y 的边缘概率分布函数。

\
\
[[$I\left( X;Y \right) = \sum_{y \in Y}\sum_{x \in X}p\left( x,y \right)\log\left( \frac{p\left( x,y \right)}{p\left( x \right)\, p\left( y \right)} \right),\,$*{{y\\in
Y}}\\sum* {{x\\in X}}p(x,y)\\log {\\left({\\frac
{p(x,y)}{p(x)\\,p(y)}}\\right)},\\,!]{.katex-mathml}[[[]{.strut
style="height:1em;vertical-align:-0.25em;"}[I]{.mord .mathit
style="margin-right:0.07847em;"}[(]{.mopen}[X]{.mord .mathit
style="margin-right:0.07847em;"}[;]{.mpunct}[]{.mspace
style="margin-right:0.16666666666666666em;"}[Y]{.mord .mathit
style="margin-right:0.22222em;"}[)]{.mclose}[]{.mspace
style="margin-right:0.2777777777777778em;"}[=]{.mrel}[]{.mspace
style="margin-right:0.2777777777777778em;"}]{.base}[[]{.strut
style="height:1.80002em;vertical-align:-0.65002em;"}[[∑]{.mop .op-symbol
.small-op
style="position:relative;top:-0.0000050000000000050004em;"}[[[[[[]{.pstrut
style="height:2.7em;"}[[[[y]{.mord .mathit .mtight
style="margin-right:0.03588em;"}[∈]{.mrel .mtight}[Y]{.mord .mathit
.mtight style="margin-right:0.22222em;"}]{.mord .mtight}]{.mord
.mtight}]{.sizing .reset-size6 .size3
.mtight}]{style="top:-2.40029em;margin-left:0em;margin-right:0.05em;"}]{.vlist
style="height:0.17862099999999992em;"}[​]{.vlist-s}]{.vlist-r}[[[]{}]{.vlist
style="height:0.43581800000000004em;"}]{.vlist-r}]{.vlist-t
.vlist-t2}]{.msupsub}]{.mop}[]{.mspace
style="margin-right:0.16666666666666666em;"}[[∑]{.mop .op-symbol
.small-op
style="position:relative;top:-0.0000050000000000050004em;"}[[[[[[]{.pstrut
style="height:2.7em;"}[[[[x]{.mord .mathit .mtight}[∈]{.mrel
.mtight}[X]{.mord .mathit .mtight
style="margin-right:0.07847em;"}]{.mord .mtight}]{.mord
.mtight}]{.sizing .reset-size6 .size3
.mtight}]{style="top:-2.40029em;margin-left:0em;margin-right:0.05em;"}]{.vlist
style="height:0.17862099999999992em;"}[​]{.vlist-s}]{.vlist-r}[[[]{}]{.vlist
style="height:0.32708000000000004em;"}]{.vlist-r}]{.vlist-t
.vlist-t2}]{.msupsub}]{.mop}[]{.mspace
style="margin-right:0.16666666666666666em;"}[p]{.mord
.mathit}[(]{.mopen}[x]{.mord .mathit}[,]{.mpunct}[]{.mspace
style="margin-right:0.16666666666666666em;"}[y]{.mord .mathit
style="margin-right:0.03588em;"}[)]{.mclose}[]{.mspace
style="margin-right:0.16666666666666666em;"}[lo[g]{style="margin-right:0.01389em;"}]{.mop}[]{.mspace
style="margin-right:0.16666666666666666em;"}[[[[(]{.delimsizing
.size2}]{.mopen .delimcenter style="top:0em;"}[[[]{.mopen
.nulldelimiter}[[[[[[]{.pstrut style="height:3em;"}[[[p]{.mord .mathit
.mtight}[(]{.mopen .mtight}[x]{.mord .mathit .mtight}[)]{.mclose
.mtight}[]{.mspace .mtight
style="margin-right:0.19516666666666668em;"}[p]{.mord .mathit
.mtight}[(]{.mopen .mtight}[y]{.mord .mathit .mtight
style="margin-right:0.03588em;"}[)]{.mclose .mtight}]{.mord
.mtight}]{.sizing .reset-size6 .size3
.mtight}]{style="top:-2.655em;"}[[]{.pstrut
style="height:3em;"}[]{.frac-line
style="border-bottom-width:0.04em;"}]{style="top:-3.23em;"}[[]{.pstrut
style="height:3em;"}[[[p]{.mord .mathit .mtight}[(]{.mopen
.mtight}[x]{.mord .mathit .mtight}[,]{.mpunct .mtight}[y]{.mord .mathit
.mtight style="margin-right:0.03588em;"}[)]{.mclose .mtight}]{.mord
.mtight}]{.sizing .reset-size6 .size3
.mtight}]{style="top:-3.485em;"}]{.vlist
style="height:1.01em;"}[​]{.vlist-s}]{.vlist-r}[[[]{}]{.vlist
style="height:0.52em;"}]{.vlist-r}]{.vlist-t
.vlist-t2}]{.mfrac}[]{.mclose
.nulldelimiter}]{.mord}]{.mord}[[)]{.delimsizing .size2}]{.mclose
.delimcenter style="top:0em;"}]{.minner}]{.mord}[,]{.mpunct}[]{.mspace
style="margin-right:0.16666666666666666em;"}[]{.mspace
style="margin-right:-0.16666666666666666em;"}]{.base}]{.katex-html
aria-hidden="true"}]{.katex}\
\

在连续随机变量的情形下，求和被替换成了二重定积分：

\
\
[[$I\left( X;Y \right) = \int_{Y}\int_{X}p\left( x,y \right)\log\left( \frac{p\left( x,y \right)}{p\left( x \right)\, p\left( y \right)} \right)\text{ \,}dx\, dy,$*{Y}\\int*
{X}p(x,y)\\log {\\left({\\frac
{p(x,y)}{p(x)\\,p(y)}}\\right)}\\;dx\\,dy,}]{.katex-mathml}[[[]{.strut
style="height:2.40003em;vertical-align:-0.95003em;"}[[I]{.mord .mathit
style="margin-right:0.07847em;"}[(]{.mopen}[X]{.mord .mathit
style="margin-right:0.07847em;"}[;]{.mpunct}[]{.mspace
style="margin-right:0.16666666666666666em;"}[Y]{.mord .mathit
style="margin-right:0.22222em;"}[)]{.mclose}[]{.mspace
style="margin-right:0.2777777777777778em;"}[=]{.mrel}[]{.mspace
style="margin-right:0.2777777777777778em;"}[[∫]{.mop .op-symbol
.large-op
style="margin-right:0.44445em;position:relative;top:-0.0011249999999999316em;"}[[[[[[]{.pstrut
style="height:2.7em;"}[[[Y]{.mord .mathit .mtight
style="margin-right:0.22222em;"}]{.mord .mtight}]{.sizing .reset-size6
.size3
.mtight}]{style="top:-1.7880500000000001em;margin-left:-0.44445em;margin-right:0.05em;"}]{.vlist
style="height:-0.433619em;"}[​]{.vlist-s}]{.vlist-r}[[[]{}]{.vlist
style="height:0.9119499999999999em;"}]{.vlist-r}]{.vlist-t
.vlist-t2}]{.msupsub}]{.mop}[]{.mspace
style="margin-right:0.16666666666666666em;"}[[∫]{.mop .op-symbol
.large-op
style="margin-right:0.44445em;position:relative;top:-0.0011249999999999316em;"}[[[[[[]{.pstrut
style="height:2.7em;"}[[[X]{.mord .mathit .mtight
style="margin-right:0.07847em;"}]{.mord .mtight}]{.sizing .reset-size6
.size3
.mtight}]{style="top:-1.7880500000000001em;margin-left:-0.44445em;margin-right:0.05em;"}]{.vlist
style="height:-0.433619em;"}[​]{.vlist-s}]{.vlist-r}[[[]{}]{.vlist
style="height:0.9119499999999999em;"}]{.vlist-r}]{.vlist-t
.vlist-t2}]{.msupsub}]{.mop}[]{.mspace
style="margin-right:0.16666666666666666em;"}[p]{.mord
.mathit}[(]{.mopen}[x]{.mord .mathit}[,]{.mpunct}[]{.mspace
style="margin-right:0.16666666666666666em;"}[y]{.mord .mathit
style="margin-right:0.03588em;"}[)]{.mclose}[]{.mspace
style="margin-right:0.16666666666666666em;"}[lo[g]{style="margin-right:0.01389em;"}]{.mop}[]{.mspace
style="margin-right:0.16666666666666666em;"}[[[[(]{.delimsizing
.size3}]{.mopen .delimcenter style="top:0em;"}[[[]{.mopen
.nulldelimiter}[[[[[[]{.pstrut style="height:3em;"}[[p]{.mord
.mathit}[(]{.mopen}[x]{.mord .mathit}[)]{.mclose}[]{.mspace
style="margin-right:0.16666666666666666em;"}[p]{.mord
.mathit}[(]{.mopen}[y]{.mord .mathit
style="margin-right:0.03588em;"}[)]{.mclose}]{.mord}]{style="top:-2.314em;"}[[]{.pstrut
style="height:3em;"}[]{.frac-line
style="border-bottom-width:0.04em;"}]{style="top:-3.23em;"}[[]{.pstrut
style="height:3em;"}[[p]{.mord .mathit}[(]{.mopen}[x]{.mord
.mathit}[,]{.mpunct}[]{.mspace
style="margin-right:0.16666666666666666em;"}[y]{.mord .mathit
style="margin-right:0.03588em;"}[)]{.mclose}]{.mord}]{style="top:-3.677em;"}]{.vlist
style="height:1.427em;"}[​]{.vlist-s}]{.vlist-r}[[[]{}]{.vlist
style="height:0.936em;"}]{.vlist-r}]{.vlist-t
.vlist-t2}]{.mfrac}[]{.mclose
.nulldelimiter}]{.mord}]{.mord}[[)]{.delimsizing .size3}]{.mclose
.delimcenter style="top:0em;"}]{.minner}]{.mord}[]{.mspace
style="margin-right:0.2777777777777778em;"}[d]{.mord .mathit}[x]{.mord
.mathit}[]{.mspace style="margin-right:0.16666666666666666em;"}[d]{.mord
.mathit}[y]{.mord .mathit
style="margin-right:0.03588em;"}[,]{.mpunct}]{.mord}]{.base}]{.katex-html
aria-hidden="true"}]{.katex}\
\

其中 p(x,y) 当前是 X 和 Y
的联合概率密度函数，而[[${p\left( x \right)}p\left( x \right){p\left( y \right)}p\left( y \right)$]{.katex-mathml}[[[]{.strut
style="height:1em;vertical-align:-0.25em;"}[[p]{.mord
.mathit}[(]{.mopen}[x]{.mord .mathit}[)]{.mclose}]{.mord}[p]{.mord
.mathit}[(]{.mopen}[x]{.mord .mathit}[)]{.mclose}[和]{.mord
.cjk_fallback}[[p]{.mord .mathit}[(]{.mopen}[y]{.mord .mathit
style="margin-right:0.03588em;"}[)]{.mclose}]{.mord}[p]{.mord
.mathit}[(]{.mopen}[y]{.mord .mathit
style="margin-right:0.03588em;"}[)]{.mclose}]{.base}]{.katex-html
aria-hidden="true"}]{.katex}分别是 X 和 Y 的边缘概率密度函数。

\
\
[[$I\left( X;Y \right) = \int_{Y}\int_{X}p\left( x,y \right)\log\left( \frac{p\left( x,y \right)}{p\left( x \right)\, p\left( y \right)} \right)\text{ \,}dx\, dy,$*{Y}\\int*
{X}p(x,y)\\log {\\left({\\frac
{p(x,y)}{p(x)\\,p(y)}}\\right)}\\;dx\\,dy,]{.katex-mathml}[[[]{.strut
style="height:1em;vertical-align:-0.25em;"}[I]{.mord .mathit
style="margin-right:0.07847em;"}[(]{.mopen}[X]{.mord .mathit
style="margin-right:0.07847em;"}[;]{.mpunct}[]{.mspace
style="margin-right:0.16666666666666666em;"}[Y]{.mord .mathit
style="margin-right:0.22222em;"}[)]{.mclose}[]{.mspace
style="margin-right:0.2777777777777778em;"}[=]{.mrel}[]{.mspace
style="margin-right:0.2777777777777778em;"}]{.base}[[]{.strut
style="height:1.80002em;vertical-align:-0.65002em;"}[[∫]{.mop .op-symbol
.small-op
style="margin-right:0.19445em;position:relative;top:-0.0005599999999999772em;"}[[[[[[]{.pstrut
style="height:2.7em;"}[[[Y]{.mord .mathit .mtight
style="margin-right:0.22222em;"}]{.mord .mtight}]{.sizing .reset-size6
.size3
.mtight}]{style="top:-2.34418em;margin-left:-0.19445em;margin-right:0.05em;"}]{.vlist
style="height:0.12251099999999993em;"}[​]{.vlist-s}]{.vlist-r}[[[]{}]{.vlist
style="height:0.35582em;"}]{.vlist-r}]{.vlist-t
.vlist-t2}]{.msupsub}]{.mop}[]{.mspace
style="margin-right:0.16666666666666666em;"}[[∫]{.mop .op-symbol
.small-op
style="margin-right:0.19445em;position:relative;top:-0.0005599999999999772em;"}[[[[[[]{.pstrut
style="height:2.7em;"}[[[X]{.mord .mathit .mtight
style="margin-right:0.07847em;"}]{.mord .mtight}]{.sizing .reset-size6
.size3
.mtight}]{style="top:-2.34418em;margin-left:-0.19445em;margin-right:0.05em;"}]{.vlist
style="height:0.12251099999999993em;"}[​]{.vlist-s}]{.vlist-r}[[[]{}]{.vlist
style="height:0.35582em;"}]{.vlist-r}]{.vlist-t
.vlist-t2}]{.msupsub}]{.mop}[]{.mspace
style="margin-right:0.16666666666666666em;"}[p]{.mord
.mathit}[(]{.mopen}[x]{.mord .mathit}[,]{.mpunct}[]{.mspace
style="margin-right:0.16666666666666666em;"}[y]{.mord .mathit
style="margin-right:0.03588em;"}[)]{.mclose}[]{.mspace
style="margin-right:0.16666666666666666em;"}[lo[g]{style="margin-right:0.01389em;"}]{.mop}[]{.mspace
style="margin-right:0.16666666666666666em;"}[[[[(]{.delimsizing
.size2}]{.mopen .delimcenter style="top:0em;"}[[[]{.mopen
.nulldelimiter}[[[[[[]{.pstrut style="height:3em;"}[[[p]{.mord .mathit
.mtight}[(]{.mopen .mtight}[x]{.mord .mathit .mtight}[)]{.mclose
.mtight}[]{.mspace .mtight
style="margin-right:0.19516666666666668em;"}[p]{.mord .mathit
.mtight}[(]{.mopen .mtight}[y]{.mord .mathit .mtight
style="margin-right:0.03588em;"}[)]{.mclose .mtight}]{.mord
.mtight}]{.sizing .reset-size6 .size3
.mtight}]{style="top:-2.655em;"}[[]{.pstrut
style="height:3em;"}[]{.frac-line
style="border-bottom-width:0.04em;"}]{style="top:-3.23em;"}[[]{.pstrut
style="height:3em;"}[[[p]{.mord .mathit .mtight}[(]{.mopen
.mtight}[x]{.mord .mathit .mtight}[,]{.mpunct .mtight}[y]{.mord .mathit
.mtight style="margin-right:0.03588em;"}[)]{.mclose .mtight}]{.mord
.mtight}]{.sizing .reset-size6 .size3
.mtight}]{style="top:-3.485em;"}]{.vlist
style="height:1.01em;"}[​]{.vlist-s}]{.vlist-r}[[[]{}]{.vlist
style="height:0.52em;"}]{.vlist-r}]{.vlist-t
.vlist-t2}]{.mfrac}[]{.mclose
.nulldelimiter}]{.mord}]{.mord}[[)]{.delimsizing .size2}]{.mclose
.delimcenter style="top:0em;"}]{.minner}]{.mord}[]{.mspace
style="margin-right:0.2777777777777778em;"}[d]{.mord .mathit}[x]{.mord
.mathit}[]{.mspace style="margin-right:0.16666666666666666em;"}[d]{.mord
.mathit}[y]{.mord .mathit
style="margin-right:0.03588em;"}[,]{.mpunct}]{.base}]{.katex-html
aria-hidden="true"}]{.katex}\
\

如果对数以 2 为基底，互信息的单位是bit。

直观上，互信息度量 X 和 Y
共享的信息：它度量知道这两个变量其中一个，对另一个不确定度减少的程度。例如，如果
X 和 Y 相互独立，则知道 X 不对 Y
提供任何信息，反之亦然，所以它们的互信息为零。在另一个极端，如果 X 是 Y
的一个确定性函数，且 Y 也是 X 的一个确定性函数，那么传递的所有信息被 X
和 Y 共享：知道 X 决定 Y 的值，反之亦然。因此，在此情形互信息与 Y（或
X）单独包含的不确定度相同，称作 Y（或 X）的熵。而且，这个互信息与 X
的熵和 Y 的熵相同。（这种情形的一个非常特殊的情况是当 X 和 Y
为相同随机变量时。）

互信息是 X 和 Y 的联合分布相对于假定 X 和 Y
独立情况下的联合分布之间的内在依赖性。
于是互信息以下面方式度量依赖性：I(X; Y) = 0 当且仅当 X 和 Y
为独立随机变量。从一个方向很容易看出：当 X 和 Y 独立时，p(x,y) = p(x)
p(y)，因此：

\
\
[[${\log\left( \frac{p\left( x,y \right)}{p\left( x \right)\, p\left( y \right)} \right) = \log 1 = 0.\,}\log\left( \frac{p\left( x,y \right)}{p\left( x \right)\, p\left( y \right)} \right) = \log 1 = 0.\,$]{.katex-mathml}[[[]{.strut
style="height:2.40003em;vertical-align:-0.95003em;"}[[lo[g]{style="margin-right:0.01389em;"}]{.mop}[]{.mspace
style="margin-right:0.16666666666666666em;"}[[[[(]{.delimsizing
.size3}]{.mopen .delimcenter style="top:0em;"}[[[]{.mopen
.nulldelimiter}[[[[[[]{.pstrut style="height:3em;"}[[p]{.mord
.mathit}[(]{.mopen}[x]{.mord .mathit}[)]{.mclose}[]{.mspace
style="margin-right:0.16666666666666666em;"}[p]{.mord
.mathit}[(]{.mopen}[y]{.mord .mathit
style="margin-right:0.03588em;"}[)]{.mclose}]{.mord}]{style="top:-2.314em;"}[[]{.pstrut
style="height:3em;"}[]{.frac-line
style="border-bottom-width:0.04em;"}]{style="top:-3.23em;"}[[]{.pstrut
style="height:3em;"}[[p]{.mord .mathit}[(]{.mopen}[x]{.mord
.mathit}[,]{.mpunct}[]{.mspace
style="margin-right:0.16666666666666666em;"}[y]{.mord .mathit
style="margin-right:0.03588em;"}[)]{.mclose}]{.mord}]{style="top:-3.677em;"}]{.vlist
style="height:1.427em;"}[​]{.vlist-s}]{.vlist-r}[[[]{}]{.vlist
style="height:0.936em;"}]{.vlist-r}]{.vlist-t
.vlist-t2}]{.mfrac}[]{.mclose
.nulldelimiter}]{.mord}]{.mord}[[)]{.delimsizing .size3}]{.mclose
.delimcenter style="top:0em;"}]{.minner}]{.mord}[]{.mspace
style="margin-right:0.2777777777777778em;"}[=]{.mrel}[]{.mspace
style="margin-right:0.2777777777777778em;"}[lo[g]{style="margin-right:0.01389em;"}]{.mop}[]{.mspace
style="margin-right:0.16666666666666666em;"}[1]{.mord}[]{.mspace
style="margin-right:0.2777777777777778em;"}[=]{.mrel}[]{.mspace
style="margin-right:0.2777777777777778em;"}[0]{.mord}[.]{.mord}[]{.mspace
style="margin-right:0.16666666666666666em;"}[]{.mspace
style="margin-right:-0.16666666666666666em;"}]{.mord}[]{.mspace
style="margin-right:0.16666666666666666em;"}[lo[g]{style="margin-right:0.01389em;"}]{.mop}[]{.mspace
style="margin-right:0.16666666666666666em;"}[[[[(]{.delimsizing
.size2}]{.mopen .delimcenter style="top:0em;"}[[[]{.mopen
.nulldelimiter}[[[[[[]{.pstrut style="height:3em;"}[[[p]{.mord .mathit
.mtight}[(]{.mopen .mtight}[x]{.mord .mathit .mtight}[)]{.mclose
.mtight}[]{.mspace .mtight
style="margin-right:0.19516666666666668em;"}[p]{.mord .mathit
.mtight}[(]{.mopen .mtight}[y]{.mord .mathit .mtight
style="margin-right:0.03588em;"}[)]{.mclose .mtight}]{.mord
.mtight}]{.sizing .reset-size6 .size3
.mtight}]{style="top:-2.655em;"}[[]{.pstrut
style="height:3em;"}[]{.frac-line
style="border-bottom-width:0.04em;"}]{style="top:-3.23em;"}[[]{.pstrut
style="height:3em;"}[[[p]{.mord .mathit .mtight}[(]{.mopen
.mtight}[x]{.mord .mathit .mtight}[,]{.mpunct .mtight}[y]{.mord .mathit
.mtight style="margin-right:0.03588em;"}[)]{.mclose .mtight}]{.mord
.mtight}]{.sizing .reset-size6 .size3
.mtight}]{style="top:-3.485em;"}]{.vlist
style="height:1.01em;"}[​]{.vlist-s}]{.vlist-r}[[[]{}]{.vlist
style="height:0.52em;"}]{.vlist-r}]{.vlist-t
.vlist-t2}]{.mfrac}[]{.mclose
.nulldelimiter}]{.mord}]{.mord}[[)]{.delimsizing .size2}]{.mclose
.delimcenter style="top:0em;"}]{.minner}]{.mord}[]{.mspace
style="margin-right:0.2777777777777778em;"}[=]{.mrel}[]{.mspace
style="margin-right:0.2777777777777778em;"}]{.base}[[]{.strut
style="height:0.8888799999999999em;vertical-align:-0.19444em;"}[lo[g]{style="margin-right:0.01389em;"}]{.mop}[]{.mspace
style="margin-right:0.16666666666666666em;"}[1]{.mord}[]{.mspace
style="margin-right:0.2777777777777778em;"}[=]{.mrel}[]{.mspace
style="margin-right:0.2777777777777778em;"}]{.base}[[]{.strut
style="height:0.64444em;vertical-align:0em;"}[0]{.mord}[.]{.mord}[]{.mspace
style="margin-right:0.16666666666666666em;"}[]{.mspace
style="margin-right:-0.16666666666666666em;"}]{.base}]{.katex-html
aria-hidden="true"}]{.katex}\
\

此外，互信息是非负的，而且是对称的（即 I(X;Y) = I(Y;X)）。

### [实现细节]{} {#h-1}

    # Build and the 判别器 and 识别网络
    self.discriminator, self.auxilliary =self.build_disk_and_q_net()

    # Build the generator
    self.generator = self.build_generator()

    # The generator takes noise and the target label as input
    # and generates the corresponding digit of that label
    gen_input = Input(shape=(self.latent_dim,))
    img = self.generator(gen_input)
    #生成器依赖标签信息

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(self.channels, kernel_size=3, padding='same'))
        model.add(Activation("tanh"))

        gen_input = Input(shape=(self.latent_dim,))
        img = model(gen_input)

        model.summary()

        return Model(gen_input, img)

生成器的输入是一个72维的向量,输出是一张图片

------------------------------------------------------------------------

    def build_disk_and_q_net(self):

        img = Input(shape=self.img_shape)

        # Shared layers between discriminator and recognition network
        model = Sequential()
        model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(512, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Flatten())

        img_embedding = model(img)

        # Discriminator
        validity = Dense(1, activation='sigmoid')(img_embedding)

        # Recognition
        q_net = Dense(128, activation='relu')(img_embedding)
        label = Dense(self.num_classes, activation='softmax')(q_net)

        # Return discriminator and recognition network
        return Model(img, validity), Model(img, label)

判别器要判别真假,q\_net 和要判别 label.

    def sample_generator_input(self, batch_size):
        # Generator inputs
        sampled_noise = np.random.normal(0, 1, (batch_size, 62))
        sampled_labels = np.random.randint(0, self.num_classes, batch_size).reshape(-1, 1)
        sampled_labels = to_categorical(sampled_labels, num_classes=self.num_classes)

噪声 Z 的输入包括高斯噪声和一个10维的 label-one-hot 向量.

------------------------------------------------------------------------

其实写到这里,我发现了一点啊,他真的和 ACGAN 好像啊……

![image-20190407011433732](https://ws1.sinaimg.cn/large/006tNc79ly1g1tf1c5b31j31980k0qby.jpg "image-20190407011433732")
image-20190407011433732
像到什么程度,自行体会,不过说实话原理还是有所不同的,说实话,我很无语

接下来继续填 semi-supervised GAN 的坑吧,最近self supervised GAN
出来了,行有余力可以看看.

总感觉接下来要开 cycleGAN
系列了,这个有意思.看到这里了要是看官老爷觉得interesting,不如麻烦关注转发好看三连一波,你的每一点小小的鼓励都是对作者莫大的鼓舞鸭😄.

参考:

https://www.zhihu.com/question/24059517/answer/37430101

https://blog.csdn.net/wspba/article/details/54808833

<https://www.jiqizhixin.com/articles/2018-10-29-21>

<https://aistudio.baidu.com/aistudio/#/projectdetail/29156>

<https://aistudio.baidu.com/aistudio/#/projectdetail/29156>

<https://zh.wikipedia.org/zh-hant/%E4%BA%92%E4%BF%A1%E6%81%AF>

</div>

</div>
