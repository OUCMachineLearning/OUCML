# 深入理解风格迁移三部曲(一)--UNIT

标签（空格分隔）： 陈扬

---

> MUNIT: Multimodal Unsupervised Image-to-Image Translation
>
> github:https://github.com/NVlabs/MUNIT
>
> 作者:[陈扬](https://www.zhihu.com/people/ba-la-ba-la-82-47/activities)

[toc]

## 简介

近期我研究的方向转向了GAN的应用, 其中图像的风格迁移是GAN中一个非常有意思的应用,传统的方法基于拉普拉斯金字塔对成对的图像进行纹理上的风格迁移.随着2014年GAN的爆火,研究者发现GAN通过判别器D学习两个图像域的关系,实现了unpaired image-to-image(非成对图像数据集的风格迁移)的功能,其中有两个广为人知的应用分别是pix2pix和cycleGAN,今天我们另辟蹊径,从NVIDIA-Lab提出的UNIT框架来探索image-to-image的实现原理.

---

## 前言

在开始说UNIT之前,我们先来简要的回顾一下GAN和VAE,这也是在我之前的[ODOG](https://github.com/OUCMachineLearning/OUCML/tree/master/One_Day_One_GAN)项目中有过详细的介绍.

### GAN

**生成对抗网络**（英语：**G**enerative **A**dversarial **N**etwork，简称GAN）是[非监督式学习](https://zh.wikipedia.org/wiki/%E9%9D%9E%E7%9B%91%E7%9D%A3%E5%BC%8F%E5%AD%A6%E4%B9%A0)的一种方法，通过让两个[神经网络](https://zh.wikipedia.org/wiki/%E4%BA%BA%E5%B7%A5%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C)相互[博弈](https://zh.wikipedia.org/wiki/%E5%8D%9A%E5%BC%88%E8%AE%BA)的方式进行学习。该方法由[伊恩·古德费洛](https://zh.wikipedia.org/wiki/%E4%BC%8A%E6%81%A9%C2%B7%E5%8F%A4%E5%BE%B7%E8%B4%B9%E6%B4%9B)等人于2014年提出。[[1\]](https://zh.wikipedia.org/w/index.php?title=%E7%94%9F%E6%88%90%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9C&oldid=52710805#cite_note-MyUser_Arxiv.org_April_7_2016c-1)

核心公式:

![image_1ct4sn8kqg8ftika3b1nmj6i1j.png-43kB](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-12-08-140757.jpg)
这个公式我们要分成两个部分来看:
先看前半部分:![image_1ct4ss3bu18gs14fs5af1qgrhf420.png-32.1kB][4]
这个公式的意思是,先看加号前面$\Epsilon_{x\sim p_{data}(x)}[\log D(x)]+\Epsilon_{z\sim p_z(z)}[log(1-D(G(z)))]$

 ,我们希望D最大,所以log(D(x))应该最大,意味着我的判别器可以很好的识别出,真实世界图像是"true",在看加号后面$\Epsilon_{z\sim p_z(z)}[\log(1-D(G(z)))]$,要让log尽可能的大,需要的是D(G(z))尽可能的小,意味着我们生成模型的图片应该尽可能的被判别模型视为"FALSE".

再看后半部分部分![image_1ct5064nkclm14jh1o401pdm1v349.png-29.7kB][5],
我们应该让G尽可能的小,加号前面的式子并没有G,所以无关,在看加号后面的式子\Epsilon_{z\sim p_z(z)}[\log(1-D(G(z)))],要让G尽可能地小,就要D(G(Z))尽可能的大,也就是说本来就一张→噪声生成的图片,判别器却被迷惑了,以为是一张真实世界图片.这就是所谓的以假乱真.
![image_1ct56911djif1ae81ebq10sf1l3om.png-91.4kB][6]

### VAE

VAE也叫变分自动编码机, 其基础是自编码机(autoencoder),AE通过对输入X进行编码后得到一个低维的向量y，然后根据这个向量还原出输入X，。通过对比X与$ \widetilde {X}$，的误差，再利用神经网络去训练使得误差逐渐 减小从而达到非监督学习的目的. 

![image-20191208221145293](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-12-08-141145.png)

## UNIT

在我们展开深入理解MUNIT之前,我们来看一下他的前身UNIT,由NVIDIA-Lab在NIPS 2017年提出,该文章首次提Image-Image Translation这个概念，将计算机视觉和计算机图形学的许多任务总结进去，分为一对多和多对一的两类转换任务，包括CV里的边缘检测，图像分割，语义标签以及CG里的mapping labels or sparse user inputs to realistic images.

![image-20191124143200833](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-12-08-135441.png)

该文章定义了$\chi_1$和$\chi_2$作为两个图像域.传统的supervised Image-to-image 通过对图像域进行采样,求其联合概率分布$$P_{(\chi_1,\chi_2)}(x_1,x_2)$$,通过Encoder-Decoder的思想,作者定义了两个E和G,希望使得z=E(X)在latent space上近可能的分布一致.意味着当我们同时对$$Sample(\chi_1 ,\chi_2)$$时,我们希望得出:
$$
z=E_{1}^{*}\left(x_{1}\right)=E_{2}^{*}\left(x_{2}\right)
$$

这样,我们得到了两个Domain下image的一致表示,再通过令$$G=D$$,从latent space中重构$\hat{x}=G(z)$,

因此,我们两个采样下的$\{x_1,x_2\}$经过$\{<E_1,G_1>,<E_2,G_1>,<E_1,G_2>,<E_2,G_1>\}$后得到了$\{\hat{x}^{1\rightarrow1}_1,\hat{x}^{2\rightarrow1}_2,\hat{x}^{1\rightarrow2}_1,\hat{x}^{2\rightarrow2}_2\}$,再把:
$$
\hat{x}^{1\rightarrow1}_1,\hat{x}^{2\rightarrow1}_2\rightarrow D_1\rightarrow T/F
$$

$$
\hat{x}^{1\rightarrow2}_1,\hat{x}^{2\rightarrow2}_2\rightarrow D_2\rightarrow T/F
$$

通过Adv_loss对抗学习跨域生成图片的效果.

可能细心的你以及发现了这是不是很类似VAE-GAN吗?是的.

作者通过联合训练4个网络$VAE_1, VAE_2, GAN_1, GAN_2$的三个$loss function$来训练整个网络:
$$
\begin{aligned} \min _{E_{1}, E_{2}, G_{1}, G_{2}} \max _{D_{1}, D_{2}} \mathcal{L}_{\mathrm{VAE}_{1}}\left(E_{1}, G_{1}\right)+\mathcal{L}_{\mathrm{GAN}_{1}}\left(E_{2}, G_{1}, D_{1}\right)+\mathcal{L}_{\mathrm{CC}_{1}}\left(E_{1}, G_{1}, E_{2}, G_{2}\right) \\ \mathcal{L}_{\mathrm{VAE}_{2}}\left(E_{2}, G_{2}\right)+\mathcal{L}_{\mathrm{GAN}_{2}}\left(E_{1}, G_{2}, D_{2}\right)+\mathcal{L}_{\mathrm{CC}_{2}}\left(E_{2}, G_{2}, E_{1}, G_{1}\right) \end{aligned}
$$
**VAE**的目标是minimize source domain to latent space's KL diversity and latent space to destination domain's KL diversity(我觉得中文太拗口了,这句话实在是说不来)来最小化变分上界,VAE的定义如下:
$$
\begin{array}{l}{\mathcal{L}_{\mathrm{VAE}_{1}}\left(E_{1}, G_{1}\right)=\lambda_{1} \operatorname{KL}\left(q_{1}\left(z_{1} | x_{1}\right) \| p_{\eta}(z)\right)-\lambda_{2} \mathbb{E}_{z_{1} \sim q_{1}\left(z_{1} | x_{1}\right)}\left[\log p_{G_{1}}\left(x_{1} | z_{1}\right)\right]} \\ {\mathcal{L}_{\mathrm{VAE}_{2}}\left(E_{2}, G_{2}\right)=\lambda_{1} \operatorname{KL}\left(q_{2}\left(z_{2} | x_{2}\right) \| p_{\eta}(z)\right)-\lambda_{2} \mathbb{E}_{z_{2} \sim q_{2}\left(z_{2} | x_{2}\right)}\left[\log p_{G_{2}}\left(x_{2} | z_{2}\right)\right]}\end{array}
$$
**对抗**:GAN_LOSS被用于确保翻译图像类似图像在目标域.定义如下:
$$
\begin{array}{l}{\mathcal{L}_{\mathrm{GAN}_{1}}\left(E_{2}, G_{1}, D_{1}\right)=\lambda_{0} \mathbb{E}_{x_{1} \sim P_{\mathcal{X}_{1}}}\left[\log D_{1}\left(x_{1}\right)\right]+\lambda_{0} \mathbb{E}_{z_{2} \sim q_{2}\left(z_{2} | x_{2}\right)}\left[\log \left(1-D_{1}\left(G_{1}\left(z_{2}\right)\right)\right)\right]} \\ {\mathcal{L}_{\mathrm{GAN}_{2}}\left(E_{1}, G_{2}, D_{2}\right)=\lambda_{0} \mathbb{E}_{x_{2} \sim P_{\mathcal{X}_{2}}}\left[\log D_{2}\left(x_{2}\right)\right]+\lambda_{0} \mathbb{E}_{z_{1} \sim q_{1}\left(z_{1} | x_{1}\right)}\left[\log \left(1-D_{2}\left(G_{2}\left(z_{1}\right)\right)\right)\right]}\end{array}
$$
**循环一致性**:由于shared latent-space假设暗含了循环一致性约束，因此我们在提出的框架中实施循环一致性约束，以进一步规范不适定的无监督图像间转换问题。产生的信息处理流称为循环重建流,定义如下:
$$
\begin{aligned} \mathcal{L}_{\mathrm{CC}_{1}}\left(E_{1}, G_{1}, E_{2}, G_{2}\right)=&\left.\left.\lambda_{3} \operatorname{KL}\left(q_{1}\left(z_{1} | x_{1}\right) \| p_{\eta}(z)\right)+\lambda_{3} \operatorname{KL}\left(q_{2} | x_{1}^{1 \rightarrow 2}\right)\right) \| p_{\eta}(z)\right)-\\ & \lambda_{4} \mathbb{E}_{z_{2} \sim q_{2}\left(z_{2} | x_{1}^{1 \rightarrow 2}\right)}\left[\log p_{G_{1}}\left(x_{1} | z_{2}\right)\right] \\ \mathcal{L}_{\mathrm{CC}_{2}}\left(E_{2}, G_{2}, E_{1}, G_{1}\right)=&\left.\lambda_{3} \operatorname{KL}\left(q_{2}\left(z_{2} | x_{2}\right) \| p_{\eta}(z)\right)+\lambda_{3} \operatorname{KL}\left(q_{1}\left(z_{1} | x_{2}^{2 \rightarrow 1}\right)\right) \| p_{\eta}(z)\right)-\\ & \lambda_{4} \mathbb{E}_{z_{1} \sim q_{1}\left(z_{1} | x_{2}^{2} \rightarrow 1\right)}\left[\log p_{G_{2}}\left(x_{2} | z_{1}\right)\right] \end{aligned}
$$
训练好的网络,我们可以通过对latent sapce的latent variable重编码,进而把输入图像迁移到各个域中:

![image-20191124154047242](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-12-08-135443.png)

![image-20191124153813482](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-12-08-135442.png)

---

## 



[1]: http://static.zybuluo.com/Team/mmbpgzozxzaesexm03rkvt3a/image_1ct4sdm7c171la761q1rf881aak9.png
[2]: http://static.zybuluo.com/Team/lvg8hb8uip1yut64h5n8reh0/image_1ct4si11fu6o2lg1m6ub6110o6m.png
[3]: http://static.zybuluo.com/Team/t1p9ulca7z9y5vyg9yhwucta/image_1ct4sn8kqg8ftika3b1nmj6i1j.png
[4]: http://static.zybuluo.com/Team/i791fay3szt4melmamnt9bb7/image_1ct4ss3bu18gs14fs5af1qgrhf420.png
[5]: http://static.zybuluo.com/Team/9ex5978j1gqod4ffjd2jmiti/image_1ct5064nkclm14jh1o401pdm1v349.png
[6]: http://static.zybuluo.com/Team/rt34igjxwylimw3ps2j4zjs8/image_1ct56911djif1ae81ebq10sf1l3om.png
[7]: http://static.zybuluo.com/Team/hvuexh4izp5tvjud2g4o3z9t/image_1ct566r7419d7ai41rmaa6qqqt9.png

