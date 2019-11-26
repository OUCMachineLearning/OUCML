# 深入理解风格迁移三部曲(二)--MUNIT

标签（空格分隔）： 陈扬

---

> MUNIT: Multimodal Unsupervised Image-to-Image Translation
>
> github:https://github.com/NVlabs/MUNIT
>
> 作者:[陈扬](https://www.zhihu.com/people/ba-la-ba-la-82-47/activities)

[toc]

## 简介

无监督的图像到图像转换方法学习利用图像的非结构化(UNlabel)数据集将给定类中的图像映射到不同类中的类似图像。在ECCV2018上,NVIDIA-lab发表了MUNIT,意在解决unpair image-to-image中的一对多的转换.原本UNIT可以看作是猫和狗的 1 to 1 对应,但MUNIT，多了一个 M -> Multimodal的过程，可以看作是 1 to Many。

通过实验证明,该算法有效的提升了生产模型的多样性.

---

## 前言

在我们展开深入理解MUNIT之前,我们来看一下他的前身UNIT,由NVIDIA-Lab在NIPS 2017年提出,该文章首次提Image-Image Translation这个概念，将计算机视觉和计算机图形学的许多任务总结进去，分为一对多和多对一的两类转换任务，包括CV里的边缘检测，图像分割，语义标签以及CG里的mapping labels or sparse user inputs to realistic images.

![image-20191124143200833](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-24-103533.png)

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

![image-20191124154047242](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-24-103536.png)

![image-20191124153813482](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-24-103538.png)

---

## Multi-modal

虽然UNIT以及其变种(cycleGAN等等)已经表现得非常成功，但是其存在一个缺点就是,没办法实现image-to-image的多样性,每一张图片只能生成目标域中一张特定的图片.本文提出一种方法,通过讲源域图像进行编码得到内容编码和风格编码,再通过对风格编码进行随机采样,再和内容编码一起重构,实现一对多的的图像转换任务.

![image-20191124192114138](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-24-112114.png)

Let x1 ∈ X1 and x2 ∈ X2 be images from two diﬀerent image domains. In the  unsupervised image-to-image translation setting, we are given samples drawn from two marginal distributions p(x1 ) and p(x2 ), without access to the joint  distribution p(x1 , x2 ). Our goal is to estimate the two conditionals p(x2 |x1 )  and p(x1 |x2 ) with learned image-to-image translation models p(x1→2 |x1 ) and  p(x2→1 |x2 ), where x1→2 is a sample produced by translating x1 to X2 (similar  for x2→1 ). In general, p(x2 |x1 ) and p(x1 |x2 ) are complex and multimodal distributions,  in  which  case  a  deterministic  translation  model  does  not  work  well.



## Model

![image-20191124194738432](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-24-114739.png)

![image-20191124194930752](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-24-114930.png)

![image-20191124193808118](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-24-113808.png)

our translation model consists of an encoder Ei and a decoder Gi for each domain Xi (i = 1, 2).

the latent code of each auto-encoder is factorized into a content code ci and a style code si , where

 (ci , si ) = (Eic (xi ), Eis (xi )) = Ei (xi ). 

by swapping encoder-decoder pairs,translate an image x1 ∈ X1 to X2 , we ﬁrst extract its content latent code c1 = E1c (x1 ) and  randomly draw a style latent code s2 from the prior distribution q(s2 ) ∼ N (0, I). 
We then use G2 to produce the ﬁnal output image x1→2 = G2 (c1 , s2 ). We note  that  although  the  prior  distribution  is  unimodal,  the  output  image  distribution can  be  multimodal  thanks  to  the  nonlinearity  of  the  decoder.

![image-20191124194031132](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-24-114031.png)

## Loss

**Bidirectional reconstruction loss**. To learn pairs of encoder and decoder that are inverses of each other, we use objective functions that encourage reconstruc- tion in both image → latent → image and latent → image → latent directions:

- Image  reconstruction.  Given  an  image  sampled  from  the  data  distribution, we  should  be  able  to  reconstruct  it  after  encoding  and  decoding.

$$
\mathcal{L}_{\mathrm{recon}}^{x_{1}}=\mathbb{E}_{x_{1} \sim p\left(x_{1}\right)}\left[\left\|G_{1}\left(E_{1}^{c}\left(x_{1}\right), E_{1}^{s}\left(x_{1}\right)\right)-x_{1}\right\|_{1}\right]
$$
- Latent reconstruction. Given a latent code (style and content) sampled from the latent distribution at translation time, we should be able to recon- struct it after decoding and encoding. 
  $$
  \begin{array}{l}{\mathcal{L}_{\text {recon }}^{c_{1}}=\mathbb{E}_{c_{1} \sim p\left(c_{1}\right), s_{2} \sim q\left(s_{2}\right)}\left[\left\|E_{2}^{c}\left(G_{2}\left(c_{1}, s_{2}\right)\right)-c_{1}\right\|_{1}\right]} \\ {\mathcal{L}_{\text {recon }}^{s_{2}}=\mathbb{E}_{c_{1} \sim p\left(c_{1}\right), s_{2} \sim q\left(s_{2}\right)}\left[\left\|E_{2}^{s}\left(G_{2}\left(c_{1}, s_{2}\right)\right)-s_{2}\right\|_{1}\right]}\end{array}
  $$

- We  use  L1    reconstruction  loss  as  it  encourages  sharp  output  images. 

**Adversarial loss**. We employ GANs to match the distribution of translated images to the target data distribution. In other words, images generated by our model  should  be  indistinguishable  from  real  images  in  the  target  domain.
$$
\mathcal{L}_{\mathrm{GAN}}^{x_{2}}=\mathbb{E}_{c_{1} \sim p\left(c_{1}\right), s_{2} \sim q\left(s_{2}\right)}\left[\log \left(1-D_{2}\left(G_{2}\left(c_{1}, s_{2}\right)\right)\right)\right]+\mathbb{E}_{x_{2} \sim p\left(x_{2}\right)}\left[\log D_{2}\left(x_{2}\right)\right]
$$
**Total loss**. We jointly train the encoders, decoders, and discriminators to opti- mize the ﬁnal objective, which is a weighted sum of the adversarial loss and the bidirectional reconstruction loss terms.
$$
\begin{array}{l}{\min _{E_{1}, E_{2}, G_{1}, G_{2}} \max _{D_{1}, D_{2}} \mathcal{L}\left(E_{1}, E_{2}, G_{1}, G_{2}, D_{1}, D_{2}\right)=\mathcal{L}_{\mathrm{GAN}}^{x_{1}}+\mathcal{L}_{\mathrm{GAN}}^{x_{2}}+} \\ {\quad \lambda_{x}\left(\mathcal{L}_{\mathrm{recon}}^{x_{1}}+\mathcal{L}_{\mathrm{recon}}^{x_{2}}\right)+\lambda_{c}\left(\mathcal{L}_{\mathrm{recon}}^{c_{1}}+\mathcal{L}_{\mathrm{reccon}}^{c_{2}}\right)+\lambda_{s}\left(\mathcal{L}_{\mathrm{recon}}^{s_{1}}+\mathcal{L}_{\mathrm{recon}}^{s_{2}}\right)}\end{array}
$$
where  λx ,  λc ,  λs   are weights  that  control the  importance  of reconstruction terms. 



## Theoretical   Analysis

![image-20191124194557308](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-24-114557.png)

![image-20191124194605682](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-24-114606.png)

![image-20191124194625306](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-24-114625.png)

Style-augmented Cycle Consistency:

![image-20191124194654337](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-24-114654.png)
$$
\mathcal{L}_{\mathrm{cc}}^{x_{1}}=\mathbb{E}_{x_{1} \sim p\left(x_{1}\right), s_{2} \sim q\left(s_{2}\right)}\left[\left\|G_{1}\left(E_{2}^{c}\left(G_{2}\left(E_{1}^{c}\left(x_{1}\right), s_{2}\right)\right), E_{1}^{s}\left(x_{1}\right)\right)-x_{1}\right\|_{1}\right]
$$


## Evaluation   Metrics

![image-20191124195042444](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-24-115042.png)

IS我不太理解为什么用这个指标...

![image-20191124195132820](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-24-115133.png)

## Domain-invariant  Perceptual   Loss

We conduct an experiment to verify if applying IN before computing the feature distance can indeed make the distance more domain-invariant. We experiment on the day ↔ dataset used by Isola et al. [6] and originally proposed by Laﬀont et al. [4]. We randomly sample two sets of image pairs: 1) images from the same domain (both day or both night) but diﬀerent scenes, 2) images from the same scene but diﬀerent domains. Fig. 11 shows examples from the two sets of image pairs. We then compute the VGG feature (relu4 3) distance between each image pair, with IN either applied or not before computing the distance. In Fig. 12, we show histograms of the distance computed either with or without IN, and from image pairs either of the same domain or the same scene. Without applying IN before computing the distance, the distribution of feature distance is similar for both sets of image pairs. With IN enabled, however, image pairs from the same scene have clearly smaller distance, even they come from diﬀerent domains. The results suggest that applying IN before computing the distance makes the feature distance much more domain-invariant.

![image-20191124195320389](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-24-115320.png)

