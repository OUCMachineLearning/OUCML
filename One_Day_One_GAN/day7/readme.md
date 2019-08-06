# ESRGAN

#### 前言

> **超分辨率成像**（Super-resolution imaging，缩写SR），是一种提高影片分辨率的技术。在一些称为“光学SR”的SR技术中，系统的衍射极限被超越；而在其他所谓的“几何SR”中，数位[感光元件](https://zh.wikipedia.org/wiki/%E6%84%9F%E5%85%89%E5%85%83%E4%BB%B6)的分辨率因而提高。超分辨率成像技术用于一般图像处理和超高分辨率显微镜。

> **生成对抗网络**（英语：**G**enerative **A**dversarial **N**etwork，简称GAN）是[非监督式学习](https://zh.wikipedia.org/wiki/%E9%9D%9E%E7%9B%91%E7%9D%A3%E5%BC%8F%E5%AD%A6%E4%B9%A0)的一种方法，通过让两个[神经网络](https://zh.wikipedia.org/wiki/%E4%BA%BA%E5%B7%A5%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C)相互[博弈](https://zh.wikipedia.org/wiki/%E5%8D%9A%E5%BC%88%E8%AE%BA)的方式进行学习。该方法由[伊恩·古德费洛](https://zh.wikipedia.org/wiki/%E4%BC%8A%E6%81%A9%C2%B7%E5%8F%A4%E5%BE%B7%E8%B4%B9%E6%B4%9B)等人于2014年提出。[[1\]](https://zh.wikipedia.org/wiki/%E7%94%9F%E6%88%90%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9C#cite_note-MyUser_Arxiv.org_April_7_2016c-1)
>
> 生成对抗网络由一个[生成网络](https://zh.wikipedia.org/wiki/%E7%94%9F%E6%88%90%E6%A8%A1%E5%9E%8B)与一个[判别网络](https://zh.wikipedia.org/wiki/%E5%88%A4%E5%88%AB%E6%A8%A1%E5%9E%8B)组成。生成网络从潜在空间（latent space）中随机采样作为输入，其输出结果需要尽量模仿训练集中的真实样本。判别网络的输入则为真实样本或生成网络的输出，其目的是将生成网络的输出从真实样本中尽可能分辨出来。而生成网络则要尽可能地欺骗判别网络。两个网络相互对抗、不断调整参数，最终目的是使判别网络无法判断生成网络的输出结果是否真实。[[2\]](https://zh.wikipedia.org/wiki/%E7%94%9F%E6%88%90%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9C#cite_note-2)[[1\]](https://zh.wikipedia.org/wiki/%E7%94%9F%E6%88%90%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9C#cite_note-MyUser_Arxiv.org_April_7_2016c-1)[[3\]](https://zh.wikipedia.org/wiki/%E7%94%9F%E6%88%90%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9C#cite_note-OpenAI_com-3)

关于GAN:我我之前写过一篇:[手把手教你用keras搭建GAN](https://mp.weixin.qq.com/s?__biz=MzUyMjE2MTE0Mw==&mid=2247487223&idx=1&sn=4edd1d39e6cd22799668f766532d82af&chksm=f9d1506fcea6d9797f2eef951962e90a412afa5106c8531493505bb4cfb13d411e44c9c3d4f2&mpshare=1&scene=1&srcid=#rd)

### 直入主 题

ESRGAN的全名叫 **Enhanced Super-Resolution Generative Adversarial Networks**,发表于ECCV2018,它是基于SRGAN改进而来到,相比于SRGAN它在三个方面进行了改进

1. 改进了网络结构,对抗损失,感知损失
2. 引入**Residual-in-Residu Dense Block**(RRDB)
3. 使用激活前的VGG特征来改善感知损失

在开始讲这个ESRGAN的具体实现之前,我先来看一下他和他的前辈SRGAN的对比效果:

![屏幕快照 2019-03-01 上午1.23.31](https://ws1.sinaimg.cn/large/006tKfTcly1g0mnrnehh9j316s0mi7fx.jpg)

![image-20190301013829937](https://ws2.sinaimg.cn/large/006tKfTcly1g0mnstwwsvj317v0jt0x5.jpg)

我们可以从上图看出,

1. ESRGAN在锐度和边缘信息上优于SRGAN,且去除了"伪影"
2. 从PI和PMSE两个指标来看,ESRGAN也可以当之无愧地称得上是超分辨率复原任务中的the State-of-the-Art

### 庖丁解牛

#### RRDB,对residual blocks的改进:

我们可以看出这个残差块是很传统的Conv-BN-relu-Conv-BN的结构,而作者在文章中是这么说到的:

> We empirically observe that BN layers tend to bring artifacts. These artifacts,
> namely BN artifacts, occasionally appear among iterations and different settings,
> violating the needs for a stable performance over training. In this section, we
> present that the network depth, BN position, training dataset and training loss
> have impact on the occurrence of BN artifacts and show corresponding visual

什么意思呢?就是说作者认为SRGAN之所以会产生伪影,就是因为使用了[Batchnormalization](https://en.wikipedia.org/wiki/Batch_normalization),所以作者做出了去除BN的改进

![image-20190301143914007](https://ws1.sinaimg.cn/large/006tKfTcly1g0nad8mwg5j30er088mxp.jpg)

而且我们再来看,SRGAN的残差块是顺序连接的,而作者可能哎,受denseNet的启发,他就把这些残差块用密集连接的方式连在一起.那么他的生成器里的特征提取部分最终变成了这样子:

![image-20190301144350837](https://ws3.sinaimg.cn/large/006tKfTcly1g0nai1nnjzj30ow07wjt9.jpg)

既然我们知道他的网络结构是这样子设计的,那么他的实现其实就很简单:

```python
class DenseResidualBlock(nn.Module):
    """
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, filters, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_features, non_linearity=True):
            layers = [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]
            if non_linearity:
                layers += [nn.LeakyReLU()]
            return nn.Sequential(*layers)

        self.b1 = block(in_features=1 * filters)
        self.b2 = block(in_features=2 * filters)
        self.b3 = block(in_features=3 * filters)
        self.b4 = block(in_features=4 * filters)
        self.b5 = block(in_features=5 * filters, non_linearity=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

    def forward(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)
        return out.mul(self.res_scale) + x


```

先定义了DenseResidualBlock块,我们可以看出来作者在里面移除了 BN 层再把b1-b5 密集连接起来.

```python
class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(filters), DenseResidualBlock(filters), DenseResidualBlock(filters)
        )

    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x

```

再三个DRB 块级联成一个 RDB 块.

```python
class GeneratorRRDB(nn.Module):
    def __init__(self, channels, filters=64, num_res_blocks=16, num_upsample=2):
        super(GeneratorRRDB, self).__init__()

        # First layer
        self.conv1 = nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1)
        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])
        # Second conv layer post residual blocks
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        # Upsampling layers
        upsample_layers = []
        for _ in range(num_upsample):
            upsample_layers += [
                nn.Conv2d(filters, filters * 4, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
                nn.PixelShuffle(upscale_factor=2),
            ]
        self.upsampling = nn.Sequential(*upsample_layers)
        # Final output block
        self.conv3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out


```

生成器再根据num_res_blocks变量控制 RDB 块的数量.

#### 对损失函数的改进

说到损失函数啊,我们之前在SRGAN的文章里我也介绍过,这个判别器它判断的是你输入的图片是"真的"高清图像,还是"假的"高清图像,而且作者他就提出一种新的思考模式,就是说我的判别器是来**估计真实图像相对来说比fake图像更逼真的概率。**

怎么来理解这句话呢?

![image-20190301145004428](https://ws2.sinaimg.cn/large/006tKfTcly1g0naoh6ormj316308qwhs.jpg)

具体而言，作者把标准的判别器换成Relativistic average Discriminator（RaD），所以判别器的损失函数定义为：

![image-20190301151447125](https://ws3.sinaimg.cn/large/006tKfTcly1g0nbe6fbmcj313w02laa9.jpg)

对应的生成器的对抗损失函数为：

![image-20190301151456042](https://ws1.sinaimg.cn/large/006tKfTcly1g0nbeboqc1j3153027mxd.jpg)求MSE的操作是通过对$mini-batch$中的所有数据求平均得到的，$x_f$是原始低分辨图像经过生成器以后的图像,由于对抗的损失包含了$x_r$和$x_f$,所以生成器受益于对抗训练中的生成数据和实际数据的梯度，这种调整会使得网络学习到更尖锐的边缘和更细节的纹理。

看起来这个原理十分的高大上.

```python
pred_real = discriminator(imgs_hr)
pred_fake = discriminator(gen_hr.detach())

# Adversarial loss for real and fake images (relativistic average GAN)
loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

# Total loss
loss_D = (loss_real + loss_fake) / 2


```

其实就加多两行代码的事情….

#### 对感知损失的改进

我们之前看SRGAN的时候看到哎,它是用来一个训练好的VGG16来给出超分辨率复原所需要的特征,作者通过对损失域的研究发现,激活前的特征，这样会克服两个缺点。

> Perceptual
> loss is previously defined on the activation layers of a pre-trained deep network,
> where the distance between two activated features is minimized. Contrary to
> the convention, we propose to use features before the activation layers, which
> will overcome two drawbacks of the original design. 

1. 激活后的特征是非常稀疏的，特别是在很深的网络中。这种稀疏的激活提供的监督效果是很弱的，会造成性能低下；
2. 使用激活后的特征会导致重建图像与GT的亮度不一致。

![image-20190301154741733](https://ws4.sinaimg.cn/large/006tKfTcly1g0nccfnrxxj30z60bcamd.jpg)

与此同时,作者还在loss函数中加入了$L_1=E_{xi}||G(x_i)-y||_1$,也就是$L_1$损失,最终损失函数由三部分组成:



![image-20190301153434204](https://ws2.sinaimg.cn/large/006tKfTcly1g0nbyrii6uj312q02iq30.jpg)

#### 网络插值

为了平衡感知质量和PSNR等评价值，作者提出了一个灵活且有效的方法---网络插值。具体而言，作者首先基于PSNR方法训练的得到的网络G_PSNR，然后再用基于GAN的网络G_GAN进行整合



![image-20190301153806957](https://ws4.sinaimg.cn/large/006tKfTcly1g0nc2g2zduj312m02rmxa.jpg)

![image-20190301154824312](https://ws3.sinaimg.cn/large/006tKfTcly1g0ncd5ua86j30zh0mb4qp.jpg)

它的具体实现就是:

```python
for k, v_PSNR in net_PSNR.items():
    v_ESRGAN = net_ESRGAN[k]
    net_interp[k] = (1 - alpha) * v_PSNR + alpha * v_ESRGAN
```

#### 训练细节

```python
# ----------
#  Training
# ----------

for epoch in range(opt.epoch, opt.n_epochs):
    for i, imgs in enumerate(dataloader):

        batches_done = epoch * len(dataloader) + i

        # Configure model input
        imgs_lr = Variable(imgs["lr"].type(Tensor))
        imgs_hr = Variable(imgs["hr"].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr)

        # Measure pixel-wise loss against ground truth
        loss_pixel = criterion_pixel(gen_hr, imgs_hr)

        if batches_done < opt.warmup_batches:
            # Warm-up (pixel-wise loss only)
            loss_pixel.backward()
            optimizer_G.step()
            print(
                "[Epoch %d/%d] [Batch %d/%d] [G pixel: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), loss_pixel.item())
            )
            continue

        # Extract validity predictions from discriminator
        pred_real = discriminator(imgs_hr).detach()
        pred_fake = discriminator(gen_hr)

        # Adversarial loss (relativistic average GAN)
        loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)

        # Content loss
        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr).detach()
        loss_content = criterion_content(gen_features, real_features)

        # Total generator loss
        loss_G = loss_content + opt.lambda_adv * loss_GAN + opt.lambda_pixel * loss_pixel

        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        pred_real = discriminator(imgs_hr)
        pred_fake = discriminator(gen_hr.detach())

        # Adversarial loss for real and fake images (relativistic average GAN)
        loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
        loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()
```

在训练的过程中,为了使得 G 和 D 能够 平衡训练,我们先引入了一个一个 warmup的过程,在 warmup 的训练过程中,我们首先只是考虑对 G 进行训练,训练的时候只对生成gen_hr 和 img_hr 计算 L1loss,等到生成器能够有一定的学习能力的时候,我们引入 VGGloss 和对抗 loss.在训练判别器的时候,我们把原来判别真假的 loss 改成判别相对真假的 loss.

在实际的实验的过程中,网络大概跑完 5 个 epoch 就可以达到不错的复原效果.

### 总结

实际上 ESRGAN 相对于 SRGAN 还是做出了不少改进的,同时让我刚到眼前一亮的是他在训练的过程中加入了不少训练的技巧,是的生成器和判别器的学习更加的平衡.

不过带来的问题也是存在的,就是 G生成器的网络参数相对于 SRGAN,计算量要多很多,如果想要进一步的运用到工程实践中,我的建议还是要考虑网络的精简.

