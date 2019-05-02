# One Day One GAN

Hi ,my name is Chen Yang ,I am a sophomore in ocean university of China .I do some scientific research in my spare time. Based on the current hot direction of artificial intelligence, I hope to share my research progress with **Generative adversarial network**

嗨，我的名字是陈扬，我是中国海洋大学的二年级学生。我在业余时间做了一些科学研究。基于当前人工智能的热点方向，我希望与**生成对抗网络分享我的研究进展**

## 前言

**ODOG**,顾名思义就我我希望能每天抽出一个小时的时间来讲讲到目前为止,GAN的前沿发展和研究,笔者观察了很多深度学习的应用,特别是在图像这一方面,GAN已经在扮演着越来越重要的角色,我们经常可以看到老黄的NVIDIA做了各种各样的application,而且其中涉及到了大量GAN的理论及其实现,再者笔者个人也觉得目前国内缺少GAN在pytorch,keras,tensorflow等主流的框架下的实现教学.

我的老师曾经对我说过:"**深度学习是一块未知的新大陆,它是一个大的黑箱系统,而GAN则是黑箱中的黑箱,谁要是能打开这个盒子,将会引领一个新的时代**"

## cycleGAN

大名鼎鼎的 cycleGAN 就不用我多介绍了吧,那个大家每一次一讲到深度学习,总有人会放出来的那张图片,马变斑马的那个,就是出自于大名鼎鼎的 cycleGAN 了

![horse2zebra](./cyclegan/horse2zebra.gif)

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

最后,判别器还是和寻常的一样,分别判别 A 和 B 是真是假

好了,正经的来看一下损失函数:

![img](http://media.paperweekly.site/LUOHAO_1513259309.1659012.png?imageView2/2/w/800/q/70|imageslim)

X→Y的判别器损失为，字母换了一下，和上面的单向GAN是一样的： 
$$
L_{G A N}\left(G, D_{Y}, X, Y\right)=\mathbb{E} y \sim p_{d a t a}(y)\left[\log D_{Y}(y)\right]+\mathbb{E} x \sim p_{d a t a}(x)\left[\log \left(1-D_{Y}(G(x))\right)\right]
$$
同理Y→X的判别器损失为 :
$$
L_{G A N}\left(G, D_{X}, X, Y\right)=\mathbb{E} x \sim p_{d a t a}(x)\left[\log D_{X}(x)\right]+\mathbb{E} y \sim p_{d a t a}(y)\left[\log \left(1-D_{X}(F(y))\right)\right]
$$
再循环回来和自己比一轮就是:
$$
\begin{aligned} \mathcal{L}_{\mathrm{cyc}}(G, F) &=\mathbb{E}_{x \sim p_{\text { tata }}(x)}\left[\|F(G(x))-x\|_{1}\right] \\ &+\mathbb{E}_{y \sim p_{\text { data }}(y)}\left[\|G(F(y))-y\|_{1}\right] \end{aligned}
$$
全部合在一起就是:
$$
\begin{aligned} \mathcal{L}\left(G, F, D_{X}, D_{Y}\right) &=\mathcal{L}_{\text { GAN }}\left(G, D_{Y}, X, Y\right) \\ &+\mathcal{L}_{\text { GAN }}\left(F, D_{X}, Y, X\right) \\ &+\lambda \mathcal{L}_{\text { cyc }}(G, F) \end{aligned}
$$
那么我们的目标就是:
$$
G^{*}, F^{*}=\arg \min _{G, F} \max _{D_{x}, D_{Y}} \mathcal{L}\left(G, F, D_{X}, D_{Y}\right)
$$

### 代码实现

我选了[eriklindernoren](https://github.com/eriklindernoren)大佬的代码来讲,因为简单…...

先看看

#### 生成器 G :

```python

##############################
#           RESNET
##############################


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

```

emmmmm,我觉得已经说的很清楚了吧,就是一个普普通通的 resnet 啦

就不可视化出来了,免得说我水字数…..

#### 判别器:

```python

##############################
#        Discriminator
##############################

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)
```

这个其实是有点意思的,这个叫 PatchGAN,就是说判别器 D 的输出不是简单的 0/1,而是输出一张比原来输入size小 16 倍的通道数为 1 的特征图,特征图的每个点,1 表示real,0 表示 fake.他的感受野计算你们就别算了,我告诉你最后一层是 70.

### 再来看看训练

首先有三个损失函数:

```python
# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()
```

```python
# Set model input
real_A = Variable(batch["A"].type(Tensor))
real_B = Variable(batch["B"].type(Tensor))

# Adversarial ground truths
valid = Variable(Tensor(np.ones((real_A.size(0), *D_A.output_shape))), requires_grad=False)
fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A.output_shape))), requires_grad=False)
```

这个 label 就是对应了 PatchGAN 的输出,有点意思

#### 训练生成器

```python
# ------------------
#  Train Generators
# ------------------

G_AB.train()
G_BA.train()

optimizer_G.zero_grad()

# Identity loss
loss_id_A = criterion_identity(G_BA(real_A), real_A)
loss_id_B = criterion_identity(G_AB(real_B), real_B)

loss_identity = (loss_id_A + loss_id_B) / 2

# GAN loss
fake_B = G_AB(real_A)
loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
fake_A = G_BA(real_B)
loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

# Cycle loss
recov_A = G_BA(fake_B)
loss_cycle_A = criterion_cycle(recov_A, real_A)
recov_B = G_AB(fake_A)
loss_cycle_B = criterion_cycle(recov_B, real_B)

loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

# Total loss
loss_G = loss_GAN + opt.lambda_cyc * loss_cycle + opt.lambda_id * loss_identity
loss_G.backward()
optimizer_G.step()

```

这个训练生成器比较复杂,我们得先把他拆解成 3 个部分:

1. identity_loss:是指我把 A 集合中的图片过生成器 $G_{BA}$后还是 A,意思是 A 就是 A,B 就是 B(相当于证明你妈是你妈,没有那个意思哈)
2. GAN_loss:经典的以假乱真 loss了,就是希望判别器把生成的图片以为是ground truth.
3. Cycle_loss:循环一致性损失,就是说,我希望斑马变成野马后,再从野马变成斑马还是原来那样的斑马(其实从信息论的角度来看这是不可能的,毕竟信息流动了就会损失,就好比你曾经爱上了她,后来你俩分手了,即便是再后来你们和好了,也不可能再回到当初那个"人生若只如初见,何事西风悲画扇"的状态了)

好了,生成器也就这么回事吧.

再来看看判别器:

```python
# -----------------------
#  Train Discriminator A
# -----------------------

optimizer_D_A.zero_grad()

# Real loss
loss_real = criterion_GAN(D_A(real_A), valid)
# Fake loss (on batch of previously generated samples)
fake_A_ = fake_A_buffer.push_and_pop(fake_A)
loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
# Total loss
loss_D_A = (loss_real + loss_fake) / 2

loss_D_A.backward()
optimizer_D_A.step()

# -----------------------
#  Train Discriminator B
# -----------------------

optimizer_D_B.zero_grad()

# Real loss
loss_real = criterion_GAN(D_B(real_B), valid)
# Fake loss (on batch of previously generated samples)
fake_B_ = fake_B_buffer.push_and_pop(fake_B)
loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
# Total loss
loss_D_B = (loss_real + loss_fake) / 2

loss_D_B.backward()
optimizer_D_B.step()

loss_D = (loss_D_A + loss_D_B) / 2
```

也是有两个判别器,分别判别 A,B 是真是假.

如何我再聊聊代码这么跑起来吧

你先 clone 我上传上去的代码,我改了一些,

额外帮你们写了一个 test.py

```shell
cd data/
sh download_cyclegan_dataset.sh monet2photo
cd ..
python cyclegan.py

after long long time............

python test --checkpiont saved_models/monet2photo/G_BA_100.pth --image_path 你的图片路径

```

这是一篇很有灵魂的文章,在文章最后我想说,图像翻译的坑很深,训练也是真的很难,不过真的挺有意思的,很难得写这么有灵魂的文章了.

来一首[董小姐](https://y.qq.com/portal/player.html)吧

