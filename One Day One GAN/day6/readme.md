# One Day One GAN

Hi ,my name is Chen Yang ,I am a sophomore in ocean university of China .I do some scientific research in my spare time. Based on the current hot direction of artificial intelligence, I hope to share my research progress with **Generative adversarial network**

嗨，我的名字是陈扬，我是中国海洋大学的二年级学生。我在业余时间做了一些科学研究。基于当前人工智能的热点方向，我希望与**生成对抗网络分享我的研究进展**

## 前言

**ODOG**,顾名思义就我我希望能每天抽出一个小时的时间来讲讲到目前为止,GAN的前沿发展和研究,笔者观察了很多深度学习的应用,特别是在图像这一方面,GAN已经在扮演着越来越重要的角色,我们经常可以看到老黄的NVIDIA做了各种各样的application,而且其中涉及到了大量GAN的理论及其实现,再者笔者个人也觉得目前国内缺少GAN在pytorch,keras,tensorflow等主流的框架下的实现教学.

我的老师曾经对我说过:"**深度学习是一块未知的新大陆,它是一个大的黑箱系统,而GAN则是黑箱中的黑箱,谁要是能打开这个盒子,将会引领一个新的时代**"

## SRGAN

SRGAN 是在 GAN 在 图像超分辨率复原领域的 baseline,作为一个 baseline,我不想在文章中过多的聊比如说什么是超分辨率复原的前世今生,因为实在是可以讲太多了,有兴趣的朋友我建议你直接在我的 github 上 fellow 我即可,或者说电邮我.

在这里我想简要的介绍一下 GAN 是如何实现超分辨率的复原的,还有就是如何使用目前我们研究的时候经常使用的 celebA 数据集.

### 数据集

CeleA是香港中文大学的开放数据，包含10177个名人身份的202599张图片，并且都做好了特征标记，这对人脸相关的训练是非常好用的数据集。[网盘链接](https://pan.baidu.com/s/1eSNpdRG#list/path=%2FCelebA)

img文件夹下有两个压缩包

img_align_celeba.zip & img_align_celeba_png.7z

我选择下载的是

img_align_celeba.zip

解压后的内容是包含202599张图片

Anno文件夹下有个文档identity_CelebA，部分内容如下：

```
000001.jpg 2880
000002.jpg 2937
000003.jpg 8692
000004.jpg 5805
000005.jpg 9295
000006.jpg 4153
000007.jpg 9040
000008.jpg 6369
000009.jpg 3332
000010.jpg 612
```

此文档是10,177个名人身份标识，每张图片后面的数字即是该图片对应的标签；

更多介绍看<https://zhuanlan.zhihu.com/p/35975956>

### Super-Resolution IMAGE

简单点说,就是给你一张模糊的图片,让你复原一张高清的图片.
![image_1ctqd73lo1vkka11cgu1kpt1qd316.png-907kB][8]

------



------

### 我们如何用生成对抗网络来做呢?

这个时候,我们可以把$LR_{img}$看成是一个噪声$z$的输入,G生成的是一个$Fake_{HR-img} $,我们让D分辨$fake_{HR-img}$ && $original_{HR-img}$.

------

### 定义一个目标函数

Our ultimate goal is to train a generating function G that estimates for a given LR input image its corresponding HR counterpart. To achieve this, we train a generator network as a feed-forward CNN GθG parametrized by θG. Here θG = {W1:L ; b1:L } denotes the weights and biases of a L-layer deep network and is obtained by optimizing a SR-specific
loss function lSR. For training images IHR , n = 1, . . . , N n
withcorrespondingILR,n=1,...,N,wesolve:
![image_1ctqe1ovcrng10u110bidd9q1r20.png-7.1kB][9]

------

## 提出perceptual loss  

作者认为这更接近人的主观感受，因为使用pixel-wise的MSE使得图像变得平滑，而如果先用VGG来抓取到高级特征（feature）表示，再对feature使用MSE，可以更好的抓取不变特征。
![image_1ctqe4ifno4g16hkiuo15114f37.png-170kB][10]

------

![lovemusicge_1505531635.8098323.JPG-27.9kB][11]

------

![lovemusicge_1505531647.0589468.JPG-31.4kB][12]

------

![lovemusicge_1505531655.8268762.JPG-25.3kB][13]

------

### GAN核心公式

![image_1ctqdkcva82dabec4k1abboe51j.png-8.6kB][14]

------

### 网络设计



------

### loss函数

```python
###vgg用于提取特征
self.vgg.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
###生成器
self.combined.compile(loss=['binary_crossentropy', 'mse'],
            loss_weights=[1e-3, 1],
            optimizer=optimizer)
###判别器
self.discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
```

------

### train

#### 训练判别器

```python
d_loss_real = self.discriminator.train_on_batch(imgs_hr, valid)
d_loss_fake = self.discriminator.train_on_batch(fake_hr, fake)
d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

```

#### 训练生成器

```python
image_features = self.vgg.predict(imgs_hr)
# Train the generators
g_loss = self.combined.train_on_batch([imgs_lr, imgs_hr], [valid, image_features])

```

------

### 实际结果

![5000.png-193.8kB][16]
5000 batchsize

------

### 对比实验

![image_1ctqid1qr1rf1agii1l1csf16tt82.png-189.3kB][17]

------

## question and answer, 

Enjoy writing slides! :+1:

[1]: http://static.zybuluo.com/Team/mmbpgzozxzaesexm03rkvt3a/image_1ct4sdm7c171la761q1rf881aak9.png
[2]: http://static.zybuluo.com/Team/lvg8hb8uip1yut64h5n8reh0/image_1ct4si11fu6o2lg1m6ub6110o6m.png
[3]: http://static.zybuluo.com/Team/t1p9ulca7z9y5vyg9yhwucta/image_1ct4sn8kqg8ftika3b1nmj6i1j.png
[4]: http://static.zybuluo.com/Team/i791fay3szt4melmamnt9bb7/image_1ct4ss3bu18gs14fs5af1qgrhf420.png
[5]: http://static.zybuluo.com/Team/9ex5978j1gqod4ffjd2jmiti/image_1ct5064nkclm14jh1o401pdm1v349.png
[6]: http://static.zybuluo.com/Team/rt34igjxwylimw3ps2j4zjs8/image_1ct56911djif1ae81ebq10sf1l3om.png
[7]: http://static.zybuluo.com/Team/5m4idtsufahdg5xiw8m33dcy/image_1ctqctu7u852vooutn17075obp.png
[8]: http://static.zybuluo.com/Team/ywnqa4zufq9izyle2lma1snd/image_1ctqd73lo1vkka11cgu1kpt1qd316.png
[9]: http://static.zybuluo.com/Team/fou5iy5p33g5l0p5m498bbdc/image_1ctqhfu351lqh1sns1kfh196liq878.png
[10]: http://static.zybuluo.com/Team/uwdxl0d5usi8gex42pmzd72e/image_1ctqe4ifno4g16hkiuo15114f37.png
[11]: http://static.zybuluo.com/Team/tsg6nq9mo32swqtzletzr28r/lovemusicge_1505531635.8098323.JPG
[12]: http://static.zybuluo.com/Team/g99s4760ttwjhkhhi4wcmjqo/lovemusicge_1505531647.0589468.JPG
[13]: http://static.zybuluo.com/Team/fq8dgbotm6pi1ci40r51q9q9/lovemusicge_1505531655.8268762.JPG
[14]: http://static.zybuluo.com/Team/xdewpuod7pv4zyd1gesfko0a/image_1ctqhvd2a1eoe1c6cecccmc1ars7l.png

[15]: http://static.zybuluo.com/Team/fdq2ull9hrv6du5o651nbro0/image_1ctqe9ihsrc7cbv75k16nl1oco44.png

[16]: http://static.zybuluo.com/Team/gxtzo4736idwaoyacknmhlj9/5000.png
[17]: http://static.zybuluo.com/Team/pqojy7mxnfmg2h4tnfyx951d/image_1ctqikdcj13h21l1u151c1rab1pra8r.png
[8]: http://static.zybuluo.com/Team/ywnqa4zufq9izyle2lma1snd/image_1ctqd73lo1vkka11cgu1kpt1qd316.png