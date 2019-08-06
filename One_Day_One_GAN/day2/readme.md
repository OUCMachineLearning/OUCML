# One Day One GAN

Hi ,my name is Chen Yang ,I am a sophomore in ocean university of China .I do some scientific research in my spare time. Based on the current hot direction of artificial intelligence, I hope to share my research progress with **Generative adversarial network**

嗨，我的名字是陈扬，我是中国海洋大学的二年级学生。我在业余时间做了一些科学研究。基于当前人工智能的热点方向，我希望与**生成对抗网络分享我的研究进展**

## 前言

**ODOG**,顾名思义就我我希望能每天抽出一个小时的时间来讲讲到目前为止,GAN的前沿发展和研究,笔者观察了很多深度学习的应用,特别是在图像这一方面,GAN已经在扮演着越来越重要的角色,我们经常可以看到老黄的NVIDIA做了各种各样的application,而且其中涉及到了大量GAN的理论及其实现,再者笔者个人也觉得目前国内缺少GAN在pytorch,keras,tensorflow等主流的框架下的实现教学.

我的老师曾经对我说过:"**深度学习是一块未知的新大陆,它是一个大的黑箱系统,而GAN则是黑箱中的黑箱,谁要是能打开这个盒子,将会引领一个新的时代**"

## DCGAN

arxiv:https://arxiv.org/pdf/1511.06434

code:<https://github.com/eriklindernoren/Keras-GAN/blob/master/dcgan/dcgan.py>(简易实现)

DCGAN,全称叫**Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks**,顾名思义.就是在生成器和判别器特征提取层用卷积神经网络代替了原始GAN中的多层感知机

### 代码实现

```python
    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)
```

![DCGAN](https://ws4.sinaimg.cn/large/006tKfTcly1g1i91e489uj31100fc0wj.jpg)

我们可以看到,他用的是卷积+上采样的方法来实现将100维的噪声输入$z$经过多层的卷积和上采样后得到的一张$64*64*3$大小的一张图片,因为作者在15年写这篇文章的时候就已经使用了celebA数据集来生成人脸图像.

![image-20190328093017036](https://ws3.sinaimg.cn/large/006tKfTcly1g1i9631thgj30k50ckds8.jpg)

具体的公式和原始GAN的几乎也一样,涉及到的原理并没有什么太多的变化,仅仅是是把生成器和判别器里面的多层感知机换成了深度卷积网络,就可以在人脸数据集上做生成了,可以看出来原始的GAN他在原理和公式上是很强大的,

### 判别器

```python
def build_discriminator(self):

    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    img = Input(shape=self.img_shape)
    validity = model(img)

    return Model(img, validity)

```

#### 总结

DCGAN几乎是生成对抗网络的入门模板了,不多赘述.

### 参考

<https://blog.csdn.net/liuxiao214/article/details/74502975>

<https://github.com/carpedm20/DCGAN-tensorflow>

<https://github.com/eriklindernoren/Keras-GAN/blob/master/gan/gan.py>

