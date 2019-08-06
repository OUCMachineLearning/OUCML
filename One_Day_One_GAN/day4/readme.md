# One Day One GAN

Hi ,my name is Chen Yang ,I am a sophomore in ocean university of China .I do some scientific research in my spare time. Based on the current hot direction of artificial intelligence, I hope to share my research progress with **Generative adversarial network**

嗨，我的名字是陈扬，我是中国海洋大学的二年级学生。我在业余时间做了一些科学研究。基于当前人工智能的热点方向，我希望与**生成对抗网络分享我的研究进展**

## 前言

**ODOG**,顾名思义就我我希望能每天抽出一个小时的时间来讲讲到目前为止,GAN的前沿发展和研究,笔者观察了很多深度学习的应用,特别是在图像这一方面,GAN已经在扮演着越来越重要的角色,我们经常可以看到老黄的NVIDIA做了各种各样的application,而且其中涉及到了大量GAN的理论及其实现,再者笔者个人也觉得目前国内缺少GAN在pytorch,keras,tensorflow等主流的框架下的实现教学.

我的老师曾经对我说过:"**深度学习是一块未知的新大陆,它是一个大的黑箱系统,而GAN则是黑箱中的黑箱,谁要是能打开这个盒子,将会引领一个新的时代**"

### ACGAN

ACGAN的全称叫Auxiliary Classifier Generative Adversarial Network,翻译成汉语的意思就是带辅助分类器的GAN,其实他的思想和昨天说到的CGAN很想,也是利用label的信息作为噪声的输入的条件概率,但是相比较于CGAN,ACGAN在设计上更为巧妙,他很好地利用了判别器使得不但可以判别真假,也可以判别类别,通过对生成图像类别的判断,判别器可以更好地传递loss函数使得生成器能够更加准确地找到label对应的噪声分布,通过下图告诉了我们ACGAN与CGAN的异同之处.

![image-20190330011810374](https://ws4.sinaimg.cn/large/006tKfTcly1g1k66mb278j31980k0jye.jpg)

为此,我们先回顾一下我们昨天讲的CGAN的loss函数:

![image-20190330012259653](https://ws4.sinaimg.cn/large/006tKfTcly1g1k6bmwocuj31e806qaco.jpg)

那么在对比一下我们ACGAN他的loss函数:

![image-20190330012347134](https://ws3.sinaimg.cn/large/006tKfTcly1g1k6cgeiyij30xe06egnv.jpg)

$L_S$表示的是真实样本对应ground truth,假的样本对应fake.

$L_c$表示的是真实样本对应他真实的类别信息,假的样本对应的也是真实样本的类别信息

**在训练判别器的时候,我们希望$L_S+L_C$最大化**

**在训练生成器的时候,我们希望$L_C-L_S$最大化**

当然了,也有人是从JS散度优化的角度来看待这个问题的,我不想把问题将得太复杂,有兴趣的同学可以瞄一瞄:

![image-20190330013253663](https://ws3.sinaimg.cn/large/006tKfTcly1g1k6lxsfuzj31ao0t27ec.jpg)

### 代码实现

其实和之前的CGAN,只是做了很微小的改变,生成器和CGAN相比,就是加入了卷积层,相当于把原来CGAN里面的多层感知机换成了DCGAN里面一样的深度卷积神经网络,那么判别器同理.

#### 生成器

```python
def build_generator(self):

    model = Sequential()

    model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
    model.add(Reshape((7, 7, 128)))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(self.channels, kernel_size=3, padding='same'))
    model.add(Activation("tanh"))

    model.summary()

    noise = Input(shape=(self.latent_dim,))
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(self.num_classes, 100)(label))

    model_input = multiply([noise, label_embedding])
    img = model(model_input)

    return Model([noise, label], img)
```

#### 判别器

判别器倒是有些有趣,他在卷积层的末尾相当于做了一个分叉,一边是判断真假,一边是还得判断类别.

```python
def build_discriminator(self):

    model = Sequential()

    model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.summary()

    img = Input(shape=self.img_shape)

    # Extract feature representation
    features = model(img)

    # Determine validity and label of the image
    validity = Dense(1, activation="sigmoid")(features)
    label = Dense(self.num_classes+1, activation="softmax")(features)

    return Model(img, [validity, label])
#所以判别器的输出有两个,一个是判断真假的validity,一个图片对应的label信息


# 对应的,他的loss函数定义也挺有意思的,不得不说keras是真的方便😊 
losses = ['binary_crossentropy', 'sparse_categorical_crossentropy']
self.discriminator = self.build_discriminator()
self.discriminator.compile(loss=losses,
    optimizer=optimizer,
    metrics=['accuracy'])

self.combined = Model([noise, label], [valid, target_label])
self.combined.compile(loss=losses,
    optimizer=optimizer)

```

#### 训练细节

```python
# ---------------------
#  训练判别器
# ---------------------

# 随机选择batch个图片
idx = np.random.randint(0, X_train.shape[0], batch_size)
imgs = X_train[idx]

# 生成高斯噪声noise
noise = np.random.normal(0, 1, (batch_size, 100))

# 生成随机标签
# image representation of
sampled_labels = np.random.randint(0, 10, (batch_size, 1))

# 应用随机标签和noise生成图片
gen_imgs = self.generator.predict([noise, sampled_labels])

# 判别器要判别10(0~9)+1(fake)=11种类别
img_labels = y_train[idx]
fake_labels = 10 * np.ones(img_labels.shape)

# 训练判别器
d_loss_real = self.discriminator.train_on_batch(imgs, [valid, img_labels])
d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, fake_labels])
d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

# ---------------------
#  训练生成器
# ---------------------

# 训练生成器
g_loss = self.combined.train_on_batch([noise, sampled_labels], [valid, sampled_labels])

```

### 实验结果

![image-20190330015005462](https://ws4.sinaimg.cn/large/006tKfTcly1g1k73wcgkij313g0u01l4.jpg)

作者从1000类的imageNet数据集上挑选了10种类别的图像,并在不同的尺度上做了实验,我们看得出来,在彩色图像上的生成结果还是马马虎虎不错的,不过看起来还是显得图像没什么协调感,在之后的GAN中,我们会讲到如何利用GAN生成高质量的图像.看到这里了要是看官老爷觉得interesting,不如麻烦关注转发好看三连一波,你的每一点小小的鼓励都是对作者莫大的鼓舞鸭😄.

### 参考

<https://zhuanlan.zhihu.com/p/44177576>

<http://ruishu.io/2017/12/26/acgan/>

<https://www.cnblogs.com/punkcure/p/7873566.html>

<https://github.com/eriklindernoren/Keras-GAN/blob/master/acgan/acgan.py>

