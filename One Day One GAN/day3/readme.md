# One Day One GAN

Hi ,my name is Chen Yang ,I am a sophomore in ocean university of China .I do some scientific research in my spare time. Based on the current hot direction of artificial intelligence, I hope to share my research progress with **Generative adversarial network**

嗨，我的名字是陈扬，我是中国海洋大学的二年级学生。我在业余时间做了一些科学研究。基于当前人工智能的热点方向，我希望与**生成对抗网络分享我的研究进展**

## 前言

**ODOG**,顾名思义就我我希望能每天抽出一个小时的时间来讲讲到目前为止,GAN的前沿发展和研究,笔者观察了很多深度学习的应用,特别是在图像这一方面,GAN已经在扮演着越来越重要的角色,我们经常可以看到老黄的NVIDIA做了各种各样的application,而且其中涉及到了大量GAN的理论及其实现,再者笔者个人也觉得目前国内缺少GAN在pytorch,keras,tensorflow等主流的框架下的实现教学.

我的老师曾经对我说过:"**深度学习是一块未知的新大陆,它是一个大的黑箱系统,而GAN则是黑箱中的黑箱,谁要是能打开这个盒子,将会引领一个新的时代**"

## CGAN

CGAN的全称叫**Conditional Generative Adversarial Nets**,condition的意思是就是条件,我们其实可以理解成概率统计里一个很基本的概念叫做条件概率分布.举个例子:

假设在桌子上抛掷一枚普通的骰子，则其点数结果的概率分布是集合  \{1,2,3,4,5,6\}的均匀分布：每个点数出现的概率都是均等的六分之一。然而，如果据某个坐在桌边的人观察，向着他的侧面是6点，那么，在此条件下，向上的一面不可能是6点，也不可能是6点对面的1点。因此，在此条件下，抛骰子的点数结果是集合 \{2,3,4,5\}的均匀分布：有四分之一的可能性出现 2,3,4,5四种点数中的一种。可以看出，增加的条件或信息量（某个侧面是6点）导致了点数结果的概率分布的变化。这个新的概率分布就是条件概率分布。

那么回过头来看原始的GAN的噪声分布和ground truth的分布构成的核心公式:

![image-20190330002722695](https://ws1.sinaimg.cn/large/006tKfTcly1g1k4prcvzdj312i06m3za.jpg)

如果我们已知输入的ground truth的label信息,那么我们便可以在这个基础上结合条件概率的公式得到CGAN的目标函数:

![image-20190330001423117](https://ws1.sinaimg.cn/large/006tKfTcly1g1k4c8vwu6j31fa0a2q5c.jpg)

如下图所示，通过将额外信息y输送给判别模型和生成模型,作为输入层的一部分,从而实现条件GAN。在生成模型中,先验输入噪声p(z)和条件信息y联合组成了联合隐层表征。对抗训练框架在隐层表征的组成方式方面相当地灵活。类似地，条件GAN的目标函数是带有条件概率的二人极小极大值博弈（two-player minimax game ）： 

![image-20190330001521830](https://ws3.sinaimg.cn/large/006tKfTcly1g1k4d9g482j30u00uk0xq.jpg)

## 核心代码

### 生成器

```python
    def build_generator(self):

        model = Sequential()
#一个由多层感知机构成的生成网络
        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()
#生成器的输入有两个,一个是高斯噪声noise,一个是由我希望生成的图片的label信息,通过embedding的方法把label调整到和噪声相同的维度,在乘起来这样便使得noise的输入是建立在label作为条件的基础上
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))

        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)
```



### 判别器:

```python
    def build_discriminator(self):

        model = Sequential()
#一个多层感知机的判别网络
        model.add(Dense(512, input_dim=np.prod(self.img_shape)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        label = Input(shape=(1,), dtype='int32')
# 我们把label通过embedding扩展到和image一样的维度 
        label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape))(label))
        flat_img = Flatten()(img)
#判别器的输入包含了图片信息和起对应的标签,我们的判别器不但要判别是否真假,还需要判别是不是图片符合对应的类别信息

        model_input = multiply([flat_img, label_embedding])

        validity = model(model_input)

        return Model([img, label], validity)
```

### 训练细节

```python
# 选择要batch个训练的图片和label
idx = np.random.randint(0, X_train.shape[0], batch_size)
imgs, labels = X_train[idx], y_train[idx]

# 生成100维的高斯噪声
noise = np.random.normal(0, 1, (batch_size, 100))

# 生成器根据label和noise生成图片
gen_imgs = self.generator.predict([noise, labels])

# 训练判别器
d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

# ---------------------
#  生成器训练部分
# ---------------------

# 生成随机的标签
sampled_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)

# 训练生成器
g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)

```

### 实验结果

![image-20190330004627555](https://ws1.sinaimg.cn/large/006tKfTcly1g1k59mdqtjj30we0hg7b4.jpg)

我们可以清晰的看出来,生成器实际上是可以根据我们提供的label信息很好的生成对应的图像,虽然CGAN用的是很原始的多层感知机,但是我们仍然可以清晰的看出来,这个多层感知机实际上是可以生成所有label对应的分布的图像的,而且作为一种监督学习,condition GAN告诉我们了,要想GAN训练成功,condition是很重要的,具体在以后会说为什么,看到这里了要是看官老爷觉得interesting,不如麻烦关注转发好看三连一波,你的每一点小小的鼓励都是对作者莫大的鼓舞鸭😄.

### 参考

<https://github.com/eriklindernoren/Keras-GAN/blob/master/cgan/cgan.py>

<https://blog.csdn.net/u011534057/article/details/53409968>