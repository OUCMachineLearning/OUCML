**导读**

> 图像增强是CV领域非常常用的技术，这里找到一个非常好用的图像增强的工具，可以用于Pytorch和Keras，而且功能强大，使用简单，更重要的是可以成对的进行图像增强，简直是实战利器，有了这个，妈妈再也不用担心我的数据不够了。

Augmentor是一个Python的图像增强库。这是一个独立的库，不依赖与某个平台或某个框架，非常的方便，可以进行细粒度的增强控制，而且实现了大部分的增强技术。使用了随机的方法来构建基础的模块，用户可以把这些模块组成pipline使用。

## **安装**

Augmentor是Python写的。还有一个Julia的版本，链接：[https://github.com/Evizero/Augmentor.jl](https://link.zhihu.com/?target=https%3A//github.com/Evizero/Augmentor.jl)

使用pip安装：

```text
pip install Augmentor
```

从源码安装的话，请看编译文档。升级版本的话：

```text
pip install Augmentor --upgrade
```

## **文档**

完整的文档链接: [http://augmentor.readthedocs.io](https://link.zhihu.com/?target=http%3A//augmentor.readthedocs.io)

## **快速指南和使用**

*Augmentor*的目的是进行自动的图像增强（生成人造数据）为了扩展数据集作为机器学习算法的输入，特别是神经网络和深度学习。

这个包通过创建一个增强的pipeline，即定义一系列的操作。这些操作有比如旋转和变换，一个加一个成为一个增强的pipeline，当完成的时候，pipeline可以执行，增强之后的数据也创建成功。

开始时，需要初始化pipeline对象，指向一个文件夹。

```python
import Augmentor
p = Augmentor.Pipeline("/path/to/images")
```

然后可以在pipeline对象中添加操作：

```python
p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
```

每个函数需要制定一个概率，用来决定是否需要对这个图像进行这个操作。

一旦你创建了pipeline，可以从中进行采样，就像这样：

```python
p.sample(10000)
```

这样会产生10000个增强之后的图像。默认会写到指定文件夹中的名为output的目录中，这个指定文件夹就是初始化时指定的那个。

如果你想进行一次图像的增强操作，可以使用process():

```python
p.process()
```

这个函数在进行数据集缩放的时候会有用。可以创建一个pipeline，其中所有的操作的概率都设置为1，然后使用process()方法。

### **多线程**

Augmentor (version >=0.2.1) 现在使用多线程技术来提高速度。

对于原始图像非常小的图像来说，某些pipeline可能会变慢。如果发现这种情况，可以设置multi_threaded为False。

```python
p.sample(100, multi_threaded=False)
```

默认的情况下，sample()函数是使用多线程的。这个只在保存到磁盘的时候实现。生成器也会在下个版本使用多线程。

### **Ground Truth数据**

图像可以两个一组的通过pipeline，所以ground truth的图像可以同等的进行增强。

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-10-07-101011.jpg)

为了并行的对原始数据进行ground truth的增强，可以使用ground_truth()方法增加一个ground truth的文件夹到pipeline中：

```python
p = Augmentor.Pipeline("/path/to/images")
# Point to a directory containing ground truth data.
# Images with the same file names will be added as ground truth data
# and augmented in parallel to the original data.
p.ground_truth("/path/to/ground_truth_images")
# Add operations to the pipeline as normal:
p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
p.flip_left_right(probability=0.5)
p.zoom_random(probability=0.5, percentage_area=0.8)
p.flip_top_bottom(probability=0.5)
p.sample(50)
```

### **多掩模/图像增强**

使用DataPipeline类 (Augmentor version >= 0.2.3)，可以对有多个相关的掩模的图像进行增强：

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-10-07-101017.jpg)

任意长度的图像列表都可以成组的通过pipeline，并且使用DataPipeline类同样的进行增强。这个对于ground truth图像有好几个掩模的时候非常有用。举个例子。

下面的例子中，图像和掩模包含在一个images的数据结构中，对应的标签在y中：

```python
p = Augmentor.DataPipeline(images, y)
p.rotate(1, max_left_rotation=5, max_right_rotation=5)
p.flip_top_bottom(0.5)
p.zoom_random(1, percentage_area=0.5)

augmented_images, labels = p.sample(100)
```

DataPipeline直接返回图像，并不存储在磁盘中，也不从磁盘中读取数据。图像通过初始化直接传到DataPipeline中。images的数据结构的创建细节，可以参考[https://github.com/mdbloice/Augmentor/blob/master/notebooks/Multiple-Mask-Augmentation.ipynb](https://link.zhihu.com/?target=https%3A//github.com/mdbloice/Augmentor/blob/master/notebooks/Multiple-Mask-Augmentation.ipynb)。

### **Keras和Pytorch的生成器**

如果你不想将图像存储到硬盘中，可以使用生成器，generator，使用Keras的情况：

```python
g = p.keras_generator(batch_size=128)
images, labels = next(g)
```

返回的图像的batchsize是128，还有对应的labels。Generator返回的数据是不确定的，可以用来在线生成增强的数据，用在训练神经网络中。

同样的，你可以使用Pytorch：

```text
import torchvision
transforms = torchvision.transforms.Compose([
   p.torch_transform(),
   torchvision.transforms.ToTensor(),
])
```

## **主要功能**

### **弹性畸变**

使用弹性畸变，一张图像可以生成许多图像。

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-10-07-101021.jpg)

这个输入图像有一个像素宽的黑边，表明了在进行畸变的时候，没有改变尺寸，也没有在新的图像上进行任何的padding。

具体的功能可以在这里看到:

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-10-07-101012.jpg)

### **透视变换**

总共有12个不同类型的透视变换。4中最常用的如下：

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-10-07-101022.jpg)

剩下的8种透视变换:

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-10-07-101018.jpg)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-10-07-101013.jpg)

### **保持大小的旋转**

默认保持原始文件大小的旋转：

![img](https://pic2.zhimg.com/v2-7bd0fbbed067fc6d9a653ced46bf9705_b.jpg)

对比其他软件的旋转:

![img](https://pic2.zhimg.com/v2-7bd0fbbed067fc6d9a653ced46bf9705_b.jpg)

### **保持大小的剪切**

剪切的同时也会自动从剪切图像中裁剪正确的区域，所以图像中没有黑的区域或者padding。

![img](https://pic2.zhimg.com/v2-7bd0fbbed067fc6d9a653ced46bf9705_b.jpg)

对比普通的剪切操作：

![img](https://pic2.zhimg.com/v2-7bd0fbbed067fc6d9a653ced46bf9705_b.jpg)

### **裁剪**

裁剪同样也使用了一种更加适合机器学习的方法：

![img](https://pic2.zhimg.com/v2-7bd0fbbed067fc6d9a653ced46bf9705_b.jpg)

### **随机擦除**

随机擦除是一种使模型对遮挡更加鲁棒的技术。这个对使用神经网络训练物体检测的时候非常有用：

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-10-07-101015.jpg)

看 Pipeline.random_erasing() 文档了解更多的用法。

### **把操作串成Pipeline**

使用几个操作，单个图像可以增强成许多的新图像，对应同样的label：

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-10-07-101014.jpg)

在上面的例子中，我们使用了3个操作：首先做了畸变操作，然后进行了左右的镜像，概率为0.5，最后以0.5的概率做了上下的翻转。然后从这个pipeline中采样了100次，得到了100个数据。

```python
p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)
p.flip_left_right(probability=0.5)
p.flip_top_bottom(probability=0.5)
p.sample(100)
```

## **指南**

### **使用生成器和Keras集成**

Augmentor 可以用来替换Keras中的augmentation功能。Augmentor 可以创建一个生产器来产生增强后的图像，细节可以查看下面的notebook：

- 从本地文件夹中读取图像进行增强，然后使用生成器将增强的图像流送到卷积神经网络中，参见 [https://github.com/mdbloice/Augmentor/blob/master/notebooks/Augmentor_Keras.ipynb](https://link.zhihu.com/?target=https%3A//github.com/mdbloice/Augmentor/blob/master/notebooks/Augmentor_Keras.ipynb)
- 增强内存中的图像，使用生成器将新的图像送到Keras的网络中，参见 [https://github.com/mdbloice/Augmentor/blob/master/notebooks/Augmentor_Keras_Array_Data.ipynb](https://link.zhihu.com/?target=https%3A//github.com/mdbloice/Augmentor/blob/master/notebooks/Augmentor_Keras_Array_Data.ipynb)

Augmentor 允许每个类定义不同的pipelines，这意味着你可以在分类问题中为不同的类别定义不同的增强策略。

例子在这里：[https://github.com/mdbloice/Augmentor/blob/master/notebooks/Per_Class_Augmentation_Strategy.ipynb](https://link.zhihu.com/?target=https%3A//github.com/mdbloice/Augmentor/blob/master/notebooks/Per_Class_Augmentation_Strategy.ipynb)

## **完整的例子**

我们可以使用一张图像来完成一个增强的任务，演示一下Augmentor的pipeline和一些功能。

首先，导入包，初始化Pipeline对象，指定一个文件夹，这个文件夹里放着你的图像。

```python
import Augmentor

p = Augmentor.Pipeline("/home/user/augmentor_data_tests")
```

然后你可以在pipeline中添加各种操作：

```text
p.rotate90(probability=0.5)
p.rotate270(probability=0.5)
p.flip_left_right(probability=0.8)
p.flip_top_bottom(probability=0.3)
p.crop_random(probability=1, percentage_area=0.5)
p.resize(probability=1.0, width=120, height=120)
```

操作添加完了之后，可以进行采样：

```text
p.sample(100)
```

其中的几个

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-10-07-101024.jpg)

增强的图像对边缘检测任务也许很有用