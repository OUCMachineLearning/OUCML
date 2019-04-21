pytorch 技巧收集

# 使用 Spatial Pyramid Pooling 让 CNN 接受可变尺寸的图像

<https://oidiotlin.com/sppnet-tutorial/>

### 目录

- [传统 CNN 的弊端](https://oidiotlin.com/sppnet-tutorial/#technical-issue)
- [SPP-Net 概述](https://oidiotlin.com/sppnet-tutorial/#sppnet-overview)
- [SPP-Net 结构细节](https://oidiotlin.com/sppnet-tutorial/#sppnet-architecture)
- [SPP-Net 训练方法](https://oidiotlin.com/sppnet-tutorial/#sppnet-training)
- [在 pytorch 框架中实现 SPP](https://oidiotlin.com/sppnet-tutorial/#pytorch-implementation)

参考论文：[Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](https://arxiv.org/pdf/1406.4729.pdf)

### 传统 CNN 的弊端

在传统 CNN 中，由于 Fully-Connected 层的存在，输入图像的尺寸受到了严格限制。通常情况下，我们需要对原始图片进行**裁剪（crop）**或**变形（warp）**的操作来调整其尺寸使其适配于 CNN。然而裁剪过的图片可能包含不了所需的所有信息，而改变纵横比的变形操作也可能会使关键部分产生非期望的形变。由于图片内容的丢失或失真，模型的准确度会受到很大的影响。![image-20190421103723218](https://ws1.sinaimg.cn/large/006tNc79ly1g2a1zc9wl1j31tg0g2b10.jpg)

上图中分别表现了两种 resize 的方法：裁剪（左）、变形（右）。它们对原图都造成了非期望的影响。

### SPP-Net 概述

从 CNN 的结构来看，我们需要让图像在进入 FC 层前就将尺度固定到指定大小。通过修改卷积层或池化层参数可以改变图片大小，其中池化层不具有学习功能，其参数不会随着训练过程变化，自然而然承担起改变 spatial size 的工作。我们在第一个 FC 层前加入一个特殊的池化层，其参数是随着输入大小而成比例变化的。![image-20190421103738697](https://ws4.sinaimg.cn/large/006tNc79ly1g2a1zk0uaxj31d207cwg6.jpg)

图1. 使用 crop 或 warp 方法的 CNN 的层级结构。图2. 在卷积层与第一个全连接层之间加入 SPP 层。

SPP-Net 中有若干个并行的池化层，将卷积层的结果 [w×h×d][w×h×d] 池化成 [1×1],[2×2],[4×4],⋯[1×1],[2×2],[4×4],⋯ 的一层层结果，再将其所有结果与 FC 层相连。

![image-20190421103805080](https://ws4.sinaimg.cn/large/006tNc79ly1g2a200j1ixj30jg08xdit.jpg)

当输入为任意大小的图片时，我们可以随意进行卷积、池化，在 FC 层之前，通过 SPP 层，将图片抽象出固定大小的特征（即多尺度特征下的固定特征向量抽取）。

### SPP-Net 结构细节

![image-20190421103853981](https://ws1.sinaimg.cn/large/006tNc79ly1g2a20v0c2gj31130ok7ch.jpg)

结构如上所示，已知输入 conv5 的大小是 [w×h×d][w×h×d]，SPP 中某一层输出结果大小为 [n×n×d][n×n×d]，那么如何设定该层的参数呢？

- **感受野大小** [wr×hr][wr×hr]：wr=⌈wn⌉wr=⌈wn⌉，hr=⌈hn⌉hr=⌈hn⌉
- **步长** (sw,sh)(sw,sh)：sw=⌊wn⌋sw=⌊wn⌋，sh=⌊hn⌋sh=⌊hn⌋

假设输入是 [30×42×256][30×42×256]，对于 SPP 中 [4×4][4×4] 的层而言，其：

- 感受野大小应为 [⌈304⌉×⌈424⌉]=[8×11][⌈304⌉×⌈424⌉]=[8×11]
- 步长应为 (⌊304⌋,⌊424⌋)=(7,10)(⌊304⌋,⌊424⌋)=(7,10)

最后再将 SPP 中所有层的池化结果（池化操作通常是取感受野内的 max）变成 1 维向量，并与 FC 层中的神经元连接。

如上图中的 SPP 有三层([1×1],[2×2],[4×4][1×1],[2×2],[4×4])，则通过 SPP 后的特征有 (1+4+16)×256(1+4+16)×256 个。

### SPP-Net 训练方法

虽然使用了 SPP，理论上可以直接用变尺度的图像集作为输入进行训练，但是常用的一些框架（如 CUDA-convnet、Caffe等）在底层实现中更适合固定尺寸的计算（效率更高）。原论文中提及了两种训练方法：

- Single-Size：将所有的图片固定到同一尺度。
- Multi-Size：将原图片通过 crop 得到某一尺度 A，再把 A 通过 warp 放缩成更小的尺寸 B。之后用 A 尺度训练一个 epoch，再用 B 尺度训练一个 epoch，交替迭代。

由何凯明等人的实验结果可以发现，采用 Multi-Size 方法训练得到的模型错误率更低，且收敛速度更快。

### 在 pytorch 框架中实现 SPP

```python
class SpatialPyramidPool2D(nn.Module):
    """
    Args:
        out_side (tuple): Length of side in the pooling results of each pyramid layer. 
    
    Inputs:
        - `input`: the input Tensor to invert ([batch, channel, width, height])
    """
    def __init__(self, out_side):
        super(SpatialPyramidPool2D, self).__init__()
        self.out_side = out_side
    
    def forward(self, x):
        out = None
        for n in self.out_side:
            w_r, h_r = map(lambda s: math.ceil(s/n), x.size()[2:])    # Receptive Field Size
            s_w, s_h = map(lambda s: math.floor(s/n), x.size()[2:])   # Stride
            max_pool = nn.MaxPool2d(kernel_size=(w_r, h_r), stride=(s_w, s_h))
            y = max_pool(x)
            if out is None:
                out = y.view(y.size()[0], -1)
            else:
                out = torch.cat((out, y.view(y.size()[0], -1)), 1)
        return out
```

可以在模型中插入该模块，如：

```python
nn.Sequential(
    nn.Conv2d(
        in_channels=1,
        out_channels=16,
        kernel_size=5,
        stride=1,
        padding=2,
    ),
    nn.ReLU(),
    SpatialPyramidPool2D(out_side=(1,2,4))
)
```

# 在 pytorch 中建立自己的图片数据集

### 目录

- [图片文件在同一目录下](https://oidiotlin.com/create-custom-dataset-in-pytorch/#one-directory)
- [图片文件在不同目录下](https://oidiotlin.com/create-custom-dataset-in-pytorch/#multi-directories)

------

通常情况下，待处理的图片数据有两种存放方式：

- 所有图片在**同一目录**下，另有一份文本文件记录了标签。
- 不同标签的图片放在**不同目录**下，文件夹名就是标签。

对于这两种情况，我们有不同的解决方法。

### 图片文件在同一目录下

假设在 `./data/` 目录下有所需的所有的图片，以及一份标记了图片标签的文本文件（列为图片路径+标签）`./labels.txt`

```plain-text
./data/IZvVCYcuOkcu6Ufj.jpg 0
./data/2wuPp4yYoc2wJbZI.jpg 0
./data/vzlBbG4Z1KKJ4P6L.jpg 1
./data/nR8VZBPbjF92wNGC.jpg 2
......
```

思路是继承 `torch.utils.data.Dataset`，并重点重写其 `__getitem__` 方法，示例代码如下：

```python
class CustomDataset(Dataset):
    def __init__(self, label_file_path):
        with open(label_file_path, 'r') as f:
            # (image_path(str), image_label(str))
            self.imgs = list(map(lambda line: line.strip().split(' '), f))
    
    def __getitem__(self, index):
        path, label = self.imgs[index]
        img = transforms.Compose([transforms.Scale(224),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),])(Image.open(path).convert('RGB'))
        label = int(label)
        return img, label
    
    def __len__(self):
        return len(self.imgs)

dataset = CustomDataset('./labels.txt')
loader = DataLoader(dataset, batch_size=64, shuffle=True)
```

至此，可以用 `enumerate(loader)` 的方式迭代数据了。需要注意的是，在 `__getitem__`时要确保 **batch 内图片尺寸相同**（上面的例子用了 Scale+CenterCrop 的方法），否则会出现 `RuntimeError: inconsistent tensor sizes at ...` 的错误。

### 图片文件在不同目录下

当图片文件依据 label 处于不同文件下时，如：

```plain-text
─── data
    ├── 虾饺
    │   ├── 00856315f0df13536183d8ae6cbaf8d6a54f37ce.jpg
    │   └── 00ce9dccdf9a218d3b891e006c81f8e66524b1b3.jpg
    ├── 八宝粥
    │   ├── 055133235f649411e599ce5dba83627d58996209.jpg
    │   └── 0a72473884cb6c03191ca929a9aa0b2bbe4abb3d.jpg
    └── 钵仔糕
        ├── 1237b1e7b7e7da0ac78f9e1c8317b9462fe92803.jpg
        └── 14a7d6c1a881d1dcfe855bf783064ad2c9d5aba4.jpg
```

此时我们可以利用 `torchvision.datasets.ImageFolder` 来直接构造出 dataset，代码如下：

```python
dataset = ImageFolder(path)
loader = DataLoader(dataset)
```

ImageFolder 会将目录中的文件夹名自动转化成序列，那么 `loader` 载入时，标签就是整数序列了。

作者：龙鹏-言有三

链接：https://zhuanlan.zhihu.com/p/39455807

来源：知乎

著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

下面一个一个解释，完整代码请移步 Git 工程。

（1）datasets.ImageFolder

Pytorch 的 torchvision 模块中提供了一个 dataset 包，它包含了一些基本的数据集如 mnist、coco、imagenet 和一个通用的数据加载器 ImageFolder。

它会以这样的形式组织数据，具体的请到 Git 工程中查看。

```text
root/left/1.png
root/left/2.png
root/left/3.png

root/right/1.png
root/right/2.png
root/right/3.png
```

imagefolder 有3个成员变量。

- `self.classes`：用一个 list 保存类名，就是文件夹的名字。
- `self.class_to_idx`：类名对应的索引，可以理解为 0、1、2、3 等。
- `self.imgs`：保存（imgpath，class），是图片和类别的数组。

不同文件夹下的图，会被当作不同的类，天生就用于图像分类任务。

（2）Transforms

这一点跟 Caffe 非常类似，就是定义了一系列数据集的预处理和增强操作。到此，数据接口就定义完毕了，接下来在训练代码中看如何使用迭代器进行数据读取就可以了，包括 scale、减均值等。

（3）torch.utils.data.DataLoader

这就是创建了一个 batch，生成真正网络的输入。关于更多 Pytorch 的数据读取方法，请移步知乎专栏和公众号，链接在前面的课程中有给出。

**2.2 模型定义**

如下：

```text
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    class simpleconv3(nn.Module):`
    def __init__(self):
        super(simpleconv3,self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 3, 2)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 24, 3, 2)
        self.bn2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 48, 3, 2)
        self.bn3 = nn.BatchNorm2d(48)
        self.fc1 = nn.Linear(48 * 5 * 5 , 1200)
        self.fc2 = nn.Linear(1200 , 128)
        self.fc3 = nn.Linear(128 , 2)
    def forward(self , x):
        x = F.relu(self.bn1(self.conv1(x)))
        #print "bn1 shape",x.shape        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1 , 48 * 5 * 5) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

这三节课的任务，都是采用一个简单的 3 层卷积 + 2 层全连接层的网络结构。根据上面的网络结构的定义，需要做以下事情。

（1）simpleconv3(nn.Module)

继承 nn.Module，前面已经说过，Pytorch 的网络层是包含在 nn.Module 里，所以所有的网络定义，都需要继承该网络层。

并实现 super 方法，如下：

```text
super(simpleconv3,self).__init__()
```

这个，就当作一个标准，执行就可以了。

（2）网络结构的定义

都在 nn 包里，举例说明：

```text
torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
```

完整的接口如上，定义的第一个卷积层如下：

```text
nn.Conv2d(3, 12, 3, 2)
```

即输入通道为3，输出通道为12，卷积核大小为3，stride=2，其他的层就不一一介绍了，大家可以自己去看 nn 的 API。

（3）forward

backward 方法不需要自己实现，但是 forward 函数是必须要自己实现的，从上面可以看出，forward 函数也是非常简单，串接各个网络层就可以了。

对比 Caffe 和 TensorFlow 可以看出，Pytorch 的网络定义更加简单，初始化方法都没有显示出现，因为 Pytorch 已经提供了默认初始化。

如果我们想实现自己的初始化，可以这么做：

```text
init.xavier_uniform(self.conv1.weight)init.constant(self.conv1.bias, 0.1)
```

它会对 conv1 的权重和偏置进行初始化。如果要对所有 conv 层使用 xavier 初始化呢？可以定义一个函数：

```text
def weights_init(m):    
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        xavier(m.bias.data)  
    net = Net()   
    net.apply(weights_init)
```

###  模型训练

网络定义和数据加载都定义好之后，就可以进行训练了，老规矩先上代码：

```python
    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    model.train(True)  
                else:
                    model.train(False)  
                    running_loss = 0.0                running_corrects = 0.0
                for data in dataloders[phase]:
                    inputs, labels = data
                    if use_gpu:
                        inputs = Variable(inputs.cuda())
                        labels = Variable(labels.cuda())
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.data.item()
                    running_corrects += torch.sum(preds == labels).item()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects / dataset_sizes[phase]

                if phase == 'train':
                    writer.add_scalar('data/trainloss', epoch_loss, epoch)
                    writer.add_scalar('data/trainacc', epoch_acc, epoch)
                else:
                    writer.add_scalar('data/valloss', epoch_loss, epoch)
                    writer.add_scalar('data/valacc', epoch_acc, epoch)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
        writer.export_scalars_to_json("./all_scalars.json")
        writer.close()
        return model
```

分析一下上面的代码，外层循环是 epoches，然后利用 for data in dataloders[phase] 循环取一个 epoch 的数据，并塞入 variable，送入 model。需要注意的是，每一次 forward 要将梯度清零，即 `optimizer.zero_grad()`，因为梯度会记录前一次的状态，然后计算 loss，反向传播。

```text
loss.backward()
optimizer.step()
```

下面可以分别得到预测结果和 loss，每一次 epoch 完成计算。

```python
epoch_loss = running_loss / dataset_sizes[phase]
epoch_acc = running_corrects / dataset_sizes[phase]
      
_, preds = torch.max(outputs.data, 1)
loss = criterion(outputs, labels)
```