# Tensor to img && imge to tensor



在pytorch中经常会遇到图像格式的转化，例如将PIL库读取出来的图片转化为Tensor，亦或者将Tensor转化为numpy格式的图片。而且使用不同图像处理库读取出来的图片格式也不相同，因此，如何在pytorch中正确转化各种图片格式(PIL、numpy、Tensor)是一个在调试中比较重要的问题。

本文主要说明在pytorch中如何正确将图片格式在各种图像库读取格式以及tensor向量之间转化的问题。以下代码经过测试都可以在Pytorch-0.4.0或0.3.0版本直接使用。

对python不同的图像库读取格式有疑问可以看这里：<https://oldpan.me/archives/pytorch-transforms-opencv-scikit-image>

# 格式转换

我们一般在pytorch或者python中处理的图像无非这几种格式：

- PIL：使用python自带图像处理库读取出来的图片格式
- numpy：使用python-opencv库读取出来的图片格式
- tensor：pytorch中训练时所采取的向量格式（当然也可以说图片）

注意，之后的讲解图片格式皆为RGB三通道，24-bit真彩色，也就是我们平常使用的图片形式。

## PIL与Tensor

PIL与Tensor的转换相对容易些，因为pytorch已经提供了相关的代码，我们只需要搭配使用即可：

所有代码都已经引用了（之后的代码省略引用部分）：

```python
import torch
from PIL import Image
import matplotlib.pyplot as plt

# loader使用torchvision中自带的transforms函数
loader = transforms.Compose([
    transforms.ToTensor()])  

unloader = transforms.ToPILImage()
```

### 1 PIL读取图片转化为Tensor

```python
# 输入图片地址
# 返回tensor变量
def image_loader(image_name):
    image = Image.open(image_name).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)
```

### 2 将PIL图片转化为Tensor

```python
# 输入PIL格式图片
# 返回tensor变量
def PIL_to_tensor(image):
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)
```

### 3 Tensor转化为PIL图片

```python
# 输入tensor变量
# 输出PIL格式图片
def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image
```

### 4 直接展示tensor格式图片

```python
def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
```

### 5 直接保存tensor格式图片

```python
def save_image(tensor, **para):
    dir = 'results'
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    if not osp.exists(dir):
        os.makedirs(dir)
    image.save('results_{}/s{}-c{}-l{}-e{}-sl{:4f}-cl{:4f}.jpg'
               .format(num, para['style_weight'], para['content_weight'], para['lr'], para['epoch'],
                       para['style_loss'], para['content_loss']))
```

## numpy与Tensor

numpy格式是使用cv2，也就是python-opencv库读取出来的图片格式，需要注意的是用python-opencv读取出来的图片和使用PIL读取出来的图片数据略微不同，经测试用python-opencv读取出来的图片在训练时的效果比使用PIL读取出来的略差一些(详细过程之后发布)。

之后所有代码引用：

```python
import cv2
import torch
import matplotlib.pyplot as plt
```

### numpy转化为tensor

```python
def toTensor(img):
    assert type(img) == np.ndarray,'the img type is {}, but ndarry expected'.format(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255).unsqueeze(0)  # 255也可以改为256
```

### tensor转化为numpy

```python
def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    return img
```

### 展示numpy格式图片

```python
def show_from_cv(img, title=None):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
```

### 展示tensor格式图片

```python
def show_from_tensor(tensor, title=None):
    img = tensor.clone()
    img = tensor_to_np(img)
    plt.figure()
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
```

## 注意

上面介绍的都是一张图片的转化，如果是n张图片一起的话，只需要修改一下相应代码即可。

举个例子，将之前说过的修改略微修改一下即可：

```python
# 将 N x H x W X C 的numpy格式图片转化为相应的tensor格式
def toTensor(img):
    img = torch.from_numpy(img.transpose((0, 3, 1, 2)))
    return img.float().div(255).unsqueeze(0)
```