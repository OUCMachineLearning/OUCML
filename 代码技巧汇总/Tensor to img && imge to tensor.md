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

# orchvision transforms 总结

2018年11月16日 17:00:04 [Hansry](https://me.csdn.net/Hansry) 阅读数：2399



## 一.torchvision.transforms

Transfoms 是很常用的图片变换方式，可以通过`compose`将各个变换串联起来
**1. class torchvision.transforms.Compose (transforms) **
这个类将多个变换方式结合在一起
参数：各个变换的实例对象
举例：

```
transforms.Compose([
			transforms.CenterCrop(10),
			transforms.ToTensor(), 
			])
1234
```

## 二. 在PIL格式图片上的转换

**1.class torchvision.transforms.CenterCrop(size)**
剪切并返回PIL图片上中心区域
参数：size (序列或者整型)　—　输出的中心区域的大小。如果输入的size是整型而不是类似于 (h,w)的序列，那么将会转成类似(size, size)的序列。

**2.class torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)**
随机改变图片的亮度、对比度和饱和度
**参数：**

- brightness(亮度，float类型)——调整亮度的程度，亮度因子(brightness_factor)从 **[max(0,1-brightness), 1+brightness]** 中均匀选取。
- contrast(对比度，float类型)——调整对比度的程度，对比度因子(contrast_factor)从 **[max(0,1-contrast),1+contrast]** 中均匀选取。
- saturation(饱和度，float类型)——调整饱和度的程度，饱和度因子(saturation_factor) **[max(0,1-saturation),1+saturation]** 中均匀选取。
- hue(色相，float类型) —— 调整色相的程度，色相因子(hue_factor)从 **[-hue,hue]** 等均匀选择, 其中hue的大小为 **[0, 0.5]**。

**对比度：**　对比度指不同颜色之间的差别。对比度越大，不同颜色之间的反差越大，所谓黑白分明，对比度过大，图像就会显得很刺眼。对比度越小，不同颜色之间的反差就越小。
**亮度：**　亮度是指照射在景物或者图像上光线的明暗程度，图像亮度增加时，会显得刺眼或耀眼，亮度越小，会显得灰暗。
**色相：**　色相就是颜色，调整色相就是调整景物的颜色。
**饱和度：**　饱和度指图像颜色的浓度。饱和度越高，颜色越饱满，所谓的青翠欲滴的感觉。饱和度越低，颜色就会越陈旧，惨淡，饱和度为0时，图像就为灰度图像。

**3. class torchvision.transforms.FiveCrop(size)**
将给定的PIL图像剪裁成四个角落区域和中心区域
**注意：**　这个变换返回的是一个图像元组(tuple of images), 因此其输出跟输出的数量会不匹配。
**参数：**　size(序列或者整型) —— 需要返回的剪裁区域的尺寸。如果输入的是整型，那么会被转成(size,size)序列。

例子：

```
transform = Compose([
	FiveCrop(size),
	Lambda(lambda crops:torch.stack([ToTensor()(crop) for crop in crops]))  #return a 4D tensor
])
#in your test loop you can do the following:
input ,target = batch #input is a 5d tensor, target is 2d　
bs, ncrops,c ,h ,w = input.size()   
result = model(input.view(-1,c,h,w))  　#fuse batch size and ncrops  转成(bs*ncrops, c, h , w)
result_avg = result.view(bs, ncrops, -1).mean(1)      #avg over crops 转成(bs，ncrops, c*h*w)
123456789
```

**４. class torchvision.transforms.Ｇrayscale(num_output_channels=1)**
将图片转成灰度图
**参数：**　num_output_channels(int) ——　(1或者3)，输出图片的通道数量
**返回：**　输入图片的灰度图，如果num_output_channels=1, 返回的图片为单通道. 如果 num_output_channels=3, 返回的图片为3通道图片，且r=g=b
返回类型：PIL图片类型

**5.　class torchvision.transforms.Pad(padding, fill=0, padding_mode=‘constant’)**
对给定的PIL图像的边缘进行填充，填充的数值为给定填充数值
**参数：**

- padding(int或者tuple)——填充每一个边界。如果只输入了一个int类型的数值，那么这个数值会被用来填充所有的边界。如果输入的是tuple且长度为2，那么俩个数值分别被用于填充left/right 和 top/bottom。如果输入的数组为4，那么分别被用来填充left, top ,right 和 bottom边界。
- fill (int 或者 tuple) ——　填充的像素的数值为fill。默认为0，如果输入的元组的长度为3，那么分别被用来填充R,G,B通道。这个数值当padding_mode 等于‘constant’　的时候才会被使用。
- padding_mode (string) ——　填充的类型，必须为：constant, edge, reflect or symmetric，默认为 constant.
    **constant:** 以常量值进行填充，常量值由 fill 确定。
    **edge:** 用图片边界最后一个值进行填充
    **reflect:** pads with reflection of image without repeating the last value on the edge (这句不知怎么翻译，看下面例子)
    例子: 用俩个元素填充[1,2,3,4], 将会返回[3,2,1,2,3,4,3,2]
    symmetric: pads with reflection of image repeating the last value on the edge
    例子：用俩个函数元素填充 [1,2,3,4]，将会返回[2,1,1,2,3,4,4,3]

**6. class torchvision.transforms.RandomAffine(degrees, translate=None, scale=None)**
保持中心不变的对图片进行随机仿射变化
**参数：**[添加链接描述](https://pytorch.org/docs/stable/torchvision/transforms.html?highlight=transforms)

- degree (旋转，squence或者float或者int) ——　旋转的角度范围。如果角度是数值而不是类似于(min,max)的序列，那么将会转换成(-degree, +degree)序列。设为0则取消旋转。
- transalate (平移，tuple，可选)　——　数组，其中元素为代表水平和垂直变换的最大绝对分数。例如translate=(a,b),那么水平位移数值为从　**-image_widtha<dx<image_widtha**　随机采样的，同时垂直位移是从　**-img_heightb<dy<image_heightb** 随机采样的。默认情况下没有平移。
- scale (缩放，tuple, 可选)　——　缩放因子区间。若scale=(a,b), 则缩放的值在**a<=scale<=b** 随机采样。默认情况下没有缩放。
- shear (错切，sequence 或者 float 或者 int, 可选)　——　错切的程度。如果错切的程度是一个值，那么将会转换为序列即(—degree, +degree)。默认情况下不使用错切。
- resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, 可选)。
- fillcolor(整型) —— 可选择的在输出图片中填充变换以外的区域。(Pillow>=5.0.0)

**7.torchvision.transforms.RandomApply(transforms, p=0.5)**
随机选取变换中(各种变换存储在列表中)的其中一个，同时给定一定的概率
**参数：**　
变换（list或者tuple） ——　转换的列表
p (float 类型) —— 概率,选取某个变化需要的概率

**8.transforms.RandomSizedCrop() RandomApply() RandomChoice() RadomCrop RamdomGrayscale() RamdomHorizontalFlip(p=0.5) RamdomRotation()** … 还有各种Random，详细请查看[torch.transforms](https://pytorch.org/docs/stable/torchvision/transforms.html?highlight=transforms)

**9.torchvision.transforms.Resize(size,interpolation=2)**
将输入的PIL图片转换成给定的尺寸的大小
参数：

- size(sequence 或者 int)　——　需要输出的图片的大小。如果size是类似于(h,w)的序列，输出的尺寸将会跟(h,w)一致。如果size是整型，图片较小的边界将会被置为这个尺寸。例如，如果height->width, 图片将会被置为 (size*height/width, size)
- Interpolation (int, 可选) —— 默认为 PIL.Image.BILINEAR

# 三. 在torch.*Tensor上的转换

**1. class torchvision.transforms.Normalize(mean,std)**
用均值和标准差对张量图像进行标准化处理。给定n通道的均值(M1, … , Mn) 和标准差(S1, … ,Sn), 这个变化将会归一化根据均值和标准差归一化每个通道值。例如，input[channel] = (input[channel]-mean[channel])/std(channel)
参数：

- mean (squence) ——　每个通道的均值
- std (sequence) —— 每个通道的标准差

```
__call__(tensor) 参数：tensor(Tensor) , 尺寸为(C,H,W)的图片将会被归一化 ; 返回：归一化后的Tensor类型图片 ; 返回类型：Ｔensor
```

# 四. 类型转换变换 (Conversion Transforms)

**1. class torchvision.transforms.ToPILImage(mode=None)**
将tensor类型或者ndarray转换成PIL图片
将 CxHxW大小的torch.*Tensor或者ＨxWxC 大小的numpy 矩阵转成PIL图片
参数：如果model为Ｎone,那么如果输入有三个通道，那么mode为RGB; 如果input有4个通道，mode为RGBA. 如果输入是1通道，mode为数据类型，如int, float, short

```
__call__(pic) 参数：pic (Tensor或者numpy.ndarray类型的)　——　转换成PIL图片;  返回PIL图片; 返回类型为PIL类型
1
```

**2. torchvision.transforms.ToTensor**
将PIL图片或者numpy.ndarray转成Tensor类型的
将PIL图片或者numpy.ndarray(HxWxC) (范围在0-255) 转成torch.FloatTensor (CxHxW) (范围为0.0-1.0)

```
__call__(pic) 参数：pic(PIL图片或者numpy.ndarray) —— 将图片转成向量; 返回Tensor类型的图片
1
```

## 五. 一般变换 (Generic Transforms)

**1. torchvision.transforms.Lambda(lambd)**
使用用户定义的lambda作为转换
参数：lambd(function) ——　用Lambda/funtion 作为变换

**2. torchvision.transforms.functional.adjust_brightness(img, brightness_factor)**
调整图片的亮度
参数：

- img(PIL 图片)——PIL图片
- brightness_factor(float)——亮度调整程度。不能为负数, ０代表黑色图片，1代表原始图片，2代表增加了2个因子的亮度。
- returns: 返回调整完的图片

**3.torchvision.transforms.functional.adjust_contrast(img,contrast_factor)**
调整图片的对比度
参数：

- img —— 需要调整的PIL 图片
- constrast_factor(float) —— 调整对比度的程度。可以是非负的数。0为灰度图，1为原图，2为增加图片2个对比因子的图片。
- returns　——　返回调整后的对比度图片

**4. torchvison.transforms.function.adjust_gamma(img, gamma, gain=1)**
对图片进行gamma校正，[gamma校正详情](https://en.wikipedia.org/wiki/Gamma_correction)
Iout=255∗gain∗(Iin/255)γI_{out}=255*gain*(I_{in}/255)^{\gamma}*I**o**u**t*​=255∗*g**a**i**n*∗(*I**i**n*​/255)*γ*

参数：

- img(PIL图片)——需要调整的PIL图片
- gamma (float类型)——非零实数，公式中的γ\gamma*γ*也是非零实数。gamma大于1使得阴影部分更暗，gamma小于1使得暗的区域亮些。
- gain(float) ——　常量乘数

**5.torchvision.transforms.functional.ajust_hue (img,hue_factor)**
调整图片的色相
通过将图像转换为HSV来调整图像的色调，并在色调通道(H)中循环移动强度，然后将图像转换回原始图像模式。
色相因子是Ｈ通道平移量，其必须在区间[-0.5,0.5]中。
参数

- img (PIL 图片)　——　需要调整的PIL图片
- hue_factor (float类型) —— 色相通道平移的量，必须在[-0.5,0.5]之间。0.5和-0.5分别代表在HSV空间中正负方向完全相反的色相通道。0代表没有平移。

**5. tochvision.transforms.functional.adjust_saturation(img, hue_factor)**
调整图片的颜色饱和度
参数：

- img (PIL图片)——需要调整的PIL图片
- 饱和度因子(float类型)——调整饱和度的程度。0将会输出黑白图片，1将会输出原始图片，2将会增强2个因子的饱和度。
- 返回调整后的图片。

**6. torchvision.transforms.functional.affine(img, angle, translate, scale, shear, resample=0, fillcolor=None)**
对图片进行放射变换，保持中心不变。
参数：

- img (PIL图片)——需要变换的PIL图片
- angle(float 或者 int)——旋转的的角度，角度范围为 (-180,180), 正方向为顺时针方向。
- translate(list 或者 tuple)——水平或者垂直平移
- scale(float)——总体缩放
- shear(错切，float)——错切的角度位于(-180,180)，顺时针方向。
- resample（这个有点看不懂，应该比较少用到——PIL.Image.NEAREST or PIL.Image.BILINEAR or PIL.Image.BICUBIC, optional
- fillcolor (int) —— 填充输出图片中超过变换的区域(Pillow>=5.0)

**7.torchvision.transforms.functional.crop(img,i,j,h,w)**
剪裁给定的PIL图片
参数：

- img(PIL图片)——被剪裁的图片
- （i, j) ——左上角图片坐标
- (h,w)——剪裁的图片的高和宽
    returns: 返回剪彩的图片

**8. torchvision.transforms.functional.normalize(tensor, mean, std)**
根据给定的标准差和方差归一化tensor图片
参数：

- tensor(Tensor)—— 形状为(C,H,W)的Tensor图片
- mean(squence) —— 每个通道的均值，序列
- std (sequence) —— 每个通道的标准差，序列
    返回：返回归一化后的Tensor图片。

**9.torchvision.transforms.functional.pad(img, padding, fill=0, padding_mode=‘constant’)、torchvision.transforms.functional.resize(img, size, interpolation=2)、torchvision.transforms.functional.rotate(img, angle, resample=False, expand=False, center=None)、torchvision.transforms.functional.to_grayscale(img, num_output_channels＝１)**
等均与上述函数类似，这里不再重复。

**10.torchvision.transforms.functional.to_pil_image(pic, mode=None) 将tensor或者numpy.ndarray转成PIL图片torchvision.transforms.functional.to_tensor(pic) 将PIL图片或者numpy.ndarray转成tensor**

