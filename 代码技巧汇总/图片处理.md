前言：用CNN进行训练模型的时候，通常需要对图像进行处理，有时候也叫做数据增强，常见的图像处理的Python库：OpenCV、PIL、matplotlib、tensorflow等，这里用TensorFlow介绍图像处理的过程

# 图片处理

- 展示一张图片  注意需要对图像进行解码，然后进行展示，用tf.image.decode_png  先定义一个图片展示的函数代码如下：

```javascript
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
def show_image_tensor(image_tensor):
    #使用交互式回话
    image = image_tensor.eval()
    print("图片的大小为：{}".format(image.shape))
    if len(image.shape)==3 and image.shape[2]==1:
        plt.imshow(image[:,:,0],cmap="Greys_r")
        plt.show()
    elif len(image.shape)==3:
        plt.imshow(image)
        plt.show()
```

进行图像的读取和解码，然后调用函数进行展示

```javascript
#1读取、编码、展示
file_content=tf.read_file(image_path)
image_tensor = tf.image.decode_png(file_content,channels=3)
show_image_tensor(image_tensor)
```

结果如下：  图片的大小为：(512, 512, 3)

![image-20190427002618186](https://ws2.sinaimg.cn/large/006tNc79ly1g2gi1ewe7yj30xn0u0wwg.jpg)

- 修改大小，压缩或者放大  用tf.image.resize_images

```javascript
"""
BILINEAR = 0 线性插值，默认
NEAREST_NEIGHBOR = 1 最近邻插值，失真最小
BICUBIC = 2 三次插值
AREA = 3 面积插值
# images: 给定需要进行大小转换的图像对应的tensor对象，格式为：[height, width, num_channels]或者[batch, height, width, num_channels]
# API返回值和images格式一样，唯一区别是height和width变化为给定的值
"""
resize_image_tensor = tf.image.resize_images(images=image_tensor,size=(20,20),
                                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
show_image_tensor(resize_image_tensor)#注意前面进行解码的时候一定要用tf.image.decode_png
```

结果：  图片的大小为：(20, 20, 3)

![image-20190427002645298](https://ws4.sinaimg.cn/large/006tNc79ly1g2gi1t5k5qj30xh0u0djh.jpg)

注意：当放大时候，几乎图像不失真

- 剪切  或者是填充用tf.image.resize_image_with_crop_or_pad

```javascript
# 图片重置大小，通过图片的剪切或者填充（从中间开始计算新图片的大小）
corp_pad_image_tensor = tf.image.resize_image_with_crop_or_pad(image_tensor,300,300)
show_image_tensor(corp_pad_image_tensor)
```



上述为中间位置剪切或者填充，下面介绍任意位置剪切或者填充

```javascript
# 填充数据（给定位置开始填充）
pad_image_tensor = tf.image.pad_to_bounding_box(image=image_tensor, offset_height=200, offset_width=50,
                                                target_height=1000,target_width=1000)
# show_image_tensor(pad_image_tensor)
corp_to_bounding_box_image_tensor=tf.image.crop_to_bounding_box(image=image_tensor, offset_height=20, offset_width=50,
                                                target_height=300,target_width=400)
show_image_tensor(corp_to_bounding_box_image_tensor)
```

![image-20190427002834668](https://ws2.sinaimg.cn/large/006tNc79ly1g2gi3o21vcj30u014zha6.jpg)

# 数据增强

当训练数据有限的时候，可以通过一些变换来从已有的训 练数据集中生成一些新的数据，来扩大训练数据。数据增强的方法有：

-  镜像，翻转  例如：以垂直平面为对称轴如下：  

-  代码如下：  

    ```javascript
    # 上下交换
    filp_up_down_image_tensor = tf.image.flip_up_down(image_tensor)
    # show_image_tensor(filp_up_down_image_tensor)
    filp_left_right_image_tensor = tf.image.flip_left_right(image_tensor)
    show_image_tensor(filp_left_right_image_tensor)
    ```

    以水平面为对称轴如下：

    转置，相当于矩阵的转置,90度转换

    ```javascript
    # 转置
    transpose_image_tensor = tf.image.transpose_image(image_tensor)
    # show_image_tensor(transpose_image_tensor)
    
    # 旋转（90度、180度、270度....）
    # k*90度旋转，逆时针旋转
    k_rot90_image_tensor = tf.image.rot90(image_tensor, k=4)
    # show_image_tensor(k_rot90_image_tensor)
    ```

# 颜色空间转换

注意：颜色空间的转换必须讲image的值转换为float32类型，不能使用unit8类型  图像基本格式：  rgb（颜色）0-255,三个255为白色，转化为float32就是把区间变为0-1  hsv（h: 图像的色彩/色度，s:图像的饱和度，v：图像的亮度）  grab（灰度）

```javascript
# 颜色空间的转换必须讲image的值转换为float32类型，不能使用unit8类型
float32_image_tensor = tf.image.convert_image_dtype(image_tensor, dtype=tf.float32)
# show_image_tensor(float32_image_tensor)
# rgb -> hsv（h: 图像的色彩/色度，s:图像的饱和度，v：图像的亮度）
hsv_image_tensor= tf.image.rgb_to_hsv(float32_image_tensor)
show_image_tensor(hsv_image_tensor)
# hsv -> rgb
rgb_image_tensor = tf.image.hsv_to_rgb(float32_image_tensor)
# show_image_tensor(rgb_image_tensor)

# rgb -> gray
gray_image_tensor = tf.image.rgb_to_grayscale(rgb_image_tensor)
show_image_tensor(gray_image_tensor)
```

- 可以从颜色空间中提取图像的轮廓信息(图像的二值化)

```javascript
a = gray_image_tensor
b = tf.less_equal(a,0.4)
# 0是黑，1是白
# condition?true:false
# condition、x、y格式必须一模一样，当condition中的值为true的之后，返回x对应位置的值，否则返回y对应位置的值
# 对于a中所有大于0.4的像素值，设置为0
c = tf.where(condition=b,x=a,y=a-a)
# 对于a中所有小于等于0.4的像素值，设置为1
d= tf.where(condition=b,x=c-c+1,y=c)
show_image_tensor(d)
```

这样的方法，可以运用到车牌设别的过程中，对车牌自动进行截取。

- 图像调整（亮度调整，对比度调整，gammer调整，归一化操作）  亮度调整  image: RGB图像信息，设置为float类型和unit8类型的效果不一样，一般建议设置为float类型  delta: 取值范围(-1,1）之间的float类型的值，表示对于亮度的减弱或者增强的系数值  底层执行：rgb -> hsv -> h,s,v*delta -> rgb  同理还有色调和饱和度

```javascript
adiust_brightness_image_tensor = tf.image.adjust_brightness(image=image_tensor, delta=-0.8)
# show_image_tensor(adiust_brightness_image_tensor)
# 色调调整
# image: RGB图像信息，设置为float类型和unit8类型的效果不一样，一般建议设置为float类型
# delta: 取值范围(-1,1）之间的float类型的值，表示对于色调的减弱或者增强的系数值
# 底层执行：rgb -> hsv -> h*delta,s,v -> rgb
adjust_hue_image_tensor = tf.image.adjust_hue(image_tensor, delta=-0.8)
# show_image_tensor(adjust_hue_image_tensor)

# 饱和度调整
# image: RGB图像信息，设置为float类型和unit8类型的效果不一样，一般建议设置为float类型
# saturation_factor: 一个float类型的值，表示对于饱和度的减弱或者增强的系数值，饱和因子
# 底层执行：rgb -> hsv -> h,s*saturation_factor,v -> rgb
adjust_saturation_image_tensor = tf.image.adjust_saturation(image_tensor, saturation_factor=20)
show_image_tensor(adjust_saturation_image_tensor)
# 对比度调整，公式：(x-mean) * contrast_factor + mean(小的更小，大的更大）
adiust_contrast_image_tensor=tf.image.adjust_contrast(images=image_tensor, contrast_factor=1000)
show_image_tensor(adiust_contrast_image_tensor)
# 图像的gamma校正
# images: 要求必须是float类型的数据
# gamma：任意值，Oup = In * Gamma只要不是白色，都加深
adjust_gamma_image_tensor = tf.image.adjust_gamma(float32_image_tensor, gamma=10)
# show_image_tensor(adjust_gamma_image_tensor)
# 图像的归一化(x-mean)/adjusted_sttdev, adjusted_sttdev=max(stddev, 1.0/sqrt(image.NumElements()))
# per_image_standardization_image_tensor = tf.image.per_image_standardization(image_tensor)
# show_image_tensor(per_image_standardization_image_tensor)
```

# 噪音数据的加入

高斯噪声、模糊处理

```javascript
# noisy_image_tensor = image_tensor + tf.cast(50 * tf.random_normal(shape=[512, 512, 3], mean=0, stddev=0.1), tf.uint8)
noisy_image_tensor = image_tensor + tf.cast( tf.random_uniform(shape=[512, 512, 3],
                   minval=60,
                   maxval=70), tf.uint8)
show_image_tensor(noisy_image_tensor)
```

![image-20190427003007008](https://ws1.sinaimg.cn/large/006tNc79ly1g2gi5aqhhuj311v0u01kx.jpg)

