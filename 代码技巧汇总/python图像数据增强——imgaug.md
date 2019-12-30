

# python图像数据增强——imgaug

在这章我们展示一个涵盖了大部分数据增强方法的例子。这里有大量的代码，可能会引起部分读者的不适，但是大家可以主要看注释，以及最后的总结性的话语，在实际上使用的时候再详细的看具体的实现，有一些。

```python
from imgaug import augmenters as iaa #引入数据增强的包

sometimes = lambda aug: iaa.Sometimes(0.5, aug) #建立lambda表达式，

```

这里定义sometimes意味有时候做的操作，上一讲中我们有看到加入高斯模糊的例子，在那里，实际上是对每一张图片都进行了高斯模糊的处理，然而实际上在深度学习的模型训练中，数据增强不能喧宾夺主，如果对每一张图片都加入高斯模糊的话实际上是毁坏了原来数据的特征，因此，我们需要“有时候”做，给这个操作加一个概率。

```python
seq = iaa.Sequential(
    [
        iaa.Fliplr(0.5), # 对50%的图像进行镜像翻转
        iaa.Flipud(0.2), # 对20%的图像做左右翻转

        sometimes(iaa.Crop(percent=(0, 0.1))), 
	   #这里沿袭我们上面提到的sometimes，对随机的一部分图像做crop操作
        # crop的幅度为0到10%

        sometimes(iaa.Affine(                          #对一部分图像做仿射变换
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},#图像缩放为80%到120%之间
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, #平移±20%之间
            rotate=(-45, 45),   #旋转±45度之间
            shear=(-16, 16),    #剪切变换±16度，（矩形变平行四边形）
            order=[0, 1],   #使用最邻近差值或者双线性差值
            cval=(0, 255),  #全白全黑填充
            mode=ia.ALL    #定义填充图像外区域的方法
        )),
		
        # 使用下面的0个到5个之间的方法去增强图像。注意SomeOf的用法
        iaa.SomeOf((0, 5),
            [
                # 将部分图像进行超像素的表示。o(╥﹏╥)o用超像素增强作者还是第一次见，比较孤陋寡闻
                sometimes(
                    iaa.Superpixels(
                        p_replace=(0, 1.0),
                        n_segments=(20, 200)
                    )
                ),

                #用高斯模糊，均值模糊，中值模糊中的一种增强。注意OneOf的用法
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)),
                    iaa.AverageBlur(k=(2, 7)), # 核大小2~7之间，k=((5, 7), (1, 3))时，核高度5~7，宽度1~3
                    iaa.MedianBlur(k=(3, 11)),
                ]),

                #锐化处理
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

                #浮雕效果
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                #边缘检测，将检测到的赋值0或者255然后叠在原图上
                sometimes(iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0, 0.7)),
                    iaa.DirectedEdgeDetect(
                        alpha=(0, 0.7), direction=(0.0, 1.0)
                    ),
                ])),

                # 加入高斯噪声
                iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, 0.05*255), per_channel=0.5
                ),

                # 将1%到10%的像素设置为黑色
			  # 或者将3%到15%的像素用原图大小2%到5%的黑色方块覆盖
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5),
                    iaa.CoarseDropout(
                        (0.03, 0.15), size_percent=(0.02, 0.05),
                        per_channel=0.2
                    ),
                ]),

                #5%的概率反转像素的强度，即原来的强度为v那么现在的就是255-v
                iaa.Invert(0.05, per_channel=True), 

                # 每个像素随机加减-10到10之间的数
                iaa.Add((-10, 10), per_channel=0.5),

                # 像素乘上0.5或者1.5之间的数字.
                iaa.Multiply((0.5, 1.5), per_channel=0.5),

                # 将整个图像的对比度变为原来的一半或者二倍
                iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),

                # 将RGB变成灰度图然后乘alpha加在原图上
                iaa.Grayscale(alpha=(0.0, 1.0)),

                #把像素移动到周围的地方。这个方法在mnist数据集增强中有见到
                sometimes(
                    iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                ),

                # 扭曲图像的局部区域
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
            ],
            
            random_order=True # 随机的顺序把这些操作用在图像上
        )
    ],
    random_order=True # 随机的顺序把这些操作用在图像上
)

images_aug = seq.augment_images(images) #实现图像增强

```

