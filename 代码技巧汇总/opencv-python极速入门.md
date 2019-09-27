## opencv-python 极速入门

## **什么是OpenCV-Python?**

OpenCV是一个开源的计算机视觉（computer vision）和机器学习库。它拥有超过2500个优化算法，包括经典和最先进的计算机视觉和机器学习算法。它有很多语言接口，包括Python、Java、c++和Matlab。

这里，我们将处理Python接口。

## **安装**

- 在Windows上, 请在这里查看指南。地址：https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_setup/py_setup_in_windows/py_setup_in_windows.html 
- 在 Linux上, 请在这里查看指南。地址：https://docs.opencv.org/trunk/d7/d9f/tutorial_linux_install.html 

## **图像导入&显示**

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-27-082332.jpg)

警告1: 通过openCV读取图像时，它不是以RGB 颜色空间来读取，而是以BGR 颜色空间。有时候这对你来说不是问题，只有当你想在图片中添加一些颜色时，你才会遇到问题。

有两种解决方案:

1. 将R — 第一个颜色值(红色)和B  — 第三个颜色值(蓝色) 交换, 这样红色就是 (0,0,255) 而不是(255,0,0)。
2. 将颜色空间变成RGB:

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-27-082318.jpg)

**使用rgb_image代替image继续处理代码。**

警告2: 要关闭显示图像的窗口，请按任意按钮。如果你使用关闭按钮，它可能会导致窗口冻结(我在Jupyter笔记本上运行代码时发生了这种情况)。

为了简单起见，在整个教程中，我将使用这种方法来查看图像:

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-27-082345.jpg)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-27-082342.jpg)

来源：Pixabay

## 裁剪

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-27-082343.jpg)

来源：Pixabay

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-27-082344.jpg)

裁剪后的狗狗

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-27-082319.jpg)

其中： image[10:500,500:200] 是 image[y:y+h,x:x+w]。

## 调整大小

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-27-082337.jpg)

来源：Pexels

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-27-082323.jpg)

调整大小到20%后

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-27-082339.jpg)

这个调整大小函数会保持原始图像的尺寸比例。

更多图像缩放函数，请查看这里。（https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/  ）

## 旋转

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-27-082320.jpg)

左图: 图片来自Pexels的Jonathan Meyer。右图: 进行180度旋转之后的狗狗。

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-27-082347.jpg)

image.shape输出高度、宽度和通道。M是旋转矩阵——它将图像围绕其中心旋转180度。

-ve表示顺时针旋转图像的角度  & +ve逆表示逆时针旋转图像的角度。

## 灰度和阈值(黑白效果)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-27-082346.jpg)

来源：Pexels

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-27-082340.jpg)

gray_image 是灰度图像的单通道版本。

这个threshold函数将把所有比127深(小)的像素点阴影值设定为0，所有比127亮(大)的像素点阴影值设定为255。

另一个例子:

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-27-82320.jpg)

这将把所有阴影值小于150的像素点设定为10和所有大于150的像素点设定为200。

更多有关thresholding函数的内容，请查看这里。（https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html  ）

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-27-082324.jpg)

左图：灰阶狗狗。右图：黑白狗狗。

## 模糊/平滑

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-27-82326.jpg)

左图：图像来自Pixabay。右图：模糊后的狗狗。

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-27-082326.jpg)

高斯模糊函数接受3个参数:

1. 第一个参数是要模糊的图像。
2. 第二个参数必须是一个由两个正奇数组成的元组。当它们增加，模糊效果也会增加。
3. 第三个参数是sigmaX和sigmaY。当左边位于0时，它们会自动从内部大小计算出来。

更多关于模糊函数的内容，请查看这里。（https://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html  ）

## 在图像上绘制矩形框或边框

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-27-082333.jpg)

 左图：图像来自Pixabay。右图：脸上有一个矩形框的狗狗。

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-27-082338.jpg)

rectangle函数接受5个参数:

1. 第一个参数是图像。
2. 第二个参数是x1, y1 -左上角坐标。
3. 第三个参数是x2, y2 -右下角坐标。
4. 第四个参数是矩形颜色(GBR/RGB，取决于你如何导入图像)。
5. 第五个参数是矩形线宽。

## 绘制一条线

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-27-82344.jpg)

左图：图像来自Pixabay。右图：两只狗狗用一条线分开。

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-27-082327.jpg)

line函数接受5个参数:

- 第一个参数是要画的线所在的图像。
- 第二个参数是x1, y1。
- 第三个参数是x2, y2。
- 第四个参数是线条颜色(GBR/RGB，取决于你如何导入图像)。
- 第五个参数是线宽。

## 在图片上写入文字

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-27-082325.jpg)

左图：图像来自Pixabay。右图：两只狗狗用一条线分开。

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-27-082321.jpg)

putText函数接受 七个参数：

- 第一个参数是要写入文本的图像。
- 第二个参数是待写入文本。
- 第三个参数是x, y——文本开始的左下角坐标。
- 第四个参数是字体类型。
- 第五个参数是字体大小。
- 第六个参数是颜色(GBR/RGB，取决于你如何导入图像)。
- 第七个参数是文本线条的粗细。

## **人脸检测**

这里没有找到狗狗照片，很遗憾：（

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-27-082328.jpg)

图片来自Pixabay,作者：Free-Photos。

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-27-82341.jpg)

detectMultiScale函数是一种检测对象的通用函数。因为我们调用的是人脸级联，所以它会检测到人脸。

detectMultiScale函数接受4个参数：

- 第一个参数是灰阶图像。
- 第二个参数是scaleFactor。因为有些人脸可能离镜头更近，所以看起来会比后台的人脸更大。比例系数弥补了这一点。
- 检测算法使用一个移动窗口来检测对象。minNeighbors定义在当前对象附近检测到多少对象，然后再声明检测到人脸。
- 与此同时，minsize给出了每个窗口的大小。

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-27-082331.jpg)

检测到两张人脸。

## 轮廓——一种对象检测方法

使用基于颜色的图像分割，你可以来检测对象。

cv2.findContours & cv2.drawContours 这两个函数可以帮助你做到这一点。

最近，我写了一篇非常详细的文章，叫做《使用Python通过基于颜色的图像分割来进行对象检测》。你需要知道的关于轮廓的一切都在那里。（https://towardsdatascience.com/object-detection-via-color-based-image-segmentation-using-python-e9b7c72f0e11  ）

## 最终,保存图片

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-27-082341.jpg)

## **总结**

OpenCV是一个非常容易使用的算法库，可以用于3D建模、高级图像和视频编辑、跟踪视频中的标识对象、对视频中正在做某个动作的人进行分类、从图像数据集中找到相似的图像，等等。

最重要的是，学习OpenCV对于那些想要参与与图像相关的机器学习项目的人来说是至关重要的。