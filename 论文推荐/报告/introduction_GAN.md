---
marp: true
---

# introduction 
`Marcus`


---
![](https://cy-1256894686.cos.ap-beijing.myqcloud.com/uPic/i1aSu5.png)

----

# Application

----

### **[姿势引导人形像生成](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1705.09368.pdf)**

通过姿势的附加输入，我们可以将图像转换为不同的姿势。例如，右上角图像是基础姿势，右下角是生成的图像。

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/uPic/v2-0a459f8b54d77aa22201159120fbc64f_b.png)


---
下面的优化结果列是生成的图像。



![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/uPic/v2-f98966f0b16fe84e46c91c824d71c405_b.png)


----
该设计由二级图像发生器和鉴频器组成。生成器使用元数据（姿势）和原始图像重建图像。判别器使用原始图像作为[CGAN](https://link.zhihu.com/?target=https%3A//medium.com/%40jonathan_hui/gan-cgan-infogan-using-labels-to-improve-gan-8ba4de5f9c3d)设计标签输入的一部分。



![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/uPic/v2-ff89e58856c495d6bef825ab81744b3f_b.png)

----

### **CycleGAN**

跨域名转让将很可能成为第一批商业应用。GANs将图像从一个领域（如真实的风景）转换为另一个领域（莫奈绘画或梵高）。



![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/uPic/v2-0bb3ad1d9803ba4bd483e0a38f18c3e3_b.png)

---

例如，它可以在斑马和马之间转换图片。



![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/uPic/v2-ec0c46ae59d032ba3d43d84a44c1fcd3_b.png)

----

Cyclegan构建了两个网络G和F来构建从一个域到另一个域以及反向的图像。它使用判别器d来批评生成的图像有多好。例如，G将真实图像转换为梵高风格的绘画，并且DY用于区分图像是真实的还是生成的。

域A到域B：------------------------>


![bg right width:600px](https://cy-1256894686.cos.ap-beijing.myqcloud.com/uPic/v2-cd0d9b998dd532514f1311104a53d63f_b.png)



我们在反向域B域A中重复该过程：

![width:600px](https://cy-1256894686.cos.ap-beijing.myqcloud.com/uPic/v2-556739c5d0557ff65cb5ffc3946c96ed_b.png)

----

**PixelDTGAN**

根据名人图片推荐商品已经成为时尚博客和电子商务的热门话题。Pixeldtgan的作用就是从图像中创建服装图像和样式。



![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/uPic/v2-48b4cf374eeaa1d9a6ad48485b6c2d42_b.png)

---

![bg  width:600px](https://cy-1256894686.cos.ap-beijing.myqcloud.com/uPic/v2-a3ab30d7ee2a33e91a9566ff49cc1f93_b.png)


![bg width:600px](https://cy-1256894686.cos.ap-beijing.myqcloud.com/uPic/v2-82da972689bae5d50bd3720e112730c2_b.png)

---

**超分辨率**

从低分辨率创建超分辨率图像。这是GAN显示出非常令人印象深刻的结果，也是具有直接商业可能性的一个领域。



![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/uPic/v2-554afcc3da74ca15e3ed4bb66d02f395_b.png)


---

与许多GAN的设计类似，它是由多层卷积层、批标准化、高级relu和跳跃连接组成。

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/uPic/v2-933d08753407d48b2d9ffe6620183c7e_b.png)

---

**PGGAN**

Progressive GAN可能是第一个展示商业化图像质量的GAN之一。以下是由GAN创建的1024×1024名人形象。



![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/uPic/v2-20685837d4e713dcd5ebf638f0ae8ca6_b.png)

----

它采用分而治之的策略，使训练更加可行。卷积层的一次又一次训练构建出2倍分辨率的图像。


![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/uPic/v2-aeac3c10cba9e4675fcb764257465b84_b.png)

----

在9个阶段中，生成1024×1024图像。

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/uPic/v2-207fc48709b7578032e87a9005bc5833_b.png)

---

### **高分辨率图像合成**

需要注意的是这并非图像分割，而是从语义图上生成图像。由于采集样本非常昂贵，我们采用生成的数据来补充培训数据集，以降低开发成本。在训练自动驾驶汽车时可以自动生成视频，而不是看到它们在附近巡航，这就为我们的生活带来了便捷。

网络设计:

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/uPic/v2-35502c5e53a24658b828cd7d9c1c1eb4_b.png)



![bg right:30% width:600px](https://cy-1256894686.cos.ap-beijing.myqcloud.com/uPic/v2-f9cb726cb00abfa861eb571672b2c088_b.png)

---

## **文本到图像（[StackGAN](https://link.zhihu.com/?target=https%3A//github.com/hanzhanggit/StackGAN)）**

文本到图像是域转移GAN的早期应用之一。比如，我们输入一个句子就可以生成多个符合描述的图像。


![bg vertical width:600px](https://cy-1256894686.cos.ap-beijing.myqcloud.com/uPic/v2-4a5f7cd14021eb479deede0d41489386_b.png)

![bg right:60% width:600px](https://cy-1256894686.cos.ap-beijing.myqcloud.com/uPic/v2-daaf25d2b5db0b69c7034f6ef508bece_b.png)

---

### **文本到图像合成**

另一个比较通用的实现：

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/uPic/v2-8359529d27d5aedaec4b17b08bb5f482_b.png)

---

### **人脸合成**

不同姿态下的合成面：使用单个输入图像，我们可以在不同的视角下创建面。例如，我们可以使用它来转换更容易进行人脸识别图像。



![ ](https://cy-1256894686.cos.ap-beijing.myqcloud.com/uPic/v2-fada3ec3b9072148d87eb2ef419c995f_b.png)

![bg right width:700px](https://cy-1256894686.cos.ap-beijing.myqcloud.com/uPic/v2-c0bd8f724c4f858a4bd426e6f5636827_b.png)

----

### **图像修复**

几十年前，修复图像一直是一个重要的课题。gan就可以用于修复图像并用创建的“内容”填充缺失的部分。



![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/uPic/v2-0e1646f5337dbd2ed477ca694ad6826a_b.png)

----

### **学习联合分配**

用面部字符P（金发，女性，微笑，戴眼镜），P（棕色，男性，微笑，没有眼镜）等不同组合创建GAN是很不现实的。维数的诅咒使得GAN的数量呈指数增长。但我们可以学习单个数据分布并将它们组合以形成不同的分布，即不同的属性组合。


![width:800px](https://cy-1256894686.cos.ap-beijing.myqcloud.com/uPic/v2-470e2ce39ff9f07ffed590293bcd8c21_b.png)

![bg right width:600px](https://cy-1256894686.cos.ap-beijing.myqcloud.com/uPic/v2-f7678e5d027735bc8d34b45ce0653ed3_b.png)

----

### **DiscoGAN**

DiscoGAN提供了匹配的风格：许多潜在的应用程序。DiscoGAN在没有标签或配对的情况下学习跨域关系。例如，它成功地将样式（或图案）从一个域（手提包）传输到另一个域（鞋子）。



![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/uPic/v2-3e5d9abd9640e776f35e13308627f50c_b.png)


DiscoGAN和cyclegan在网络设计中非常相似。

----


![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/uPic/v2-fcc18c0d909e03a21195267998413068_b.png)


----

### **[Pix2Pix](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1611.07004.pdf)**

PIX2PIx是一种图像到图像的翻译，在跨域Gan的论文中经常被引用。例如，它可以将卫星图像转换为地图（图片左下角）。

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/uPic/v2-331774f96a52bcf9bfb7174e5f166485_b.png)


----

### **[DTN](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1611.02200.pdf)**

从图片中创建表情符号。



![bg right:40% width:500px](https://cy-1256894686.cos.ap-beijing.myqcloud.com/uPic/v2-84cd8e7a96a15d8e978379447f7ba146_b.png)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/uPic/v2-7ecd94ff06afc7f49bd7a59d7a74f627_b.png)

----

### **纹理合成**

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/uPic/v2-5bf7a7d1671d5676dafaf54ec6cae291_b.png)

----

### **图像编辑 ([IcGAN](https://link.zhihu.com/?target=https%3A//github.com/Guim3/IcGAN))**

重建或编辑具有特定属性的图像。



![width:500px](https://cy-1256894686.cos.ap-beijing.myqcloud.com/uPic/v2-3ec0dc038368d1a492c184a3badd68d7_b.png)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/uPic/v2-36c8a303e4962a68889d1e7355b72932_b.png)

----

### **人脸老化([Age-cGAN](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1702.01983.pdf))**



![width:500px](https://cy-1256894686.cos.ap-beijing.myqcloud.com/uPic/v2-2ab323908286ac437d67c670a437cebf_b.png)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/uPic/v2-0f704fde93036dace842170438070b10_b.png)

---


### **创建动画角色**

众所周知，游戏开发和动画制作成本很高，并且雇佣了许多制作艺术家来完成相对常规的任务。但通过GAN就可以自动生成动画角色并为其上色。

![center width:400px](https://cy-1256894686.cos.ap-beijing.myqcloud.com/uPic/v2-6a78a59e34050647628347fa75601d48_b.png)

使用Generative Adversarial Networks创建自动动画人物角色

----


生成器和判别器由多层卷积层、批标准化和具有跳过链接的relu组成。


![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/uPic/v2-c54205a3c85390dc1e1da15e2cfaf187_b.png)

----

### **神经照片编辑器**

基于内容的图像编辑：例如，扩展发带。



![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/uPic/v2-74439d38f3ced0a65bb80999ece86620_b.png)神经照片编辑

---

### **细化图像**



![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/uPic/v2-3f43dfbe9ee90b5e1bc5df479f13e00e_b.png)



### **目标检测**

这是用gan增强现有解决方案的一个应用程序。



![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/uPic/v2-a3a8daef84d685b1c3a0f506a5e54101_b.png)

---

### **图像融合**

将图像混合在一起。

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/uPic/v2-6c12e0b5e3a897aff8398ea9a3c672ea_b.png)


### **生成三维对象**

这是用gan创建三维对象时经常引用的一篇文章。



![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/uPic/v2-4066e57281af5778ab6db3377119a431_b.png)

----

### **音乐的产生**

GaN可以应用于非图像领域，如作曲。



![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/uPic/v2-40bba7f1a5521675927b7baa4ffdf4d9_b.png)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/uPic/v2-6a56ebb4c3f45e571968ffc4b9fc1928_b.png)

----

### **医疗（异常检测）**

GAN还可以扩展到其他行业，例如医学中的肿瘤检测。


![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/uPic/v2-8e00353b12ba7fc9dd3ee61690b321cf_b.png)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/uPic/v2-17ee72b03b9ca7ffc13780bb05f1881c_b.png)

