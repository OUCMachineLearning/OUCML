# 最便捷的神经网络可视化工具之一--Flashtorch

【磐创AI 导读】：FlashTorch是PyTorch中用于神经网络的开源特征可视化工具包,本文介绍了如何使用FlashTorch揭示神经网络看到的内容，欢迎大家转发、留言。想要更多电子杂志的机器学习，深度学习资源，大家欢迎点击上方蓝字关注我们的公众号：**磐创AI。**

### **前言**

几周前，我在[http://AnitaB.org](https://link.zhihu.com/?target=http%3A//AnitaB.org)组织的**Hopperx1 London**上发表了演讲作为伦敦科技周的一部分。

在演讲结束后，我收到了热烈的反馈，所以我决定写一个稍微长一点的演讲版本来介绍FlashTorch

该软件包可通过pip安装。Github仓库链接([https://github.com/MisaOgura/flashtorch](https://link.zhihu.com/?target=https%3A//github.com/MisaOgura/flashtorch))。你也可以在Google Colab上运行([https://colab.research.google.com/github/MisaOgura/flashtorch/blob/master/examples/visualise_saliency_with_backprop_colab.ipynb](https://link.zhihu.com/?target=https%3A//colab.research.google.com/github/MisaOgura/flashtorch/blob/master/examples/visualise_saliency_with_backprop_colab.ipynb))，从而无需安装任何东西！
但首先，我将简要介绍一下特征可视化的历史，为你提供更好的背景信息。

### **特征可视化简介**

**特征可视化**是一个活跃的研究领域，旨在探索我们观看"神经网络看到的图像"的方法来了解神经网络如何感知图像。它的出现和发展是为了响应人们越来越希望神经网络能够被人类解读。

最早的工作包括分析输入图像中神经网络正在关注的内容。例如，**图像特定类显著图**(**image-specific class saliency maps**)([https://arxiv.org/abs/1312.6034](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1312.6034))通过反向传播计算相对于输入图像的类的梯度，可视化输入*图像中对对应类别贡献最大的区域*(在后面有更多关于显著图的内容) 。

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-04-26-123216.jpg)

特征可视化中的另一个技术是**激活最大化(Activation maximization)**。这允许我们迭代地更新输入图像(最初由一些随机噪声产生)以生成最大程度地激活目标神经元的图像。它提供了一些关于个体神经元如何响应输入图像的直觉。这就是谷歌推广的所谓的Deep Dream背后的技术。

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-04-26-123205.jpg)

这是一个巨大的进步，但是有一些缺点，因为它没有提供观察整个网络如何运作的能力，因为神经元不是孤立运作。这导致了对神经元之间相互作用的可视化研究。Olah等人通过在两个神经元之间进行添加或插值来演示激活空间的算术性质。

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-04-26-123217.jpg)

然后Olah通过分析在给定特定输入时每个神经元在隐藏层内的贡献，进一步定义了更有意义的可视化单元。观察一组同时被强烈激活的神经元，发现似乎有一组神经元负责捕捉诸如耷拉的耳朵、毛茸茸的腿和草之类的概念。

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-04-26-123215.jpg)

该领域的最新发展之一是Activation Atlas(Carter等，2019)([https://distill.pub/2019/activation-atlas/](https://link.zhihu.com/?target=https%3A//distill.pub/2019/activation-atlas/))。在这项研究中，作者指出了可视化过滤器激活的一个主要缺点，因为它只给出了一个有限的网络如何响应单个输入的视图。为了看到一个大的网络如何感知大量的对象和这些对象之间的联系,他们设计了一种方法，通过显示神经元的常见组合，来创建一个通过神经网络可以看到的全局图。

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-04-26-123208.jpg)

### **FlashTorch实现的动机**

当我发现特征可视化时，我立即被吸引这项技术使神经网络更易于解释的潜力。然后我很快意识到没有工具可以轻松地将这些技术应用到我在PyTorch中构建的神经网络。

所以我决定建立一个FlashTorch，现在它可以通过pip进行安装！我实现的第一个特征可视化技术是**显著图**。

我们将在下面详细介绍哪些显著图，以及如何使用FlashTorch它们与神经网络一起实现它们。

### **显著图**

人类视觉感知中的显着性是一种主观能力，使得视野中的某些事物脱颖而出并引起我们的注意。计算机视觉中的显著图可以指示图像中最显着的区域。

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-04-26-123209.jpg)

从卷积神经网络(CNN)创建显著图的方法最初于2013年在Deep Inside Convolutional Networks：Visualizing Image Classification Models and Saliency Maps([https://arxiv.org/abs/1312.6034](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1312.6034))中引入。作者报告说，通过计算目标类相对于输入图像的梯度，我们可以可视化输入图像中的区域，这些区域对该类的预测值有影响。

### **使用FlashTorch的显着性**

不用多说，让我们自己使用FlashTorch可视化显著图！

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-04-26-123207.jpg)

FlashTorch附带了一些utils功能，使数据处理更容易。我们将以灰色的猫头鹰这个图像为例。

然后我们将对图像应用一些变换，使其形状，类型和值适合作为CNN的输入。

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-04-26-123214.jpg)

我将使用被ImageNet分类数据集进行预训练的AlexNet来进行可视化。事实上，FlashTorch支持所有随torchvision推出的模型，所以我鼓励您也尝试其他模型！

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-04-26-123213.jpg)

该*Backprop*类的核心就是创建显著图。

在实例化时，它接收模型Backprop(model)并将自定义钩子注册到网络中感兴趣的层，以便我们可以从计算图中获取中间梯度以进行可视化。由于PyTorch的设计方式，这些中间梯度并不是立即可用的。FlashTorch会帮你整理。

现在，在计算梯度之前我们需要的最后一件事，目标类索引。

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-04-26-123206.jpg)

回顾一下，我们对目标类相对于输入图像的梯度感兴趣。然而，该模型使用ImageNet数据集进行预训练，因此其预测实际上是1000个类别的概率。我们希望从这1000个值中找出目标类的值(在我们的例子中是灰色的猫头鹰)，以避免不必要的计算，并且只关注输入图像和目标类之间的关系。

为此，我还实现了一个名为ImageNetIndex的类。如果你不想下载整个数据集，只想根据类名找出类索引，这是一个方便的工具。如果给它一个类名，它将找到相应的类索引target_class = imagenet['great grey owl']。如果你确实要下载数据集，请使用最新版本中提供的ImageNet类torchvision==0.3.0。

现在，我们有输入图像和目标类索引(24)，所以我们准备计算梯度！

这两行是关键：

```text
gradients = backprop.calculate_gradients(input_, target_class)

max_gradients = backprop.calculate_gradients(input_, target_class, take_max=True)
```

默认情况下，每个颜色通道都会计算梯度，所以它的s形与我们的输入图像(3,224,224)相同。有时候，如果我们在不同的颜色通道上设置最大的梯度，就可以更容易地看到梯度。我们可以通过将take_max=True参数传递给方法调用来实现这一点。梯度的形状将是(1,224,224)

最后，让我们看看我们得到了什么！

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-04-26-123204.jpg)

我们可以看到，动物所在区域的像素对预测值的影响最大。

但是这是一种噪声信号，它不能告诉我们很多关于神经网络对猫头鹰的感知。

有什么方法可以改善这一点吗

### **通过引导反向传播来改善**

答案是肯定的！

在"Striving for Simplicity: The All Convolutional Net"([https://arxiv.org/abs/1412.6806](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1412.6806))一文中，作者介绍了一种降低梯度计算噪声的方法。

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-04-26-123211.jpg)

实质上，在**引导反向传播**(**Guided backproagation**)中，对目标类的预测值没有影响或有负面影响的神经元被屏蔽并忽略。通过这样做，我们可以防止梯度通过这样的神经元，从而减少噪音。

你可以FlashTorch通过传递guided=True参数来调用方法calculate_gradients来使用带引导的反向传播，如下所示：

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-04-26-123202.jpg)

让我们可视化进行引导后的梯度。

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-04-26-123203.jpg)

差异是惊人的！

现在我们可以清楚地看到网络正在关注凹陷的眼睛和猫头鹰的圆头。这些是说服该神经网络将对象分类为的特征灰色的猫头鹰!

但它并不总是专注于眼睛或头部……

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-04-26-123210.jpg)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-04-26-123212.jpg)

正如你所看到的，网络已经学会专注于特征，这些特征与我们认为这些鸟类最有特色的东西大致相符。

### **特征可视化的应用**

通过特征可视化，我们不仅可以更好地了解神经网络对物体的了解，而且我们还可以更好地：

- 诊断网络出错的原因
- 找出并纠正算法中的偏差
- 从只关注神经网络的精确度向前迈进
- 了解网络行为的原因
- 阐明神经网络如何学习的机制

