作者：ycszen

链接：https://zhuanlan.zhihu.com/p/25980324

来源：知乎

著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

# 加载部分预训练模型

其实大多数时候我们需要根据我们的任务调节我们的模型，所以很难保证模型和公开的模型完全一样，但是预训练模型的参数确实有助于提高训练的准确率，为了结合二者的优点，就需要我们加载部分预训练模型。

```python
pretrained_dict = model_zoo.load_url(model_urls['resnet152'])
model_dict = model.state_dict()
# 将pretrained_dict里不属于model_dict的键剔除掉
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# 更新现有的model_dict
model_dict.update(pretrained_dict)
# 加载我们真正需要的state_dict
model.load_state_dict(model_dict)
```

因为需要剔除原模型中不匹配的键，也就是层的名字，所以我们的新模型改变了的层需要和原模型对应层的名字不一样，比如：resnet最后一层的名字是fc(PyTorch中)，那么我们修改过的resnet的最后一层就不能取这个名字，可以叫fc_

---

## 微改基础模型

PyTorch中的torchvision里已经有很多常用的模型了，可以直接调用：

- AlexNet
- VGG
- ResNet
- SqueezeNet
- DenseNet

```python
import torchvision.models as models
resnet18 = models.resnet18()
alexnet = models.alexnet()
squeezenet = models.squeezenet1_0()
densenet = models.densenet_161()
```

但是对于我们的任务而言有些层并不是直接能用，需要我们微微改一下，比如，resnet最后的全连接层是分1000类，而我们只有21类；又比如，resnet第一层卷积接收的通道是3， 我们可能输入图片的通道是4，那么可以通过以下方法修改：

```python
resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
resnet.fc = nn.Linear(2048, 21)
```

---

