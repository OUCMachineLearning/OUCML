加载部分预训练模型

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

微改基础模型

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

逐层解冻(我在 ECCV2020 上提出的训练方法)

加载一个FeatureExtractor

以 densenet 的 dense block 为例:

```python
dense = densenet121(pretrained=True)
self.loss_network = nn.Sequential(*list(dense.features)[:5])
self.loss_network.apply(add_sn)
for param in self.loss_network.parameters():
param.requires_grad = False

# vgg=vgg16(pretrained=True)
# self.loss_network = nn.Sequential(*list(vgg.features)[:16]).eval()
# self.loss_network.apply(add_sn)
# for param in self.loss_network.parameters():
#     param.requires_grad = True
```



设置一个 list,储存冻结的层,方便后期解冻

```python
self.grad = []
for name, value in self.loss_network.named_parameters():
   	print('name: {0},\t grad: {1}'.format(name, value.requires_grad))
if value.requires_grad == False:
  	self.grad.append(name)
```

解冻,unfreeze1 函数(自底向上解冻一层)

```
def unfreeze1(self):
		if len(self.grad)==0:
				print("All of network have been unfreeze!")
    		return
 		self.grad.pop()
  	print("-------------------")
  	for name, value in self.loss_network.named_parameters():
    		if name not in self.grad:
      			value.requires_grad = True
		for name, value in self.loss_network.named_parameters():
  		print('name: {0},\t grad: {1}'.format(name, value.requires_grad))
		print("---------------------")
```

展示冻结情况:

```
def showfreeze(self):
		for i in self.grad:
    		print(i)
		print("-------------------")
```

训练中解冻

```
network.unfreeze1()
```

