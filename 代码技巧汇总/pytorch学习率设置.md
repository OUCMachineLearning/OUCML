# pytorch在不同的层使用不同的学习率

有时候我们希望某些层的学习率与整个网络有些差别，这里我简单介绍一下在pytorch里如何设置，方法略麻烦，如果有更好的方法，请务必教我：

首先我们定义一个网络：

```python
class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 1)
        self.conv2 = nn.Conv2d(64, 64, 1)
        self.conv3 = nn.Conv2d(64, 64, 1)
        self.conv4 = nn.Conv2d(64, 64, 1)
        self.conv5 = nn.Conv2d(64, 64, 1)
    def forward(self, x):
        out = conv5(conv4(conv3(conv2(conv1(x)))))
        return out
123456789101112
```

我们希望conv5学习率是其他层的100倍，我们可以：

```python
net = net()
lr = 0.001

conv5_params = list(map(id, net.conv5.parameters()))
base_params = filter(lambda p: id(p) not in conv5_params,
                     net.parameters())
optimizer = torch.optim.SGD([
            {'params': base_params},
            {'params': net.conv5.parameters(), 'lr': lr * 100},
, lr=lr, momentum=0.9)12345678910
```

如果多层，则：

```python
conv5_params = list(map(id, net.conv5.parameters()))
conv4_params = list(map(id, net.conv4.parameters()))
base_params = filter(lambda p: id(p) not in conv5_params + conv4_params,
                     net.parameters())
optimizer = torch.optim.SGD([
            {'params': base_params},
            {'params': net.conv5.parameters(), 'lr': lr * 100},
            {'params': net.conv4.parameters(), 'lr': lr * 100},
            , lr=lr, momentum=0.9)
```

