# Get started with PyTorch, Cloud TPUs, and Colab

Author:Joe Spisak (PyTorch Product Lead)(转载)

PyTorch aims to make machine learning research fun and interactive by supporting all kinds of cutting-edge hardware accelerators. We [announced](https://www.youtube.com/watch?v=zXAzkqFXclM) support for Cloud TPUs at the 2019 PyTorch Developer Conference, and this blog post shows you how to use a Cloud TPU for free via Colab to speed up your PyTorch programs right in your browser.

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-04-10-042643.png)

All of the code we’ll walk through below is available in [this Colab notebook](https://colab.research.google.com/github/pytorch/xla/blob/master/contrib/colab/getting-started.ipynb) — we recommend opening it up and selecting “Runtime > Run all” right now before you read the rest of this post! Then you can visit the notebook to edit code and re-run cells interactively after you’ve gotten an overview below. (Many thanks to Mike Ruberry from Facebook and Daniel Sohn from Google for putting this notebook together!)

## **Colab notes**

When you select a TPU backend in Colab, which the notebook above does automatically, Colab currently provides you with access to a full Cloud TPU v2 device, which consists of a network-attached CPU host plus four TPU v2 chips with two cores each. You are encouraged to use all eight of these TPU cores, but the introductory notebook above only drives a single core for maximum simplicity. Additional Colab notebooks are available that demonstrate how to use multiple TPU cores, including [this one](https://colab.research.google.com/github/pytorch/xla/blob/master/contrib/colab/mnist-training-xrt-1-15.ipynb#scrollTo=Afwo4H7kSd8P) which trains a network on the MNIST dataset and [this one](https://colab.research.google.com/github/pytorch/xla/blob/master/contrib/colab/resnet18-training-xrt-1-15.ipynb#scrollTo=_2nL4HmloEyl) which trains a ResNet-18 architecture on the CIFAR-10 dataset. You can also find additional Colabs and links to PyTorch Cloud TPU tutorials [here](https://github.com/pytorch/xla/tree/master/contrib/colab). All of these Colab notebooks are intended for people who are already familiar with PyTorch. If you haven’t used PyTorch before, we recommend that you check out the tutorials at https://pytorch.org/ before continuing.

## Technical details

The PyTorch support for Cloud TPUs is achieved via an integration with XLA, a compiler for linear algebra that can target multiple types of hardware, including CPU, GPU, and TPU. You can follow the ongoing development of the PyTorch/XLA integration on GitHub [here](https://github.com/pytorch/xla).

## **Large-scale training**

In addition to supporting individual Cloud TPU devices, PyTorch/XLA supports “slices” of Cloud TPU Pods, which are multi-rack supercomputers that can deliver more than 100 petaflops of sustained performance. You can learn more about scaling up PyTorch training jobs on Cloud TPU Pods [here](https://github.com/pytorch/xla#Pod).

And now, let’s get started with PyTorch on a Cloud TPU via Colab!

# Installing PyTorch/XLA

The PyTorch/XLA package lets PyTorch connect to Cloud TPUs. In particular, PyTorch/XLA makes TPU cores available as PyTorch devices. This lets PyTorch create and manipulate tensors on TPUs.

```python
VERSION = "20200220" #@param ["20200220","nightly", "xrt==1.15.0"]!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py!python pytorch-xla-env-setup.py --version $VERSION
```

# Import the needed libraries

```
# imports pytorch
import torch
# imports the torch_xla package
import torch_xla
import torch_xla.core.xla_model as xm
```

# Creating and Manipulating Tensors on TPUs

PyTorch uses Cloud TPUs just like it uses CPU or CUDA devices, as the next few cells will show. Each core of a Cloud TPU is treated as a different PyTorch device.

```
# Creates a random tensor on xla:1 (a Cloud TPU core)
dev = xm.xla_device()
t1 = torch.ones(3, 3, device = dev)
print(t1)
```

See the documentation at http://pytorch.org/xla/ for a description of all public PyTorch/XLA functions. Here `xm.xla_device()` acquired the first Cloud TPU core ('xla:1'). Other cores can also be directly acquired:

```
# Creating a tensor on the second Cloud TPU core
second_dev = xm.xla_device(n=2, devkind='TPU')
t2 = torch.zeros(3, 3, device = second_dev)
print(t2)
```

It is recommended that you use functions like `xm.xla_device()` over directly specifying TPU cores. A future Colab tutorial will show how to easily train a network using multiple cores (or you can look at [an example](https://colab.research.google.com/github/pytorch/xla/blob/master/contrib/colab/mnist-training-xrt-1-15.ipynb#scrollTo=Afwo4H7kSd8Phttps://)). Tensors on TPUs can be manipulated like any other PyTorch tensor. The following cell adds, multiplies, and matrix multiplies two tensors on a TPU core:

```
a = torch.randn(2, 2, device = dev)
b = torch.randn(2, 2, device = dev)
print(a + b)print(b * 2)
print(torch.matmul(a, b))
```

This next cell runs a 1D convolution on a TPU core:

```
# Creates random filters and inputs to a 1D convolution
filters = torch.randn(33, 16, 3, device = dev)
inputs = torch.randn(20, 16, 50, device = dev)
torch.nn.functional.conv1d(inputs, filters)
```

And tensors can be transferred between CPU and TPU. In the following cell, a tensor on the CPU is copied to a TPU core, and then copied back to the CPU again. Note that PyTorch makes copies of tensors when transferring them across devices, so `t_cpu` and `t_cpu_again` are different tensors.

```
# Creates a tensor on the CPU (device='cpu' is unnecessary and only added for clarity)
t_cpu = torch.randn(2, 2, device='cpu')
print(t_cpu)

t_tpu = t_cpu.to(dev)
print(t_tpu)

t_cpu_again = t_tpu.to('cpu')
print(t_cpu_again)
```

# Running PyTorch Modules and Autograd on TPUs

Modules and autograd are fundamental PyTorch components. In PyTorch, every stateful function is a module. Modules are Python classes augmented with metadata that lets PyTorch understand how to use them in a neural network. For example, linear layers are modules, as are entire networks. Since modules are stateful, they can also be placed onto devices. PyTorch/XLA lets us place them on TPU cores:

```
Creates a linear module
 fc = torch.nn.Linear(5, 2, bias=True)
 
 # Copies the module to the XLA device (the first Cloud TPU core)
 fc = fc.to(dev)
 
 # Creates a random feature tensor
 features = torch.randn(3, 5, device=dev, requires_grad=True)
 
 # Runs and prints the module
 output = fc(features)
 print(output)
```

Autograd is the system PyTorch uses to populate the gradients of weights in a neural network. See [here](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#sphx-glr-beginner-blitz-autograd-tutorial-py) for details about PyTorch’s autograd. When a module is run on a TPU core, its gradients are also populated on the same TPU core by autograd. The following cell demonstrates this:

```
output.backward(torch.ones_like(output))
print(fc.weight.grad)
```

# Running PyTorch Networks on TPUs

As mentioned above, PyTorch networks are also modules, and so they’re run in the same way. The following cell runs a relatively simple PyTorch network from the [PyTorch tutorial docs](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py) on a TPU core:

<iframe src="https://medium.com/media/e4dc51472d651509413ec103f556d11f" allowfullscreen="" frameborder="0" height="1055" width="680" title="tpu10.py" class="s t u ig ai" scrolling="auto" style="box-sizing: inherit; position: absolute; top: 0px; left: 0px; width: 680px; height: 1055px;"></iframe>

As in the previous snippets, running PyTorch on a TPU just requires specifying a TPU core as a device.

# Further learning

Thank you for reading! If you run into any issues with the Colabs above or have ideas on how to improve PyTorch support for Cloud TPUs and Cloud TPU Pods, please file an issue in our [Github repo](https://github.com/pytorch/xla/issues). As a reminder, you can also try out the more advanced Colab notebooks [here](https://github.com/pytorch/xla/tree/master/contrib/colab). Stay tuned for future posts about accelerating PyTorch with Cloud TPUs!