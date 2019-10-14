# torchprof

A minimal dependency library for layer-by-layer profiling of Pytorch models.

All metrics are derived using the PyTorch autograd profiler.

## Quickstart

```
pip install torchprof
import torch
import torchvision
import torchprof

model = torchvision.models.alexnet(pretrained=False).cuda()
x = torch.rand([1, 3, 224, 224]).cuda()

with torchprof.Profile(model, use_cuda=True) as prof:
    model(x)

print(prof.display(show_events=False)) # equivalent to `print(prof)` and `print(prof.display())`
Module         | Self CPU total | CPU total | CUDA total
---------------|----------------|-----------|-----------
AlexNet        |                |           |
├── features   |                |           |
│├── 0         |        1.956ms |   7.714ms |    7.787ms
│├── 1         |       68.880us |  68.880us |   69.632us
│├── 2         |       85.639us | 155.948us |  155.648us
│├── 3         |      253.419us | 970.386us |    1.747ms
│├── 4         |       18.919us |  18.919us |   19.584us
│├── 5         |       30.910us |  54.900us |   55.296us
│├── 6         |      132.839us | 492.367us |  652.192us
│├── 7         |       17.990us |  17.990us |   18.432us
│├── 8         |       87.219us | 310.776us |  552.544us
│├── 9         |       17.620us |  17.620us |   17.536us
│├── 10        |       85.690us | 303.120us |  437.248us
│├── 11        |       17.910us |  17.910us |   18.400us
│└── 12        |       29.239us |  51.488us |   52.288us
├── avgpool    |       49.230us |  85.740us |   88.960us
└── classifier |                |           |
 ├── 0         |      626.236us |   1.239ms |    1.362ms
 ├── 1         |      235.669us | 235.669us |  635.008us
 ├── 2         |       17.990us |  17.990us |   18.432us
 ├── 3         |       31.890us |  56.770us |   57.344us
 ├── 4         |       39.280us |  39.280us |  212.128us
 ├── 5         |       16.800us |  16.800us |   17.600us
 └── 6         |       38.459us |  38.459us |   79.872us
```

To see the low level operations that occur within each layer, print the contents of `prof.display(show_events=True)`.

```
Module                        | Self CPU total | CPU total | CUDA total
------------------------------|----------------|-----------|-----------
AlexNet                       |                |           |
├── features                  |                |           |
│├── 0                        |                |           |
││├── conv2d                  |       15.740us |   1.956ms |    1.972ms
││├── convolution             |       12.000us |   1.940ms |    1.957ms
││├── _convolution            |       36.590us |   1.928ms |    1.946ms
││├── contiguous              |        6.600us |   6.600us |    6.464us
││└── cudnn_convolution       |        1.885ms |   1.885ms |    1.906ms
│├── 1                        |                |           |
││└── relu_                   |       68.880us |  68.880us |   69.632us
│├── 2                        |                |           |
││├── max_pool2d              |       15.330us |  85.639us |   84.992us
││└── max_pool2d_with_indices |       70.309us |  70.309us |   70.656us
│├── 3                        |                |           |
...
```

The original [Pytorch EventList](https://pytorch.org/docs/stable/autograd.html#torch.autograd.profiler.profile) can be returned by calling `raw()` on the profile instance.

```
trace, event_lists_dict = prof.raw()
print(trace[2])
# Trace(path=('AlexNet', 'features', '0'), leaf=True, module=Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)))

print(event_lists_dict[trace[2].path][0])
---------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------
Name                   Self CPU total %   Self CPU total      CPU total %        CPU total     CPU time avg     CUDA total %       CUDA total    CUDA time avg  Number of Calls
---------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------
conv2d                           0.80%         15.740us          100.00%          1.956ms          1.956ms           25.32%          1.972ms          1.972ms                1
convolution                      0.61%         12.000us           99.20%          1.940ms          1.940ms           25.14%          1.957ms          1.957ms                1
_convolution                     1.87%         36.590us           98.58%          1.928ms          1.928ms           24.99%          1.946ms          1.946ms                1
contiguous                       0.34%          6.600us            0.34%          6.600us          6.600us            0.08%          6.464us          6.464us                1
cudnn_convolution               96.37%          1.885ms           96.37%          1.885ms          1.885ms           24.47%          1.906ms          1.906ms                1
---------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------
Self CPU time total: 1.956ms
CUDA time total: 7.787ms
```

Layers can be selected for individually using the optional `paths` kwarg. Profiling is ignored for all other layers.

```
model = torchvision.models.alexnet(pretrained=False)
x = torch.rand([1, 3, 224, 224])

# Layer does not have to be a leaf layer
paths = [("AlexNet", "features", "3"), ("AlexNet", "classifier")]

with torchprof.Profile(model, paths=paths) as prof:
    model(x)

print(prof)
Module         | Self CPU total | CPU total | CUDA total
---------------|----------------|-----------|-----------
AlexNet        |                |           |           
├── features   |                |           |           
│├── 0         |                |           |           
│├── 1         |                |           |           
│├── 2         |                |           |           
│├── 3         |        2.846ms |  11.368ms |    0.000us
│├── 4         |                |           |           
│├── 5         |                |           |           
│├── 6         |                |           |           
│├── 7         |                |           |           
│├── 8         |                |           |           
│├── 9         |                |           |           
│├── 10        |                |           |           
│├── 11        |                |           |           
│└── 12        |                |           |           
├── avgpool    |                |           |           
└── classifier |       12.016ms |  12.206ms |    0.000us
 ├── 0         |                |           |           
 ├── 1         |                |           |           
 ├── 2         |                |           |           
 ├── 3         |                |           |           
 ├── 4         |                |           |           
 ├── 5         |                |           |           
 └── 6         |                |           |           
```

- [Self CPU Time vs CPU Time](https://software.intel.com/en-us/vtune-amplifier-help-self-time-and-total-time)