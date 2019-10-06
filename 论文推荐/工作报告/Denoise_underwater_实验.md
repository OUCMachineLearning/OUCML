# Denoise_underwater 实验

任务:单纯的水下去噪

---

我们的问题:

UNET 的改进太少

|             | PSNR | SSIM |   备注   |  SN  |  DA  |  TODO  | URL |
| :---------: | :--: | :--: | :--: | :--: | :--: | :------: | :------: |
|    DNGAN    | 29.984474 | 0.920650 | 去海洋雪 |  no  |  no  |  doing  | sftp://222.195.151.21:3900//home/ouc/cy/work_jiang/DNGAN/DNGAN |
|    DNGAN    | 31.468112 | 0.938993 | 去海洋雪 | yes  |  no  |  doing  | sftp://222.195.151.21:3900//home/ouc/cy/work_jiang/DNGAN/DNGAN+sn |
|    DNGAN    | 29.749487 | 0.921075 | 去海洋雪 |  no  | yes  | doing | sftp://222.195.151.21:3900//home/ouc/cy/work_jiang/DNGAN/DNGAN+da |
|    DNGAN    | 31.505804 | 0.936642 | 去海洋雪 | yes  | yes  | doing | sftp://222.195.151.21:3900//home/ouc/cy/work_jiang/DNGAN/DNGAN+sn+da |
| noise2noise | 28.219122 | 0.947879 | 对比实验 |  no  |  no  |  yes  | sftp://222.195.151.21:3902//home/hx/cy/work_jiang/noise2noise-pytorch |
|   ID-CGAN   | 18.677912 | 0.782176 |  对比实验  |  no  |  no | yes | sftp://222.195.151.21:3902//home/hx/cy/work_jiang/Single-Image-De-Raining-Keras |
|             |      |      |          |      |      |      |      |
|             |      |      |          |      |      |      |      |
|             |      |      |          |      |      |      |      |

需要补充的实验

- SN 的作用--YES
- DA 的作用--YES
- 训练稳定性曲线变化--TODO
- 
- 用表格进一步描述网络的具体结构--TODO
- Noise2noise 对比实验—yes
- ID-CGAN对比实验—yes

— 似乎最好的结果比表上的还高,你写的时候查

> /draw_line

>```python
>import sys
>sys.path.append('../')
>from pycore.tikzeng import *
>from pycore.blocks  import *
>
>arch = [ 
>        # 开头
>        to_head('..'), 
>        to_cor(),
>        to_begin(),
>        
>        #　添加输入层
>        
>        to_Conv( name='input_rgb', s_filer=256, n_filer=(3), offset="(0,0,0)", to="(0,0,0)", height=256, depth=256,width=1,caption="Noise IMG"),
>        to_input( 'input.png' ,width=51.2, height=51.2,to='(0,-0.2,-0.5)',name='0'),
>        #  block1
>        to_Conv( name='block1-1', s_filer=256, n_filer=(48), offset="(6,0,0)", to="(input_rgb-east)", height=256, depth=256,width=8,caption="Block1"),
>        to_Conv( name='block1-2', s_filer=128, n_filer=(48), offset="(3,0,0)", to="(block1-1-east)",height=128,depth=128,width=8),
>        to_connection('block1-1', 'block1-2'),
>#        to_connection('being', 'block1-1'),ß	
>        #  block2
>        to_Conv( name='block2-1', s_filer=128, n_filer=(48), offset="(0,0,0)", to="(block1-2-east)", height=128, depth=128,width=8,caption="Block2"),
>        to_Conv( name='block2-2', s_filer=64, n_filer=(48), offset="(3,0,0)", to="(block2-1-east)",height=64,depth=64,width=8),
>        to_connection('block2-1', 'block2-2'),	
>        #  block3
>        to_Conv( name='block3-1', s_filer=64, n_filer=(48), offset="(0,0,0)", to="(block2-2-east)", height=64, depth=64,width=8,caption="Block3"),
>        to_Conv( name='block3-2', s_filer=32, n_filer=(48), offset="(3,0,0)", to="(block3-1-east)",height=32,depth=32,width=8),
>        to_connection("block3-1", "block3-2"),	
>        #  block4
>        to_Conv( name='block4-1', s_filer=32, n_filer=(48), offset="(0,0,0)", to="(block3-2-east)", height=32, depth=32,width=8,caption="Block4"),
>        to_Conv( name='block4-2', s_filer=16, n_filer=(48), offset="(3,0,0)", to="(block4-1-east)",height=16,depth=16,width=8),	
>        to_connection("block4-1", "block4-2"),
>        
>        #  block5
>        to_Conv( name='block5-1', s_filer=16, n_filer=(48), offset="(0,0,0)", to="(block4-2-east)", height=16, depth=16,width=8,caption="Block5"),
>       
>        to_Conv( name='block5-2', s_filer=8, n_filer=(48), offset="(3,0,0)", to="(block5-1-east)",height=8,depth=8,width=8),	
>        to_connection("block5-1", "block5-2"),
>        
>        # unblock6
>        to_Conv( name='block6', s_filer=16, n_filer=(48), offset="(3,0,0)", to="(block5-2-east)",height=16,depth=16,width=8,caption="Block6"),	
>        to_connection('block5-2', 'block6'),
>        # PAM CAM
>        #  block5
>        to_ConvRes( name='PAM', s_filer=16, n_filer=(48), offset="(2,-3,5)", to="(block6-east)", height=16, depth=16,width=8,caption="PAM"),
>        to_ConvRes( name='CAM', s_filer=16, n_filer=(48), offset="(1,3,-5)", to="(block6-east)", height=16, depth=16,width=8,caption="CAM"),
>        to_connection("block6", "CAM"),
>        to_connection("block6", "PAM"),
>        
>         # unblock7
>        *block_Unconv( name="block7", botton="block6", top='block7', s_filer=16,  n_filer=96, offset="(6,0,0)", size=(16,16,16), opacity=0.5,caption='block7' ),
>        to_connection("CAM",'ccr_res_block7'),
>        to_connection("PAM",'ccr_res_block7'),
>        # unblock8
>        *block_Unconv( name="block8", botton="block7", top='block8', s_filer=32,  n_filer=144, offset="(5,0,0)", size=(32,32,24), opacity=0.5,caption='block8'  ),
>        to_connection('block7','block8'),
>        # unblock9
>        *block_Unconv( name="block9", botton="block8", top='block9', s_filer=64,  n_filer=144, offset="(5,0,0)", size=(64,64,24), opacity=0.5 ,caption='block9' ),
>        to_connection('block8', 'block9'),
>        # unblock10
>        *block_Unconv( name="block10", botton='block9', top='block10', s_filer=128,  n_filer=144, offset="(7,0,0)", size=(128,128,24), opacity=0.5,caption='block10'  ),
>        to_connection('block9', 'block10'),
>        # unblock10
>        *block_Unconv( name='block11', botton='block10', top='block11', s_filer=256,  n_filer=144, offset="(7,0,0)", size=(256,256,24), opacity=0.5,caption='block11'  ),
>        to_connection('block10', 'block11'),
>        
>        to_Conv(name='block12',s_filer=256, n_filer=(99), offset="(3,0,0)", to=
>        '(block11-east)', height=256, depth=256,width=16,caption=""),
>        to_Conv(name='block13',s_filer=256, n_filer=(64), offset="(0,0,0)", to='(block12-east)', height=256, depth=256,width=12,caption="Block12"),
>        to_Conv(name='block14',s_filer=256, n_filer=(32), offset="(0,0,0)", to='(block13-east)', height=256, depth=256,width=4,caption=""),
>        to_Conv(name='block15',s_filer=256, n_filer=(3), offset="(3,0,0)", to='(block14-east)', height=256, depth=256,width=1,caption="TO RGB"),
>        to_connection('block11', 'block15'),
>        to_output('output.png',width=51.2, height=51.2,to='(0,-0.2,-0.5)',offset="137.2"),
>        to_skip('block5-1', 'ccr_res_{}'.format('block7'),pos=3),
>        to_skip('block4-1', 'ccr_res_{}'.format('block8'),pos=2.35),
>        to_skip('block3-1', 'ccr_res_{}'.format('block9'),pos=1.55),
>        to_skip('block2-1', 'ccr_res_{}'.format('block10'),pos=1.25),
>        to_skip('block1-1', 'ccr_res_{}'.format('block11'),pos=1),
>        to_skip('input_rgb', 'block12',pos=1.25),
>#        to_connection("soft1","output"),
>        #  结束
>        to_end(), 
>        ]
>
>
>def main():
>        namefile = str(sys.argv[0]).split('.')[0]
>        to_generate(arch, namefile + '.tex' )
>
>if __name__ == '__main__':
>        main()
>        
>
>```
>
>
>
>```
>device =  cuda
>----------------------------------------------------------------
>        Layer (type)               Output Shape         Param #
>================================================================
>            Conv2d-1         [-1, 48, 256, 256]           1,344
>    InstanceNorm2d-2         [-1, 48, 256, 256]               0
>         LeakyReLU-3         [-1, 48, 256, 256]               0
>            Conv2d-4         [-1, 48, 128, 128]          20,784
>    InstanceNorm2d-5         [-1, 48, 128, 128]               0
>         LeakyReLU-6         [-1, 48, 128, 128]               0
>            Conv2d-7         [-1, 48, 128, 128]          20,784
>    InstanceNorm2d-8         [-1, 48, 128, 128]               0
>         LeakyReLU-9         [-1, 48, 128, 128]               0
>           Conv2d-10           [-1, 48, 64, 64]          20,784
>   InstanceNorm2d-11           [-1, 48, 64, 64]               0
>        LeakyReLU-12           [-1, 48, 64, 64]               0
>           Conv2d-13           [-1, 48, 64, 64]          20,784
>   InstanceNorm2d-14           [-1, 48, 64, 64]               0
>        LeakyReLU-15           [-1, 48, 64, 64]               0
>           Conv2d-16           [-1, 48, 32, 32]          20,784
>   InstanceNorm2d-17           [-1, 48, 32, 32]               0
>        LeakyReLU-18           [-1, 48, 32, 32]               0
>           Conv2d-19           [-1, 48, 32, 32]          20,784
>   InstanceNorm2d-20           [-1, 48, 32, 32]               0
>        LeakyReLU-21           [-1, 48, 32, 32]               0
>           Conv2d-22           [-1, 48, 16, 16]          20,784
>   InstanceNorm2d-23           [-1, 48, 16, 16]               0
>        LeakyReLU-24           [-1, 48, 16, 16]               0
>           Conv2d-25           [-1, 48, 16, 16]          20,784
>   InstanceNorm2d-26           [-1, 48, 16, 16]               0
>        LeakyReLU-27           [-1, 48, 16, 16]               0
>           Conv2d-28             [-1, 48, 8, 8]          20,784
>   InstanceNorm2d-29             [-1, 48, 8, 8]               0
>        LeakyReLU-30             [-1, 48, 8, 8]               0
>           Conv2d-31             [-1, 48, 8, 8]          20,784
>   InstanceNorm2d-32             [-1, 48, 8, 8]               0
>        LeakyReLU-33             [-1, 48, 8, 8]               0
>         Upsample-34           [-1, 48, 16, 16]               0
>           Conv2d-35           [-1, 48, 16, 16]          20,784
>   InstanceNorm2d-36           [-1, 48, 16, 16]               0
>        LeakyReLU-37           [-1, 48, 16, 16]               0
>           Conv2d-38           [-1, 96, 16, 16]          83,040
>   InstanceNorm2d-39           [-1, 96, 16, 16]               0
>        LeakyReLU-40           [-1, 96, 16, 16]               0
>           Conv2d-41           [-1, 96, 16, 16]          83,040
>   InstanceNorm2d-42           [-1, 96, 16, 16]               0
>        LeakyReLU-43           [-1, 96, 16, 16]               0
>         Upsample-44           [-1, 96, 32, 32]               0
>           Conv2d-45           [-1, 96, 32, 32]          83,040
>   InstanceNorm2d-46           [-1, 96, 32, 32]               0
>        LeakyReLU-47           [-1, 96, 32, 32]               0
>           Conv2d-48           [-1, 96, 32, 32]         124,512
>   InstanceNorm2d-49           [-1, 96, 32, 32]               0
>        LeakyReLU-50           [-1, 96, 32, 32]               0
>           Conv2d-51           [-1, 96, 32, 32]          83,040
>   InstanceNorm2d-52           [-1, 96, 32, 32]               0
>        LeakyReLU-53           [-1, 96, 32, 32]               0
>         Upsample-54           [-1, 96, 64, 64]               0
>           Conv2d-55           [-1, 96, 64, 64]          83,040
>   InstanceNorm2d-56           [-1, 96, 64, 64]               0
>        LeakyReLU-57           [-1, 96, 64, 64]               0
>           Conv2d-58           [-1, 96, 64, 64]         124,512
>   InstanceNorm2d-59           [-1, 96, 64, 64]               0
>        LeakyReLU-60           [-1, 96, 64, 64]               0
>           Conv2d-61           [-1, 96, 64, 64]          83,040
>   InstanceNorm2d-62           [-1, 96, 64, 64]               0
>        LeakyReLU-63           [-1, 96, 64, 64]               0
>         Upsample-64         [-1, 96, 128, 128]               0
>           Conv2d-65         [-1, 96, 128, 128]          83,040
>   InstanceNorm2d-66         [-1, 96, 128, 128]               0
>        LeakyReLU-67         [-1, 96, 128, 128]               0
>           Conv2d-68         [-1, 96, 128, 128]         124,512
>   InstanceNorm2d-69         [-1, 96, 128, 128]               0
>        LeakyReLU-70         [-1, 96, 128, 128]               0
>           Conv2d-71         [-1, 96, 128, 128]          83,040
>   InstanceNorm2d-72         [-1, 96, 128, 128]               0
>        LeakyReLU-73         [-1, 96, 128, 128]               0
>         Upsample-74         [-1, 96, 256, 256]               0
>           Conv2d-75         [-1, 96, 256, 256]          83,040
>   InstanceNorm2d-76         [-1, 96, 256, 256]               0
>        LeakyReLU-77         [-1, 96, 256, 256]               0
>           Conv2d-78         [-1, 64, 256, 256]          57,088
>   InstanceNorm2d-79         [-1, 64, 256, 256]               0
>        LeakyReLU-80         [-1, 64, 256, 256]               0
>           Conv2d-81         [-1, 32, 256, 256]          18,464
>   InstanceNorm2d-82         [-1, 32, 256, 256]               0
>        LeakyReLU-83         [-1, 32, 256, 256]               0
>           Conv2d-84          [-1, 3, 256, 256]             867
>        LeakyReLU-85          [-1, 3, 256, 256]               0
>================================================================
>Total params: 1,427,283
>Trainable params: 1,427,283
>Non-trainable params: 0
>----------------------------------------------------------------
>Input size (MB): 0.75
>Forward/backward pass size (MB): 617.95
>Params size (MB): 5.44
>Estimated Total Size (MB): 624.15
>----------------------------------------------------------------
>
>```
>
>![image-20191006133054963](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-10-06-053055.png)



---

>```
>import sys
>import os
>sys.path.append('../')
>from pycore.tikzeng import *
>from pycore.blocks  import *
>
>arch = [ 
>	# 开头
>	to_head('..'), 
>	to_cor(),
>	to_begin(),
>	#　添加输入层
>	to_input( 'output.png',width=13, height=13),
>	#  CRCRP
>	to_ConvConvRelu( name='Conv2d-1', s_filer=256, n_filer=(64,64), offset="(0,0,0)", to="(0,0,0)", width=(2,2), height=64, depth=64,caption="CRCRM"),
>	to_Pool(name="MaxPool2d-5", offset="(0,0,0)", to="(Conv2d-1-east)", width=1, height=32, depth=32, opacity=0.5),
>	#  CRCRP
>	to_ConvConvRelu( name='Conv2d-6', s_filer=128, n_filer=(128,128), offset="(2,0,0)", to="(MaxPool2d-5-east)", width=(4,4), height=32, depth=32,caption="CRCRM"),
>	to_Pool(name="MaxPool2d-10", offset="(0,0,0)", to="(Conv2d-6-east)", width=1, height=16, depth=16, opacity=0.5),	
>	to_connection("MaxPool2d-5", "Conv2d-6"),    
>	#  CRCRCRCRP
>	to_ConvConvRelu( name='Conv2d-11', s_filer=64, n_filer=(256,256), offset="(2,0,0)", to="(MaxPool2d-10-east)", width=(8,8), height=16, depth=16,caption="CRCRCRCRM"),
>	to_ConvConvRelu( name='Conv2d-15', s_filer=64, n_filer=(256,256), offset="(0,0,0)", to="(Conv2d-11-east)", width=(8,8), height=16, depth=16,caption=""),
>	to_Pool(name="MaxPool2d-19", offset="(0,0,0)", to="(Conv2d-15-east)", width=1, height=8, depth=8, opacity=0.5),	
>	to_connection("MaxPool2d-10", "Conv2d-11"), 
>	#  CRCRCRCRP
>	to_ConvConvRelu( name='Conv2d-20', s_filer=32, n_filer=(512,512), offset="(2,0,0)", to="(MaxPool2d-19-east)", width=(16,16), height=8, depth=8,caption="CRCRCRCRM"),
>	to_ConvConvRelu( name='Conv2d-24', s_filer=32, n_filer=(512,512), offset="(0,0,0)", to="(Conv2d-20-east)", width=(16,16), height=8, depth=8,caption=""),
>	to_Pool(name="MaxPool2d-28", offset="(0,0,0)", to="(Conv2d-24-east)", width=1, height=4, depth=4, opacity=0.5),	
>	to_connection("MaxPool2d-19", "Conv2d-20"),
>	#  CRCRCR
>	to_ConvConvRelu( name='Conv2d-29', s_filer=32, n_filer=(512,512), offset="(2,0,0)", to="(MaxPool2d-28-east)", width=(16,16), height=4, depth=4,caption="CRCRCRC"),
>	to_Conv( name='Conv2d-34', s_filer=64, n_filer=(512), offset="(0,0,0)", to="(Conv2d-29-east)", width=(16), height=4, depth=4,caption=""),	
>	to_connection("MaxPool2d-28", "Conv2d-29"),
>	# output_feature
>	to_Conv( name='Conv2d-35', s_filer=64, n_filer=(512), offset="(2,0,0)", to="(Conv2d-34-east)", width=(16), height=4, depth=4,caption="VGG-Feature"),
>	to_connection("Conv2d-34", "Conv2d-35"),
>	#  结束
>	to_end(), 
>	]
>
>
>def main():
>	namefile = str(sys.argv[0]).split('.')[0]
>	to_generate(arch, namefile + '.tex' )
>
>if __name__ == '__main__':
>	main()
>
>```
>
>```
>
>device =  cuda
>----------------------------------------------------------------
>				Layer (type)               Output Shape         Param #
>================================================================
>						Conv2d-1         [-1, 64, 256, 256]           1,792
>							ReLU-2         [-1, 64, 256, 256]               0
>						Conv2d-3         [-1, 64, 256, 256]          36,928
>							ReLU-4         [-1, 64, 256, 256]               0
>				 MaxPool2d-5         [-1, 64, 128, 128]               0
>						Conv2d-6        [-1, 128, 128, 128]          73,856
>							ReLU-7        [-1, 128, 128, 128]               0
>						Conv2d-8        [-1, 128, 128, 128]         147,584
>							ReLU-9        [-1, 128, 128, 128]               0
>				MaxPool2d-10          [-1, 128, 64, 64]               0
>					 Conv2d-11          [-1, 256, 64, 64]         295,168
>						 ReLU-12          [-1, 256, 64, 64]               0
>					 Conv2d-13          [-1, 256, 64, 64]         590,080
>						 ReLU-14          [-1, 256, 64, 64]               0
>					 Conv2d-15          [-1, 256, 64, 64]         590,080
>						 ReLU-16          [-1, 256, 64, 64]               0
>					 Conv2d-17          [-1, 256, 64, 64]         590,080
>						 ReLU-18          [-1, 256, 64, 64]               0
>				MaxPool2d-19          [-1, 256, 32, 32]               0
>					 Conv2d-20          [-1, 512, 32, 32]       1,180,160
>						 ReLU-21          [-1, 512, 32, 32]               0
>					 Conv2d-22          [-1, 512, 32, 32]       2,359,808
>						 ReLU-23          [-1, 512, 32, 32]               0
>					 Conv2d-24          [-1, 512, 32, 32]       2,359,808
>						 ReLU-25          [-1, 512, 32, 32]               0
>					 Conv2d-26          [-1, 512, 32, 32]       2,359,808
>						 ReLU-27          [-1, 512, 32, 32]               0
>				MaxPool2d-28          [-1, 512, 16, 16]               0
>					 Conv2d-29          [-1, 512, 16, 16]       2,359,808
>						 ReLU-30          [-1, 512, 16, 16]               0
>					 Conv2d-31          [-1, 512, 16, 16]       2,359,808
>						 ReLU-32          [-1, 512, 16, 16]               0
>					 Conv2d-33          [-1, 512, 16, 16]       2,359,808
>						 ReLU-34          [-1, 512, 16, 16]               0
>					 Conv2d-35          [-1, 512, 16, 16]       2,359,808
>================================================================
>Total params: 20,024,384
>Trainable params: 20,024,384
>Non-trainable params: 0
>----------------------------------------------------------------
>Input size (MB): 0.75
>Forward/backward pass size (MB): 310.00
>Params size (MB): 76.39
>Estimated Total Size (MB): 387.14
>----------------------------------------------------------------
>```
>
>![image-20191006132918585](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-10-06-053006.png)

>```python
>import sys
>sys.path.append('../')
>from pycore.tikzeng import *
>from pycore.blocks  import *
># defined your arch
>arch = [
>	to_head( '..' ),
>	to_cor(),
>	to_begin(),
>	#　添加输入层
>	to_input( 'output.png',width=13,height=13 ),
>	# CL
>	to_Conv( name='Conv2d-1', s_filer=256, n_filer=(16), offset="(0,0,0)", to="(0,0,0)", width=(2), height=64, depth=64,caption="C"), 
>	# CSICSI
>	to_ConvConvRelu( name='Conv2d-3', s_filer=128, n_filer=(16,32), offset="(2,0,0)", to="(Conv2d-1-east)", width=(2,4), height=64, depth=64,caption="CSICSI"),
>	to_connection("Conv2d-1", "Conv2d-3"), 
>	# CSICSI
>	to_ConvConvRelu( name='Conv2d-9', s_filer=64, n_filer=(32,64), offset="(2,0,0)", to="(Conv2d-3-east)", width=(4,8), height=32, depth=32,caption="CSICSI"),
>	to_connection("Conv2d-3", "Conv2d-9"),
>	# CSICSI
>	to_ConvConvRelu( name='Conv2d-15', s_filer=32, n_filer=(64,64), offset="(2,0,0)", to="(Conv2d-9-east)", width=(8,8), height=16, depth=16,caption="CSICSI"),
>	to_connection("Conv2d-9", "Conv2d-15"),
>	# CSI
>	to_Conv( name='Conv2d-21', s_filer=16, n_filer=(64), offset="(2,0,0)", to="(Conv2d-15-east)", width=(8), height=8, depth=8,caption="CSI"),
>	to_connection("Conv2d-15", "Conv2d-21"),
>	# output_patch   
>	to_Conv(name='patch', s_filer=16, n_filer=(1), offset="(2,0,0)", to="(Conv2d-21-east)", width=(1), height=8, depth=8,caption="patch"),
>	to_connection("Conv2d-21", "patch"),
>	to_end(),
>	]
>
>def main():
>	namefile = str(sys.argv[0]).split('.')[0]
>	to_generate(arch, namefile + '.tex' )
>
>if __name__ == '__main__':
>	main()
>```
>
>```
>device =  cuda
>----------------------------------------------------------------
>		Layer (type)               Output Shape         Param #
>================================================================
>			Conv2d-1         [-1, 16, 256, 256]             448
>		 LeakyReLU-2         [-1, 16, 256, 256]               0
>			Conv2d-3         [-1, 16, 128, 128]           2,320
>	InstanceNorm2d-4         [-1, 16, 128, 128]               0
>		 LeakyReLU-5         [-1, 16, 128, 128]               0
>			Conv2d-6         [-1, 32, 128, 128]           4,640
>	InstanceNorm2d-7         [-1, 32, 128, 128]               0
>		 LeakyReLU-8         [-1, 32, 128, 128]               0
>			Conv2d-9           [-1, 32, 64, 64]           9,248
>   InstanceNorm2d-10           [-1, 32, 64, 64]               0
>		LeakyReLU-11           [-1, 32, 64, 64]               0
>		   Conv2d-12           [-1, 64, 64, 64]          18,496
>   InstanceNorm2d-13           [-1, 64, 64, 64]               0
>		LeakyReLU-14           [-1, 64, 64, 64]               0
>		   Conv2d-15           [-1, 64, 32, 32]          36,928
>   InstanceNorm2d-16           [-1, 64, 32, 32]               0
>		LeakyReLU-17           [-1, 64, 32, 32]               0
>		   Conv2d-18           [-1, 64, 32, 32]          36,928
>   InstanceNorm2d-19           [-1, 64, 32, 32]               0
>		LeakyReLU-20           [-1, 64, 32, 32]               0
>		   Conv2d-21           [-1, 64, 16, 16]          36,928
>   InstanceNorm2d-22           [-1, 64, 16, 16]               0
>		LeakyReLU-23           [-1, 64, 16, 16]               0
>		   Conv2d-24            [-1, 1, 16, 16]             577
>================================================================
>Total params: 146,513
>Trainable params: 146,513
>Non-trainable params: 0
>----------------------------------------------------------------
>Input size (MB): 0.75
>Forward/backward pass size (MB): 46.38
>Params size (MB): 0.56
>Estimated Total Size (MB): 47.69
>----------------------------------------------------------------
>```
>
>![image-20191006133214824](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-10-06-053215.png)
>
>

