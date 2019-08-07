# ROI

ROIs Pooling顾名思义，是Pooling层的一种，而且是针对RoIs的Pooling，他的特点是**输入特征图**尺寸不固定，但是**输出特征图**尺寸固定；

![è¿éåå¾çæè¿°](http://ww4.sinaimg.cn/large/006tNc79ly1g5owglddd1j315s0dv461.jpg)

## ROI Pooling的输入

输入有两部分组成： 

1. 特征图：指的是图1中所示的特征图，在Fast RCNN中，它位于RoI Pooling之前，在Faster RCNN中，它是与RPN共享那个特征图，通常我们常常称之为“share_conv”； 
2. rois：在Fast RCNN中，指的是Selective Search的输出；在Faster RCNN中指的是RPN的输出，一堆矩形候选框框，形状为1x5x1x1（4个坐标+索引index），其中值得注意的是：坐标的参考系不是针对feature map这张图的，而是针对原图的（神经网络最开始的输入）

## ROI Pooling的输出

输出是batch个vector，其中batch的值等于RoI的个数，vector的大小为channel * w * h；RoI Pooling的过程就是将一个个大小不同的box矩形框，都映射成大小固定（w * h）的矩形框

---

​      **一）、RoIPooling**

​      这个可以在Faster RCNN中使用以便使生成的候选框region proposal映射产生固定大小的feature map

​      先贴出一张图，接着通过这图解释RoiPooling的工作原理   

​      ![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-08-07-031051.png)

​      针对上图

​      1)Conv layers使用的是VGG16，feat_stride=32(即表示，经过网络层后图片缩小为原图的1/32),原图800*800,最后一层特征图feature map大小:25*25

​      2)假定原图中有一region proposal，大小为665*665，这样，映射到特征图中的大小：665/32=20.78,即20.78*20.78，如果你看过Caffe的Roi Pooling的C++源码，在计算的时候会进行取整操作，于是，进行所谓的第一次量化，即映射的特征图大小为20*20

​      3)假定pooled_w=7,pooled_h=7,即pooling后固定成7*7大小的特征图，所以，将上面在 feature map上映射的20*20的 region  proposal划分成49个同等大小的小区域，每个小区域的大小20/7=2.86,即2.86*2.86，此时，进行第二次量化，故小区域大小变成2*2

​      4)每个2*2的小区域里，取出其中最大的像素值，作为这一个区域的‘代表’，这样，49个小区域就输出49个像素值，组成7*7大小的feature map

​     总结，所以，通过上面可以看出，经过两次量化，即将浮点数取整，原本在特征图上映射的20*20大小的region proposal，偏差成大小为14*14的，这样的像素偏差势必会对后层的回归定位产生影响

​     所以，产生了替代方案，RoiAlign

 

​     **二）、RoIAlign**

​      这个是在Mask RCNN中使用以便使生成的候选框region proposal映射产生固定大小的feature map时提出的

​      先贴出一张图，接着通过这图解释RoiAlign的工作原理

​      ![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-08-07-031053.png)

​    同样，针对上图，有着类似的映射

​     1)Conv layers使用的是VGG16，feat_stride=32(即表示，经过网络层后图片缩小为原图的1/32),原图800*800,最后一层特征图feature map大小:25*25

​      2)假定原图中有一region proposal，大小为665*665，这样，映射到特征图中的大小：665/32=20.78,即20.78*20.78，此时，没有像RoiPooling那样就行取整操作，保留浮点数

​      3)假定pooled_w=7,pooled_h=7,即pooling后固定成7*7大小的特征图，所以，将在 feature map上映射的20.78*20.78的region proposal 划分成49个同等大小的小区域，每个小区域的大小20.78/7=2.97,即2.97*2.97

​      4)假定采样点数为4，即表示，对于每个2.97*2.97的小区域，平分四份，每一份取其中心点位置，而中心点位置的像素，采用双线性插值法进行计算，这样，就会得到四个点的像素值，如下图

​      ![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-08-07-031052.png)

​      上图中，四个红色叉叉‘×’的像素值是通过双线性插值算法计算得到的

​       最后，取四个像素值中最大值作为这个小区域(即：2.97*2.97大小的区域)的像素值，如此类推，同样是49个小区域得到49个像素值，组成7*7大小的feature map

​    

​     总结：知道了RoiPooling和RoiAlign实现原理，在以后的项目中可以根据实际情况进行方案的选择；对于检测图片中大目标物体时，两种方案的差别不大，而如果是图片中有较多小目标物体需要检测，则优先选择RoiAlign，更精准些....