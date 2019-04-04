论文题目：Attentive Generative Adversarial Network for Raindrop Removal from A Single Image

论文下载地址：<https://arxiv.org/abs/1711.10098>

Github代码：<https://github.com/MaybeShewill-CV/attentive-gan-derainnet>

数据集下载地址： <https://drive.google.com/drive/folders/1e7R76s6vwUJxILOcAsthgDLPSnOrQ49K>

![img](https://ws3.sinaimg.cn/large/006tNc79ly1g1r0yqhucxj30bs0dsq43.jpg)

这是一篇结合attention机制和GAN去除雨滴的文章。从上图可以看出这个方法去除雨滴的效果非常好。下面介绍一下这个网络。

![img](https://ws4.sinaimg.cn/large/006tNc79ly1g1r0owakpmj30n408wtd5.jpg) 

该网络共包含三个部分，分别为： Attention-recurrent Network，Context Autoencoder 和 Dicriminator Network。前两个部分构成了生成器。

第一部分主要的工作是做检测（即检测雨滴在图片中的位置），然后生成 attention map，它由先前的图片中得出，每一层由若干个Residual Block，一个LSTM和一个Convs组成。需要说明的是，图中的每一个“Block”由5层ResNet组成，其作用是得到输入图片的精确特征与前一个Block的模。具体做法是首先使用 Residual block 从雨滴图片中抽取 feature，渐进式地使用 Convs 来检测 attentive 的区域。训练数据集中图片都是成对的，所以可以很容易计算出相应的 mask（M），由此可以构建出 Loss 函数；由于不同的 attention 网络刻画 feature 的准确度不同，所以给每个 loss 一个指数的衰减。相应的 loss 函数如下：

![img](https://ws3.sinaimg.cn/large/006tNc79ly1g1r0oxrp4ij30au01uq33.jpg) 

![img](https://ws1.sinaimg.cn/large/006tNc79ly1g1r0ox5vdyj301y00w0si.jpg)![img](https://ws1.sinaimg.cn/large/006tNc79ly1g1r0ou3qvcj306u0123yh.jpg) 

LSTM公式如下：

![img](https://ws3.sinaimg.cn/large/006tNc79ly1g1r0ovti2ej30em04a75n.jpg) 

![img](https://ws1.sinaimg.cn/large/006tNc79ly1g1r0oyx9txj30is0aqq6r.jpg) 

随后将 attention map 和雨滴图像一起送给 autoencoder，生成去雨滴的超分辨率图像。由于attention map的给出，这一部分的工作类似于将attention map中attention值较高的部分通过该部分周围的图片信息形成的新的色块进行替换，从而实现图片信息的还原。autoencoder 的结构用了 16 个 Con和 Relu。为了避免网络本身造成的 blur，作者使用了 skip connection，因为在低级层次会带来很好的效果。在构建 loss 方面，除了多尺度的考虑，还加上了一个高精度的 loss，分别是：Multi-scale loss 和perceptual loss.

Multi-scale loss:

 ![img](https://ws2.sinaimg.cn/large/006tNc79ly1g1r0ozo3klj308e01o0ss.jpg)

Perceptual loss:

![img](https://ws2.sinaimg.cn/large/006tNc79ly1g1r0p1fnmtj30ak010mx9.jpg) 

![img](https://ws4.sinaimg.cn/large/006tNc79ly1g1r0p0wxesj303000sa9w.jpg) 

结合两个部分，得到生成器的损失，如下：

![img](https://ws4.sinaimg.cn/large/006tNc79ly1g1r0p03lm1j30aa01udg5.jpg) 

![img](https://ws3.sinaimg.cn/large/006tNc79ly1g1r0p0gef1j307w016t8p.jpg) 

![img](https://ws1.sinaimg.cn/large/006tNc79ly1g1r0p3dvjyj30gs05kwfn.jpg) 

最后一个是 discriminator。这个步骤有两部分，一种是只使用 autoencoder 生成的无雨滴图像，进行判断；另一种则是加入 attention map 作为指导。

判别器的损失函数是：


![img](https://ws1.sinaimg.cn/large/006tNc79ly1g1r0p25o58j30ca01qdg6.jpg)

![img](https://ws3.sinaimg.cn/large/006tNc79ly1g1r0p2wua1j309m01q0sy.jpg) 

综上，整体损失为：

![img](https://ws3.sinaimg.cn/large/006tNc79ly1g1r0owpms7j30b402awet.jpg) 

 下面说一下实验部分。

生成器中残差块的实现：

```python
 def _residual_block(self, input_tensor, name):  
    """ 
    attentive recurrent net中的residual block 
    :param input_tensor: 
    :param name: 
    :return: 
    """  
    output = None  
    with tf.variable_scope(name):  
         inputs = input_tensor  
         shortcut = input_tensor  
         for i in range(5):  
             if i == 0:  
                 inputs = self.conv2d(inputdata=inputs,out_channel=32,
 nel_size=3,padding='SAME',stride=1,use_bias=False,
 e='block_{:d}_conv_1'.format(i))  
                 # TODO reimplement residual block  
                 inputs = self.lrelu(inputdata=inputs, name='block_{:d}_relu_1'.format(i + 1))  
                 output = inputs  
                 shortcut = output  
             else:  
                 inputs = self.conv2d(inputdata=inputs,out_channel=32,
                         kernel_size=1, padding='SAME', stride=1, 
                         use_bias=False, name='block_{:d}_conv_1'.format(i))  
                 inputs = self.lrelu(inputdata=inputs, name='block_{:d}_conv_1'.format(i + 1))  
                 inputs = self.conv2d(inputdata=inputs, out_channel=32,  
                          kernel_size=1, padding='SAME', stride=1,  
                          use_bias=False, name='block_{:d}_conv_2'.format(i))  
                 inputs = self.lrelu(inputdata=inputs, name='block_{:d}_conv_2'.format(i + 1))  
   
                 output = self.lrelu(inputdata=tf.add(inputs, shortcut),  
                                     name='block_{:d}_add'.format(i))  
                 shortcut = output  
     return output  
```



## 注意力机制的实现：

```python
 def build_attentive_rnn(self, input_tensor, name, reuse=False)
    """
    Generator的attentive recurrent部分, 主要是为了找到attention
    :param input_tenso
    :param nam
    :param reus
    :retur
    """
    [batch_size, tensor_h, tensor_w, _] = input_tensor.get_shape().as_lis
     with tf.variable_scope(name, reuse=reuse)
         init_attention_map = tf.constant(0.5, dtype=tf.float32
                                 shape=[batch_size, tensor_h, tensor_w, 1]
         init_cell_state = tf.constant(0.0, dtype=tf.float32
                              shape=[batch_size, tensor_h, tensor_w, 32]
         init_lstm_feats = tf.constant(0.0, dtype=tf.float32
                              shape=[batch_size, tensor_h, tensor_w, 32]
         attention_map_list = [

         **for** i **in** range(4)
             attention_input = tf.concat((input_tensor, init_attention_map), axis=-1
             conv_feats = self._residual_block(input_tensor=attention_input,                                     name='residual_block_{:d}'.format(i + 1)
             lstm_ret = self._conv_lstm(input_tensor=conv_feats
                                   input_cell_state=init_cell_state
                                  name='conv_lstm_block_{:d}'.format(i +
             init_attention_map = lstm_ret['attention_map'
             init_cell_state = lstm_ret['cell_state'
             init_lstm_feats = lstm_ret['lstm_feats'
             attention_map_list.append(lstm_ret['attention_map']
     ret = 
         'final_attention_map': init_attention_map
         'final_lstm_feats': init_lstm_feats
         'attention_map_list': attention_map_lis
     
     **return** ret  
```



## **自编码器的实现：**

```python
def build_autoencoder(self, input_tensor, name, reuse=False):  

        """  
        Generator的autoencoder部分, 负责获取图像上下文信息  
        :param input_tensor:  
        :param name:  
        :param reuse:  
        :**return**:  
        """  
        with tf.variable_scope(name, reuse=reuse):  
			conv_1 = self.conv2d(inputdata=input_tensor, out_channel=64, kernel_size=5,  padding='SAME', stride=1, use_bias=False, name='conv_1')  
			relu_1 = self.lrelu(inputdata=conv_1, name='relu_1')  
			conv_2 = self.conv2d(inputdata=relu_1, out_channel=128, kernel_size=3, padding='SAME', stride=2, use_bias=False, name='conv_2')  
			relu_2 = self.lrelu(inputdata=conv_2, name='relu_2')  
			conv_3 = self.conv2d(inputdata=relu_2, out_channel=128, kernel_size=3, padding='SAME',  stride=1, use_bias=False, name='conv_3')  
			relu_3 = self.lrelu(inputdata=conv_3, name='relu_3')  
			conv_4 = self.conv2d(inputdata=relu_3, out_channel=128, kernel_size=3,  padding='SAME', stride=2, use_bias=False, name='conv_4')  
			relu_4 = self.lrelu(inputdata=conv_4, name='relu_4')  
			conv_5 = self.conv2d(inputdata=relu_4, out_channel=256, kernel_size=3, padding='SAME',  stride=1, use_bias=False, name='conv_5')  
			relu_5 = self.lrelu(inputdata=conv_5, name='relu_5)
			conv_6 = self.conv2d(inputdata=relu_5, out_channel=256, kernel_size=3, padding='SAME',  stride=1, use_bias=False, name='conv_6')
			relu_6 = self.lrelu(inputdata=conv_6, name='relu_6')  
			dia_conv1 = self.dilation_conv(input_tensor=relu_6, k_size=3, out_dims=256, rate=2, padding='SAME', use_bias=False, name=' dia_conv_1') 
			relu_7 = self.lrelu(dia_conv1, name='relu_7')  
			dia_conv2 = self.dilation_conv(input_tensor=relu_7, k_size=3, out_dims=256, rate=4,  padding='SAME', use_bias=False, name=' dia_conv_2')
			relu_8 = self.lrelu(dia_conv2, name='relu_8') 
			dia_conv3 = self.dilation_conv(input_tensor=relu_8, k_size=3, out_dims=256, rate=8, padding='SAME', use_bias=False, name=' dia_conv_3')
			relu_9 = self.lrelu(dia_conv3, name='relu_9') 
			dia_conv4 = self.dilation_conv(input_tensor=relu_9, k_size=3, out_dims=256, rate=16,  padding='SAME', use_bias=False, name=' dia_conv_4')
			relu_10 = self.lrelu(dia_conv4, name='relu_10') 
			conv_7 = self.conv2d(inputdata=relu_10, out_channel=256, kernel_size=3,  padding='SAME', use_bias=False, stride=1,name=' conv_7')
			relu_11 = self.lrelu(inputdata=conv_7, name='relu_11')
			conv_8 = self.conv2d(inputdata=relu_11, out_channel=256, kernel_size=3,  padding='SAME', use_bias=False, stride=1,name=' conv_8')
			relu_12 = self.lrelu(inputdata=conv_8, name='relu_12') 
			deconv_1 = self.deconv2d(inputdata=relu_12, out_channel=128, kernel_size=4,  stride=2, padding='SAME', use_bias=False, name=' deconv_1')
			avg_pool_1 = self.avgpooling(inputdata=deconv_1, kernel_size=2, stride=1, padding='SAME',  name='avg_pool_1')                            
			relu_13 = self.lrelu(inputdata=avg_pool_1, name='relu_13') 
			conv_9 = self.conv2d(inputdata=tf.add(relu_13, relu_3), out_chan                     nel=128, kernel_size=3,padding='SAME', stride=1, use_bias=False,name=' conv_9')
			relu_14 = self.lrelu(inputdata=conv_9, name='relu_14')  
			deconv_2 = self.deconv2d(inputdata=relu_14, out_channel=64, kernel_size=4, stride=2, padding='SAME', use_bias=False, name='deconv_2')       
			avg_pool_2 = self.avgpooling(inputdata=deconv_2, kernel_size=2, stride=1, padding='SAME',    name='avg_pool_2')  
			relu_15 = self.lrelu(inputdata=avg_pool_2, name='relu_15')
			conv_10 = self.conv2d(inputdata=tf.add(relu_15, relu_1), out_channel=32, kernel_size=3, padding='SAME', stride=1, use_bias=False,  name='conv_10')                               
			relu_16 = self.lrelu(inputdata=conv_10, name='relu_16')  
			skip_output_1 = self.conv2d(inputdata=relu_12, out_channel=3, kernel_size=3,  padding='SAME', stride=1, use_bias=False,name='skip_ouput_1') 
			skip_output_2 = self.conv2d(inputdata=relu_14, out_channel=3, kernel_size=3, padding='SAME', stride=1, use_bias=False,name='skip_ouput_2') 
			skip_output_3 = self.conv2d(inputdata=relu_16, out_channel=3, kernel_size=3, padding='SAME', stride=1, use_bias=False,name='skip_ouput_3') 
			# 传统GAN输出层都使用tanh函数激活  
			skip_output_3 = tf.nn.tanh(skip_output_3, name='skip_output_3_tanh')  

   
			ret = {  
			    'skip_1': skip_output_1,  
			    'skip_2': skip_output_2,  
			    'skip_3': skip_output_3  
			}    
			rn ret  
```



## **判别器的实现：**

```python
 with tf.variable_scope(name, reuse=reuse):  
	conv_stage_1 = self._conv_stage(input_tensor=input_tensor, k_size=5,  stride=1, out_dims=8,  group_size=0, name='conv_stage_1')
	conv_stage_2 = self._conv_stage(input_tensor=conv_stage_1, k_size=5,  stride=1, out_dims=16, group_size=0, name='conv_stage_2')  
	conv_stage_3 = self._conv_stage(input_tensor=conv_stage_2, k_size=5, stride=1, out_dims=32, group_size=0, name='conv_stage_3')  
	conv_stage_4 = self._conv_stage(input_tensor=conv_stage_3, k_size=5, stride=1, out_dims=64, group_size=0, name='conv_stage_4')  
	conv_stage_5 = self._conv_stage(input_tensor=conv_stage_4, k_size=5, stride=1, out_dims=128, group_size=0, name='conv_stage_5')  
	conv_stage_6 = self._conv_stage(input_tensor=conv_stage_5, k_size=5,  stride=1, out_dims=128, group_size=0, name='conv_stage_6')  
	attention_map = self.conv2d(inputdata=conv_stage_6, out_channel=1, kernel_size=5,  padding='SAME', stride=1, use_bias=False, name='attention_map')  
	conv_stage_7 = self._conv_stage(input_tensor=attention_map * conv_stage_6, k_size=5,  stride=4, out_dims=64, group_size=0, name='conv_stage_7')  
	onv_stage_8 = self._conv_stage(input_tensor=conv_stage_7, k_size=5,stride=4, out_dims=64, group_size=0, name='conv_stage_8')  
	onv_stage_9 = self._conv_stage(input_tensor=conv_stage_8, k_size=5,  stride=4, out_dims=32, group_size=0, name='conv_stage_9')  
	c_1 = self.fullyconnect(inputdata=conv_stage_9, out_dim=1024, use_bias=False, name='fc_1')  
	c_2 = self.fullyconnect(inputdata=fc_1, out_dim=1, use_bias=False, name='fc_2')  
	c_out = self.sigmoid(inputdata=fc_2, name='fc_out') 
	c_out = tf.where(tf.not_equal(fc_out, 1.0), fc_out, fc_out - 0.0000001)  
	c_out = tf.where(tf.not_equal(fc_out, 0.0), fc_out, fc_out + 0.0000001)  
	return fc_out, attention_map, fc_2  
```



## **想要了解更多代码请看github**