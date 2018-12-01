## celebA
首先,rm dataset/celebA
然后python download.py celebA ......你会发现下载不了哈哈哈哈哈哈哈哈哈
然后,我发下可以直接用kaggle下载
先注册一个kaggle账号,然后install kaggel ,再按github上的教程编辑kaggle api的json
>>kaggle datasets download -d jessicali9530/celeba-dataset 

把/img_align_celeba文件夹解压放到datasets下面

```
hx@hx-b412:~$ python main.py --phase train --dataset img_align_celeba  --gan_type wgan-gp --img 64

```

## cifar10
直接就能跑
python main.py --phase train --dataset cifar10 --gan_type wgan-gp --img 32

---

# Self-Attention-GAN-Tensorflow
Simple Tensorflow implementation of ["Self-Attention Generative Adversarial Networks" (SAGAN)](https://arxiv.org/pdf/1805.08318.pdf)


## Requirements
* Tensorflow 1.8
* Python 3.6

## Summary
### Framework
![framework](./assests/framework.PNG)

### Code
```python
    def attention(self, x, ch):
      f = conv(x, ch // 8, kernel=1, stride=1, sn=self.sn, scope='f_conv') # [bs, h, w, c']
      g = conv(x, ch // 8, kernel=1, stride=1, sn=self.sn, scope='g_conv') # [bs, h, w, c']
      h = conv(x, ch, kernel=1, stride=1, sn=self.sn, scope='h_conv') # [bs, h, w, c]

      # N = h * w
      s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True) # # [bs, N, N]

      beta = tf.nn.softmax(s, axis=-1)  # attention map

      o = tf.matmul(beta, hw_flatten(h)) # [bs, N, C]
      gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

      o = tf.reshape(o, shape=x.shape) # [bs, h, w, C]
      x = gamma * o + x

      return x
```
## Usage
### dataset

```python
> python download.py celebA
```

* `mnist` and `cifar10` are used inside keras
* For `your dataset`, put images like this:

```
├── dataset
   └── YOUR_DATASET_NAME
       ├── xxx.jpg (name, format doesn't matter)
       ├── yyy.png
       └── ...
```

### train
* python main.py --phase train --dataset celebA --gan_type hinge

### test
* python main.py --phase test --dataset celebA --gan_type hinge

## Results
### ImageNet
<div align="">
   <img src="./assests/result_.png" width="420">
</div>

### CelebA (100K iteration, hinge loss)
![celebA](./assests/celebA.png)

## Author
Junho Kim
