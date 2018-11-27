# Wasserstein GAN

This is a tensorflow implementation of WGAN on mnist and SVHN.

## Requirement

tensorflow==1.0.0+

numpy

matplotlib

cv2

## Usage

Train: Use WGAN.ipynb, set the parameters in the second cell and choose the dataset you want to run on. You can use tensorboard to visualize the training. 

Generation : Use generate_from_ckpt.ipynb, set  `ckpt_dir` in the second cell. Don't forget to change the dataset type accordingly.

## Note

1. All data will be downloaded automatically, the SVHN script is modified from [this](https://github.com/openai/improved-gan/blob/master/mnist_svhn_cifar10/svhn_data.py).

2. All parameters are set to the values the original paper recommends by default. `Diters`  represents the number of critic updates in one step, in [Original PyTorch version](https://github.com/martinarjovsky/WassersteinGAN) it was set to 5 unless iterstep < 25 or iterstep % 500 == 0 , I guess since the critic is free to fully optimized, it's reasonable to make more updates to critic at the beginning and every 500 steps, so I borrowed it without tuning. The learning rates for generator and critic are both set to 5e-5 , since during the training time the gradient norms are always relatively high(around 1e3), I suggest no drastic change on learning rates.

3. MLP version could take longer time to generate sharp image.

4. In this implementation, the critic loss is `tf.reduce_mean(fake_logit - true_logit)`, and generator loss is `tf.reduce_mean(-fake_logit)` . Actually, the whole system still works if you add a `-` before both of them, it doesn't matter. Recall that the critic loss in duality form is ![](https://ww2.sinaimg.cn/large/006tKfTcly1fcewxqyfvwj307i00kglj.jpg), and the set![](https://ww2.sinaimg.cn/large/006tKfTcly1fcex6p638ij302200o3yd.jpg) is symmetric about the sign. Substitute $f$ with $-f$ gives us ![](https://ww3.sinaimg.cn/large/006tKfTcly1fcewyhols5j307g00odfr.jpg), the opposite number of original form. The original PyTorch implementation takes the second form, this implementation takes the first, both will work equally. You might want to add the `-` and try it out.

5. Please set your device you want to run on in the code, search `tf.device` and change accordingly. It runs on gpu:0 by default.

6. Inproved-WGAN is added, but somehow the gradient norm is close to 1, so the square-gradient normalizer doesn't work. Couldn't figure out why.   ​
