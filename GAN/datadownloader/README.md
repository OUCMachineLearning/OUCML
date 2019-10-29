# DCGAN in TensorFlow

TensorFlow / TensorLayer implementation of [Deep Convolutional Generative Adversarial Networks](http://arxiv.org/abs/1511.06434) which is a stabilize Generative Adversarial Networks.

Looking for Text to Image Synthesis ? [click here](https://github.com/zsdonghao/text-to-image)

![alt tag](img/DCGAN.png)

* [Brandon Amos](http://bamos.github.io/) wrote an excellent [blog post](http://bamos.github.io/2016/08/09/deep-completion/) and [image completion code](https://github.com/bamos/dcgan-completion.tensorflow) based on this repo.
* *To avoid the fast convergence of D (discriminator) network, G (generator) network is updated twice for each D network update, which differs from original paper.*


## Prerequisites

- Python 2.7 or Python 3.3+
- [TensorFlow==1.10.0+](https://www.tensorflow.org/)
- [TensorLayer==1.10.1+](https://github.com/tensorlayer/tensorlayer)


## Usage

First, download images to `data/celebA`:

    $ python download.py celebA		[202599 face images]

Second, train the GAN:

    $ python main.py

## Result on celebA


<a href="http://tensorlayer.readthedocs.io">
<div align="center">
	<img src="img/result.png" width="90%" height="90%"/>
</div>
</a>