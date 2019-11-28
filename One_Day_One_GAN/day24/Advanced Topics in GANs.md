# Advanced Topics in GANs

Want to turn horses into zebras? Make DIY anime characters or celebrities? Generative adversarial networks (GANs) are your new best friend.

*“Generative Adversarial Networks is the most interesting idea in the last 10 years in Machine Learning.”* — **Yann LeCun, Director of AI Research at Facebook AI**This is the second part of a 3 part tutorial on creating deep generative models specifically using generative adversarial networks. This is a natural extension to the previous topic on variational autoencoders (found [here](https://towardsdatascience.com/generating-images-with-autoencoders-77fd3a8dd368)). We will see that GANs are largely superior to variational autoencoders, but are notoriously difficult to work with.

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-021612.png)

Throughout this tutorial, we will tackle the following topics:

- Short Review of GANs
- GAN Applications
- Problems with GANs
- Other Types of GANs
- Building an Image GAN
- Coding Tutorial

The bulk of this article will be about coding GANs as well as a broad introduction into some of the more advanced implementations of GANs.

All code used in this tutorial can be found on my GAN-Tutorial GitHub repository.

https://github.com/mrdragonbear/GAN-Tutorial?source=post_page-----fd8e4a70775----------------------

## Short Review of GANs

For those of you who have read the first part, you can probably skip over this section as I will just be reiterating how GANs work mechanistically using my previous example with pandas.

Generative adversarial nets were recently introduced (in the past 5 years or so) as a novel way to train a generative model, i.e. create a model that is able to generate data. They consist of two ‘adversarial’ models: a generative model *G* that captures the data distribution, and a discriminative model *D* that estimates the probability that a sample came from the training data rather than *G*. Both *G* and *D* could be a non-linear mapping function, such as a multi-layer perceptron

In a generative adversarial network (GAN) we have two neural networks that are pitted against each other in a zero-sum game, whereby the first network, the generator, is tasked with fooling the second network, the discriminator. The generator creates ‘fake’ pieces of data, in this case, images of pandas, to try and fool the discriminator into classifying the image as a real panda. We can iteratively improve these two networks in order to produce photorealistic images as well as performing some other very cool applications, some of which we will discuss later in this tutorial.

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-021707.png)

Initially, the images may be fairly obviously faked, but as the networks get better, it becomes harder to distinguish between real and fake images, even for humans!

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-021709.png)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-21710.png)

The two networks can be considered black boxes representing some arbitrarily complicated function that is applied to either noise or real data. The input to the generator is some random noise, which produces a fake image, and the input to the discriminator is both fake samples as well as samples from our real data set. The discriminator then makes a binary decision about whether the image supplied, *z,* is a real image *D(z)=1*, or a fake image, *D(z)=0.*

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-021714.png)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-021715.png)

To train the two networks we must have a loss function, and the loss function for each network depends on the second network. To train the networks we perform backpropagation whilst freezing the other network’s neuron weights.

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-021717.png)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-021716.png)

If you are having trouble understanding what is going on, I recommend you go back and read part 1 to get a better intuition of how this training procedure works.

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-021718.png)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-021719.png)

It is important to note that in general, the discriminator and generator networks can be any form of mapping function, such as a support vector machine. It is this generalization of GANs which is often sometimes referred to as Turing learning. In practice, however, neural networks are the most common as they are generalized function approximators for arbitrary non-linear functions.

I will now discuss some of the coolest applications of GANs before jumping into some of the more advanced topics and also a code walkthrough of GANs aimed at generating celebrity faces as well as anime characters.

# **GAN Applications**

In this section, I will briefly talk about some of the most interesting applications of GANs that I have found during the course of my data science research. The most common topics are:

- **(Conditional) Synthesis** — This includes font generation, Text2Image, as well as 3D Object generation.
- **Data Augmentation** — Aiming to reduce the need for labeled data (GAN is only used as a tool for enhancing the training process of another model).
- **Style Transfer and Manipulation —** Face Aging, painting, pose estimation and manipulation, inpainting, and blending.
- **Signal Super Resolution —** Artificially increasing the resolution of images.

## Conditional Synthesis

Conditional synthesis is fairly incredible in my opinion. Arguably the most fascinating application of conditional synthesis is Image2Text and Text2Image, which is able to translate a picture into words (the whole picture says a thousand words malarky..), or vice versa.

The applications of this are far-reaching, if not only for the analysis of medical images to describe the features of the image, thus removing the subjective analysis of images by doctors (this is actually something I am currently trying looking into myself with mammograms as part of a side project).

This also works the other way (assuming we give our networks words that it understands), and images can be generated purely from words. Here is an example of a Multi-Condition GAN (MC-GAN) used to perform this text-to-image conditional synthesis:

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-023301.png)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-023314.png)

Implementation of MC-GAN for translating words into images. Source: https://arxiv.org/pdf/1902.06068.pdf

------

## Data Augmentation

This one is fairly self-explanatory. GANs learn a data generating distribution like we did when looking at VAEs. Because of this, we are able to sample from our generator and produce additional samples which we can use to augment our training set. Thus, GANs provide an additional method to perform data augmentation (besides rotating and distorting images).

------

## **Style Transfer and Manipulation**

Style transfer is quite self-explanatory. It involves the transfer of the ‘style’ of one image onto another image. This is very similar to neural style transfer, which I will be discussing in a future article.

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-023307.png)

![img](https://miro.medium.com/max/1438/1*1SNfqePTlfrVcYd4HoOAuQ.png)

Example of style transfer using a GAN.

This works very nicely for background scenery and is similar to image filtering except that we can manipulate aspects of the actual image (compare the clouds in the above images for the input and output).

How do GANs perform on other objects such as animals or fruit?

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-023346.png)

Pretty well it seems! If I knew nothing about GANs I would probably believe that the image of a horse was, in fact, a zebra after the style transfer.

We can also change scenery to manipulate the season, a potentially useful manipulation for things like video games and virtual reality simulators.

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-023340.png)

We can also change the depth of field using GANs.

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-023333.png)

We can also manipulate drawings to turn them into real objects with relative ease (however, this probably requires substantially more drawing skills than I currently possess).

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-023344.png)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-023343.png)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-023337.png)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-023339.png)

Self-driving cars see the world in a similar view to the image below, which allows objects to be viewed in a more contrasting manner (typically called a semantic map).

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-023342.png)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-023345.png)

![image-20191128103531713](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-023532.png)

That is enough about style transfer and image manipulation. There are many good applications of this but we have already seen several malicious uses of the technology, with people impersonating political figures and creating fake phone conversations, emails, etc.

This is actually such a problem that the U.S. military is developing a new field of forensics to study videos and similar examples of media in order to determine whether they were generated by GANs (oh my, what a world we live in.

## Image Super-Resolution

Image super-resolution (SR), which refers to the process of recovering high-resolution (HR) images from low-resolution (LR) images, is an important class of image processing techniques in computer vision and image processing.

In general, this problem is very challenging and inherently ill-posed since there are always multiple HR images corresponding to a single LR image.

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-023554.png)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-023559.png)

A variety of deep learning methods have been applied to tackle SR tasks, ranging from the early Convolutional Neural Networks (CNN) based method (e.g., SRCNN) to recent promising SR approaches using GANs (e.g., SRGAN). In general, the family of SR algorithms using deep learning techniques differs from each other in the following major aspects: different types of network architectures, different types of loss functions, different types of learning principles and strategies, etc.

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-023605.png)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-023602.png)

The procedure to do this is actually quite involved, so I will not go into too much detail in this article (although if readers are interested I will cover this in the future).

A comprehensive overview of style transfer with GANs can be found [here](https://arxiv.org/pdf/1902.06068.pdf).

# **Problems with GANs**

We discussed some of the most fundamental issues with GANs in the previous article, mainly the high amount of computational power necessary, the difficulty of training on large images, sensitivity, as well as modal collapse.

I wanted to reiterate these problems because training a GAN is very difficult and time-consuming. There are so many other problems with GANs that in academia (at least in Harvard), there is a running joke that if you want to train a GAN, you pick an unsuspecting and naive graduate student to do it for you (I have been on the receiving end of such a joke).

## Oscillations

Oscillations can occur when both the generator and discriminator are jointly searching for equilibrium, but the model updates are independent. There is no theoretical guarantee of convergence and in fact, the outcome can oscillate.

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-024215.png)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-24217.png)

The solution to this is an extensive hyperparameter-search, which sometimes may require manual intervention. An extensive hyperparameter-search on something that already takes 10 hours to run is not something I recommend lightly.

## **Vanishing Gradient**

It is possible for the discriminator to become too strong to provide a signal for the generator. What do I mean by this? If the generator gets too good too fast (it rarely occurs the other way around), the generator can learn to fool the discriminator consistently and will stop needing to learn anything.

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-24227.png)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-024216.png)

A GAN exhibiting the vanishing gradient problem.

The solution is not to pre-train the discriminator, or lower its learning rate compared to that of the generator. One can also change the number of updates for generator/discriminator per iteration (as I recommended in part 1).

It is fairly easy to see when a GAN has converged as a stabilization of the two networks will occur somewhere in the middle ground (i.e. one of the networks is not dominating).

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-024224.png)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-024221.png)

The minimax expression for Nash equilibrium in a GAN.

## Modal collapse

The generator can collapse so that it always produces the same samples. This can occur when the generator is restrained to a small subspace and hence begins generating samples of low diversity.

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-24224.png)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-024225.png)

Five of the above generated images look identical, and several others seem to occur in pairs.

The solution to this is to encourage diversity through mini-batch discrimination (presenting the whole batch to the discriminator for review) or by feature matching (i.e. adding a generator penalty for low diversity) or to use multiple GANs.

## **Evaluation Metrics**

This is one I did not mention in the previous article. GANs are still evaluated on a very qualitative basis — essentially, does this image look good? Defining proper metrics that are somewhat objective is surprisingly challenging. How does a “good” generator look?

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-024226.png)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-024223.png)

There is no definitive solution for this and it is still an active research field and quite domain specific. Strong classification models are commonly used to judge the quality of generated samples. Two common scores that are used are the **inception score**, and the **TSTR score** (Train on Synthetic, Test on Real).

Jonathan Hui has a fairly comprehensive article explaining the most common metrics used to evaluate GAN performance, which you can find here:https://medium.com/@jonathan_hui/gan-how-to-measure-gan-performance-64b988c47732?source=post_page-----fd8e4a70775----------------------

In the next section, I will outline some of the most important types of GANs that are surfacing from the academic field.

# **Other Types of GANs**

There are many other types of GAN that have surfaced for tackling domain-specific problems as well as different types of data (for example, time series, images, or normal csv-style data).

The types of GAN I will discuss in this section are:

- Wasserstein GAN
- CycleGAN
- Conditional GAN (briefly discussed previously)

## Wasserstein GAN

In my opinion, this is the most important type of GAN, so pay attention!

Using the standard GAN formulation, we have already observed that training is extremely unstable. The discriminator often improves too quickly for the generator to catch up, which is why we need to regulate the learning rates or perform multiple epochs on one of the two networks. To get a decent output we need careful balancing and even then, modal collapse is quite frequent.

**Source of information:** Arjovsky, M., Chintala, S. and Bottou, L., 2017. Wasserstein GAN. arXiv preprint arXiv:1701.07875.

In general, generative models seek to minimize the distance between real and learned distribution (distance is everything!). The Wasserstein (also EM, Earth-Mover) distance, in informal terms, refers to when distributions are interpreted as two different ways of piling up a certain amount of dirt over a region *D.* The Wasserstein distance is the minimum cost of turning one pile into the other; where the cost is assumed to be the amount of dirt moved times the distance by which it is moved.

Unfortunately, the exact computation is intractable in this case. However, we can use CNNs to approximate the Wasserstein distance. Here, we reuse the discriminator, whose outputs are now unbounded. We define a custom loss function corresponding to the Wasserstein loss:

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-024317.png)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-024318.png)

What is the idea here? We can make predictions for one type as large as possible, and predictions for other types as small as possible.

The authors of the Wasserstein paper claim that:

- Higher stability during training, less need for carefully balancing generator and discriminator.
- Meaningful loss metric, correlating well with sample quality.
- Modal collapse is rare.

**Tips for implementing Wasserstein GAN in Keras.**

- Leave the discriminator output unbounded, i.e. apply linear activation.
- Initialize with small weights to not run into clipping issues from the start.
- Remember to run sufficient discriminator updates. This is crucial in the WGAN setup.
- You can use the Wasserstein surrogate loss implementation.
- Clip discriminator weights by implementing your own Keras constraint.

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-024327.png)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-024320.png)

This is a complicated subject, and implementing a Wasserstein GAN is not a trivial task. If you are interested in pursuing one of these for a personal project I recommend reading the original paper that I referenced previously.

------

## CycleGAN

Remember the first image we saw of a horse being interchanged with a zebra? That was a CycleGAN. CycleGANs transfer styles to images. This is essentially the same idea as performing neural style transfer, which I will cover in a future article.

As an example, imagine taking a famous, such as a picture of the Golden Gate Bridge, and then extracting the style from another image, which could be a famous painting, and redrawing the picture of the bridge in the style of the said famous painting.

As a teaser for my neural transfer learning article, here is an example of one I did previously.

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-24342.png)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-024326.png)

Combining the style of the famed “Great Wave off Kanagawa” with the Chicago skyline.

Applying a GAN to these types of problems is relatively simple, it is essentially an image reconstruction. We use a first network ***G\*** to convert image ***x\*** to ***y\***. We reverse the process with another deep network ***F\*** to reconstruct the image. Then, we use a mean square error MSE to guide the training of ***G\*** and ***F\***.

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-024338.png)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-024344.png)

The difference here is that we do not actually care about reconstructing the images, we are trying to mix the styles of the two images. In the GAN implementation, a discriminator ***D\*** is added to an existing design to guide the generator network to perform better. ***D\*** acts as a critic between the training samples and the generated images. Through this criticism, we use backpropagation to modify the generator to produce images that address the shortcoming identified by the discriminator. In this problem, we introduce a discriminator ***D\*** to make sure ***Y\*** resemble Van Gogh’s paintings.

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-024330.jpg)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-024332.jpg)

CycleGANs transfer pictures from one domain to another. To transform pictures between real images and Van Gogh paintings. We build three networks.

- A generator ***G\*** to convert a real image to a Van Gogh style picture.
- A generator ***F\*** to convert a Van Gogh style picture to a real image.
- A discriminator ***D\*** to identify real or generated Van Gogh pictures.

For the reverse direction, we just reverse the data flow and build an additional discriminator to identify real images.

An example of this implementation in creating the zebra/horse images is shown below.

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-024337.png)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-024332.png)

------

## Conditional GAN

As in VAEs, GANs can simply be conditioned to generate a certain mode of data. A great reference paper for conditional GANs can be found [here](https://arxiv.org/pdf/1411.1784.pdf).

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-24339.png)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-024340.png)

We can extend the generative model of a GAN to a conditional model if both the generator and discriminator are conditioned on some extra information *c*. This *c* could be any kind of auxiliary information, such as class labels or data from other modalities. We can perform the conditioning by feeding *c* into both the discriminator and generator as an additional input layer.

In the generator, the prior input noise *p(z)*, and *c* are combined in joint hidden representation, and the adversarial training framework allows for considerable flexibility in how this hidden representation is composed. In the discriminator *x* and *c* are presented as inputs and to a discriminative function.

------

## Troubleshooting GANs

Just a brief overview of all the methods of troubleshooting that we have discussed up to now, for those of you who like summaries.

**[1] Models.** Make sure models are correctly defined. You can debug the discriminator alone by training on a vanilla image-classification task.

**[2] Data.** Normalize inputs properly to [-1, 1]. Make sure to use tanh as final activation for the generator in this case.

**[3] Noise.** Try sampling the noise vector from a normal distribution (not uniform).

**[4] Normalization.** Apply BatchNorm when possible, and send the real and fake samples in separate mini-batches.

**[5] Activations.** Use LeakyRelu instead of Relu.

**[6] Smoothing.** Apply label smoothing to avoid overconfidence when updating the discriminator, i.e. set targets for real images to less than 1.

**[7] Diagnostics.** Monitor the magnitude of gradients constantly.

**[8] Vanishing gradients.** If the discriminator becomes too strong (discriminator loss = 0), try decreasing its learning rate or update the generator more often.

Now let’s get into the fun part, actually building a GAN.

------

# **Building an Image GAN**

As we have already discussed several times, training a GAN can be frustrating and time-intensive. We will walk through a clean minimal example in Keras. The results are only on the proof-of-concept level to enhance understanding. In the code example, if you don’t tune parameters carefully, you won’t surpass this level (see below) of image generation by much:

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-024336.png)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-024341.png)

The network takes an **image \*[H, W, C]\*** and outputs a **vector of \*[M]\***, either class scores (classification) or single score quantifying photorealism. Can be any image classification network, e.g. ResNet or DenseNet. We use minimalistic custom architecture.

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-024333.png)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-024334.png)

Takes a vector of **noise \*[N]\*** and outputs an **image of \*[H, W, C]\***. The network has to perform synthesis. Again, we use a very minimalistic custom architecture.

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-024335.png)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-024339.png)

It is important to define the models properly in Keras so that the weights of the respective models are fixed at the right time.

[1] Define the discriminator model, and compile it.

[2] Define the generator model, no need to compile.

[3] Define an overall model comprised of these two, setting the discriminator to not trainable before the compilation:

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-024342.png)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-024343.png)

In the simplest form, this is all that you need to do to train a GAN.

The training loop has to be executed manually:

[1] Select R real images from the training set.

[2] Generate *F* fake images by sampling random vectors of size *N*, and predicting images from them using the generator.

[3] Train the discriminator using train_on_batch: call it separately for the batch of *R* real images and *F* fake images, with the ground truth being 1 and 0, respectively.

[4] Sample new random vectors of size *N*.

[5] Train the full model on the new vectors using train_on_batch with targets of 1. This will update the generator.