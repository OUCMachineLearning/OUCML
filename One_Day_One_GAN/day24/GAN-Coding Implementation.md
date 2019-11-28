## **GAN-Coding Implementation**

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

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-030702.png)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-030703.png)

The network takes an **image \*[H, W, C]\*** and outputs a **vector of \*[M]\***, either class scores (classification) or single score quantifying photorealism. Can be any image classification network, e.g. ResNet or DenseNet. We use minimalistic custom architecture.

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-030707.png)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-030708.png)

Takes a vector of **noise \*[N]\*** and outputs an **image of \*[H, W, C]\***. The network has to perform synthesis. Again, we use a very minimalistic custom architecture.

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-030709.png)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-030710.png)

It is important to define the models properly in Keras so that the weights of the respective models are fixed at the right time.

[1] Define the discriminator model, and compile it.

[2] Define the generator model, no need to compile.

[3] Define an overall model comprised of these two, setting the discriminator to not trainable before the compilation:

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-030705.png)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-30706.png)

In the simplest form, this is all that you need to do to train a GAN.

The training loop has to be executed manually:

[1] Select R real images from the training set.

[2] Generate *F* fake images by sampling random vectors of size *N*, and predicting images from them using the generator.

[3] Train the discriminator using train_on_batch: call it separately for the batch of *R* real images and *F* fake images, with the ground truth being 1 and 0, respectively.

[4] Sample new random vectors of size *N*.

[5] Train the full model on the new vectors using train_on_batch with targets of 1. This will update the generator.

Now we will go through the above minimalistic implementation in code format using Keras on the well-known CelebA dataset. You may need to reference the above procedure if you are confused about the way the code is structured, although I will do my best to walk you through this.

The full code implementation is freely available on my corresponding GitHub repository for this 3-part tutorial.

Note that in this code I have not optimized this in any way, and I have ignored several of the rules of thumb. I will discuss these more in part 3 which will involve a more in-depth discussion of the GAN code.

**Imports**

First, we import the necessary packages.

```
import keras
from keras.layers import *
from keras.datasets import cifar10
import glob, cv2, os
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from IPython.display import clear_output
```

**Global Parameters**

It is always best (in my opinion) to specify these parameters at the start to avoid later confusion, as these networks can get fairly messy and involved.

```
SPATIAL_DIM = 64 # Spatial dimensions of the images.
LATENT_DIM = 100 # Dimensionality of the noise vector.
BATCH_SIZE = 32 # Batchsize to use for training.
DISC_UPDATES = 1  # Number of discriminator updates per training iteration.
GEN_UPDATES = 1 # Nmber of generator updates per training iteration.

FILTER_SIZE = 5 # Filter size to be applied throughout all convolutional layers.
NUM_LOAD = 10000 # Number of images to load from CelebA. Fit also according to the available memory on your machine.
NET_CAPACITY = 16 # General factor to globally change the number of convolutional filters.

PROGRESS_INTERVAL = 80 # Number of iterations after which current samples will be plotted.
ROOT_DIR = 'visualization' # Directory where generated samples should be saved to.

if not os.path.isdir(ROOT_DIR):
    os.mkdir(ROOT_DIR)
```

**Prepare Data**

We now do some image preprocessing in order to normalize the images, and also plot an image to make sure our implementation is working correctly.

```
def plot_image(x):
    plt.imshow(x * 0.5 + 0.5)X = []
# Reference to CelebA dataset here. I recommend downloading from the Harvard 2019 ComputeFest GitHub page (there is also some good coding tutorials here)faces = glob.glob('../Harvard/ComputeFest 2019/celeba/img_align_celeba/*.jpg')

for i, f in enumerate(faces):
    img = cv2.imread(f)
    img = cv2.resize(img, (SPATIAL_DIM, SPATIAL_DIM))
    img = np.flip(img, axis=2)
    img = img.astype(np.float32) / 127.5 - 1.0
    X.append(img)
    if i >= NUM_LOAD - 1:
        break
X = np.array(X)
plot_image(X[4])
X.shape, X.min(), X.max()
```

**Define Architecture**

The architecture is fairly simple in the Keras format. It is good to write the code in blocks to keep things as simple as possible.

First, we add the encoder block portion. Notice here that we are using ‘same’ padding so that the input and output dimensions are the same, as well as batch normalization and leaky ReLU. Stride is mostly optional, as is the magnitude for the leaky ReLU argument. The values I have put here are not optimized but did give me a reasonable result.

```
def add_encoder_block(x, filters, filter_size):
    x = Conv2D(filters, filter_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, filter_size, padding='same', strides=2)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.3)(x)
    return x
```

Followed by the discriminator itself — notice how we have recycled the encoder block segment and are gradually increasing the filter size to solve the problem we previously discussed for training on (large) images (best practice to do it for all images).

```
def build_discriminator(start_filters, spatial_dim, filter_size):
    inp = Input(shape=(spatial_dim, spatial_dim, 3))
    
    # Encoding blocks downsample the image.
    x = add_encoder_block(inp, start_filters, filter_size)
    x = add_encoder_block(x, start_filters * 2, filter_size)
    x = add_encoder_block(x, start_filters * 4, filter_size)
    x = add_encoder_block(x, start_filters * 8, filter_size)
    
    x = GlobalAveragePooling2D()(x)
    x = Dense(1, activation='sigmoid')(x)
    return keras.Model(inputs=inp, outputs=x)
```

Now for the decoder block segment. This time we are performing the opposite of the convolutional layers, i.e. deconvolution. Notice that the strides and padding are the same for ease of implementation, and we again use batch normalization and leaky ReLU.

```
def add_decoder_block(x, filters, filter_size):
    x = Deconvolution2D(filters, filter_size, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.3)(x)
    return x
```

Now build the generator, notice that this time we are using decoder blocks and gradually decreasing the filter size.

```
def build_generator(start_filters, filter_size, latent_dim):
    inp = Input(shape=(latent_dim,))
    
    # Projection.
    x = Dense(4 * 4 * (start_filters * 8), input_dim=latent_dim)(inp)
    x = BatchNormalization()(x)
    x = Reshape(target_shape=(4, 4, start_filters * 8))(x)
    
    # Decoding blocks upsample the image.
    x = add_decoder_block(x, start_filters * 4, filter_size)
    x = add_decoder_block(x, start_filters * 2, filter_size)
    x = add_decoder_block(x, start_filters, filter_size)
    x = add_decoder_block(x, start_filters, filter_size)    
    
    x = Conv2D(3, kernel_size=5, padding='same', activation='tanh')(x)
    return keras.Model(inputs=inp, outputs=x)
```

**Training**

Now that we have set up the network architectures, we can outline the training procedure, this can be where people get confused easily. I think this is probably because there are functions within functions within more functions.

```
def construct_models(verbose=False):
    # 1. Build discriminator.
    discriminator = build_discriminator(NET_CAPACITY, SPATIAL_DIM, FILTER_SIZE)
    discriminator.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0002), metrics=['mae'])

    # 2. Build generator.
    generator = build_generator(NET_CAPACITY, FILTER_SIZE, LATENT_DIM)

    # 3. Build full GAN setup by stacking generator and discriminator.
    gan = keras.Sequential()
    gan.add(generator)
    gan.add(discriminator)
    discriminator.trainable = False # Fix the discriminator part in the full setup.
    gan.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0002), metrics=['mae'])

    if verbose: # Print model summaries for debugging purposes.
        generator.summary()
        discriminator.summary()
        gan.summary()
    return generator, discriminator, gan
```

Essentially, what we did above was design a function that creates a GAN model based on our global parameters. Notice that we compile the discriminator, but not the generator, but we do compile the GAN as a whole at the end.

**Also, notice that we have set the discriminator not to be trainable, this is a very common thing people forget when constructing GANs!**

The reason this is so important is that you cannot train both networks at once, it is like trying to calibrate something with multiple variables changing, you will get sporadic results. You require a fixed setup for the rest of the model when training any single network.

Now the full model is constructed, we can get on with the training.

```
def run_training(start_it=0, num_epochs=1000):# Save configuration file with global parameters    config_name = 'gan_cap' + str(NET_CAPACITY) + '_batch' + str(BATCH_SIZE) + '_filt' + str(FILTER_SIZE) + '_disc' + str(DISC_UPDATES) + '_gen' + str(GEN_UPDATES)
    folder = os.path.join(ROOT_DIR, config_name) 

    if not os.path.isdir(folder):
        os.mkdir(folder)# Initiate loop variables    avg_loss_discriminator = []
    avg_loss_generator = []
    total_it = start_it    # Start of training loop
    for epoch in range(num_epochs):
        loss_discriminator = []
        loss_generator = []
        for it in range(200): 

            # Update discriminator.
            for i in range(DISC_UPDATES): 
                # Fetch real examples (you could sample unique entries, too).
                imgs_real = X[np.random.randint(0, X.shape[0], size=BATCH_SIZE)]

                # Generate fake examples.
                noise = np.random.randn(BATCH_SIZE, LATENT_DIM)
                imgs_fake = generator.predict(noise)

                d_loss_real = discriminator.train_on_batch(imgs_real, np.ones([BATCH_SIZE]))[1]
                d_loss_fake = discriminator.train_on_batch(imgs_fake, np.zeros([BATCH_SIZE]))[1]
            
            # Progress visualizations.
            if total_it % PROGRESS_INTERVAL == 0:
                plt.figure(figsize=(5,2))
                # We sample separate images.
                num_vis = min(BATCH_SIZE, 8)
                imgs_real = X[np.random.randint(0, X.shape[0], size=num_vis)]
                noise = np.random.randn(num_vis, LATENT_DIM)
                imgs_fake = generator.predict(noise)
                for obj_plot in [imgs_fake, imgs_real]:
                    plt.figure(figsize=(num_vis * 3, 3))
                    for b in range(num_vis):
                        disc_score = float(discriminator.predict(np.expand_dims(obj_plot[b], axis=0))[0])
                        plt.subplot(1, num_vis, b + 1)
                        plt.title(str(round(disc_score, 3)))
                        plot_image(obj_plot[b]) 
                    if obj_plot is imgs_fake:
                        plt.savefig(os.path.join(folder, str(total_it).zfill(10) + '.jpg'), format='jpg', bbox_inches='tight')
                    plt.show()  

            # Update generator.
            loss = 0
            y = np.ones([BATCH_SIZE, 1]) 
            for j in range(GEN_UPDATES):
                noise = np.random.randn(BATCH_SIZE, LATENT_DIM)
                loss += gan.train_on_batch(noise, y)[1]

            loss_discriminator.append((d_loss_real + d_loss_fake) / 2.)        
            loss_generator.append(loss / GEN_UPDATES)
            total_it += 1

        # Progress visualization.
        clear_output(True)
        print('Epoch', epoch)
        avg_loss_discriminator.append(np.mean(loss_discriminator))
        avg_loss_generator.append(np.mean(loss_generator))
        plt.plot(range(len(avg_loss_discriminator)), avg_loss_discriminator)
        plt.plot(range(len(avg_loss_generator)), avg_loss_generator)
        plt.legend(['discriminator loss', 'generator loss'])
        plt.show()
```

The above code may look very confusing to you. There are several items in the above code that are just helpful for managing the running of the GAN. For example, the first part sets up a configuration file and saves it, so you are able to reference it in the future and know exactly what the architecture and hyperparameters of your network were.

There is also the progress visualization steps, which prints the output of the notebook in real-time so that you can access the current performance of the GAN.

If you ignore the configuration file and progress visualization, the code is relatively simple. First, we update the discriminator, and then we update the generator, and then we iterate between these two scenarios.

Now, due to our smart use of functions, we can run the model in two lines.

```
generator, discriminator, gan = construct_models(verbose=True)
run_training()
```

The only thing left to do is wait (possibly for hours or days) and then test the output of the network.

------

**Sample from Trained GAN**

Now after waiting for the network to finish training, we are able to take a number of samples from the network.

```
NUM_SAMPLES = 7
plt.figure(figsize=(NUM_SAMPLES * 3, 3))

for i in range(NUM_SAMPLES):
    noise = np.random.randn(1, LATENT_DIM) 
    pred_raw = generator.predict(noise)[0]
    pred = pred_raw * 0.5 + 0.5
    plt.subplot(1, NUM_SAMPLES, i + 1)
    plt.imshow(pred)
plt.show()
```

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-025328.png)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-11-28-025329.png)

Samples from trained GAN

Voila! That is our first implementation of a basic GAN. This is probably the simplest way of constructing a fully-functioning GAN.

------

# **Final Comments**

In this tutorial, I have gone through some more of the advanced topics of GANs, their architectures, and their current applications, as well as covered the coding implementation of a simple GAN. In the final part of this tutorial, we will compare the performance of VAEs, GANs, and the implementation of a VAE-GAN for the purpose of generating anime images.

Thank you for reading! If you want to continue to part 3 of the tutorial, you can find it here:

[GANs vs. Autoencoders: Comparison of Deep Generative ModelsWant to turn horses into zebras? Make DIY anime characters or celebrities? Generative adversarial networks (GANs) are…medium.com](https://medium.com/@matthew_stewart/gans-vs-autoencoders-comparison-of-deep-generative-models-985cf15936ea?source=post_page-----fd8e4a70775----------------------)

------

# Further Reading

**Run BigGAN in COLAB:**

- https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/biggan_generation_with_tf_hub.ipynb

**More code help + examples:**

- https://www.jessicayung.com/explaining-tensorflow-code-for-a-convolutional-neural-network/
- https://lilianweng.github.io/lil-log/2017/08/20/from-GAN-to-WGAN.html
- https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
- https://github.com/tensorlayer/srgan
- https://junyanz.github.io/CycleGAN/ https://affinelayer.com/pixsrv/
- https://tcwang0509.github.io/pix2pixHD/

**Influential Papers:**

- DCGAN https://arxiv.org/pdf/1511.06434v2.pdf
- Wasserstein GAN (WGAN) https://arxiv.org/pdf/1701.07875.pdf
- Conditional Generative Adversarial Nets (CGAN) https://arxiv.org/pdf/1411.1784v1.pdf
- Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks (LAPGAN) https://arxiv.org/pdf/1506.05751.pdf
- Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network (SRGAN) https://arxiv.org/pdf/1609.04802.pdf
- Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (CycleGAN) https://arxiv.org/pdf/1703.10593.pdf
- InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets https://arxiv.org/pdf/1606.03657
- DCGAN https://arxiv.org/pdf/1704.00028.pdf
- Improved Training of Wasserstein GANs (WGAN-GP) https://arxiv.org/pdf/1701.07875.pdf
- Energy-based Generative Adversarial Network (EBGAN) https://arxiv.org/pdf/1609.03126.pdf
- Autoencoding beyond pixels using a learned similarity metric (VAE-GAN) https://arxiv.org/pdf/1512.09300.pdf
- Adversarial Feature Learning (BiGAN) https://arxiv.org/pdf/1605.09782v6.pdf
- Stacked Generative Adversarial Networks (SGAN) https://arxiv.org/pdf/1612.04357.pdf
- StackGAN++: Realistic Image Synthesis with Stacked Generative Adversarial Networks https://arxiv.org/pdf/1710.10916.pdf
- Learning from Simulated and Unsupervised Images through Adversarial Training (SimGAN) https://arxiv.org/pdf/1612.07828v1.pdf