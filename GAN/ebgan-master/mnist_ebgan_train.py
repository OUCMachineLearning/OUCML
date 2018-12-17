import sugartensor as tf
import numpy as np
from model import *


__author__ = 'namju.kim@kakaobrain.com'


# set log level to debug
tf.sg_verbosity(10)

#
# hyper parameters
#

batch_size = 128   # batch size

#
# inputs
#

# MNIST input tensor ( with QueueRunner )
data = tf.sg_data.Mnist(batch_size=batch_size)

# input images
x = data.train.image

# random uniform seed
z = tf.random_uniform((batch_size, z_dim))

#
# Computational graph
#

# generator
gen = generator(z)

# add image summary
tf.sg_summary_image(x, name='real')
tf.sg_summary_image(gen, name='fake')

# discriminator
disc_real = discriminator(x)
disc_fake = discriminator(gen)

#
# pull-away term ( PT ) regularizer
#

sample = gen.sg_flatten()
nom = tf.matmul(sample, tf.transpose(sample, perm=[1, 0]))
denom = tf.reduce_sum(tf.square(sample), reduction_indices=[1], keep_dims=True)
pt = tf.square(nom/denom)
pt -= tf.diag(tf.diag_part(pt))
pt = tf.reduce_sum(pt) / (batch_size * (batch_size - 1))


#
# loss & train ops
#

# mean squared errors
mse_real = tf.reduce_mean(tf.square(disc_real - x), reduction_indices=[1, 2, 3])
mse_fake = tf.reduce_mean(tf.square(disc_fake - gen), reduction_indices=[1, 2, 3])

# discriminator loss
loss_disc = mse_real + tf.maximum(margin - mse_fake, 0)
# generator loss + PT regularizer
loss_gen = mse_fake + pt * pt_weight

train_disc = tf.sg_optim(loss_disc, lr=0.001, category='discriminator')  # discriminator train ops
train_gen = tf.sg_optim(loss_gen, lr=0.001, category='generator')  # generator train ops

# add summary
tf.sg_summary_loss(loss_disc, name='disc')
tf.sg_summary_loss(loss_gen, name='gen')


#
# training
#

# def alternate training func
@tf.sg_train_func
def alt_train(sess, opt):
    l_disc = sess.run([loss_disc, train_disc])[0]  # training discriminator
    l_gen = sess.run([loss_gen, train_gen])[0]     # training generator
    return np.mean(l_disc) + np.mean(l_gen)

# do training
alt_train(log_interval=10, max_ep=30, ep_size=data.train.num_batch)

