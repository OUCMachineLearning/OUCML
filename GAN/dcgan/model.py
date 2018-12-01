import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import (
    InputLayer,
    DenseLayer,
    DeConv2d,
    ReshapeLayer,
    BatchNormLayer,
    Conv2d,
    FlattenLayer
)

flags = tf.app.flags
FLAGS = flags.FLAGS

def generator(inputs, is_train=True, reuse=False):
    image_size = 64
    s16 = image_size // 16
    gf_dim = 64    # Dimension of gen filters in first conv layer. [64]
    c_dim = FLAGS.c_dim    # n_color 3
    w_init = tf.glorot_normal_initializer()
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("generator", reuse=reuse):

        net_in = InputLayer(inputs, name='g/in')
        net_h0 = DenseLayer(net_in, n_units=(gf_dim * 8 * s16 * s16), W_init=w_init,
                act = tf.identity, name='g/h0/lin')
        net_h0 = ReshapeLayer(net_h0, shape=[-1, s16, s16, gf_dim*8], name='g/h0/reshape')
        net_h0 = BatchNormLayer(net_h0, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g/h0/batch_norm')

        net_h1 = DeConv2d(net_h0, gf_dim * 4, (5, 5), strides=(2, 2),
                padding='SAME', act=None, W_init=w_init, name='g/h1/decon2d')
        net_h1 = BatchNormLayer(net_h1, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g/h1/batch_norm')

        net_h2 = DeConv2d(net_h1, gf_dim * 2, (5, 5), strides=(2, 2),
                padding='SAME', act=None, W_init=w_init, name='g/h2/decon2d')
        net_h2 = BatchNormLayer(net_h2, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g/h2/batch_norm')

        net_h3 = DeConv2d(net_h2, gf_dim, (5, 5), strides=(2, 2),
                padding='SAME', act=None, W_init=w_init, name='g/h3/decon2d')
        net_h3 = BatchNormLayer(net_h3, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g/h3/batch_norm')

        net_h4 = DeConv2d(net_h3, c_dim, (5, 5), strides=(2, 2),
                padding='SAME', act=None, W_init=w_init, name='g/h4/decon2d')
        net_h4.outputs = tf.nn.tanh(net_h4.outputs)
    return net_h4

def discriminator(inputs, is_train=True, reuse=False):
    df_dim = 64   # Dimension of discrim filters in first conv layer. [64]
    w_init = tf.glorot_normal_initializer()
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("discriminator", reuse=reuse):

        net_in = InputLayer(inputs, name='d/in')
        net_h0 = Conv2d(net_in, df_dim, (5, 5), (2, 2), act=tf.nn.leaky_relu,
                padding='SAME', W_init=w_init, name='d/h0/conv2d')

        net_h1 = Conv2d(net_h0, df_dim*2, (5, 5), (2, 2), act=None,
                padding='SAME', W_init=w_init, name='d/h1/conv2d')
        net_h1 = BatchNormLayer(net_h1, act=tf.nn.leaky_relu,
                is_train=is_train, gamma_init=gamma_init, name='d/h1/batch_norm')

        net_h2 = Conv2d(net_h1, df_dim*4, (5, 5), (2, 2), act=None,
                padding='SAME', W_init=w_init, name='d/h2/conv2d')
        net_h2 = BatchNormLayer(net_h2, act=tf.nn.leaky_relu,
                is_train=is_train, gamma_init=gamma_init, name='d/h2/batch_norm')

        net_h3 = Conv2d(net_h2, df_dim*8, (5, 5), (2, 2), act=None,
                padding='SAME', W_init=w_init, name='d/h3/conv2d')
        net_h3 = BatchNormLayer(net_h3, act=tf.nn.leaky_relu,
                is_train=is_train, gamma_init=gamma_init, name='d/h3/batch_norm')

        net_h4 = FlattenLayer(net_h3, name='d/h4/flatten')
        net_h4 = DenseLayer(net_h4, n_units=1, act=tf.identity,
                W_init = w_init, name='d/h4/lin_sigmoid')
        logits = net_h4.outputs
        net_h4.outputs = tf.nn.sigmoid(net_h4.outputs)
    return net_h4, logits
