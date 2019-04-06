import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope

@add_arg_scope
def gate_conv(x_in, cnum, ksize, stride=1, rate=1, name='conv',
             padding='SAME', activation='leaky_relu', use_lrn=True,training=True):
    assert padding in ['SYMMETRIC', 'SAME', 'REFELECT']
    if padding == 'SYMMETRIC' or padding == 'REFELECT':
        p = int(rate*(ksize-1)/2)
        x = tf.pad(x_in, [[0,0], [p, p], [p, p], [0,0]], mode=padding)
        padding = 'VALID'
    x = tf.layers.conv2d(
        x_in, cnum, ksize, stride, dilation_rate=rate,
        activation=None, padding=padding, name=name)    
    if use_lrn:
        x = tf.nn.lrn(x, bias=0.00005)
    if activation=='leaky_relu':
        x = tf.nn.leaky_relu(x)

    g = tf.layers.conv2d(
        x_in, cnum, ksize, stride, dilation_rate=rate,
        activation=tf.nn.sigmoid, padding=padding, name=name+'_g')

    x = tf.multiply(x,g)
    return x, g

@add_arg_scope
def gate_deconv(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv", training=True):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                  initializer=tf.random_normal_initializer(stddev=stddev))

        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                    strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases1', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        deconv = tf.nn.leaky_relu(deconv)

        g = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                    strides=[1, d_h, d_w, 1])
        b = tf.get_variable('biases2', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        g = tf.reshape(tf.nn.bias_add(g, b), deconv.get_shape())
        g = tf.nn.sigmoid(deconv)

        deconv = tf.multiply(g,deconv)

        return deconv, g



