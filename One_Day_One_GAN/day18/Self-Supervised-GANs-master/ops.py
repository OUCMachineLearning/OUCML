import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
import warnings

def log_sum_exp(x, axis=1):
    m = tf.reduce_max(x, keep_dims=True)
    return m + tf.log(tf.reduce_sum(tf.exp(x - m), axis=axis))

#the implements of leakyRelu
def lrelu(x, alpha=0.1):
    return tf.maximum(x, alpha*x)

def conv2d(input_, output_dim,
           kernel=4, stride=2, stddev=0.02, spectural_normed=False, iter=1,
           name="conv2d", padding='SAME', with_w=False):
    with tf.variable_scope(name):

        w = tf.get_variable('w', [kernel, kernel, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        if spectural_normed:
            conv = tf.nn.conv2d(input_, spectral_norm(w, iteration=iter), strides=[1, stride, stride, 1], padding=padding)
        else:
            conv = tf.nn.conv2d(input_, w, strides=[1, stride, stride, 1], padding=padding)

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        if with_w:
            return conv, w

        else:
            return conv

def instance_norm(input, scope="instance_norm"):

    with tf.variable_scope(scope):

        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input - mean) * inv

        return scale * normalized + offset

def weight_normalization(weight, scope='weight_norm'):

  """based upon openai's https://github.com/openai/generating-reviews-discovering-sentiment/blob/master/encoder.py"""
  weight_shape_list = weight.get_shape().as_list()

  if len(weight.get_shape()) == 2: #I think you want to sum on axis [0,1,2]
    g_shape = [weight_shape_list[1]]
  else:
    raise ValueError('dimensions unacceptable for weight normalization')

  with tf.variable_scope(scope):

    g = tf.get_variable('g_scalar', shape=g_shape, initializer = tf.ones_initializer())
    weight = g * tf.nn.l2_normalize(weight, dim=0)

    return weight

def de_conv(input_, output_shape,
             k_h=4, k_w=4, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", spectural_normed=False, with_w=False):

    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]], dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        if spectural_normed:
            deconv = tf.nn.conv2d_transpose(input_, spectral_norm(w, iteration=1), output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])
        else:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], tf.float32, initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w

        else:
            return deconv

def avgpool2d(x, k=2):
    return tf.nn.avg_pool(x, ksize=[1, k, k ,1], strides=[1, k, k, 1], padding='SAME')

def upscale(x, scale):
    _, h, w, _ = get_conv_shape(x)
    return resize_nearest_neighbor(x, (h * scale, w * scale))

def get_conv_shape(tensor):
    shape = int_shape(tensor)
    return shape

def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]

def resize_nearest_neighbor(x, new_size):
    x = tf.image.resize_nearest_neighbor(x, new_size)
    return x

def fully_connect(input_, output_size, scope=None, stddev=0.02, spectural_normed=False, iter=1,
                  bias_start=0.0, with_w=False):

  shape = input_.get_shape().as_list()
  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.truncated_normal_initializer(stddev=stddev))
    bias = tf.get_variable("bias", [output_size], tf.float32,
      initializer=tf.constant_initializer(bias_start))

    if spectural_normed:
        mul = tf.matmul(input_, spectral_norm(matrix, iteration=iter))
    else:
        mul = tf.matmul(input_, matrix)
    if with_w:
      return mul + bias, matrix
    else:
      return mul + bias

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat(3 , [x , y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2] , y_shapes[3]])])

def batch_normal(input, reuse=False, is_training=True,  scope="scope"):
    return batch_norm(input , epsilon=1e-5, decay=0.9 , scale=True, scope=scope ,
                      is_training=is_training, reuse=reuse, fused=True, updates_collections=None)

def Residual_G(x, output_dims=256, kernel=3, strides=1, spectural_normed=False, up_sampling=False, residual_name='resi'):

    with tf.variable_scope('residual_{}'.format(residual_name)):

        def short_cut(x):
            x = upscale(x, 2) if up_sampling else x
            return x

        x = tf.nn.relu(batch_normal(x, scope='bn1'))
        conv1 = upscale(x, 2) if up_sampling else x
        conv1 = conv2d(conv1, output_dim=output_dims, spectural_normed=spectural_normed,
                                kernel=kernel, stride=strides, name="conv1")
        conv2 = conv2d(tf.nn.relu(batch_normal(conv1, scope='bn2')), output_dim=output_dims, spectural_normed=spectural_normed,
                       kernel=kernel, stride=strides, name="conv2")
        resi = short_cut(x) + conv2
        return resi

def Residual_D(x, output_dims=256, kernel=3, strides=1, spectural_normed=True, down_sampling=False, residual_name='resi', is_start=False):

    with tf.variable_scope('residual_{}'.format(residual_name)):

        def short_cut(x):
            x = avgpool2d(x, 2) if down_sampling else x
            x = conv2d(x, output_dim=output_dims, spectural_normed=spectural_normed, kernel=1,
                       stride=1, name='conv')
            return x

        if is_start:
            conv1 = tf.nn.relu(conv2d(x, output_dim=output_dims, spectural_normed=spectural_normed, kernel=kernel,
                           stride=strides, name="conv1"))
            conv2 = tf.nn.relu(conv2d(conv1, output_dim=output_dims, spectural_normed=spectural_normed, kernel=kernel,
                           stride=strides, name="conv2"))
            conv2 = avgpool2d(conv2, 2) if down_sampling else conv2
        else:
            conv1 = conv2d(tf.nn.relu(x), output_dim=output_dims, spectural_normed=spectural_normed, kernel=kernel, stride=strides, name="conv1")
            conv2 = conv2d(tf.nn.relu(conv1), output_dim=output_dims, spectural_normed=spectural_normed, kernel=kernel, stride=strides, name="conv2")
            conv2 = avgpool2d(conv2, 2) if down_sampling else conv2

        resi = short_cut(x) + conv2
        return resi

NO_OPS = 'NO_OPS'

def _l2normalize(v, eps=1e-12):
  return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

def spectral_norm(W, iteration=1, collections=None, return_norm=False, name='sn'):

    shape = W.get_shape().as_list()
    if len(shape) == 1:
        sigma = tf.reduce_max(tf.abs(W))
    else:
        if len(shape) == 4:
            _W = tf.reshape(W, (-1, shape[3]))
            shape = (shape[0] * shape[1] * shape[2], shape[3])
        else:
            _W = W
        u = tf.get_variable(
            name=name + "_u",
            shape=(_W.shape.as_list()[-1], shape[0]),
            initializer=tf.random_normal_initializer,
            collections=collections,
            trainable=False
        )
        _u = u
        for _ in range(iteration):

            _v = tf.nn.l2_normalize(tf.matmul(_u, _W), 1)
            _u = tf.nn.l2_normalize(tf.matmul(_v, tf.transpose(_W)), 1)

        _u = tf.stop_gradient(_u)
        _v = tf.stop_gradient(_v)
        sigma = tf.reduce_mean(tf.reduce_sum(_u * tf.transpose(tf.matmul(_W, tf.transpose(_v))), 1))
        update_u_op = tf.assign(u, _u)
        with tf.control_dependencies([update_u_op]):
            sigma = tf.identity(sigma)

    if return_norm:
        return W / sigma, sigma
    else:
        return W / sigma


def variable_summaries(var, name):

    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope("summaries" + name):

      # mean = tf.reduce_mean(var)
      # tf.summary.scalar('mean', mean)
      # with tf.name_scope('stddev'):
      #   stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      # tf.summary.scalar('stddev', stddev)
      # tf.summary.scalar('max', tf.reduce_max(var))
      # tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)