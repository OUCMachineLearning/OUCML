import sugartensor as tf

#
# hyper parameters
#

z_dim = 50        # noise dimension
margin = 1        # max-margin for hinge loss
pt_weight = 0.1   # PT regularizer's weight


#
# create generator
#

def generator(x):

    reuse = len([t for t in tf.global_variables() if t.name.startswith('generator')]) > 0
    with tf.sg_context(name='generator', size=4, stride=2, act='leaky_relu', bn=True, reuse=reuse):

        # generator network
        res = (x.sg_dense(dim=1024, name='fc_1')
               .sg_dense(dim=7*7*128, name='fc_2')
               .sg_reshape(shape=(-1, 7, 7, 128))
               .sg_upconv(dim=64, name='conv_1')
               .sg_upconv(dim=1, act='sigmoid', bn=False, name='conv_2'))
    return res


#
# create discriminator
#

def discriminator(x):

    reuse = len([t for t in tf.global_variables() if t.name.startswith('discriminator')]) > 0
    with tf.sg_context(name='discriminator', size=4, stride=2, act='leaky_relu', bn=True, reuse=reuse):
        res = (x.sg_conv(dim=64, name='conv_1')
                .sg_conv(dim=128, name='conv_2')
                .sg_upconv(dim=64, name='conv_3')
                .sg_upconv(dim=1, act='linear', name='conv_4'))

    return res
