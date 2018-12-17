import sugartensor as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import *


__author__ = 'namju.kim@kakaobrain.com'


# set log level to debug
tf.sg_verbosity(10)

#
# hyper parameters
#

batch_size = 100


# random uniform seed
z = tf.random_uniform((batch_size, z_dim))

# generator
gen = generator(z)

#
# draw samples
#

with tf.Session() as sess:

    tf.sg_init(sess)

    # restore parameters
    tf.sg_restore(sess, tf.train.latest_checkpoint('asset/train'), category='generator')

    # run generator
    imgs = sess.run(gen.sg_squeeze())

    # plot result
    _, ax = plt.subplots(10, 10, sharex=True, sharey=True)
    for i in range(10):
        for j in range(10):
            ax[i][j].imshow(imgs[i * 10 + j], 'gray')
            ax[i][j].set_axis_off()
    plt.savefig('asset/train/sample.png', dpi=600)
    tf.sg_info('Sample image saved to "asset/train/sample.png"')
    plt.close()
